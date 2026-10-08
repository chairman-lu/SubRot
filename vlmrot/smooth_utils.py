from collections import defaultdict
import torch
import torch.nn as nn
import logging, functools
import sys, pdb, copy, os
from vlmrot import utils, quant_utils, model_utils, vis_utils
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.bloom.modeling_bloom import BloomGelu
from transformers.activations import GELUActivation

def _profile(dataloader, model, image_processor, args, sample_module=0, out_mask=False):
    inps = []
    layer_kwargs = []
    inputs = []
    vision_masks = []
    caption_masks = []
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, *args, **kwargs):
            inps.append(inp)
            kwargs['use_cache'] = False
            layer_kwargs.append(kwargs)
            raise ValueError
    
    model_utils.get_layers(model)[sample_module] = Catcher(model_utils.get_layers(model)[sample_module])

    model = model.cuda()

    for batch in dataloader:
        if out_mask:
            input = model_utils.build_loss_inputs(batch, image_processor, model)
        else:
            input = model_utils.message_to_prompt(batch, image_processor, type(model))

        if out_mask:
            vision_mask, caption_mask = model_utils.get_input_masks(model, input)
            vision_masks.append(vision_mask.squeeze(0).cpu())
            caption_masks.append(caption_mask.squeeze(0).cpu())
            inputs.append(input)
        try:
            if type(model) == model_utils.LLAVA_OV_HF:
                model.generate(**input, max_new_tokens=1, use_cache=True, 
                               eos_token_id=image_processor.tokenizer.pad_token_id,
                               pad_token_id=image_processor.tokenizer.pad_token_id)
            else:
                model.generate(**input, max_new_tokens=1, use_cache=True)
        except ValueError:
            if args.profile and out_mask and len(inps) == 5:
                save_heat_map = False
                save_value_histogram = True
                tensor = inps[-1].clone().squeeze(0).cpu()
                vis_mask = vision_mask.clone().squeeze(0).cpu()
                cap_mask = caption_mask.clone().squeeze(0).cpu()
                if save_heat_map:
                    save_path = os.path.join(args.profile_dir, 
                                            args.model.split('/')[-1], 
                                            f'llm_layer{sample_module}_input_visualization.pdf')
                    channel_len = min(tensor.shape[-1], 128)
                    channel_idx = torch.linspace(0, tensor.shape[-1] - 1, steps=channel_len).long()
                    tensor = tensor[:, channel_idx].abs()
                    vis_utils.plot_heatmap_with_mask(
                        tensor, vis_mask, cap_mask, 
                        title=f'Layer {sample_module} input visualization', 
                        save_path=save_path)
                elif save_value_histogram:
                    save_path = os.path.join(args.profile_dir, 
                                             args.model.split('/')[-1],
                                             f'llm_layer{sample_module}_value_histogram.pdf')
                    tensor = tensor.abs()
                    vis_utils.plot_histogram_with_mask(
                        tensor, vis_mask, cap_mask,
                        title=f'Layer {sample_module} input value histogram',
                        save_path=save_path
                    )

    model_utils.get_layers(model)[sample_module] = model_utils.get_layers(model)[sample_module].module

    model = model.cpu()
    if out_mask:
        return inps, layer_kwargs, inputs, vision_masks, caption_masks
    else:
        return inps, layer_kwargs


@torch.no_grad()
def get_smoothquant_act_scales(dataloader, model, image_processor, args):
    model.eval()
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.reshape(-1, hidden_dim).abs().detach()
        coming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.maximum(act_scales[name], coming_max)
        else:
            act_scales[name] = coming_max

    def stat_input_hook(module, inputs, output, name):
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        stat_tensor(name, x)

    hooks = []
    lm = model.language_model
    for name, module in lm.named_modules():
        if isinstance(module, nn.Linear) and "lm_head" not in name:
            hooks.append(module.register_forward_hook(
                functools.partial(stat_input_hook, name=name)
            ))

    model = utils.dispatch_model_to_cuda(model)
    try:
        logging.info(f'Smoothquant running...')
        for batch in tqdm(dataloader):
            inputs = model_utils.message_to_prompt(batch, image_processor, type(model))
            if type(model) == model_utils.LLAVA_OV_HF:
                model.generate(**inputs, max_new_tokens=1, use_cache=True,
                               eos_token_id=image_processor.tokenizer.pad_token_id,
                               pad_token_id=image_processor.tokenizer.pad_token_id)
            else:
                model.generate(**inputs, max_new_tokens=1, use_cache=True)
            utils.cleanup_memory(verbos=False)
    finally:
        for hook in hooks:
            hook.remove()
        model = utils.move_model_to_cpu(model)

    return act_scales

class GradCacheHook:
    def __init__(self, 
                 catch_grad_mode='topk', 
                 grad_stat_mode='per_modal', 
                 threshold=0.5, 
                 vis_masks=None, 
                 cap_masks=None,
                 n_samples=None) -> None:

        self.catch_grad_mode = catch_grad_mode
        self.grad_stat_mode = grad_stat_mode
        self.threshold = threshold
        self.vis_masks = vis_masks
        self.cap_masks = cap_masks
        self.n_samples = n_samples

        self.hooks = []
        self.steps = defaultdict(lambda: 0)     # name: steps(0 ~ (nsamples - 1))
        self.cum_grad = defaultdict(lambda: 0)
        self.important_tokens = {}              # key: name, value: token_ids
        self.grad_dict = {}

    def topk(self, name):
        grad = self.cum_grad[name].mean(dim=-1).squeeze(0)  # N
        values, indices = torch.topk(grad, k=int(grad.shape[0] * self.threshold))
        self.important_tokens[name] = indices
        logging.info(f'{name}: TopK selected {len(indices)} tokens in {grad.shape[0]}')

    def topp(self, name, grad, temperature=1.5):
        grad = grad.mean(dim=-1).squeeze(0).float() # N
        z = (grad - grad.mean()) / (grad.std() + 1e-6)
        importance = torch.sigmoid(z / temperature)
        self.important_tokens.setdefault(name, []).append(importance)
    
    @staticmethod
    def pad(tensor, length, pad_value):
        assert len(tensor.shape) == 3   # BNC
        if tensor.shape[1] == length:
            return tensor
        new_tensor = torch.full(
            (tensor.shape[0], length, tensor.shape[2]),
            pad_value, 
            dtype=tensor.dtype,
            device=tensor.device
        )
        new_tensor[:, :tensor.shape[1], :] = tensor
        return new_tensor

    @torch.no_grad()
    def cache_grad_hook(self, module, inp, out, name):
        if name not in self.grad_dict:
            self.grad_dict[name] = {"vis_grad": [], "cap_grad": []}
        step = self.steps[name]
        grad = out[0].detach().float().abs()    # BNC
        B, N, C = grad.shape
        assert B == 1

        # pad_grad = grad
        # if type(self.cum_grad[name]) != int:
        #     # pdb.Pdb(stdin=sys.__stdin__, stdout=sys.__stdout__).set_trace()
        #     max_len = max(self.cum_grad[name].shape[1], grad.shape[1])
        #     self.cum_grad[name] = self.pad(self.cum_grad[name], max_len, 0)
        #     pad_grad = self.pad(grad, max_len, 0)
        # self.cum_grad[name] += pad_grad

        if self.catch_grad_mode == 'topp':
            self.topp(name, grad)
        else:
            raise NotImplementedError(f'{self.catch_grad_mode} is not supported yet!')

        # if self.steps[name] == self.n_samples - 1:
        #     if self.catch_grad_mode == 'topk':
        #         self.topk(name)
        vis_grad = grad[0][self.vis_masks[step]].mean()
        cap_grad = grad[0][self.cap_masks[step]].mean()
        self.grad_dict[name]["vis_grad"].append(vis_grad.cpu())
        self.grad_dict[name]["cap_grad"].append(cap_grad.cpu())

        self.steps[name] += 1

    def get_avg_grad_dict(self):
        avg_grad_dict = {}

        for name, grad_values in self.grad_dict.items():
            mean_vis = torch.mean(torch.stack(grad_values["vis_grad"]))
            mean_cap = torch.mean(torch.stack(grad_values["cap_grad"]))

            avg_grad_dict[name] = {
                "vis_avg_grad": mean_vis.item(),
                "cap_avg_grad": mean_cap.item()
            }

        return avg_grad_dict

    def register_hooks(self, layers):
        register_names = [
            "wo", "w2", "down_proj", "o_proj", "v_proj", "gate_proj", "up_proj", "w1", "w3"
            ]
        for n, m in layers.named_modules():
            if isinstance(m, nn.Linear) and any([_ in n for _ in register_names]):
                self.hooks.append(
                    m.register_full_backward_hook(
                        functools.partial(self.cache_grad_hook, name=f'layers.{n}')
                    )
                )
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

@torch.no_grad()
def batch1_infer(layer, inps: list, layer_kwargs: list):
    # assert len(inps) == len(layer_kwargs)
    outputs = []
    for i in range(len(inps)):
        if len(layer_kwargs) == 0:
            output = layer(inps[i])
        else:
            output = layer(inps[i], **layer_kwargs[i])
        if isinstance(output, tuple):
            output = output[0]
        outputs.append(output.detach().cpu())
        del output
    utils.cleanup_memory(verbos=False)
    return outputs

@torch.no_grad()
def get_act_scale_by_importance(x, important_tokens):
    # x.shape: BNC, important_tokens.shape: K
    important_tokens = important_tokens.to(x[0].device)
    important_tokens = torch.sort(important_tokens)
    device, _ = utils.get_free_gpu()
    x = [tmp[:, important_tokens, :].to(f'cuda:{device}') for tmp in x]
    return x.abs().view(-1, x.shape[-1]).max(0)[0]

@torch.no_grad()
def get_act_scale(x):
    x = [tmp.squeeze(0) for tmp in x]
    x = pad_sequence(x, batch_first=True, padding_value=0)
    x_max = x.abs().view(-1, x.shape[-1]).mean(0)
    return x_max


@torch.no_grad()
def auto_scale_block_wa(module, module_kwargs, w_bit, a_bit, q_config, 
                        input_feat, ans_mask, vis_mask, reweight_ratio_dict, 
                        loss_mode="mae", device=None, important_tokens=None, layer_id=None):
    if "use_cache" in module_kwargs:
        module_kwargs.pop("use_cache")
        
    def _auto_get_scale_wa(prev_op, layers, layers_name, inp,
                            w_bit, a_bit, ans_mask, vis_mask, loss_mode,
                            reweight_ratio=None, module2inspect=None, device=None, kwargs={}, important_tokens=None):
        if module2inspect is None:
            assert len(layers) == 1
            if device is None:
                module2inspect = layers[0].cuda()
            else:
                module2inspect = layers[0].to(device)
        scales = _search_module_scale_wa(module2inspect, layers, layers_name, inp,
                                            w_bit, a_bit, ans_mask, vis_mask, loss_mode,
                                            reweight_ratio, kwargs, important_tokens=important_tokens)
        scales = scales.detach().cpu()
        utils.cleanup_memory(verbos=False)
        return (
            utils.get_op_name(module, prev_op),
            tuple([utils.get_op_name(module, m) for m in layers]),
            scales,
        )

    scales_list = []  # return the searched scales

    if module.__class__.__name__ in ["Qwen2DecoderLayer", "Qwen2VLDecoderLayer"]:
        # attention_input
        scales_list.append(
            _auto_get_scale_wa(
                prev_op=module.input_layernorm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                layers_name=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                ],
                inp=input_feat["self_attn.q_proj"],
                w_bit=w_bit,
                a_bit=a_bit,
                ans_mask=ans_mask,
                vis_mask=vis_mask,
                loss_mode=loss_mode,
                important_tokens=important_tokens["self_attn.v_proj"] if important_tokens else important_tokens,
                reweight_ratio=reweight_ratio_dict["attn"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )

        # attention output
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            scales_list.append(
                _auto_get_scale_wa(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    layers_name=["o_proj"],
                    inp=input_feat["self_attn.o_proj"],
                    w_bit=w_bit,
                    a_bit=a_bit,
                    ans_mask=ans_mask,
                    vis_mask=vis_mask,
                    loss_mode=loss_mode,
                    important_tokens=important_tokens["self_attn.o_proj"] if important_tokens else important_tokens,
                    reweight_ratio=reweight_ratio_dict["attn"],
                    device=device,
                )
            )

        # fc1
        scales_list.append(
            _auto_get_scale_wa(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                layers_name=["gate_proj", "up_proj"],
                inp=input_feat["mlp.gate_proj"],
                w_bit=w_bit,
                a_bit=a_bit,
                ans_mask=ans_mask,
                vis_mask=vis_mask,
                loss_mode=loss_mode,
                important_tokens=important_tokens["mlp.gate_proj"] if important_tokens else important_tokens,
                reweight_ratio=reweight_ratio_dict["mlp"],
                module2inspect=module.mlp,
            )
        )

        # fc2
        scales_list.append(
            _auto_get_scale_wa(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                layers_name=["down_proj"],
                inp=input_feat["mlp.down_proj"],
                w_bit=w_bit,
                a_bit=a_bit,
                ans_mask=ans_mask,
                vis_mask=vis_mask,
                loss_mode=loss_mode,
                important_tokens=important_tokens["mlp.down_proj"] if important_tokens else important_tokens,
                reweight_ratio=reweight_ratio_dict["mlp"],
                device=device
            )
        )
    else:
        raise NotImplementedError(f"{type(module)} not supported yet!")
    return scales_list


@torch.no_grad()
def _search_module_scale_wa(block, linears2scale: list, layers_name, x,
                            w_bit, a_bit, ans_mask, vis_mask, loss_mode,
                            reweight_ratio=None, kwargs={}, important_tokens=None):
    utils.move_list_tensor(x, next(block.parameters()).device)
    org_out = batch1_infer(block, x, kwargs)
    # x_max = get_act_scale_by_importance(x, important_tokens)
    x_max = get_act_scale(x)

    best_error = float("inf")
    best_ratio = -1
    best_scales = None
    n_grid = 20
    history = []

    org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
    for ratio in range(n_grid):
        # try:
        ratio = ratio * 1 / n_grid
        scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
        scales = scales / (scales.max() * scales.min()).sqrt()
        if torch.isinf(scales).any() or torch.isnan(scales).any():
            print('inf', end=' ')
            continue

        if isinstance(block, nn.Linear):
            new_block = None
        overflow = False
        for fc, fc_name in zip(linears2scale, layers_name):
            fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
            if torch.isinf(fc.weight.data).any() or torch.isnan(fc.weight.data).any():
                overflow = True
                break
            new_fc = copy.deepcopy(fc)
            new_fc = quant_utils.add_actquant(new_fc)
            new_fc.quantizer.configure(bits=a_bit, groupsize=-1, sym=True, clip_ratio=0.9)  # 这里写死，且只量化激活，不量化权重
            # new_fc = WALinear.from_float(fc, weight_quant="per_channel", act_quant="per_token", w_bit=w_bit, a_bit=a_bit)
            
            if isinstance(block, nn.Linear):
                new_block = copy.deepcopy(new_fc)                   
            else:
                setattr(block, fc_name, new_fc)
                
            del new_fc
            utils.cleanup_memory(verbos=False)
        if overflow:
            print('inf', end=' ')
            continue
        
        x_scale = [(tmp / scales.view(1, 1, -1)) for tmp in x]

        if isinstance(block, nn.Linear):
            out = batch1_infer(new_block, x_scale, kwargs)
        else:
            out = batch1_infer(block, x_scale, kwargs)
        del x_scale
        utils.cleanup_memory(verbos=False)

        total_loss = []
        assert len(out) == len(org_out) == len(ans_mask) == len(vis_mask)
        # utils.cleanup_memory(verbos=False)
        if loss_mode == "mse":
            if ans_mask is not None and vis_mask is not None:
                ans_mask_expand = ans_mask.unsqueeze(-1).expand_as(out)
                vis_mask_expand = vis_mask.unsqueeze(-1).expand_as(out).cuda()
                masked_diff_ans = ((org_out - out).float().pow(2) * ans_mask_expand)
                masked_diff_vis = ((org_out - out).float().pow(2) * vis_mask_expand)
                if reweight_ratio is not None:
                    loss = masked_diff_ans.sum() / ans_mask_expand.sum() + reweight_ratio * (masked_diff_vis.sum() / vis_mask_expand.sum())
                else:
                    loss = (
                        (org_out - out).float().pow(2).mean().item()
                    ) 
            elif ans_mask is not None and vis_mask is None:
                ans_mask_expand = ans_mask.unsqueeze(-1).expand_as(out)
                masked_diff = ((org_out - out).float().pow(2) * ans_mask_expand)
                loss = masked_diff.sum() / ans_mask_expand.sum() 
            else:
                loss = (
                    (org_out - out).float().pow(2).mean().item()
                )  # float prevents overflow
        elif loss_mode == "mae":
            if len(ans_mask) > 0 and len(vis_mask) > 0:
                for i in range(len(out)):
                    tmp_out = out[i]
                    tmp_org_out = org_out[i]
                    tmp_ans_mask = ans_mask[i]
                    tmp_vis_mask = vis_mask[i]
                    assert tmp_out.shape == tmp_org_out.shape
                    assert tmp_out.shape[1] == tmp_ans_mask.shape[0] == tmp_vis_mask.shape[0]
                    assert len(tmp_vis_mask.shape) == len(tmp_vis_mask.shape) == 1

                    diff = (tmp_org_out - tmp_out).float().abs()
                    ans_mask_expand = tmp_ans_mask.unsqueeze(0).unsqueeze(-1).expand_as(tmp_out)
                    vis_mask_expand = tmp_vis_mask.unsqueeze(0).unsqueeze(-1).expand_as(tmp_out)
                    masked_diff_ans = diff * ans_mask_expand
                    masked_diff_vis = diff * vis_mask_expand
                    if reweight_ratio is not None:
                        loss = (masked_diff_ans.sum() + reweight_ratio * masked_diff_vis.sum()) / (ans_mask_expand.sum() + vis_mask_expand.sum())
                    else:
                        loss = (tmp_org_out - tmp_out).float().abs().mean().item()
                    total_loss.append(loss)
                loss = torch.stack(total_loss).mean()
            elif ans_mask is not None and vis_mask is None:
                ans_mask_expand = ans_mask.unsqueeze(-1).expand_as(out)
                masked_diff = ((org_out - out).float().abs() * ans_mask_expand)
                loss = masked_diff.sum() / ans_mask_expand.sum() 
            else:
                loss = (
                    (org_out - out).float().abs().mean().item()
                )  # float prevents overflow
        elif loss_mode == 'mae_by_token_importance':
            assert len(ans_mask) > 0 and len(vis_mask) > 0
            for i in range(len(out)):
                tmp_out = out[i]
                tmp_org_out = org_out[i]
                tmp_ans_mask = ans_mask[i]
                tmp_vis_mask = vis_mask[i]
                tmp_importance = important_tokens[i]
                assert tmp_out.shape == tmp_org_out.shape
                assert tmp_out.shape[1] == tmp_ans_mask.shape[0] == tmp_vis_mask.shape[0] == tmp_importance.shape[0]
                assert len(tmp_vis_mask.shape) == len(tmp_vis_mask.shape) == len(tmp_importance.shape) == 1

                diff = (tmp_org_out - tmp_out).float().abs().to(tmp_importance.device)
                tmp_importance = tmp_importance.unsqueeze(0).unsqueeze(-1).expand_as(diff)
                loss = (diff * tmp_importance).mean()
                total_loss.append(loss)
                del tmp_out, tmp_org_out, tmp_ans_mask, tmp_vis_mask, tmp_importance, diff, loss
            # utils.cleanup_memory(verbos=False)
            loss = torch.stack(total_loss).mean().item()

        history.append(loss)
        is_best = loss < best_error
        if is_best:
            best_error = loss
            best_ratio = ratio
            best_scales = scales.detach().cpu().clone()

        # restore the block 将量化的linear重新替换为原本的linear用于下次网格搜索
        for fc, fc_name in zip(linears2scale, layers_name):
            if isinstance(block, nn.Linear):
                continue
            else:
                setattr(block, fc_name, fc)
        
        if isinstance(block, nn.Linear):
            del new_block 
        del out, scales, loss, total_loss
        utils.cleanup_memory(verbos=False)
        block.load_state_dict(org_sd)   # 将乘了scale的权重重新初始化为原本的权重
        # except Exception as e:
        #     print(f'layer name:{layers_name}, ratio:{ratio}')
        #     print(e)
        #     if 'x_scale' in locals():
        #         del x_scale
        #     if 'diff' in locals():
        #         del diff
        #     utils.cleanup_memory(verbos=False)
        #     pass

    if best_ratio == -1:
        print(history)
        raise Exception
    del x, x_max
    del block
    del org_out
    del ans_mask
    del vis_mask
    del important_tokens
    utils.cleanup_memory(verbos=False)
    print(f'layer: {layers_name}, best_ratio:{best_ratio}, best_error:{best_error:.3f}')
    best_scales = best_scales.view(-1)

    assert torch.isnan(best_scales).sum() == 0, best_scales
    return best_scales.detach()

def apply_scale(module, scales_list, input_feat_dict=None, type_model=None):
    if type_model in [model_utils.QWEN2_VL, model_utils.LLAVA_OV_HF]:
        for i in range(len(scales_list)):
            scales_list[i] = list(scales_list[i])
            scales_list[i][0] = scales_list[i][0].replace('model', 'language_model')
            scales_list[i][1] = list(scales_list[i][1])
            for j in range(len(scales_list[i][1])):
                scales_list[i][1][j] = scales_list[i][1][j].replace('model', 'language_model')
            scales_list[i][1] = tuple(scales_list[i][1])
            scales_list[i] = tuple(scales_list[i])

    for prev_op_name, layer_names, scales in scales_list:
        prev_op = utils.get_op_by_name(module, prev_op_name)
        layers = [utils.get_op_by_name(module, name) for name in layer_names]
        if isinstance(prev_op, nn.Linear):
            assert len(layers) == 1
            scale_fc_fc(prev_op, layers[0], scales)
        elif isinstance(prev_op, (nn.LayerNorm, LlamaRMSNorm)) or prev_op.__class__.__name__ == "InternLM2RMSNorm" or prev_op.__class__.__name__ == "Qwen2RMSNorm":
            scale_ln_fcs(prev_op, layers, scales)
        elif isinstance(prev_op, (nn.GELU, BloomGelu, GELUActivation)):
            new_module = ScaledActivation(prev_op, scales)
            utils.set_op_by_name(module, prev_op_name, new_module)
            scale_gelu_fc(prev_op, layers[0], scales)
        else:
            raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")

        # apply the scaling to input feat if given
        if input_feat_dict is not None:
            for layer_name in layer_names:
                inp = input_feat_dict[layer_name]
                inp.div_(scales.view(1, -1).to(inp.device))

@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)

    ln.weight.div_(scales)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0

@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)
    # assert fc1.out_features == fc2.in_features

    scales = scales.to(fc1.weight.device)

    # fc1.weight.div_(scales.view(-1, 1))
    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0

@torch.no_grad()
def scale_gelu_fc(gelu, fc, scales):
    assert isinstance(gelu, (nn.GELU, BloomGelu, GELUActivation))
    assert isinstance(fc, nn.Linear)

    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))

    for p in fc.parameters():
        assert torch.isnan(p).sum() == 0

class ScaledActivation(nn.Module):
    def __init__(self, module, scales):
        super().__init__()
        self.act = module
        self.scales = nn.Parameter(scales.data)

    def forward(self, x):
        return self.act(x) / self.scales.view(1, 1, -1).to(x.device)
    
@torch.no_grad()
def smooth_lm(model, scales, alpha=0.5):
    from transformers.models.opt.modeling_opt import OPTDecoderLayer
    from transformers.models.bloom.modeling_bloom import BloomBlock
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    from transformers.models.falcon.modeling_falcon import FalconDecoderLayer
    from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
    from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer
    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            attn_ln = module.self_attn_layer_norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]
            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.final_layer_norm
            fc1 = module.fc1
            fc1_input_scales = scales[name + ".fc1"]
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
        elif isinstance(module, BloomBlock):
            attn_ln = module.input_layernorm
            qkv = module.self_attention.query_key_value
            qkv_input_scales = scales[name + ".self_attention.query_key_value"]
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.post_attention_layernorm
            fc1 = module.mlp.dense_h_to_4h
            fc1_input_scales = scales[name + ".mlp.dense_h_to_4h"]
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
        elif isinstance(module, FalconDecoderLayer):
            qkv = module.self_attention.query_key_value
            qkv_input_scales = scales[name + ".self_attention.query_key_value"]
            fc1_input_scales = scales[name + ".mlp.dense_h_to_4h"]
            fc1 = module.mlp.dense_h_to_4h

            if (
                not module.config.new_decoder_architecture
                and module.config.parallel_attn
            ):
                attn_ln = module.input_layernorm
                smooth_ln_fcs(attn_ln, [qkv, fc1], qkv_input_scales, alpha)
            else:
                attn_ln = (
                    module.ln_attn
                    if module.config.new_decoder_architecture
                    else module.input_layernorm
                )
                ffn_ln = (
                    module.ln_mlp
                    if module.config.new_decoder_architecture
                    else module.post_attention_layernorm
                )
                smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)
                smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
        elif isinstance(module, (LlamaDecoderLayer, MistralDecoderLayer)):
            attn_ln = module.input_layernorm  # attention forward norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]

            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.post_attention_layernorm  # feed forward norm
            fcs = [module.mlp.gate_proj, module.mlp.up_proj]
            fcs_input_scales = scales[name + ".mlp.gate_proj"]

            smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)
        elif isinstance(module, MixtralDecoderLayer):
            attn_ln = module.input_layernorm  # attention forward norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]

            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.post_attention_layernorm  # feed forward norm
            fcs = [module.block_sparse_moe.gate]
            for expert in module.block_sparse_moe.experts:
                fcs.append(expert.w1)
                fcs.append(expert.w3)
            fcs_input_scales = scales[name + ".block_sparse_moe.gate"]

            smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)
        elif module.__class__.__name__ == "Qwen2DecoderLayer":
            attn_ln = module.input_layernorm  # attention forward norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]

            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.post_attention_layernorm  # feed forward norm
            fcs = [module.mlp.gate_proj, module.mlp.up_proj]
            fcs_input_scales = scales[name + ".mlp.gate_proj"]

            smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)
        elif module.__class__.__name__ == "InternLM2DecoderLayer":
            attn_ln = module.attention_norm  # attention forward norm
            qkv = module.attention.wqkv
            qkv_input_scales = scales[name + ".attention.wqkv"]
            smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.ffn_norm  # feed forward norm
            fcs = [module.feed_forward.w1, module.feed_forward.w3]
            fcs_input_scales = scales[name + ".feed_forward.w1"]

            smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)
        elif module.__class__.__name__ == "Qwen2VLDecoderLayer":
            attn_ln = module.input_layernorm  # attention forward norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]
            name = name.replace('language_model', 'model')
            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.post_attention_layernorm  # feed forward norm
            fcs = [module.mlp.gate_proj, module.mlp.up_proj]
            fcs_input_scales = scales[name + ".mlp.gate_proj"]

            smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)

@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, nn.LayerNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    ln.weight.div_(scales)
    ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

@torch.no_grad()
def smooth_ln_fcs_llama_like(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    # assert isinstance(ln, (LlamaRMSNorm, MistralRMSNorm, MixtralRMSNorm))
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    ln.weight.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))
