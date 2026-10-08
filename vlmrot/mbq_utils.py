# MIT License
# 
# Copyright (c) 2024
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 

"""MBQ port from MBQ/qmllm/methods/mbq (MIT; license reproduced below).

Uses the original 20-point scale search and gradient-balanced reconstruction
loss, with per-sample replay for variable image/text sequence lengths.
"""
import copy
import logging
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def validate_mbq_args(args):
    if not getattr(args, 'mbq', False):
        return
    conflicts = [n for n in ('smoothquant', 'quarot', 'spinquant', 'split_rotate',
                             'flatquant', 'subspace', 'parrot', 'online_rotate')
                 if getattr(args, n, False)]
    if conflicts:
        raise ValueError('MBQ cannot be combined with: ' + ', '.join(conflicts))
    if args.quant_method != 'rtn' or args.k_bits < 16 or args.v_bits < 16:
        raise ValueError('MBQ requires quant_method=rtn and k_bits=v_bits=16')
    if any(not 2 <= getattr(args, n) <= 16 for n in ('w_bits', 'a_bits')):
        raise ValueError('MBQ supports bit widths from 2 to 16')
    if args.nsamples < 1:
        raise ValueError('MBQ requires nsamples > 0')
    if args.mbq_loss not in ('mae', 'mse'):
        raise ValueError('MBQ loss must be mae or mse')
    if any(getattr(args, n) == 0 or getattr(args, n) < -1
           for n in ('w_groupsize', 'a_groupsize')):
        raise ValueError('MBQ group sizes must be -1 or positive')
    if args.load_qmodel and not args.save_qmodel_dir:
        raise ValueError('MBQ load_qmodel requires save_qmodel_dir')
    if args.mbq_load_scales and not args.mbq_scale_path:
        raise ValueError('mbq_load_scales requires mbq_scale_path')


def _tree_to(value, device):
    if torch.is_tensor(value):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {k: _tree_to(v, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_tree_to(v, device) for v in value)
    return value


def _output(value):
    return value[0] if isinstance(value, (tuple, list)) else value


def pseudo_quantize(tensor, bits, group=-1, asymmetric=False):
    """MBQ round/dequantize, per output channel or per activation token."""
    if bits >= 16:
        return tensor
    width = tensor.shape[-1] if group <= 0 else group
    if tensor.shape[-1] % width:
        raise ValueError(f'MBQ group size {width} does not divide {tensor.shape[-1]}')
    x = tensor.float().reshape(-1, width)
    if asymmetric:
        lo, hi = 0, 2 ** bits - 1
        scale = (x.amax(-1, keepdim=True) - x.amin(-1, keepdim=True)).clamp(min=1e-5) / hi
        zero = (-torch.round(x.amin(-1, keepdim=True) / scale)).clamp(lo, hi)
    else:
        lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        scale = x.abs().amax(-1, keepdim=True).clamp(min=1e-5) / hi
        zero = 0
    return ((torch.round(x / scale) + zero).clamp(lo, hi) - zero).mul(scale).reshape_as(tensor).to(tensor.dtype)


class MBQLinear(nn.Linear):
    """Static fake-quantized weights and dynamic per-token activations."""
    @classmethod
    def from_float(cls, linear, args, name='', quantize=True):
        result = cls(linear.in_features, linear.out_features, linear.bias is not None,
                     device=linear.weight.device, dtype=linear.weight.dtype)
        down8 = args.int8_down_proj and name.endswith('down_proj')
        result.a_bits = 8 if down8 else args.a_bits
        result.a_group = args.a_groupsize
        result.a_asym = args.a_asym
        with torch.no_grad():
            weight = pseudo_quantize(linear.weight, 8 if down8 else args.w_bits,
                                     args.w_groupsize, args.w_asym) if quantize else linear.weight
            result.weight.copy_(weight)
            if linear.bias is not None:
                result.bias.copy_(linear.bias)
        return result

    def forward(self, x):
        return F.linear(pseudo_quantize(x, self.a_bits, self.a_group, self.a_asym),
                        self.weight, self.bias)


def _groups(layer):
    # Llama and Qwen2 decoder layouts, including Qwen2-VL's decoder.
    groups = [('input_layernorm', ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'), 'self_attn', 'attn'),
              ('post_attention_layernorm', ('mlp.gate_proj', 'mlp.up_proj'), 'mlp', 'mlp'),
              ('mlp.up_proj', ('mlp.down_proj',), 'mlp.down_proj', 'mlp')]
    if layer.self_attn.v_proj.weight.shape == layer.self_attn.o_proj.weight.shape:
        groups.insert(1, ('self_attn.v_proj', ('self_attn.o_proj',), 'self_attn.o_proj', 'attn'))
    return groups


def _get_layers(model):
    # HF moved language decoders under model.language_model in newer releases.
    for path in ('model.language_model.layers', 'model.language_model.model.layers',
                 'language_model.model.layers', 'language_model.layers', 'model.layers'):
        try:
            return model.get_submodule(path)
        except AttributeError:
            continue
    raise ValueError(f'Unsupported MBQ decoder layout: {type(model).__name__}')


@torch.no_grad()
def apply_scales(layer, records):
    for prev_name, names, scale in records:
        prev = layer.get_submodule(prev_name)
        scale = scale.to(prev.weight)
        prev.weight.div_(scale[:, None] if isinstance(prev, nn.Linear) else scale)
        if getattr(prev, 'bias', None) is not None:
            prev.bias.div_(scale)
        for name in names:
            layer.get_submodule(name).weight.mul_(scale[None, :])


def _messages(loader, args):
    if hasattr(loader, 'data') and hasattr(loader.data, 'iloc'):
        from vlmrot.eval_utils import build_prompt_llava
        indices = torch.randperm(len(loader.data), generator=torch.Generator().manual_seed(args.seed))[:args.nsamples]
        for idx in indices.tolist():
            row = loader.data.iloc[idx]
            dataset = args.cal_dataset[0] if isinstance(args.cal_dataset, (tuple, list)) else args.cal_dataset
            message = build_prompt_llava(row, loader, dataset)
            if 'answer' in row and str(row['answer']) not in ('nan', ''):
                message.append(dict(type='textanw', value=str(row['answer'])))
            yield message
    else:
        for idx, message in enumerate(loader):
            if idx >= args.nsamples:
                break
            yield message


def _place_collection_model(model, device):
    """Restore the loader's single-process model placement after main's .cpu()."""
    from vlmrot import utils
    device = torch.device(device)
    device_map = getattr(model, 'hf_device_map', None)
    if device.type == 'cuda' and device_map:
        # Accelerate's CPU/disk weight offload is an inference path; its weight
        # lifetime must not be used for this activation-gradient backward pass.
        if any(str(value) in ('cpu', 'disk', 'meta') for value in device_map.values()):
            raise ValueError('MBQ gradient collection requires a GPU-only hf_device_map; '
                             'CPU/disk weight offload is not supported. Activation CPU offload is supported.')
        utils.dispatch_model_to_cuda(model)
        logging.info('MBQ collection: restored hf_device_map=%s', device_map)
    else:
        model.to(device)
    placement = {}
    for parameter in model.parameters():
        name = str(parameter.device)
        placement[name] = placement.get(name, 0) + parameter.numel() * parameter.element_size()
    logging.info('MBQ collection parameter GiB by device: %s',
                 {name: round(size / 2 ** 30, 3) for name, size in placement.items()})
    # Accelerate's root hook routes inputs to its execution device; choose that
    # device up front to avoid an unnecessary copy through cuda:0.
    execution_device = getattr(getattr(model, '_hf_hook', None), 'execution_device', None)
    if execution_device is not None:
        return torch.device(f'cuda:{execution_device}' if isinstance(execution_device, int) else execution_device)
    return next(model.parameters()).device


def _modality_grad_means(grad, vis, cap):
    """Reduce in token chunks rather than materializing an entire FP32 BNC copy."""
    totals = torch.zeros(2, device=grad.device, dtype=torch.float32)
    vis, cap = vis.to(grad.device), cap.to(grad.device)
    for batch in range(grad.shape[0]):
        for start in range(0, grad.shape[1], 256):
            stop = start + 256
            means = grad[batch, start:stop].float().abs().mean(-1)
            totals[0] += means[vis[batch, start:stop]].sum()
            totals[1] += means[cap[batch, start:stop]].sum()
    return (totals / torch.stack((vis.sum(), cap.sum())).clamp(min=1)).tolist()


def _collect(model, processor, loader, args, layers, device):
    from vlmrot import model_utils, utils
    samples, stats, handles = [], {}, []
    current = {}
    def capture(module, positional, kwargs):
        hidden = positional[0] if positional else kwargs['hidden_states']
        kwargs = {k: v for k, v in kwargs.items() if k not in ('hidden_states', 'past_key_value', 'past_key_values')}
        current['sample'] = (_tree_to(hidden, 'cpu'), _tree_to(kwargs, 'cpu'))
    def grad_hook(name, module, positional, output):
        if not output.requires_grad:
            return
        def save(grad):
            vis, cap = current['masks']
            if grad.shape[:2] != vis.shape:
                raise ValueError('MBQ masks must match expanded decoder tokens; update the model processor')
            values = stats.setdefault(name, [[], []])
            vis_mean, cap_mean = _modality_grad_means(grad, vis, cap)
            values[0].append(vis_mean)
            values[1].append(cap_mean)
        output.register_hook(save)
    flags = [p.requires_grad for p in model.parameters()]
    output = inputs = loss = None
    offload = args.mbq_reweight and getattr(args, 'mbq_offload_activations', True)
    try:
        input_device = _place_collection_model(model, device)
        logging.info('MBQ collection: batch_size=1, activation_cpu_offload=%s, input_device=%s',
                     offload, input_device)
        handles.append(layers[0].register_forward_pre_hook(capture, with_kwargs=True))
        if args.mbq_reweight:
            for p in model.parameters():
                p.requires_grad_(False)
            # Retain activation gradients without allocating parameter gradients.
            def enable_grad(m, positional, kwargs):
                if positional:
                    positional = (positional[0].detach().requires_grad_(True),) + positional[1:]
                else:
                    kwargs['hidden_states'] = kwargs['hidden_states'].detach().requires_grad_(True)
                return positional, kwargs
            handles.append(layers[0].register_forward_pre_hook(enable_grad, with_kwargs=True))
            for i, layer in enumerate(layers):
                for name in ('self_attn.o_proj', 'mlp.down_proj'):
                    key = (i, 'attn' if 'o_proj' in name else 'mlp')
                    handles.append(layer.get_submodule(name).register_forward_hook(
                        lambda m, x, y, key=key: grad_hook(key, m, x, y)))
        for sample_index, message in enumerate(_messages(loader, args)):
            inputs = _tree_to(model_utils.build_loss_inputs(message, processor, model), input_device)
            inputs.pop('use_cache', None)
            vis, _ = model_utils.get_input_masks(model, inputs)
            cap = inputs['labels'].ne(-100) & ~vis
            if args.mbq_reweight and (not vis.any() or not cap.any()):
                raise ValueError('MBQ reweighting requires image tokens and supervised text in each sample')
            current['masks'] = (vis.cpu(), cap.cpu())
            if sample_index == 0 or (sample_index + 1) % 16 == 0:
                logging.info('MBQ gradient sample %d: tokens=%d, visual_tokens=%d',
                             sample_index + 1, inputs['input_ids'].shape[-1], vis.sum().item())
            saved_tensors = torch.autograd.graph.save_on_cpu() if offload else nullcontext()
            with torch.set_grad_enabled(args.mbq_reweight), saved_tensors:
                output = model(**inputs, use_cache=False, output_hidden_states=False,
                               output_attentions=False, return_dict=True)
                if args.mbq_reweight:
                    if output.loss is None or not torch.isfinite(output.loss):
                        raise ValueError('MBQ calibration produced a missing or nonfinite loss')
                    loss = output.loss
                    # In particular, don't retain the full [B, N, vocabulary]
                    # logits through backward just to keep access to the loss.
                    output = None
                    loss.backward()
                    loss = None
            hidden, kwargs = current.pop('sample')
            if hidden.shape[:2] != vis.shape:
                raise ValueError('MBQ needs processor-expanded image tokens')
            samples.append((hidden, kwargs, vis.cpu(), cap.cpu()))
            output = inputs = None
    except torch.cuda.OutOfMemoryError:
        logging.error('MBQ collection OOM: batch_size=1; saved-tensor CPU offload=%s. '
                      'Check the device placement and token counts above; attention/logits '
                      'temporary buffers can still exceed a GPU even with offload.', offload)
        raise
    finally:
        output = inputs = loss = None
        current.clear()
        for h in handles:
            h.remove()
        for p, flag in zip(model.parameters(), flags):
            p.requires_grad_(flag)
        utils.move_model_to_cpu(model)
    if not samples:
        raise ValueError('MBQ calibration dataset is empty')
    ratios = {}
    for key, (vis, cap) in stats.items():
        ratios[key] = (sum(vis) / len(vis)) / max(sum(cap) / len(cap), 1e-12)
        if not torch.isfinite(torch.tensor(ratios[key])):
            raise ValueError(f'Nonfinite MBQ modality gradient ratio at {key}')
    for kind in ('attn', 'mlp'):
        values = [v for (i, k), v in ratios.items() if k == kind]
        if values:
            median = torch.tensor(values).quantile(0.5).item()
            for key in ratios:
                if key[1] == kind:
                    ratios[key] = max(ratios[key], median)
    return samples, ratios


def _loss(reference, output, vis, cap, ratio, mode):
    diff = (reference.float() - output.float()).abs()
    if mode == 'mse':
        diff = diff.square()
    if ratio is None:
        return diff.mean()
    vis, cap = vis.to(diff.device), cap.to(diff.device)
    if mode == 'mse':
        return diff[cap].mean() + ratio * diff[vis].mean()
    return (diff[cap].sum() + ratio * diff[vis].sum()) / ((vis.sum() + cap.sum()) * diff.shape[-1])


@torch.no_grad()
def _search(layer, group, samples, args, ratio, device):
    prev, names, block_name, kind = group
    block = layer.get_submodule(block_name)
    captured = []
    def capture(m, positional, kwargs):
        x = positional[0] if positional else kwargs['hidden_states']
        captured.append((_tree_to(x, 'cpu'), _tree_to(
            {k: v for k, v in kwargs.items() if k != 'hidden_states'}, 'cpu')))
    handle = block.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        for x, kwargs, _, _ in samples:
            layer(x.to(device), **_tree_to(kwargs, device))
    finally:
        handle.remove()
    refs = [_output(block(x.to(device), **_tree_to(kw, device))).cpu() for x, kw in captured]
    count = sum(x.numel() // x.shape[-1] for x, _ in captured)
    mean = sum(x.float().abs().reshape(-1, x.shape[-1]).sum(0) for x, _ in captured) / count
    originals = {name: layer.get_submodule(name) for name in names}
    best, best_loss = None, float('inf')
    try:
        for index in range(20):
            scale = mean.pow(index / 20).clamp(min=1e-4)
            scale /= (scale.min() * scale.max()).sqrt()
            for name, linear in originals.items():
                temp = copy.deepcopy(linear)
                temp.weight.mul_(scale.to(temp.weight)[None, :])
                layer.set_submodule(name, MBQLinear.from_float(temp, args, name))
            candidate = layer.get_submodule(block_name)
            error = 0.
            for (x, kw), ref, (_, _, vis, cap) in zip(captured, refs, samples):
                out = _output(candidate(x.to(device) / scale.to(device=device, dtype=x.dtype), **_tree_to(kw, device)))
                error += _loss(ref.to(device), out, vis, cap, ratio, args.mbq_loss).item()
            if error < best_loss:
                best, best_loss = scale.clone(), error
            for name, linear in originals.items():
                layer.set_submodule(name, linear)
    finally:
        for name, linear in originals.items():
            layer.set_submodule(name, linear)
    if best is None:
        raise ValueError('All MBQ scale candidates produced nonfinite losses')
    return prev, names, best


def _metadata(args):
    return {'format_version': 1, 'model_path': getattr(args, 'model_path', None),
            **{n: getattr(args, n) for n in ('model', 'w_bits', 'a_bits', 'w_groupsize',
            'a_groupsize', 'w_asym', 'a_asym', 'int8_down_proj', 'mbq_loss', 'mbq_reweight')}}


def _load(path, args):
    result = torch.load(path, map_location='cpu', weights_only=False)
    if result.get('metadata') != _metadata(args):
        raise ValueError(f'MBQ cache configuration does not match current arguments: {path}')
    return result


@torch.no_grad()
def run_mbq(model, processor, args):
    from vlmrot import model_utils, data_utils, utils
    validate_mbq_args(args)
    layers = _get_layers(model)
    for layer in layers:
        _groups(layer)  # Validate layout before modifying the model.
        for linear in layer.modules():
            if isinstance(linear, nn.Linear):
                for size in (args.w_groupsize, args.a_groupsize):
                    if size > 0 and linear.in_features % size:
                        raise ValueError(f'MBQ group size {size} does not divide {linear.in_features}')
    checkpoint = None
    if args.save_qmodel_dir:
        checkpoint = Path(args.save_qmodel_dir) / f'{args.model.split("/")[-1]}_MBQ_W{args.w_bits}A{args.a_bits}.pt'
    if args.load_qmodel:
        saved = _load(checkpoint, args)
    else:
        if args.mbq_load_scales:
            saved = _load(args.mbq_scale_path, args)
            if len(saved['scales']) != len(layers):
                raise ValueError('MBQ cache layer count mismatch')
            for layer, records in zip(layers, saved['scales']):
                apply_scales(layer, records)
        else:
            utils.set_seed(args.seed)
            dataset = args.cal_dataset
            if isinstance(dataset, (list, tuple)):
                if len(dataset) != 1:
                    raise ValueError('MBQ expects one calibration dataset')
                dataset = dataset[0]
            loader = data_utils.get_loaders(dataset, args.nsamples, args.seed, model.seqlen,
                                           args.question_file, args.image_base)
            device = utils.get_dev()
            samples, ratios = _collect(model, processor, loader, args, layers, device)
            saved = {'metadata': _metadata(args), 'scales': []}
            for i, layer in enumerate(layers):
                logging.info('MBQ scale search: layer %d/%d', i + 1, len(layers))
                layer.to(device)
                try:
                    # Preserve FP layer outputs for the next layer, as in upstream MBQ.
                    following = [(_output(layer(x.to(device), **_tree_to(kw, device))).cpu(), kw, vis, cap)
                                 for x, kw, vis, cap in samples]
                    records = []
                    for group in _groups(layer):
                        record = _search(layer, group, samples, args, ratios.get((i, group[3])), device)
                        apply_scales(layer, [record])
                        records.append(record)
                    saved['scales'].append(records)
                    samples = following
                finally:
                    layer.cpu()
            if args.mbq_scale_path:
                Path(args.mbq_scale_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(saved, args.mbq_scale_path)
    for layer in layers:
        for name, linear in list(layer.named_modules()):
            if isinstance(linear, nn.Linear):
                layer.set_submodule(name, MBQLinear.from_float(linear, args, name, quantize=not args.load_qmodel))
    if args.load_qmodel:
        model.load_state_dict(saved['model'], strict=True)
    elif checkpoint:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({**saved, 'model': model.state_dict()}, checkpoint)
    model.eval()
    utils.cleanup_memory(verbos=False)
