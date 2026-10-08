import math
import logging

import tqdm
import torch
import torch.nn as nn

from vlmrot import utils, quant_utils, model_utils

__all__ = ['gptq_fwrd', 'rtn_fwrd']

SUPPORTED_VLM_TYPES = (
    model_utils.LLAVA_NEXT_HF,
    model_utils.LLAVA_HF,
    model_utils.LLAVA_OV_HF,
    model_utils.QWEN2_VL,
)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        W = self.layer.weight.data.clone()
        W = W.float()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)])
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if self.dev.type == 'cuda':
            torch.cuda.synchronize(self.dev)

        if actorder:
            Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            import pprint
            pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
            raise ValueError('NaN in weights')

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)
        
        
class _StopCalibrationForward(RuntimeError):
    """Internal control flow used to stop a VLM forward at its first LM layer."""


def _validate_supported_model(model):
    if type(model) not in SUPPORTED_VLM_TYPES:
        raise TypeError(
            'Only the LLM component of LLaVA-NeXT, LLaVA-OneVision and Qwen2-VL '
            f'is supported; got {type(model)}'
        )


def _tree_to(value, device):
    """Move tensors in nested layer kwargs without touching non-tensor metadata."""
    if torch.is_tensor(value):
        return value.detach().to(device)
    if isinstance(value, tuple):
        return tuple(_tree_to(item, device) for item in value)
    if isinstance(value, list):
        return [_tree_to(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _tree_to(item, device) for key, item in value.items()}
    return value


def _layer_output_hidden_states(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    if hasattr(output, 'last_hidden_state'):
        return output.last_hidden_state
    raise TypeError(f'Unsupported transformer layer output type: {type(output)}')


def _gptq_sequential_groups(linears):
    """Resolve FlatQuant's four GPTQ groups through supported wrappers."""
    def logical_name(name):
        # ActQuantWrapper stores the real weight in ``module`` while
        # FlatQuantizedLinear stores it in ``linear``.
        while name.endswith(('.module', '.linear')):
            name = name.rsplit('.', 1)[0]
        return name

    by_logical_name = {}
    for name in linears:
        logical = logical_name(name)
        if logical in by_logical_name:
            raise ValueError(f'Multiple GPTQ weights resolve to {logical!r}')
        by_logical_name[logical] = name

    # Keep the same groups and ordering as sibling FlatQuant/gptq_utils.py.
    patterns = (
        ('self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'),
        ('self_attn.o_proj',),
        ('mlp.up_proj', 'mlp.gate_proj'),
        ('mlp.down_proj',),
    )
    groups = []
    for pattern in patterns:
        missing = [name for name in pattern if name not in by_logical_name]
        if missing:
            raise KeyError(
                f'Missing expected GPTQ weights {missing}; found {list(linears)}'
            )
        groups.append([by_logical_name[name] for name in pattern])
    return groups


def _vlm_calibration_modules(model):
    """Return modules/parameters needed before the first language-model layer."""
    language_model = model_utils.get_language_model(model)
    shell = model_utils.get_model_shell(model)
    modules = list(model_utils.get_embeddings(model))
    for owner, name in (
        (language_model, 'rotary_emb'),
        (shell, 'vision_tower'),
        (shell, 'multi_modal_projector'),
        (model, 'visual'),
    ):
        module = getattr(owner, name, None)
        if isinstance(module, nn.Module):
            modules.append(module)

    # LLaVA-NeXT stores image_newline directly on the multimodal shell.
    direct_tensors = []
    for owner in (model, shell, language_model):
        direct_tensors.extend(list(owner.parameters(recurse=False)))
        direct_tensors.extend(list(owner.buffers(recurse=False)))

    unique_modules = list(dict.fromkeys(modules))
    unique_tensors = list(dict.fromkeys(direct_tensors))
    return unique_modules, unique_tensors


@torch.no_grad()
def gptq_fwrd(model, dataloader, dev, args, image_processor=None):
    """Layer-wise GPTQ for the language model inside a supported VLM.

    Calibration uses the complete multimodal forward up to the first LM layer, so
    the captured activations include projected visual tokens.
    """
    _validate_supported_model(model)
    if image_processor is None:
        raise ValueError('VLM GPTQ requires the model image processor')

    logging.info('-----VLM GPTQ Quantization-----')
    dev = torch.device(dev)
    lm_config = model_utils.get_lm_config(model)
    use_cache = getattr(model.config, 'use_cache', None)
    lm_use_cache = getattr(lm_config, 'use_cache', None)
    model.config.use_cache = False
    lm_config.use_cache = False
    layers = model_utils.get_layers(model)
    if len(layers) == 0:
        raise ValueError('The VLM language model has no transformer layers')

    inps, layer_kwargs = [], []
    calibration_modules, direct_tensors = _vlm_calibration_modules(model)
    first_layer = layers[0]

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, hidden_states, *forward_args, **kwargs):
            if len(inps) < args.nsamples:
                inps.append(hidden_states.detach().cpu())
                # LM decoder layers in the supported models use keyword metadata.
                # Positional metadata is retained for a clear failure instead of
                # being silently ignored if a future model changes this contract.
                if forward_args:
                    kwargs['_gptq_forward_args'] = forward_args
                layer_kwargs.append(_tree_to(kwargs, 'cpu'))
            raise _StopCalibrationForward

    try:
        for module in calibration_modules:
            module.to(dev)
        for tensor in direct_tensors:
            tensor.data = tensor.data.to(dev)
        layers[0] = Catcher(first_layer.to(dev))

        source = dataloader[0] if isinstance(dataloader, tuple) else dataloader
        for batch in source:
            if len(inps) >= args.nsamples:
                break
            inputs = model_utils.message_to_prompt(batch, image_processor, type(model))
            inputs = _tree_to(inputs, dev)
            try:
                model(**inputs, use_cache=False, return_dict=True)
            except _StopCalibrationForward:
                pass
    finally:
        layers[0] = first_layer.cpu()
        for module in calibration_modules:
            module.cpu()
        for tensor in direct_tensors:
            tensor.data = tensor.data.cpu()
        if use_cache is not None:
            model.config.use_cache = use_cache
        if lm_use_cache is not None:
            lm_config.use_cache = lm_use_cache
        utils.cleanup_memory(verbos=False)

    if not inps:
        raise RuntimeError('VLM GPTQ did not capture any language-model activations')
    if len(inps) < args.nsamples:
        logging.warning(
            'VLM GPTQ requested %d samples but captured %d; continuing with the available samples',
            args.nsamples, len(inps),
        )
    token_counts = [inp.shape[-2] for inp in inps]
    logging.info(
        'VLM GPTQ calibration: %d samples, token length min/avg/max = %d/%.1f/%d',
        len(inps), min(token_counts),
        sum(token_counts) / len(token_counts), max(token_counts),
    )

    quantizers = {}
    gptq_mse = getattr(args, 'gptq_mse', False)
    logging.info(
        'GPTQ settings: percdamp=%s, act_order=%s, groupsize=%s, mse_clip=%s',
        args.percdamp, args.act_order, args.w_groupsize, gptq_mse,
    )
    try:
        for i in range(len(layers)):
            logging.info('GPTQ language layer %d/%d', i + 1, len(layers))
            layer = layers[i].to(dev)
            full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])

            for names in _gptq_sequential_groups(full):
                subset = {name: full[name] for name in names}
                gptq = {}
                for name, linear in subset.items():
                    bits = 8 if args.int8_down_proj and 'down_proj' in name else args.w_bits
                    gptq[name] = GPTQ(linear)
                    gptq[name].quantizer = quant_utils.WeightQuantizer()
                    gptq[name].quantizer.configure(
                        bits, perchannel=True, sym=not args.w_asym, mse=gptq_mse
                    )

                def add_batch(name):
                    def hook(_, hook_inputs, hook_output):
                        gptq[name].add_batch(hook_inputs[0].detach(), hook_output.detach())
                    return hook

                handles = [
                    linear.register_forward_hook(add_batch(name))
                    for name, linear in subset.items()
                ]
                try:
                    for inp, kwargs in zip(inps, layer_kwargs):
                        kwargs = _tree_to(kwargs, dev)
                        forward_args = kwargs.pop('_gptq_forward_args', ())
                        layer(inp.to(dev), *forward_args, **kwargs)
                finally:
                    for handle in handles:
                        handle.remove()

                for name in subset:
                    gptq[name].fasterquant(
                        percdamp=args.percdamp,
                        groupsize=args.w_groupsize,
                        actorder=args.act_order,
                        static_groups=False,
                    )
                    quantizers[f'model.layers.{i}.{name}'] = gptq[name].quantizer.cpu()
                    gptq[name].free()

            outs = []
            for inp, kwargs in zip(inps, layer_kwargs):
                kwargs = _tree_to(kwargs, dev)
                forward_args = kwargs.pop('_gptq_forward_args', ())
                output = layer(inp.to(dev), *forward_args, **kwargs)
                outs.append(_layer_output_hidden_states(output).detach().cpu())

            layers[i] = layer.cpu()
            del layer
            inps = outs
            utils.cleanup_memory(verbos=False)
    finally:
        if use_cache is not None:
            model.config.use_cache = use_cache
        if lm_use_cache is not None:
            lm_config.use_cache = lm_use_cache

    utils.cleanup_memory(verbos=True)
    logging.info('-----VLM GPTQ Quantization Done-----\n')
    return quantizers
       

@torch.no_grad()
def rtn_fwrd(model, dev, args, start_id=0, target_layer_idx=None):
    """RTN weight quantization for the LLM component of a supported VLM."""
    _validate_supported_model(model)
    assert args.w_groupsize == -1, "Groupsize not supported in RTN!"
    layers = model_utils.get_layers(model)
    torch.cuda.empty_cache()

    quantizers = {}

    if target_layer_idx is None:
        layer_indices = range(start_id, len(layers))
    else:
        target_layer_idx = int(target_layer_idx)
        layer_indices = [target_layer_idx] if target_layer_idx >= start_id else []

    for i in tqdm.tqdm(
        layer_indices,
        desc="(RtN Quant.) Layers",
        disable=target_layer_idx is not None,
    ):
        layer = layers[i].to(dev)

        subset = quant_utils.find_qlayers(layer,
                                            layers=[torch.nn.Linear])

        for name in subset:
            layer_weight_bits = args.w_bits
            if 'lm_head' in name:
                layer_weight_bits = 16
                continue
            if args.int8_down_proj and 'down_proj' in name:
                layer_weight_bits = 8

            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                layer_weight_bits, perchannel=True, sym=not(args.w_asym), mse=args.w_clip
            )
            W = subset[name].weight.data
            quantizer.find_params(W)
            # subset[name].weight.data = quantizer.quantize(W).to(
            #     next(iter(layer.parameters())).dtype)
            subset[name].weight.data = quantizer.quantize(W).to(W.dtype)
            quantizers['model.layers.%d.%s' % (i, name)] = quantizer.cpu()
        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        del layer
            
    utils.cleanup_memory(verbos=target_layer_idx is None)
    return quantizers
