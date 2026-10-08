import logging
import math
import os, copy
import sys
import functools
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from vlmrot import data_utils, flatquant_utils, model_utils, utils


logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)


def _init_ddp():
    if not torch.cuda.is_available():
        raise RuntimeError("FlatQuant DDP training requires CUDA")
    if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "Launch optimize_flatquant.py with torchrun and set "
            "--nproc_per_node equal to cali_bsz"
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    return (
        dist.get_rank(),
        dist.get_world_size(),
        torch.device("cuda", local_rank),
    )


def parse_args():
    original_argv = sys.argv
    try:
        args = utils.parser_gen()
    finally:
        sys.argv = original_argv
    return args


def _move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _detach_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(_detach_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_detach_to_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: _detach_to_cpu(item) for key, item in value.items()}
    return value


def _layer_forward(layer, hidden_states, layer_kwargs):
    output = layer(hidden_states, **layer_kwargs)
    if isinstance(output, tuple):
        return output[0]
    return output


def _set_layer_flatquant_mode(layer, ori_mode):
    layer.self_attn._ori_mode = ori_mode
    layer.mlp._ori_mode = ori_mode


def _validate_args(args):
    if args.model and not ("Qwen2-VL" in args.model or "llava-v1.6" in args.model or "llava-1.5" in args.model):
        raise ValueError("FlatQuant training currently supports Qwen2-VL and LLaVA-Next only")
    if args.a_groupsize > 0:
        raise NotImplementedError("FlatQuant activation groupsize is not supported yet")
    if not any([args.cali_trans, args.add_diag, args.lwc, args.lac]):
        logging.warning(
            "No FlatQuant trainable option enabled; saving initialized matrices only."
        )


@torch.no_grad()
def capture_first_layer_inputs(model, processor, dataloader, args, device):
    layers = model_utils.get_layers(model)
    inps = []
    layer_kwargs = []

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, hidden_states, *unused_args, **kwargs):
            kwargs["use_cache"] = False
            inps.append(hidden_states.detach().cpu())
            layer_kwargs.append(_detach_to_cpu(kwargs))
            raise ValueError

    layers[0] = Catcher(layers[0])
    # model = model.to(device)
    model = utils.dispatch_model_to_cuda(model)
    try:
        for batch in dataloader:
            if len(inps) >= args.nsamples:
                break
            inputs = model_utils.build_loss_inputs(batch, processor, model)
            inputs = _move_to_device(inputs, device)
            try:
                model(**inputs, use_cache=False)
            except ValueError:
                continue
    finally:
        layers[0] = layers[0].module
        # model = model.cpu()
        model = utils.move_model_to_cpu(model)
        utils.cleanup_memory(verbos=False)

    if not inps:
        raise RuntimeError("Failed to capture any FlatQuant calibration inputs")
    return inps[: args.nsamples], layer_kwargs[: args.nsamples]


def _prepare_trainable_params(layer, args):
    for param in layer.parameters():
        param.requires_grad = False
    optimizer_groups = []
    if args.cali_trans:
        params = [
            param
            for name, param in layer.named_parameters()
            if "trans" in name and "diag_scale" not in name
        ]
        if params:
            optimizer_groups.append({"params": params, "lr": args.flat_lr})
    if args.add_diag:
        params = [
            param
            for name, param in layer.named_parameters()
            if "diag_scale" in name
        ]
        if params:
            optimizer_groups.append({"params": params, "lr": args.flat_lr})
    if args.lwc:
        params = [
            param
            for name, param in layer.named_parameters()
            if "clip_factor_w" in name
        ]
        if params:
            optimizer_groups.append({"params": params, "lr": args.flat_lr * 10})
    if args.lac:
        params = [
            param
            for name, param in layer.named_parameters()
            if "clip_factor_a" in name
        ]
        if params:
            optimizer_groups.append({"params": params, "lr": args.flat_lr * 10})

    for group in optimizer_groups:
        for param in group["params"]:
            param.requires_grad = True
    return optimizer_groups or None


def _get_traincast(device):
    if device.type == "cuda":
        return functools.partial(torch.amp.autocast, device_type="cuda", dtype=torch.float16)
    if device.type == "cpu":
        return functools.partial(torch.amp.autocast, device_type="cpu", dtype=torch.bfloat16)
    return nullcontext

def _set_flat_quantizers_enabled(module, enabled):
    states = []
    for child in module.modules():
        if isinstance(
            child,
            (flatquant_utils.WeightQuantizer, flatquant_utils.ActivationQuantizer),
        ):
            states.append((child, child.enable))
            child.enable = enabled
    return states


def _restore_flat_quantizer_states(states):
    for quantizer, enabled in states:
        quantizer.enable = enabled


def _reparameterize_layer(layer):
    layer.self_attn.reparameterize()
    layer.mlp.reparameterize()
    if layer.self_attn.ln_trans is not None and layer.self_attn.ln_trans.add_diag:
        flatquant_utils.reparameterize_ln(
            layer.input_layernorm, layer.self_attn.ln_trans
        )
    if layer.mlp.up_gate_trans is not None and layer.mlp.up_gate_trans.add_diag:
        flatquant_utils.reparameterize_ln(
            layer.post_attention_layernorm, layer.mlp.up_gate_trans
        )
    return layer

@torch.no_grad()
def _check_reparameterization_consistency(
    layer, fp_inps, layer_kwargs, args, device, layer_idx
):
    """Compare online transforms with folded transforms, without quantization/RTN."""
    if not all(
        hasattr(part, "reparameterize") for part in (layer.self_attn, layer.mlp)
    ):
        return
    sample_count = min(
        len(fp_inps), int(getattr(args, "flatquant_fold_check_samples", 1))
    )
    if sample_count <= 0:
        return

    # folded_layer = copy.deepcopy(layer).to(device)
    online_states = _set_flat_quantizers_enabled(layer, False)
    # folded_states = _set_flat_quantizers_enabled(folded_layer, False)

    max_abs = 0.0
    squared_error = 0.0
    squared_reference = 0.0
    element_count = 0
    try:
        online_outputs = []
        for sample_idx in range(sample_count):
            hidden = fp_inps[sample_idx].to(device)
            kwargs = _move_to_device(layer_kwargs[sample_idx], device)
            online_outputs.append(_layer_forward(layer, hidden, kwargs).float())

        _reparameterize_layer(layer)
        for sample_idx, online_output in enumerate(online_outputs):
            hidden = fp_inps[sample_idx].to(device)
            kwargs = _move_to_device(layer_kwargs[sample_idx], device)
            folded_output = _layer_forward(layer, hidden, kwargs).float()
            diff = online_output - folded_output
            max_abs = max(max_abs, float(diff.abs().amax().cpu()))
            squared_error += float(diff.square().sum().cpu())
            squared_reference += float(online_output.square().sum().cpu())
            element_count += diff.numel()
    finally:
        _restore_flat_quantizer_states(online_states)

    rmse = math.sqrt(squared_error / max(1, element_count))
    relative_l2 = math.sqrt(squared_error / max(squared_reference, 1e-24))
    tolerance = float(getattr(args, "flatquant_fold_check_tol", 1e-3))
    logging.info(
        "[FlatQuant fold check/no RTN] layer %d samples=%d max_abs=%.6g "
        "rmse=%.6g relative_l2=%.6g tol=%.6g",
        layer_idx,
        sample_count,
        max_abs,
        rmse,
        relative_l2,
        tolerance,
    )
    if relative_l2 > tolerance:
        logging.warning(
            "FlatQuant matrix folding mismatch at layer %d: relative_l2=%.6g",
            layer_idx,
            relative_l2,
        )

    utils.cleanup_memory(verbos=False)


def calibrate_flatquant(
    model, inps, layer_kwargs, args, device, rank, world_size
):
    layers = model_utils.get_layers(model)
    loss_fn = torch.nn.MSELoss()
    traincast = _get_traincast(device)

    if len(inps) != len(layer_kwargs):
        raise ValueError(
            f"Expected matching inputs and kwargs, got {len(inps)} and "
            f"{len(layer_kwargs)}"
        )
    padded_sample_count = math.ceil(len(inps) / world_size) * world_size
    repeated_sample_count = padded_sample_count - len(inps)
    if rank == 0 and repeated_sample_count:
        logging.info(
            "Padding %d calibration samples to %d with %d repeated sample(s) "
            "cycled from the beginning, so every DDP rank executes the same "
            "number of steps",
            len(inps),
            padded_sample_count,
            repeated_sample_count,
        )

    fp_inps = inps
    layer_iterator = (
        tqdm(layers, desc="FlatQuant layers") if rank == 0 else layers
    )
    for layer_idx, layer in enumerate(layer_iterator):
        if rank == 0:
            logging.info("Calibrating FlatQuant layer %d", layer_idx)
        layer = layer.to(device)
        layer.float()
        fp_outs = []

        _set_layer_flatquant_mode(layer, True)
        with torch.no_grad():
            for sample_idx, sample in enumerate(fp_inps):
                hidden = sample.to(device)
                kwargs = _move_to_device(layer_kwargs[sample_idx], device)
                fp_outs.append(_layer_forward(layer, hidden, kwargs).detach().cpu())
        _set_layer_flatquant_mode(layer, False)

        if args.diag_init == "sq_style":
            layer.self_attn.init_diag_scale(alpha=args.diag_alpha)
            layer.mlp.init_diag_scale(alpha=args.diag_alpha)
        elif args.diag_init != "one_style":
            raise ValueError(f"Unknown diag_init: {args.diag_init}")

        optimizer_groups = _prepare_trainable_params(layer, args)
        if optimizer_groups is not None:
            ddp_layer = DistributedDataParallel(
                layer,
                device_ids=[device.index],
                output_device=device.index,
                broadcast_buffers=True,
            )
            optimizer = torch.optim.AdamW(optimizer_groups)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, args.epochs * (padded_sample_count // world_size)),
                eta_min=args.flat_lr * 1e-3,
            )
            local_indices = [
                global_idx % len(fp_inps)
                for global_idx in range(rank, padded_sample_count, world_size)
            ]
            for epoch in range(args.epochs):
                total_loss = 0.0
                local_samples = 0
                with torch.enable_grad():
                    for sample_idx in local_indices:
                        sample = fp_inps[sample_idx]
                        hidden = sample.to(device)
                        kwargs = _move_to_device(layer_kwargs[sample_idx], device)
                        target = fp_outs[sample_idx].to(device)
                        with traincast():
                            quant_out = _layer_forward(ddp_layer, hidden, kwargs)
                            loss = loss_fn(quant_out, target)
                            normalized_loss = loss / loss.detach().clamp_min(1e-12)
                        optimizer.zero_grad(set_to_none=True)
                        normalized_loss.backward()
                        optimizer.step()
                        scheduler.step()
                        total_loss += float(loss.detach().cpu())
                        local_samples += 1

                loss_stats = torch.tensor(
                    [total_loss, local_samples],
                    dtype=torch.float64,
                    device=device,
                )
                dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
                if rank == 0:
                    logging.info(
                        "layer %d epoch %d/%d mse %.8f",
                        layer_idx,
                        epoch + 1,
                        args.epochs,
                        loss_stats[0].item() / max(1.0, loss_stats[1].item()),
                    )
            del ddp_layer

        # _check_reparameterization_consistency(
        #     layer, fp_inps, layer_kwargs, args, device, layer_idx
        # )

        layers[layer_idx] = layer.cpu()
        fp_inps = fp_outs
        utils.cleanup_memory(verbos=False)

    return model


def train():
    args = parse_args()
    
    rank, world_size, device = _init_ddp()
    if world_size != args.cali_bsz:
        raise ValueError(
            "DDP world size must equal cali_bsz because every GPU processes "
            f"one sample per update; got world_size={world_size}, "
            f"cali_bsz={args.cali_bsz}"
        )
    if rank == 0:
        _validate_args(args)
    utils.set_seed(args.seed)

    # get_model currently loads with device_map="auto". Serialize this one-time
    # initialization so four processes do not concurrently occupy every GPU.
    model = None
    processor = None
    for loader_rank in range(world_size):
        if rank == loader_rank:
            model, processor = model_utils.get_model(args.model, args.model_path)
            model = utils.move_model_to_cpu(model)
            utils.cleanup_memory(verbos=False)
        dist.barrier()

    model.eval()
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False

    flatquant_utils.apply_flatquant_to_model(args, model)
    # Keep FlatQuant calibration on the official eager attention path.  This
    # covers original targets, fake-quant forwards and per-layer fold checks.
    flatquant_utils.set_flatquant_attention_backend(model, "eager")
    dataloader = data_utils.get_loaders(
        args.cal_dataset,
        args.nsamples,
        args.seed,
        model.seqlen,
        args.question_file,
        args.image_base,
    )
    inps, layer_kwargs = capture_first_layer_inputs(
        model, processor, dataloader, args, device
    )
    calibrate_flatquant(
        model,
        inps,
        layer_kwargs,
        args,
        device,
        rank,
        world_size,
    )
    output_dir = flatquant_utils.get_flatquant_output_dir(args)
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        flatquant_utils.save_flat_matrices(args, model, output_dir)
    dist.barrier()
    exit()

    # Validate the exact in-memory model produced by calibration.  The fold
    # check reparameterizes checked layers in place; fold only layers for which
    # that check was disabled, otherwise transforms would be applied twice.
    from vlmrot import eval_utils, gptq_utils

    for layer in model_utils.get_layers(model):
        if not layer.self_attn._eval_mode:
            _reparameterize_layer(layer)

    # Restore the existing low-memory validation path after all calibration
    # and fold checks have completed.
    flatquant_utils.set_flatquant_attention_backend(model, "sdpa")

    # calibrate_flatquant calls layer.float(), while the untouched projector,
    # final norm and lm_head remain FP16.  Restore only the original decoder
    # weights to the lm_head deployment dtype; keep learned FlatQuant matrices
    # and clipping parameters in FP32 as they are in the normal main.py path.
    deployment_dtype = model_utils.get_lm_head(model).weight.dtype
    for layer in model_utils.get_layers(model):
        layer.input_layernorm.to(dtype=deployment_dtype)
        layer.post_attention_layernorm.to(dtype=deployment_dtype)
        for module in layer.modules():
            if isinstance(module, flatquant_utils.FlatQuantizedLinear):
                module.linear.to(dtype=deployment_dtype)
    logging.info(
        "[FlatQuant current-model validation] restored decoder weights to %s",
        deployment_dtype,
    )

    if getattr(args, "quant_method", "rtn") != "rtn":
        raise ValueError(
            "Post-training FlatQuant validation currently requires quant_method=rtn"
        )
    if args.w_bits < 16:
        original_w_clip = getattr(args, "w_clip", False)
        args.w_clip = False
        try:
            logging.info(
                "[FlatQuant current-model validation] applying RTN W%d without extra MSE clipping",
                args.w_bits,
            )
            gptq_utils.rtn_fwrd(model, device, args)
        finally:
            args.w_clip = original_w_clip

    model.eval()
    model.config.use_cache = True
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = True

    matrix_path = os.path.join(output_dir, "flat_matrices.pth")
    validation_tag = f"flatquant_current_model_{os.stat(matrix_path).st_mtime_ns}"
    args.eval_result_dir = os.path.join(args.eval_result_dir, validation_tag)
    os.makedirs(args.eval_result_dir, exist_ok=True)

    # FlatQuantizedLinear already applies the trained activation quantizer/LAC;
    # do not add a second VLMRot ActQuantWrapper.
    for eval_dataset in args.eval_dataset:
        logging.info(
            "[FlatQuant current-model validation] evaluating %s", eval_dataset
        )
        test_loader = data_utils.get_loaders(eval_dataset)
        result = eval_utils.evaluator(
            model, test_loader, device, args, processor, eval_dataset
        )
        logging.info(
            "[FlatQuant current-model validation] %s result: %s",
            eval_dataset,
            result,
        )


if __name__ == "__main__":
    try:
        train()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
