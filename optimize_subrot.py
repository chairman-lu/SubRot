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
    model = model.to(device)
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
        model = model.cpu()
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

class SubspaceLoss(torch.nn.Module):
    def __init__(self, subspace_matrix_path) -> None:
        super().__init__()
        subspaces = torch.load(subspace_matrix_path)
        self.fishers = subspaces['fisher']
        self.ranks = subspaces['ranks']
        self.a = subspaces['a']
    
    def forward(self, quant_out, target, layer_id):
        assert quant_out.shape == target.shape
        B, N, C = quant_out.shape
        i = layer_id
        rank = self.ranks[i]
        fisher = self.fishers[i]
        U, values = fisher['U'][:, :rank].to(quant_out.device), fisher['values'][:rank].to(quant_out.device)
        a = self.a[i].to(quant_out.device)
        unstable = torch.zeros_like(a).to(a)
        unstable[a == 0] = 1
        assert a.shape[-1] == U.shape[-1] == values.shape[-1]

        e = quant_out.float() - target.float()  # 1NC
        e_high = e @ U      # 1Nr
        unstable = unstable.reshape(1, 1, -1).expand_as(e_high)
        loss = (e_high * unstable).square().mean() + (e_high * a).mean()
        return loss
        
def calibrate_flatquant(
    model, inps, layer_kwargs, args, device, rank, world_size
):
    layers = model_utils.get_layers(model)
    loss_fn = torch.nn.MSELoss()
    my_loss_fn = SubspaceLoss(args.loss_subspace_path)
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

    # Keep a stable sample shard on every rank for the whole layerwise
    # calibration.  The modulo indices append repeated samples when nsamples
    # is not divisible by world_size, so every rank still performs the same
    # number of DDP updates.
    local_global_indices = [
        global_idx % len(inps)
        for global_idx in range(rank, padded_sample_count, world_size)
    ]
    fp_inps = [inps[index] for index in local_global_indices]
    quant_inps = [inps[index] for index in local_global_indices]
    local_layer_kwargs = [
        layer_kwargs[index] for index in local_global_indices
    ]

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
                kwargs = _move_to_device(
                    local_layer_kwargs[sample_idx], device
                )
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
                T_max=max(1, args.epochs * len(fp_inps)),
                eta_min=args.flat_lr * 1e-3,
            )
            for epoch in range(args.epochs):
                total_loss = 0.0
                total_high = 0.0
                local_samples = 0
                with torch.enable_grad():
                    for sample_idx, sample in enumerate(quant_inps):
                        hidden = sample.to(device)
                        kwargs = _move_to_device(
                            local_layer_kwargs[sample_idx], device
                        )
                        target = fp_outs[sample_idx].to(device)
                        with traincast():
                            quant_out = _layer_forward(ddp_layer, hidden, kwargs)
                            mse_loss = loss_fn(quant_out, target)
                            if layer_idx < 9:
                                loss = mse_loss
                                high_loss = torch.tensor(0.0)
                            else:
                                high_loss = my_loss_fn(quant_out, target, layer_idx)
                                loss = mse_loss + args.sensitivity_beta * high_loss
                            normalized_loss = loss / mse_loss.detach().clamp_min(1e-12)
                        optimizer.zero_grad(set_to_none=True)
                        normalized_loss.backward()
                        optimizer.step()
                        scheduler.step()
                        total_loss += float(loss.detach().cpu())
                        total_high += float(high_loss.detach().cpu())
                        local_samples += 1

                loss_stats = torch.tensor(
                    [total_loss, total_high, local_samples],
                    dtype=torch.float64,
                    device=device,
                )
                dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
                if rank == 0:
                    logging.info(f'layer {layer_idx} epoch {epoch + 1}/{args.epochs} '
                                 f'total loss {loss_stats[0].item() / loss_stats[-1].item():.6f} '
                                 f'high sen loss {loss_stats[1].item() / loss_stats[-1].item():.6f} ')
            del ddp_layer

        # Advance the accumulated-quantization path with the final trained
        # fake-quant layer.  Its input differs from the clean FP path from the
        # second decoder layer onward.
        quant_outs = []
        with torch.no_grad():
            for sample_idx, sample in enumerate(quant_inps):
                hidden = sample.to(device)
                kwargs = _move_to_device(
                    local_layer_kwargs[sample_idx], device
                )
                quant_outs.append(
                    _layer_forward(layer, hidden, kwargs).detach().cpu()
                )

        layers[layer_idx] = layer.cpu()
        fp_inps = fp_outs
        quant_inps = quant_outs
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
            model, processor = model_utils.get_model(args.model, args.model_path, device_map=None)
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


if __name__ == "__main__":
    try:
        train()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()