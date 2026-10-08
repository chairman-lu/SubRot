import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import os
import torch, math
from vlmrot import model_utils


def get_split_online_rotation_plot_dir(args):
    output_root = getattr(args, "profile_dir", None) or os.path.join(".", "profile")
    model_name = getattr(args, "model", "model").split("/")[-1]
    return os.path.join(output_root, model_name)


def plot_split_online_rotation_important_counts(important_counts, args):
    if not important_counts:
        print("[online_channel_split] no important channel counts to plot")
        return None

    layers = sorted(important_counts)
    o_counts = [
        important_counts[layer].get("o_proj", float("nan"))
        for layer in layers
    ]

    output_dir = get_split_online_rotation_plot_dir(args)
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(
        output_dir,
        "online_channel_split_important_counts.png",
    )
    pdf_path = os.path.join(
        output_dir,
        "online_channel_split_important_counts.pdf",
    )

    plt.figure(figsize=(10, 5.5))
    plt.plot(layers, o_counts, marker="o", label="o_proj")
    plt.xlabel("Layer")
    plt.ylabel("Important channel count")
    plt.title("Online OV split rotation important channels by layer")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"[online_channel_split] saved important channel count plot to {png_path}")
    return png_path

def plot_heatmap(tensor, title, save_path):
    """
    Draw the heat map of the tensor.
    tensor.shape: NC
    """
    assert len(tensor.shape) == 2
    plt.figure(figsize=(8, 8))
    plt.imshow(tensor.T.numpy(), aspect='auto', cmap='hot')
    plt.colorbar()
    plt.xlabel("Token")
    plt.ylabel("Channel")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_heatmap_with_mask(tensor, vis_mask, cap_mask, title, save_path):
    """
    Draw the heat map of the tensor with vis_mask and cap_mask.
    tensor.shape: (N, C) - N tokens, C channels
    vis_mask.shape: (N,) - binary mask for vision modality
    cap_mask.shape: (N,) - binary mask for caption modality
    Masks are concatenated at the bottom of the heatmap to indicate token modality.
    """
    assert len(tensor.shape) == 2, "tensor must be 2D"
    assert len(vis_mask.shape) == 1, "vis_mask must be 1D"
    assert len(cap_mask.shape) == 1, "cap_mask must be 1D"
    assert tensor.shape[0] == vis_mask.shape[0] == cap_mask.shape[0], "First dimension must match"

    # Convert masks to tensor if needed
    if not isinstance(vis_mask, torch.Tensor):
        vis_mask = torch.tensor(vis_mask, dtype=tensor.dtype)
    if not isinstance(cap_mask, torch.Tensor):
        cap_mask = torch.tensor(cap_mask, dtype=tensor.dtype)

    # Expand masks to (1, N) for concatenation
    vis_mask_expanded = vis_mask.unsqueeze(0)  # (1, N)
    cap_mask_expanded = cap_mask.unsqueeze(0) * tensor.max()  # (1, N)
    pad = torch.zeros_like(vis_mask_expanded)

    # Concatenate: tensor.T (C, N) + masks (1, N) each -> (C+2, N)
    tensor_transposed = tensor.T  # (C, N)
    concatenated = torch.cat([tensor_transposed, pad, vis_mask_expanded, cap_mask_expanded], dim=0)

    plt.figure(figsize=(10, 8))
    plt.imshow(concatenated.numpy(), aspect='auto', cmap='hot')
    plt.colorbar()
    plt.xlabel("Token")
    plt.ylabel("Channel")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    
def plot_histogram_with_mask(tensor, vis_mask, cap_mask, title, save_path):
    """
    Draw histograms of token values for vision and caption modalities.
    tensor.shape: (N, C) - N tokens, C channels
    vis_mask.shape: (N,) - binary mask for vision modality
    cap_mask.shape: (N,) - binary mask for caption modality

    X-axis: mean values of tokens across channels, sorted in ascending order
    Y-axis: frequency of values in each bin
    """
    if len(tensor.shape) == 3:
        tensor = tensor.mean(dim=0)
    assert len(tensor.shape) == 2, "tensor must be 2D"
    assert len(vis_mask.shape) == 1, "vis_mask must be 1D"
    assert len(cap_mask.shape) == 1, "cap_mask must be 1D"
    assert tensor.shape[0] == vis_mask.shape[0] == cap_mask.shape[0], "First dimension must match"

    # Convert masks to tensor if needed
    if not isinstance(vis_mask, torch.Tensor):
        vis_mask = torch.tensor(vis_mask, dtype=torch.bool)
    if not isinstance(cap_mask, torch.Tensor):
        cap_mask = torch.tensor(cap_mask, dtype=torch.bool)

    # Compute mean values across channels for each token
    token_means = tensor.mean(dim=1)  # (N,)

    # Split into vision and caption tokens based on masks
    vis_means = token_means[vis_mask]
    cap_means = token_means[cap_mask]

    # Sort by value
    vis_means_sorted = torch.sort(vis_means)[0].cpu().numpy() if isinstance(vis_means, torch.Tensor) else np.sort(vis_means)
    cap_means_sorted = torch.sort(cap_means)[0].cpu().numpy() if isinstance(cap_means, torch.Tensor) else np.sort(cap_means)

    # Compute unified bin edges based on the overall range
    all_means = np.concatenate([vis_means_sorted, cap_means_sorted])
    bins = np.linspace(all_means.min(), all_means.max(), 101)  # 30 bins

    # Create figure
    plt.figure(figsize=(8, 5))

    # Plot histograms with different colors using the same bin edges
    plt.hist(vis_means_sorted, bins=bins, alpha=0.6, label='Visual tokens', edgecolor='black')
    plt.hist(cap_means_sorted, bins=bins, alpha=0.6, label='Text tokens', edgecolor='black')

    plt.xlabel('Token Mean Value')
    plt.ylabel('Frequency')
    # plt.title(title)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def collect_layer_quant_errors(
    model,
    inputs,
    vision_mask,
    text_mask,
    n_bits=8,
):
    """
    vision_mask/text_mask: shape [N]，N 为输入 token 长度
    inputs: tokenizer / processor 后的模型输入，dict
    """



    handles = []

    # Qwen2VL 一般是 model.model.layers
    layers = model_utils.get_layers(model)
    results = {
        "vision": [[] for _ in range(len(layers))],
        "text": [[] for _ in range(len(layers))],
    }

    def make_hook(layer_idx):
        def hook(module, inp, out):
            # decoder layer 的输入通常是 inp[0]: [B, N, C]
            x = inp[0].detach()

            v_err = relative_quant_error(
                x, current_vis_mask, n_bits=n_bits
            )
            t_err = relative_quant_error(
                x, current_cap_mask, n_bits=n_bits
            )

            results["vision"][layer_idx].append(v_err)
            results["text"][layer_idx].append(t_err)

        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(i)))

    model.eval()
    model = model.cuda()
    with torch.no_grad():
        for i in range(len(inputs)):
            input = inputs[i]
            current_vis_mask = vision_mask[i].cuda()
            current_cap_mask = text_mask[i].cuda()
            _ = model(**input)

    for h in handles:
        h.remove()
    model = model.cpu()

    return results

@torch.no_grad()
def fake_quant_per_token(x, n_bits=8, eps=1e-8):
    """
    x: [B, N, C]
    per-token symmetric quantization
    """
    qmax = 2 ** (n_bits - 1) - 1

    scale = x.abs().amax(dim=-1, keepdim=True) / qmax
    scale = scale.clamp(min=eps)

    q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    x_q = q * scale
    return x_q


@torch.no_grad()
def relative_quant_error(x, mask, n_bits=8, eps=1e-6):
    """
    x: [B, N, C]
    mask: [N] or [B, N], bool
    return scalar relative error
    """
    if mask.dim() == 1:
        mask = mask.unsqueeze(0).expand(x.shape[0], -1)

    mask = mask.to(x.device).bool()

    x_q = fake_quant_per_token(x, n_bits=n_bits, eps=eps)

    err = (x_q - x).abs()
    rel = err / x.abs().clamp(min=eps)

    # 只统计 mask 对应 token
    rel = rel[mask]

    return rel.mean().item()

def plot_quant_errors(results, save_path=None):

    vision_curve = [
        (
            sum(v for v in x if not math.isnan(v)) /
            max(sum(1 for v in x if not math.isnan(v)), 1)
        )
        for x in results["vision"]
    ]

    text_curve = [
        (
            sum(v for v in x if not math.isnan(v)) /
            max(sum(1 for v in x if not math.isnan(v)), 1)
        )
        for x in results["text"]
    ]
    layers = list(range(len(results["vision"])))

    plt.figure(figsize=(8, 5))
    plt.plot(layers, vision_curve, marker="o", label="Visual tokens")
    plt.plot(layers, text_curve, marker="o", label="Text tokens")

    plt.xlabel("Layer")
    plt.ylabel("Mean relative quantization error")
    # plt.title("Layer-wise Relative Quantization Error")
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()


def plot_channel_relative_error(results, phase, output_dir):
    import os
    os.makedirs(output_dir, exist_ok=True)

    for modality in ['vision', 'caption']:
        if not results[modality]:
            continue

        layers = sorted(results[modality].keys())
        layer_indices = list(range(len(layers)))

        important_means = []
        non_important_means = []

        for layer_name in layers:
            stats = results[modality][layer_name]
            imp_mean = sum(stats['important_mean']) / len(stats['important_mean']) if stats['important_mean'] else 0
            non_imp_mean = sum(stats['non_important_mean']) / len(stats['non_important_mean']) if stats['non_important_mean'] else 0
            important_means.append(imp_mean)
            non_important_means.append(non_imp_mean)

        plt.figure(figsize=(12, 6))
        x = np.arange(len(layer_indices))
        width = 0.35

        bars1 = plt.bar(x - width / 2, important_means, width, label='Important Channels', alpha=0.8)
        bars2 = plt.bar(x + width / 2, non_important_means, width, label='Non-important Channels', alpha=0.8)

        plt.xlabel('Layer')
        plt.ylabel('Mean Relative Error (%)')
        plt.title(f'Channel Relative Error - {modality.capitalize()} ({phase})')
        plt.xticks(x, [f'L{i}' for i in layer_indices])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f'channel_error_{phase}_{modality}.pdf')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'[vis] Saved channel error plot to {save_path}')


def _layer_sort_key(layer_name):
    try:
        return int(layer_name.split("_")[-1])
    except ValueError:
        return layer_name


def plot_layer_input_relative_error_histograms(results, phase, output_dir, bins=80):
    import os

    if not results:
        return

    os.makedirs(output_dir, exist_ok=True)
    per_layer_dir = os.path.join(output_dir, f'layer_input_error_histograms_{phase}')
    os.makedirs(per_layer_dir, exist_ok=True)

    layers = sorted(results.keys(), key=_layer_sort_key)
    layer_error_pairs = []
    for layer_name in layers:
        errors = results[layer_name]["relative_errors"].detach().float().cpu().numpy()
        errors = errors[np.isfinite(errors)]
        if errors.size == 0:
            continue
        layer_error_pairs.append((layer_name, errors))

        plt.figure(figsize=(7, 4))
        plt.hist(errors, bins=bins, density=False, alpha=0.85)
        plt.xlabel('Relative Error (%)')
        plt.ylabel('Frequency')
        plt.title(f'{layer_name} Input Activation Relative Error ({phase})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(per_layer_dir, f'{layer_name}.pdf')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    if not layer_error_pairs:
        return

    cols = min(4, len(layer_error_pairs))
    rows = math.ceil(len(layer_error_pairs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.8 * rows), squeeze=False)
    all_errors = [errors for _, errors in layer_error_pairs]
    combined_errors = np.concatenate(all_errors)
    upper = np.percentile(combined_errors, 99.5)
    if upper <= 0:
        upper = combined_errors.max() if combined_errors.size else 1.0
    hist_range = (0.0, upper)

    for ax, (layer_name, errors) in zip(axes.ravel(), layer_error_pairs):
        ax.hist(errors, bins=bins, range=hist_range, density=False, alpha=0.85)
        ax.set_title(layer_name)
        ax.set_xlabel('Relative Error (%)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.25)

    for ax in axes.ravel()[len(layer_error_pairs):]:
        ax.axis('off')

    fig.suptitle(f'Layer Input Activation Relative Error Histograms ({phase})')
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    combined_path = os.path.join(output_dir, f'layer_input_error_histograms_{phase}_all_layers.pdf')
    fig.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[vis] Saved layer input error histograms to {per_layer_dir}')
    print(f'[vis] Saved combined layer input error histogram grid to {combined_path}')


def _normalize_by_percentile(values, percentile=99.0, eps=1e-12):
    denom = np.percentile(values, percentile)
    if not np.isfinite(denom) or denom <= eps:
        denom = np.max(values) if values.size else 1.0
    return values / max(denom, eps)


def _modality_dominance_masks(vision_activations, text_activations, tau=0.1, eps=1e-12):
    score = (vision_activations - text_activations) / (
        vision_activations + text_activations + eps
    )
    vision_dominant = score > tau
    text_dominant = score < -tau
    mixed = ~(vision_dominant | text_dominant)
    return vision_dominant, mixed, text_dominant


def plot_layer_channel_diagnostics(results, phase, output_dir):
    import os
    from matplotlib.colors import LogNorm

    if not results:
        return

    diag_dir = os.path.join(output_dir, f'layer_channel_diagnostics_{phase}')
    os.makedirs(diag_dir, exist_ok=True)

    for layer_name in sorted(results.keys(), key=_layer_sort_key):
        errors = results[layer_name]["relative_errors"].detach().float().cpu().numpy()
        activations = results[layer_name]["activation_magnitudes"].detach().float().cpu().numpy()
        gradients = results[layer_name]["gradient_magnitudes"].detach().float().cpu().numpy()
        vision_activations = results[layer_name].get("vision_activation_magnitudes")
        text_activations = results[layer_name].get("text_activation_magnitudes")
        if vision_activations is None or text_activations is None:
            vision_activations = np.zeros_like(errors)
            text_activations = np.zeros_like(errors)
        else:
            vision_activations = vision_activations.detach().float().cpu().numpy()
            text_activations = text_activations.detach().float().cpu().numpy()

        finite = (
            np.isfinite(errors)
            & np.isfinite(activations)
            & np.isfinite(gradients)
            & np.isfinite(vision_activations)
            & np.isfinite(text_activations)
        )
        errors = errors[finite]
        activations = activations[finite]
        gradients = gradients[finite]
        vision_activations = vision_activations[finite]
        text_activations = text_activations[finite]
        if errors.size == 0:
            continue

        topk = min(20, errors.size)
        top_indices = np.argsort(errors)[-topk:] if topk > 0 else np.array([], dtype=int)

        scatter_errors = np.maximum(errors, 0) + 1e-9
        color_norm = LogNorm(
            vmin=max(np.min(scatter_errors), 1e-9),
            vmax=max(np.max(scatter_errors), 1e-8),
        )
        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        vision_dominant, mixed, text_dominant = _modality_dominance_masks(
            vision_activations, text_activations
        )
        sc = None
        for mask, marker, label, alpha in (
            (vision_dominant, 'o', 'Vision-dominant', 0.55),
            (mixed, 's', 'Mixed', 0.45),
            (text_dominant, '^', 'Text-dominant', 0.55),
        ):
            if not np.any(mask):
                continue
            sc = ax.scatter(
                np.maximum(activations[mask], 1e-12),
                np.maximum(gradients[mask], 1e-12),
                c=scatter_errors[mask],
                s=10,
                alpha=alpha,
                cmap='viridis',
                norm=color_norm,
                marker=marker,
                label=label,
            )
        if top_indices.size:
            ax.scatter(
                np.maximum(activations[top_indices], 1e-12),
                np.maximum(gradients[top_indices], 1e-12),
                facecolors='none',
                edgecolors='red',
                s=38,
                lw=0.8,
                label='Top-20 relative error',
            )
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Activation magnitude |x_c|')
        ax.set_ylabel('Gradient magnitude |g_c|')
        ax.set_title(f'{layer_name} error on activation-gradient plane ({phase})')
        ax.grid(alpha=0.25, which='both')
        ax.legend(fontsize=8)
        if sc is not None:
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Relative error (%)')
        fig.tight_layout()

        pdf_path = os.path.join(diag_dir, f'{layer_name}.pdf')
        fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    plot_modality_dominant_relative_error_by_layer(results, phase, diag_dir)
    plot_modality_dominant_channel_count_by_layer(results, phase, diag_dir)
    print(f'[vis] Saved layer channel diagnostics to {diag_dir}')


def _collect_modality_dominant_error_series(results):
    layer_indices = []
    vision_errors = []
    mixed_errors = []
    text_errors = []

    for layer_name in sorted(results.keys(), key=_layer_sort_key):
        errors = results[layer_name]["relative_errors"].detach().float().cpu().numpy()
        vision_activations = results[layer_name].get("vision_activation_magnitudes")
        text_activations = results[layer_name].get("text_activation_magnitudes")
        if vision_activations is None or text_activations is None:
            continue

        vision_activations = vision_activations.detach().float().cpu().numpy()
        text_activations = text_activations.detach().float().cpu().numpy()
        finite = (
            np.isfinite(errors)
            & np.isfinite(vision_activations)
            & np.isfinite(text_activations)
        )
        errors = errors[finite]
        vision_activations = vision_activations[finite]
        text_activations = text_activations[finite]
        if errors.size == 0:
            continue

        vision_dominant, mixed, text_dominant = _modality_dominance_masks(
            vision_activations, text_activations
        )

        layer_indices.append(_layer_sort_key(layer_name))
        vision_errors.append(
            float(errors[vision_dominant].mean()) if np.any(vision_dominant) else float("nan")
        )
        mixed_errors.append(
            float(errors[mixed].mean()) if np.any(mixed) else float("nan")
        )
        text_errors.append(
            float(errors[text_dominant].mean()) if np.any(text_dominant) else float("nan")
        )

    return layer_indices, vision_errors, mixed_errors, text_errors


def _collect_modality_dominant_count_series(results):
    layer_indices = []
    vision_counts = []
    mixed_counts = []
    text_counts = []

    for layer_name in sorted(results.keys(), key=_layer_sort_key):
        errors = results[layer_name]["relative_errors"].detach().float().cpu().numpy()
        vision_activations = results[layer_name].get("vision_activation_magnitudes")
        text_activations = results[layer_name].get("text_activation_magnitudes")
        if vision_activations is None or text_activations is None:
            continue

        vision_activations = vision_activations.detach().float().cpu().numpy()
        text_activations = text_activations.detach().float().cpu().numpy()
        finite = (
            np.isfinite(errors)
            & np.isfinite(vision_activations)
            & np.isfinite(text_activations)
        )
        vision_activations = vision_activations[finite]
        text_activations = text_activations[finite]
        if vision_activations.size == 0:
            continue

        vision_dominant, mixed, text_dominant = _modality_dominance_masks(
            vision_activations, text_activations
        )

        layer_indices.append(_layer_sort_key(layer_name))
        vision_counts.append(int(np.sum(vision_dominant)))
        mixed_counts.append(int(np.sum(mixed)))
        text_counts.append(int(np.sum(text_dominant)))

    return layer_indices, vision_counts, mixed_counts, text_counts


def plot_modality_dominant_relative_error_by_layer(results, phase, output_dir):
    import os

    layers, vision_errors, mixed_errors, text_errors = _collect_modality_dominant_error_series(results)
    if not layers:
        return

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(
        layers,
        vision_errors,
        marker='o',
        linewidth=1.6,
        label='Vision-dominant channels',
    )
    ax.plot(
        layers,
        text_errors,
        marker='^',
        linewidth=1.6,
        label='Text-dominant channels',
    )
    ax.plot(
        layers,
        mixed_errors,
        marker='s',
        linewidth=1.6,
        label='Mixed channels',
    )
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean channel relative error (%)')
    ax.set_title(f'Modality-dominant channel relative error by layer ({phase})')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    save_path = os.path.join(output_dir, f'modality_dominant_relative_error_by_layer_{phase}.pdf')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[vis] Saved modality-dominant layer error plot to {save_path}')


def plot_modality_dominant_channel_count_by_layer(results, phase, output_dir):
    import os

    layers, vision_counts, mixed_counts, text_counts = _collect_modality_dominant_count_series(results)
    if not layers:
        return

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(
        layers,
        vision_counts,
        marker='o',
        linewidth=1.6,
        label='Vision-dominant channels',
    )
    ax.plot(
        layers,
        text_counts,
        marker='^',
        linewidth=1.6,
        label='Text-dominant channels',
    )
    ax.plot(
        layers,
        mixed_counts,
        marker='s',
        linewidth=1.6,
        label='Mixed channels',
    )
    ax.set_xlabel('Layer')
    ax.set_ylabel('Channel count')
    ax.set_title(f'Modality-dominant channel count by layer ({phase})')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    save_path = os.path.join(output_dir, f'modality_dominant_channel_count_by_layer_{phase}.pdf')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[vis] Saved modality-dominant channel count plot to {save_path}')


def _format_split_topk_percent(value):
    return f"{float(value):g}%"


def _get_split_best_k_entry(best_k_by_layer, layer_idx):
    if not best_k_by_layer:
        return None
    if layer_idx in best_k_by_layer:
        return best_k_by_layer[layer_idx]
    str_layer_idx = str(layer_idx)
    if str_layer_idx in best_k_by_layer:
        return best_k_by_layer[str_layer_idx]
    return None


def _get_split_best_k_phase_row(phase_rows, best_k_by_layer, layer_idx):
    entry = _get_split_best_k_entry(best_k_by_layer, layer_idx)
    if entry is None:
        return None, None
    if isinstance(entry, dict):
        topk_percent = float(entry["topk_percent"])
    else:
        topk_percent = float(entry)
    for row in phase_rows:
        if float(row["topk_percent"]) == topk_percent:
            return row, topk_percent
    return None, topk_percent


def plot_split_error_sweep_layer_curves(rows, output_dir, best_k_by_layer=None):
    import os
    from collections import defaultdict

    if not rows:
        return

    sweep_dir = os.path.join(output_dir, 'split_error_sweep')

    by_layer_phase = defaultdict(list)
    for row in rows:
        by_layer_phase[(int(row["layer"]), row["phase"])].append(row)

    for (layer_idx, phase), phase_rows in sorted(by_layer_phase.items()):
        phase_rows = sorted(
            phase_rows,
            key=lambda row: (int(row["important_count"]), float(row["topk_percent"])),
        )
        x = [int(row["important_count"]) for row in phase_rows]
        mean_error = [float(row["mean_error"]) for row in phase_rows]
        p95_error = [float(row["p95_error"]) for row in phase_rows]

        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        ax.plot(
            x,
            mean_error,
            marker='o',
            linewidth=1.8,
            label='Mean relative error',
        )
        ax.plot(
            x,
            p95_error,
            marker='^',
            linewidth=1.8,
            label='P95 relative error',
        )
        best_row, best_topk_percent = _get_split_best_k_phase_row(
            phase_rows,
            best_k_by_layer,
            layer_idx,
        )
        if best_row is not None:
            selected_x = int(best_row["important_count"])
            selected_y = float(best_row["p95_error"])
            ax.axvline(
                selected_x,
                color='crimson',
                linestyle='--',
                linewidth=1.4,
                alpha=0.85,
                label=f'Selected k={_format_split_topk_percent(best_topk_percent)}',
            )
            ax.annotate(
                f'selected k={_format_split_topk_percent(best_topk_percent)}',
                xy=(selected_x, selected_y),
                xytext=(8, 12),
                textcoords='offset points',
                color='crimson',
                fontsize=9,
                arrowprops={
                    'arrowstyle': '->',
                    'color': 'crimson',
                    'lw': 1.0,
                },
            )
        ax.set_xlabel('Important channel count')
        ax.set_ylabel('Important channel relative error (%)')
        ax.set_title(f'Layer {layer_idx} important channel error ({phase})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        phase_dir = os.path.join(sweep_dir, phase)
        os.makedirs(phase_dir, exist_ok=True)
        stem = f'layer_{layer_idx}_important_channel_relative_error'
        pdf_path = os.path.join(phase_dir, f'{stem}.pdf')
        fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
