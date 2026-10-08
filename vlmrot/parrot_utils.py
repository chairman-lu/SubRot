from collections import defaultdict
import csv
import itertools
import logging
import math
import os
from pathlib import Path
import re
import tempfile

if os.name == "nt" and "WINDIR" not in os.environ:
    os.environ["WINDIR"] = os.environ.get("SystemRoot", r"C:\Windows")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "vlmrot-matplotlib")
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd.graph import save_on_cpu
from tqdm import tqdm

from vlmrot import model_utils, utils


STAT_NAMES = ("mean_abs", "rms", "std", "max_abs", "nonzero_ratio")

CHANNEL_COLUMNS = (
    "sample_id",
    "layer",
    "module",
    "modality",
    "channel",
    "token_count",
    "mean_abs",
    "rms",
    "std",
    "max_abs",
    "nonzero_ratio",
)

SAMPLE_COLUMNS = (
    "sample_id",
    "image_path",
    "question",
    "vision_tokens",
    "caption_tokens",
    "ignored_tokens",
)

SUMMARY_COLUMNS = (
    "layer",
    "module",
    "channel",
    "vision_samples",
    "caption_samples",
    "vision_mean_abs",
    "caption_mean_abs",
    "vision_std_across_samples",
    "caption_std_across_samples",
    "vision_caption_ratio",
    "modality_effect_size",
    "sample_sensitivity",
)

CORRELATION_COLUMNS = (
    "sample_id_a",
    "sample_id_b",
    "feature_count",
    "correlation",
)

TARGET_SUFFIXES = (
    "q_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def validate_token_masks(vision_mask, caption_mask):
    vision = vision_mask.detach().bool()
    caption = caption_mask.detach().bool()
    if vision.shape != caption.shape:
        raise ValueError("vision_mask and caption_mask must have the same shape")
    if torch.any(vision & caption):
        raise ValueError("vision_mask and caption_mask overlap")
    return vision, caption


def _normalize_activation(activation):
    if activation.ndim == 2:
        activation = activation.unsqueeze(0)
    if activation.ndim != 3:
        raise ValueError("activation must have shape [B, N, C] or [N, C]")
    return activation.detach().float()


def _normalize_mask(mask, batch_size):
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.shape[0] == 1 and batch_size > 1:
        mask = mask.expand(batch_size, -1)
    return mask


def _reduce_selected(selected):
    return {
        "token_count": int(selected.shape[0]),
        "mean_abs": selected.abs().mean(dim=0).cpu(),
        "rms": selected.square().mean(dim=0).sqrt().cpu(),
        "std": selected.std(dim=0, unbiased=False).cpu(),
        "max_abs": selected.abs().amax(dim=0).cpu(),
        "nonzero_ratio": selected.ne(0).float().mean(dim=0).cpu(),
    }


def compute_channel_stats(activation, vision_mask, caption_mask):
    activation = _normalize_activation(activation)
    vision, caption = validate_token_masks(vision_mask, caption_mask)
    vision = _normalize_mask(vision, activation.shape[0])
    caption = _normalize_mask(caption, activation.shape[0])
    if activation.shape[:2] != vision.shape:
        raise ValueError("mask token dimensions do not match activation")

    result = {}
    for name, mask in (("vision", vision), ("caption", caption)):
        selected = activation[mask.to(activation.device)]
        if selected.numel():
            result[name] = _reduce_selected(selected)
    return result


def build_channel_rows(sample_id, layer_idx, module_name, stats):
    rows = []
    for modality in ("vision", "caption"):
        modality_stats = stats.get(modality)
        if modality_stats is None:
            continue
        channels = modality_stats["mean_abs"].numel()
        for channel in range(channels):
            row = {
                "sample_id": sample_id,
                "layer": layer_idx,
                "module": module_name,
                "modality": modality,
                "channel": channel,
                "token_count": modality_stats["token_count"],
            }
            for statistic in STAT_NAMES:
                row[statistic] = float(modality_stats[statistic][channel])
            rows.append(row)
    return rows


def _stat_label(mode):
    if mode not in ("activation", "gradient"):
        raise ValueError("parrot mode must be 'activation' or 'gradient'")
    return mode


def _sample_columns(mode):
    return SAMPLE_COLUMNS + (f"mean_{_stat_label(mode)}",)


def describe_sample(sample_id, message, vision_mask, caption_mask, mode="activation"):
    vision, caption = validate_token_masks(vision_mask, caption_mask)
    image_paths = [item["value"] for item in message if item.get("type") == "image"]
    questions = [item["value"] for item in message if item.get("type") == "text"]
    ignored = ~(vision | caption)
    return {
        "sample_id": sample_id,
        "image_path": " | ".join(map(str, image_paths)),
        "question": " ".join(map(str, questions))[:500],
        "vision_tokens": int(vision.sum()),
        "caption_tokens": int(caption.sum()),
        "ignored_tokens": int(ignored.sum()),
        f"mean_{_stat_label(mode)}": float("nan"),
    }


def aggregate_channel_rows(rows, eps=1e-12):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (int(row["layer"]), row["module"], int(row["channel"]))
        grouped[key][row["modality"]].append(float(row["mean_abs"]))

    output = []
    for (layer, module, channel), modalities in sorted(grouped.items()):
        vision = np.asarray(modalities.get("vision", []), dtype=float)
        caption = np.asarray(modalities.get("caption", []), dtype=float)
        vision_mean = float(np.mean(vision)) if vision.size else math.nan
        caption_mean = float(np.mean(caption)) if caption.size else math.nan
        vision_std = float(np.std(vision)) if vision.size else math.nan
        caption_std = float(np.std(caption)) if caption.size else math.nan
        coefficients = [
            std / max(abs(mean), eps)
            for mean, std in (
                (vision_mean, vision_std),
                (caption_mean, caption_std),
            )
            if math.isfinite(mean) and math.isfinite(std)
        ]
        pooled = (
            math.sqrt((vision_std ** 2 + caption_std ** 2) / 2.0)
            if math.isfinite(vision_std) and math.isfinite(caption_std)
            else math.nan
        )
        output.append({
            "layer": layer,
            "module": module,
            "channel": channel,
            "vision_samples": int(vision.size),
            "caption_samples": int(caption.size),
            "vision_mean_abs": vision_mean,
            "caption_mean_abs": caption_mean,
            "vision_std_across_samples": vision_std,
            "caption_std_across_samples": caption_std,
            "vision_caption_ratio": (
                vision_mean / max(caption_mean, eps)
                if math.isfinite(vision_mean) and math.isfinite(caption_mean)
                else math.nan
            ),
            "modality_effect_size": (
                (vision_mean - caption_mean) / max(pooled, eps)
                if math.isfinite(pooled)
                else math.nan
            ),
            "sample_sensitivity": max(coefficients, default=math.nan),
        })
    return output


def _build_sample_features(rows):
    features = defaultdict(dict)
    for row in rows:
        value = float(row["mean_abs"])
        if not math.isfinite(value):
            continue
        feature = (
            int(row["layer"]),
            row["module"],
            row["modality"],
            int(row["channel"]),
        )
        features[int(row["sample_id"])][feature] = value
    return features


def _aligned_sample_values(features, sample_a, sample_b):
    feature_keys = sorted(set(features[sample_a]) & set(features[sample_b]))
    values_a = np.asarray(
        [features[sample_a][key] for key in feature_keys], dtype=float
    )
    values_b = np.asarray(
        [features[sample_b][key] for key in feature_keys], dtype=float
    )
    return feature_keys, values_a, values_b


def _correlation_rows(features, transform=None):
    output = []
    sample_ids = sorted(features)
    for sample_a in sample_ids:
        for sample_b in sample_ids:
            feature_keys, values_a, values_b = _aligned_sample_values(
                features, sample_a, sample_b
            )
            if transform is not None:
                values_a = transform(values_a)
                values_b = transform(values_b)
            if (
                len(feature_keys) < 2
                or np.std(values_a) == 0
                or np.std(values_b) == 0
            ):
                correlation = math.nan
            else:
                correlation = float(np.corrcoef(values_a, values_b)[0, 1])
            output.append({
                "sample_id_a": sample_a,
                "sample_id_b": sample_b,
                "feature_count": len(feature_keys),
                "correlation": correlation,
            })
    return output


def build_sample_correlation(rows):
    return _correlation_rows(_build_sample_features(rows))


def _average_ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def build_sample_spearman_correlation(rows):
    return _correlation_rows(
        _build_sample_features(rows), transform=_average_ranks
    )


class ActivationCollector:
    def __init__(self, layers, logger=None, mode="activation"):
        self.mode = _stat_label(mode)
        self.layers = layers
        self.logger = logger or logging.getLogger(__name__)
        self.rows = []
        self.handles = []
        self.sample_id = None
        self.vision_mask = None
        self.caption_mask = None
        self.tensor_handles = []

    def set_sample(self, sample_id, vision_mask, caption_mask):
        self.sample_id = sample_id
        self.vision_mask, self.caption_mask = validate_token_masks(
            vision_mask, caption_mask
        )

    def register(self):
        for layer_idx, layer in enumerate(self.layers):
            for name, module in layer.named_modules():
                if name.endswith(TARGET_SUFFIXES):
                    self.handles.append(module.register_forward_pre_hook(
                        self._make_hook(layer_idx, name)
                    ))
        if not self.handles:
            raise ValueError("No target linear modules found for parrot analysis")

    def _make_hook(self, layer_idx, module_name):
        def hook(module, inputs):
            if not inputs or not torch.is_tensor(inputs[0]):
                return
            sample_id = self.sample_id
            vision_mask = self.vision_mask
            caption_mask = self.caption_mask
            if vision_mask is None or caption_mask is None:
                return
            if self.mode == "activation":
                try:
                    stats = compute_channel_stats(
                        inputs[0], vision_mask, caption_mask
                    )
                except ValueError as exc:
                    self.logger.warning(
                        "Skipping sample %s layer %s module %s: %s",
                        sample_id,
                        layer_idx,
                        module_name,
                        exc,
                    )
                    return
                self.rows.extend(build_channel_rows(
                    sample_id, layer_idx, module_name, stats
                ))
                return

            if not inputs[0].requires_grad:
                return

            def grad_hook(grad):
                try:
                    stats = compute_channel_stats(
                        grad, vision_mask, caption_mask
                    )
                except ValueError as exc:
                    self.logger.warning(
                        "Skipping sample %s layer %s module %s: %s",
                        sample_id,
                        layer_idx,
                        module_name,
                        exc,
                    )
                    return
                self.rows.extend(build_channel_rows(
                    sample_id, layer_idx, module_name, stats
                ))

            try:
                self.tensor_handles.append(inputs[0].register_hook(grad_hook))
            except RuntimeError as exc:
                self.logger.warning(
                    "Skipping gradient hook for sample %s layer %s module %s: %s",
                    sample_id,
                    layer_idx,
                    module_name,
                    exc,
                )
        return hook

    def remove(self):
        for handle in self.tensor_handles:
            handle.remove()
        self.tensor_handles.clear()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _save_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _prepare_log_scatter_values(caption, vision):
    caption = np.asarray(caption, dtype=float)
    vision = np.asarray(vision, dtype=float)
    positive = np.concatenate([vision[vision > 0], caption[caption > 0]])
    epsilon = float(positive.min() / 10.0) if positive.size else 1e-12
    return np.maximum(caption, epsilon), np.maximum(vision, epsilon), epsilon


def _plot_modality_scatter(summary_rows, figures_dir, mode):
    finite = [
        row for row in summary_rows
        if math.isfinite(float(row["vision_mean_abs"]))
        and math.isfinite(float(row["caption_mean_abs"]))
    ]
    if not finite:
        return
    vision = np.asarray([float(row["vision_mean_abs"]) for row in finite])
    caption = np.asarray([float(row["caption_mean_abs"]) for row in finite])
    plot_caption, plot_vision, epsilon = _prepare_log_scatter_values(
        caption, vision
    )
    plt.figure(figsize=(6, 6))
    plt.scatter(
        plot_caption, plot_vision, s=10, alpha=0.45, color="#2878B5"
    )
    lower = min(float(plot_vision.min()), float(plot_caption.min()))
    upper = max(float(plot_vision.max()), float(plot_caption.max()))
    plt.plot(
        [lower, upper], [lower, upper], linestyle="--", color="#555555"
    )
    plt.xscale("log")
    plt.yscale("log")
    label = _stat_label(mode)
    plt.xlabel(f"Caption mean absolute {label}")
    plt.ylabel(f"Vision mean absolute {label}")
    plt.title(f"Visual vs caption channel {label}")
    plt.grid(alpha=0.2)
    plt.figtext(
        0.5,
        0.01,
        f"Non-positive values are displayed at epsilon={epsilon:.3g}; "
        "source values are unchanged.",
        ha="center",
        fontsize=8,
    )
    _save_figure(figures_dir / "visual_vs_caption_scatter.png")


def _plot_sample_correlation(
    correlation_rows, figures_dir, filename, title
):
    sample_ids = sorted({
        int(row["sample_id_a"]) for row in correlation_rows
    })
    if not sample_ids:
        return
    index = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    matrix = np.full((len(sample_ids), len(sample_ids)), np.nan)
    for row in correlation_rows:
        matrix[
            index[int(row["sample_id_a"])],
            index[int(row["sample_id_b"])],
        ] = float(row["correlation"])
    if not np.isfinite(matrix).any():
        return
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(label="Pearson correlation")
    plt.xticks(range(len(sample_ids)), sample_ids, rotation=90)
    plt.yticks(range(len(sample_ids)), sample_ids)
    plt.xlabel("Sample")
    plt.ylabel("Sample")
    plt.title(title)
    for row_idx in range(len(sample_ids)):
        for column_idx in range(len(sample_ids)):
            value = matrix[row_idx, column_idx]
            if not math.isfinite(float(value)):
                continue
            color = "white" if abs(float(value)) >= 0.55 else "black"
            plt.text(
                column_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=max(5, 10 - len(sample_ids) // 4),
            )
    _save_figure(figures_dir / filename)


def plot_representative_analyses(
    summary_rows,
    correlation_rows,
    spearman_rows,
    figures_dir,
    mode="activation",
):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    label = _stat_label(mode)
    plotters = (
        lambda: _plot_modality_scatter(summary_rows, figures_dir, label),
        lambda: _plot_sample_correlation(
            correlation_rows,
            figures_dir,
            "sample_correlation_heatmap.png",
            f"Sample {label} Pearson correlation",
        ),
        lambda: _plot_sample_correlation(
            spearman_rows,
            figures_dir,
            "sample_spearman_heatmap.png",
            f"Sample {label} Spearman correlation",
        ),
    )
    for plotter in plotters:
        try:
            plotter()
        except Exception:
            logging.exception("Parrot plot generation failed")
            plt.close("all")


def _format_channel(row):
    return f"L{row['layer']} {row['module']} channel {row['channel']}"


def write_markdown_analysis(
    path, channel_rows, sample_rows, summary_rows, correlation_rows, topk,
    mode="activation",
):
    label = _stat_label(mode)
    finite_ratios = [
        float(row["vision_caption_ratio"]) for row in summary_rows
        if math.isfinite(float(row["vision_caption_ratio"]))
    ]
    preference = sorted(
        [
            row for row in summary_rows
            if math.isfinite(float(row["modality_effect_size"]))
        ],
        key=lambda row: abs(float(row["modality_effect_size"])),
        reverse=True,
    )[:topk]
    sensitivity = sorted(
        [
            row for row in summary_rows
            if math.isfinite(float(row["sample_sensitivity"]))
        ],
        key=lambda row: float(row["sample_sensitivity"]),
        reverse=True,
    )[:topk]
    off_diagonal = sorted(
        [
            row for row in correlation_rows
            if int(row["sample_id_a"]) < int(row["sample_id_b"])
            and math.isfinite(float(row["correlation"]))
        ],
        key=lambda row: float(row["correlation"]),
    )
    ignored_tokens = sum(int(row["ignored_tokens"]) for row in sample_rows)
    median_ratio = (
        float(np.median(finite_ratios)) if finite_ratios else math.nan
    )

    lines = [
        f"# Parrot Channel {label.title()} Analysis",
        "",
        "## Dataset Summary",
        "",
        f"- Samples: {len(sample_rows)}",
        f"- Channel observations: {len(channel_rows)}",
        f"- Aggregated channels: {len(summary_rows)}",
        f"- Ignored/padding tokens: {ignored_tokens}",
        "",
        "## Visual vs Caption",
        "",
        (
            f"- Median visual/caption mean-absolute-{label} ratio: "
            f"{median_ratio:.6g}"
            if math.isfinite(median_ratio)
            else "- Median visual/caption ratio: unavailable"
        ),
        "- Strongest modality-selective channels:",
    ]
    for row in preference:
        lines.append(
            f"  - {_format_channel(row)}: effect="
            f"{float(row['modality_effect_size']):.6g}, ratio="
            f"{float(row['vision_caption_ratio']):.6g}"
        )
    lines.extend(["", "## Sample Sensitivity", ""])
    for row in sensitivity:
        lines.append(
            f"- {_format_channel(row)}: CV="
            f"{float(row['sample_sensitivity']):.6g}"
        )
    lines.extend(["", "## Sample Relationships", ""])
    if off_diagonal:
        for row in off_diagonal[:topk]:
            lines.append(
                f"- Samples {row['sample_id_a']} and {row['sample_id_b']}: "
                f"r={float(row['correlation']):.6g} "
                f"over {row['feature_count']} features"
            )
    else:
        lines.append("- Insufficient finite, non-constant features.")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_analysis_outputs(
    channel_rows, sample_rows, output_dir, topk=20, mode="activation"
):
    label = _stat_label(mode)
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for old_figure in figures_dir.glob("*.png"):
        old_figure.unlink()
    summary_rows = aggregate_channel_rows(channel_rows)
    correlation_rows = build_sample_correlation(channel_rows)
    spearman_rows = build_sample_spearman_correlation(channel_rows)

    # write_csv(output_dir / "channel_stats.csv", channel_rows, CHANNEL_COLUMNS)
    # write_csv(output_dir / "sample_summary.csv", sample_rows, _sample_columns(label))
    # write_csv(
    #     output_dir / "channel_summary.csv", summary_rows, SUMMARY_COLUMNS
    # )
    # write_csv(
    #     output_dir / "sample_correlation.csv",
    #     correlation_rows,
    #     CORRELATION_COLUMNS,
    # )
    # write_csv(
    #     output_dir / "sample_spearman_correlation.csv",
    #     spearman_rows,
    #     CORRELATION_COLUMNS,
    # )
    plot_representative_analyses(
        summary_rows, correlation_rows, spearman_rows, figures_dir, mode=label
    )
    write_markdown_analysis(
        output_dir / "analysis.md",
        channel_rows,
        sample_rows,
        summary_rows,
        correlation_rows,
        topk,
        mode=label,
    )
    return {
        "sample_count": len(sample_rows),
        "observation_count": len(channel_rows),
        "channel_count": len(summary_rows),
        "output_dir": str(output_dir),
    }


def get_parrot_output_dir(profile_dir, model_name):
    root = Path(profile_dir or "./profile")
    return root / model_name.split("/")[-1] / "parrot_analysis"


def _output_to_loss(output):
    loss = getattr(output, "loss", None)
    if loss is None and isinstance(output, (tuple, list)) and output:
        loss = output[0]
    if loss is None and torch.is_tensor(output):
        loss = output
    if loss is None:
        raise ValueError("Parrot gradient analysis requires a scalar loss")
    if loss.ndim:
        loss = loss.float().sum()
    return loss


def _enable_floating_input_grads(value):
    if torch.is_tensor(value):
        if torch.is_floating_point(value):
            value.requires_grad_(True)
        return
    if isinstance(value, dict):
        for item in value.values():
            _enable_floating_input_grads(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _enable_floating_input_grads(item)


def run_parrot_analysis(
    model,
    image_processor,
    dataloader,
    output_dir,
    sample_limit=32,
    topk=20,
    get_layers=None,
    prepare_inputs=None,
    get_masks=None,
    forward_sample=None,
    mode="activation",
):
    label = _stat_label(mode)
    if sample_limit <= 0:
        raise ValueError("parrot sample_limit must be positive")
    samples = list(itertools.islice(iter(dataloader), sample_limit))
    if not samples:
        raise ValueError("parrot analysis received an empty dataset")

    get_layers = get_layers or model_utils.get_layers
    get_masks = get_masks or model_utils.get_input_masks
    collector = ActivationCollector(get_layers(model), mode=label)
    sample_rows = []

    model.eval()
    model = utils.dispatch_model_to_cuda(model)
    try:
        collector.register()
        grad_context = (
            torch.enable_grad() if label == "gradient" else torch.no_grad()
        )
        with grad_context:
            for sample_id, message in enumerate(tqdm(samples, desc="Parrot")):
                if prepare_inputs is None:
                    inputs = (
                        model_utils.build_loss_inputs(
                            message, image_processor, model
                        )
                        if label == "gradient"
                        else model_utils.message_to_prompt(
                            message, image_processor, type(model)
                        )
                    )
                else:
                    inputs = prepare_inputs(message, image_processor, type(model))
                if label == "gradient":
                    # _enable_floating_input_grads(inputs)
                    pass
                vision_mask, caption_mask = get_masks(model, inputs)
                collector.set_sample(sample_id, vision_mask, caption_mask)
                sample_row = describe_sample(
                    sample_id, message, vision_mask, caption_mask, mode=label
                )
                row_start = len(collector.rows)
                if label == "gradient":
                    model.zero_grad(set_to_none=True)
                if forward_sample is None:
                    output = model(**inputs, use_cache=False)
                else:
                    output = forward_sample(model, inputs)
                if label == "gradient":
                    loss = _output_to_loss(output)
                    with save_on_cpu():
                        loss.backward()
                sample_values = [
                    float(row["mean_abs"])
                    for row in collector.rows[row_start:]
                    if math.isfinite(float(row["mean_abs"]))
                ]
                sample_row[f"mean_{label}"] = (
                    float(np.mean(sample_values))
                    if sample_values else math.nan
                )
                sample_rows.append(sample_row)
                if label == "gradient":
                    del loss
                del inputs, output
                utils.cleanup_memory(verbos=False)
    finally:
        collector.remove()
        model = utils.move_model_to_cpu(model)
        utils.cleanup_memory(verbos=False)

    return write_analysis_outputs(
        collector.rows, sample_rows, output_dir, topk=topk, mode=label
    )
