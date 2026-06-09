#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, FIGURES_DIR, OUTPUTS_DIR, RUNS_DIR  # noqa: E402
from src.data import make_normalizer  # noqa: E402
from src.fourier import apply_perturbation, single_fourier_noise  # noqa: E402
from src.models import get_model  # noqa: E402


DATASET_ORDER = ("cifar10", "cifar100")
LAYER_KEYS = ("block1", "block2", "block3", "block4", "logits")
LAYER_TITLES = ("Block 1", "Block 2", "Block 3", "Block 4", "Logits")
DATASETS_CFG = {
    "cifar10": {
        "label": "CIFAR-10",
        "heatmap": RUNS_DIR / "fourier_vgg_small_bn_cifar10_eps10_255" / "heatmap.npy",
        "metadata": RUNS_DIR / "fourier_vgg_small_bn_cifar10_eps10_255" / "metadata.json",
        "random": RUNS_DIR / "random_noise_vgg_small_bn_cifar10_eps10_255" / "random_noise_baseline.json",
        "sweep": RUNS_DIR / "sfa_epsilon_sweep_vgg_small_bn_cifar10" / "sfa_epsilon_sweep.json",
        "sweep_frequency": (16, 24),
    },
    "cifar100": {
        "label": "CIFAR-100",
        "heatmap": RUNS_DIR / "fourier_vgg_small_bn_cifar100_eps10_255" / "heatmap.npy",
        "metadata": RUNS_DIR / "fourier_vgg_small_bn_cifar100_eps10_255" / "metadata.json",
        "random": RUNS_DIR / "random_noise_vgg_small_bn_cifar100_eps10_255" / "random_noise_baseline.json",
        "sweep": RUNS_DIR / "sfa_epsilon_sweep_vgg_small_bn_cifar100_freq16_16" / "sfa_epsilon_sweep.json",
        "sweep_frequency": (16, 16),
    },
}
RANDOM_SWEEP_JSON = (
    RUNS_DIR
    / "sfa_epsilon_sweep_with_random_comparison_vgg_cifar10_cifar100"
    / "sfa_vs_random_epsilon_sweep.json"
)
SPECTRUM_DIR = RUNS_DIR / "mean_clean_spectrum_comparison"
VISUAL_TRACE_DIR = RUNS_DIR / "sfa_visual_progression_vgg_cifar10"
VISUAL_TRACE_PATH = VISUAL_TRACE_DIR / "sfa_visual_progression_metadata.json"
LAYERWISE_COMPARISON_DIR = RUNS_DIR / "layerwise_fourier_heatmaps_comparison_vgg_cifar10_cifar100"
MULTISEED_DIR = RUNS_DIR / "layerwise_frequency_sensitivity_vgg_cifar10_multiseed"
COLOR_SFA_DIR = RUNS_DIR / "color_sfa_vgg_small_bn_cifar10_eps10_255"
COLOR_SFA_COMPARISON_DIR = RUNS_DIR / "color_sfa_comparison" / "cifar10"
SFA_FREQUENCY = (16, 24)
SFA_EPSILON_LABELS = ["1/255", "2/255", "3/255", "5/255", "10/255"]
SFA_SELECTION_SEED = 42
NUM_VISUAL_EXAMPLES = 4
GROUP_COLORS = {
    "high": "#3b6ea8",
    "medium": "#666666",
    "low": "#2a8c6a",
    "main": "#b2182b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the final report figures from existing experimental artifacts."
    )
    parser.add_argument("--figure-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="Only verify that the artifacts needed for the final report figures exist.",
    )
    parser.add_argument(
        "--skip-visual-progression",
        action="store_true",
        help="Skip the CIFAR-10 qualitative progression figure that needs torch and the checkpoint.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    require_file(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def required_input_files(*, skip_visual_progression: bool) -> list[Path]:
    files: list[Path] = []
    for config in DATASETS_CFG.values():
        files.extend(
            [
                config["heatmap"],
                config["metadata"],
                config["random"],
                config["sweep"],
            ]
        )
    files.append(RANDOM_SWEEP_JSON)
    files.extend(
        [
            SPECTRUM_DIR / "mean_clean_spectrum_cifar10.npy",
            SPECTRUM_DIR / "mean_clean_spectrum_cifar100.npy",
            LAYERWISE_COMPARISON_DIR / "layerwise_fourier_heatmaps_comparison.json",
            LAYERWISE_COMPARISON_DIR / "layerwise_fourier_heatmaps_comparison.npz",
            MULTISEED_DIR / "layerwise_frequency_sensitivity_multiseed.json",
            MULTISEED_DIR / "layerwise_group_summary.csv",
            MULTISEED_DIR / "logits_vs_fool_ratio.csv",
            COLOR_SFA_COMPARISON_DIR / "color_direction_barplot.png",
            COLOR_SFA_COMPARISON_DIR / "color_direction_barplot.pdf",
        ]
    )
    for dataset_name in DATASET_ORDER:
        aggregate_dir = RUNS_DIR / "stability_vgg_small_bn_retrained" / dataset_name / "aggregate"
        files.extend(
            [
                aggregate_dir / "heatmap_mean.png",
                aggregate_dir / "heatmap_mean.pdf",
                aggregate_dir / "heatmap_std.png",
                aggregate_dir / "heatmap_std.pdf",
            ]
        )
    for constraint in ("linf", "l1"):
        constraint_dir = COLOR_SFA_DIR / constraint
        files.extend(
            [
                constraint_dir / "metadata.json",
                constraint_dir / "summary.json",
                constraint_dir / "best_a_index.npy",
                constraint_dir / "best_fool_ratio.npy",
            ]
        )
    if not skip_visual_progression:
        files.append(OUTPUTS_DIR / "models" / "vgg_small_bn_cifar10_final.pth")
    return files


def check_required_inputs(*, skip_visual_progression: bool) -> None:
    missing = [path for path in required_input_files(skip_visual_progression=skip_visual_progression) if not path.exists()]
    if not missing:
        print("All report figure inputs are available.")
        return

    python = sys.executable
    missing_lines = "\n".join(f"  - {path}" for path in missing)
    commands = "\n".join(
        [
            f"  make train-vgg DATASET=cifar10 PYTHON={python}",
            f"  make train-vgg DATASET=cifar100 PYTHON={python}",
            f"  make heatmap-vgg DATASET=cifar10 PYTHON={python}",
            f"  make heatmap-vgg DATASET=cifar100 PYTHON={python}",
            f"  make random-vgg DATASET=cifar10 PYTHON={python}",
            f"  make random-vgg DATASET=cifar100 PYTHON={python}",
            (
                "  make sfa-eps-sweep-vgg DATASET=cifar10 "
                f"SFA_FREQUENCY_I=16 SFA_FREQUENCY_J=24 PYTHON={python}"
            ),
            (
                "  make sfa-eps-sweep-vgg DATASET=cifar100 "
                "SFA_FREQUENCY_I=16 SFA_FREQUENCY_J=16 "
                "SFA_SWEEP_OUTPUT_DIR=outputs/runs/sfa_epsilon_sweep_vgg_small_bn_cifar100_freq16_16 "
                f"PYTHON={python}"
            ),
            (
                f"  {python} scripts/evaluate_sfa_stability.py --datasets cifar10 cifar100 "
                "--seeds 0 1 2 3 4 --checkpoint-dir outputs/models/stability_retrained "
                "--output-dir outputs/runs/stability_vgg_small_bn_retrained --download"
            ),
        ]
    )
    raise SystemExit(
        "Missing artifacts required to build the final report figures:\n"
        f"{missing_lines}\n\n"
        "Regenerate the trainable/checkpoint-based artifacts with:\n"
        f"{commands}\n\n"
        "The layerwise, multiseed and Color-SFA artifacts are long-running experiment outputs; "
        "restore them or rerun their experiment recipes before building the report figures."
    )


def save_png_pdf(fig: plt.Figure, figure_dir: Path, stem: str, *, dpi: int | None = None) -> tuple[Path, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    png_path = figure_dir / f"{stem}.png"
    pdf_path = figure_dir / f"{stem}.pdf"
    if dpi is None:
        fig.savefig(png_path, bbox_inches="tight")
    else:
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure: {png_path}")
    print(f"Figure: {pdf_path}")
    return png_path, pdf_path


def parse_epsilon_label(label: str) -> float:
    text = str(label).strip()
    if "/" in text:
        numerator, denominator = text.split("/", maxsplit=1)
        return float(numerator) / float(denominator)
    return float(text)


def epsilon_axis_values(labels: list[str]) -> np.ndarray:
    return np.asarray([255.0 * parse_epsilon_label(label) for label in labels], dtype=float)


def finite_values(heatmap: np.ndarray) -> np.ndarray:
    return heatmap[np.isfinite(heatmap)]


def conjugate_frequency(freq_i: int, freq_j: int, size: int) -> tuple[int, int]:
    return (-int(freq_i)) % size, (-int(freq_j)) % size


def format_frequency(frequency: tuple[int, int]) -> str:
    return f"({frequency[0]}, {frequency[1]})"


def frequency_ticks(size: int) -> tuple[list[int], list[str]]:
    return [0, size // 2, size - 1], [str(-size // 2), "0", str(size // 2 - 1)]


def normalize_with_range(values: np.ndarray, min_value: float | None, max_value: float | None) -> np.ndarray:
    if min_value is None or max_value is None:
        return np.full_like(values, np.nan, dtype=np.float32)
    if max_value <= min_value:
        normalized = np.zeros_like(values, dtype=np.float32)
        normalized[~np.isfinite(values)] = np.nan
        return normalized
    normalized = (values - min_value) / (max_value - min_value)
    normalized[~np.isfinite(values)] = np.nan
    return normalized.astype(np.float32)


def layer_display_names(layer_names: list[str]) -> list[str]:
    mapping = {
        "input": "Input",
        "output block 1": "Block 1",
        "output block 2": "Block 2",
        "output block 3": "Block 3",
        "output block 4": "Block 4",
        "logits": "Logits",
    }
    return [mapping.get(name, name) for name in layer_names]


def format_correlation(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.3f}"


def top_k_frequencies(heatmap: np.ndarray, k: int = 10) -> list[dict[str, Any]]:
    entries = []
    size = heatmap.shape[0]
    for i in range(size):
        for j in range(size):
            value = heatmap[i, j]
            if np.isfinite(value):
                entries.append((float(value), i, j))
    entries.sort(key=lambda item: (-item[0], item[1], item[2]))

    rows: list[dict[str, Any]] = []
    for rank, (value, i, j) in enumerate(entries[:k], start=1):
        conjugate_i, conjugate_j = conjugate_frequency(i, j, size)
        rows.append(
            {
                "rank": rank,
                "freq_i": i,
                "freq_j": j,
                "fool_ratio": value,
                "conjugate_i": conjugate_i,
                "conjugate_j": conjugate_j,
                "conjugate_fool_ratio": float(heatmap[conjugate_i, conjugate_j]),
            }
        )
    return rows


def sweep_value_at_label(sweep: dict[str, Any], target_label: str = "10/255") -> float:
    labels = [str(label) for label in sweep["epsilon_labels"]]
    if target_label in labels:
        return float(sweep["fool_ratios"][labels.index(target_label)])
    target_value = parse_epsilon_label(target_label)
    values = [parse_epsilon_label(label) for label in labels]
    index = int(np.argmin(np.abs(np.asarray(values) - target_value)))
    return float(sweep["fool_ratios"][index])


def load_experiment_results() -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for dataset_name, cfg in DATASETS_CFG.items():
        for key in ("heatmap", "metadata", "random", "sweep"):
            require_file(cfg[key])

        heatmap = np.load(cfg["heatmap"])
        if heatmap.ndim != 2 or heatmap.shape[0] != heatmap.shape[1]:
            raise ValueError(f"Invalid heatmap shape for {dataset_name}: {heatmap.shape}")

        metadata = load_json(cfg["metadata"])
        random_baseline = load_json(cfg["random"])
        sweep = load_json(cfg["sweep"])
        values = finite_values(heatmap)
        max_frequency = tuple(int(x) for x in np.unravel_index(np.nanargmax(heatmap), heatmap.shape))
        sweep_labels = [str(label) for label in sweep["epsilon_labels"]]

        results[dataset_name] = {
            "label": cfg["label"],
            "heatmap": heatmap,
            "metadata": metadata,
            "random": random_baseline,
            "sweep": sweep,
            "values": values,
            "max_frequency": max_frequency,
            "max_fool_ratio": float(heatmap[max_frequency]),
            "mean_fool_ratio": float(values.mean()),
            "random_mean": float(random_baseline["mean"]),
            "random_std": float(random_baseline["std"]),
            "clean_accuracy": float(sweep.get("clean_accuracy", np.nan)),
            "sweep_frequency": (
                int(sweep.get("frequency_i", cfg["sweep_frequency"][0])),
                int(sweep.get("frequency_j", cfg["sweep_frequency"][1])),
            ),
            "sweep_labels": sweep_labels,
            "sweep_x255": epsilon_axis_values(sweep_labels),
            "sweep_fool_ratios": np.asarray(sweep["fool_ratios"], dtype=float),
            "sweep_at_10_255": sweep_value_at_label(sweep, "10/255"),
            "top10": top_k_frequencies(heatmap, k=10),
        }
    return results


def plot_heatmap_comparison(results: dict[str, dict[str, Any]], figure_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    last_image = None
    for ax, dataset_name in zip(axes, DATASET_ORDER):
        result = results[dataset_name]
        heatmap = result["heatmap"]
        last_image = ax.imshow(heatmap, origin="lower", interpolation="nearest", cmap="magma", vmin=0.0, vmax=1.0)
        max_i, max_j = result["max_frequency"]
        ax.scatter([max_j], [max_i], marker="x", s=70, linewidths=1.8, color="white")
        ax.set_title(
            f"{result['label']} - max {format_frequency(result['max_frequency'])}, "
            f"FR={result['max_fool_ratio']:.4f}"
        )
        ax.set_xlabel("Frequence v")
        ax.set_ylabel("Frequence u")
        ax.set_xticks([0, 16, 31])
        ax.set_yticks([0, 16, 31])
    fig.colorbar(last_image, ax=axes, shrink=0.92, label="Fool ratio")
    save_png_pdf(fig, figure_dir, "fourier_heatmap_comparison_vgg_cifar10_cifar100")


def plot_fool_ratio_histogram(results: dict[str, dict[str, Any]], figure_dir: Path) -> None:
    bins = np.linspace(0.0, 1.0, 41)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharex=True, sharey=True, constrained_layout=True)
    for ax, dataset_name in zip(axes, DATASET_ORDER):
        result = results[dataset_name]
        ax.hist(result["values"], bins=bins, color="tab:blue", alpha=0.78, edgecolor="white")
        ax.axvline(
            result["mean_fool_ratio"],
            color="black",
            linestyle="-",
            linewidth=1.6,
            label=f"Moyenne heatmap: {result['mean_fool_ratio']:.3f}",
        )
        ax.axvline(
            result["random_mean"],
            color="tab:gray",
            linestyle="--",
            linewidth=1.6,
            label=f"Random: {result['random_mean']:.3f}",
        )
        ax.set_title(result["label"])
        ax.set_xlabel("Fool ratio")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Nombre de frequences")
    save_png_pdf(fig, figure_dir, "fourier_fool_ratio_histogram_comparison_vgg_cifar10_cifar100")


def plot_sfa_vs_random(results: dict[str, dict[str, Any]], figure_dir: Path) -> None:
    labels = [results[name]["label"] for name in DATASET_ORDER]
    x = np.arange(len(DATASET_ORDER))
    width = 0.34
    sfa_values = np.asarray([results[name]["max_fool_ratio"] for name in DATASET_ORDER])
    random_values = np.asarray([results[name]["random_mean"] for name in DATASET_ORDER])
    random_stds = np.asarray([results[name]["random_std"] for name in DATASET_ORDER])

    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    ax.bar(x - width / 2, sfa_values, width, label="Meilleure SFA", color="tab:blue")
    ax.bar(
        x + width / 2,
        random_values,
        width,
        yerr=random_stds,
        capsize=4,
        label="Bruit aleatoire",
        color="tab:gray",
        alpha=0.82,
    )
    for xpos, value in zip(x - width / 2, sfa_values):
        ax.text(xpos, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    for xpos, value in zip(x + width / 2, random_values):
        ax.text(xpos, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fool ratio a epsilon = 10/255")
    ax.set_title("Meilleure direction SFA vs bruit aleatoire")
    ax.grid(axis="y", alpha=0.24)
    ax.legend(frameon=False)
    save_png_pdf(fig, figure_dir, "sfa_vs_random_comparison_vgg_cifar10_cifar100")


def plot_random_sweep(results: dict[str, dict[str, Any]], figure_dir: Path) -> None:
    payload = load_json(RANDOM_SWEEP_JSON)
    x255 = np.asarray(payload["epsilons"]["x255"], dtype=float)
    plot_specs = {
        "cifar10": {
            "color": "tab:blue",
            "sfa": payload["sfa_cifar10"],
            "random_mean": payload["random_mean_cifar10"],
            "random_std": payload["random_std_cifar10"],
            "freq": results["cifar10"]["sweep_frequency"],
        },
        "cifar100": {
            "color": "tab:orange",
            "sfa": payload["sfa_cifar100"],
            "random_mean": payload["random_mean_cifar100"],
            "random_std": payload["random_std_cifar100"],
            "freq": results["cifar100"]["sweep_frequency"],
        },
    }

    fig, ax = plt.subplots(figsize=(8.8, 5.1), constrained_layout=True)
    for dataset_name in DATASET_ORDER:
        spec = plot_specs[dataset_name]
        label = results[dataset_name]["label"]
        freq_i, freq_j = spec["freq"]
        color = spec["color"]
        random_mean = np.asarray(spec["random_mean"], dtype=float)
        random_std = np.asarray(spec["random_std"], dtype=float)
        ax.plot(
            x255,
            spec["sfa"],
            marker="o",
            linewidth=2.0,
            color=color,
            label=f"{label} - SFA ({freq_i},{freq_j})",
        )
        ax.errorbar(
            x255,
            random_mean,
            yerr=random_std,
            marker="s",
            linewidth=1.7,
            linestyle="--",
            capsize=3,
            color=color,
            alpha=0.9,
            label=f"{label} - bruit aléatoire",
        )
        ax.fill_between(x255, random_mean - random_std, random_mean + random_std, color=color, alpha=0.10, linewidth=0.0)

    ax.set_title("Effet de l'amplitude : SFA vs bruit aléatoire")
    ax.set_xlabel("epsilon (niveaux / 255)")
    ax.set_ylabel("Fool ratio")
    ax.set_xticks(x255)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=1)
    save_png_pdf(fig, figure_dir, "sfa_epsilon_sweep_with_random_comparison_vgg_cifar10_cifar100")


def write_top_frequencies(results: dict[str, dict[str, Any]], figure_dir: Path) -> None:
    path = figure_dir / "top_frequencies_vgg_cifar10_cifar100.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "rank",
        "freq_i",
        "freq_j",
        "fool_ratio",
        "conjugate_i",
        "conjugate_j",
        "conjugate_fool_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset_name in DATASET_ORDER:
            for row in results[dataset_name]["top10"]:
                writer.writerow({"dataset": dataset_name, **row})
    print(f"CSV: {path}")


def plot_spatial_pattern(figure_dir: Path) -> None:
    epsilon = parse_epsilon_label("10/255")
    noise = single_fourier_noise(
        32,
        SFA_FREQUENCY[0],
        SFA_FREQUENCY[1],
        epsilon,
        channels=1,
    ).squeeze(0)
    pattern = noise.cpu().numpy()
    display = (pattern - pattern.min()) / (pattern.max() - pattern.min())

    fig, axes = plt.subplots(1, 2, figsize=(7.9, 3.9), constrained_layout=True)
    image = axes[0].imshow(pattern, origin="lower", cmap="coolwarm", vmin=-epsilon, vmax=epsilon)
    axes[0].set_title("Perturbation reelle\nbornee par epsilon=10/255")
    axes[1].imshow(display, origin="lower", cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("Version normalisee\npour visualisation")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    save_png_pdf(fig, figure_dir, "sfa_spatial_pattern_vgg_cifar10")


def safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_model_name(checkpoint: Any) -> str:
    if isinstance(checkpoint, dict):
        if checkpoint.get("model_name"):
            return str(checkpoint["model_name"])
        config = checkpoint.get("config")
        if isinstance(config, dict):
            return str(config.get("model_name") or "vgg_small_bn")
        return str(getattr(config, "model_name", "vgg_small_bn") or "vgg_small_bn")
    return "vgg_small_bn"


def load_visual_model(device: torch.device) -> tuple[torch.nn.Module, str]:
    checkpoint_path = OUTPUTS_DIR / "models" / "vgg_small_bn_cifar10_final.pth"
    require_file(checkpoint_path)
    checkpoint = safe_torch_load(checkpoint_path)
    model_name = checkpoint_model_name(checkpoint)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model = get_model(model_name, num_classes=10)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, model_name


def plot_visual_progression(figure_dir: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_name = load_visual_model(device)
    normalizer = make_normalizer("cifar10", device, torch.float32)
    cifar10_test = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )
    class_names = list(cifar10_test.classes)

    def predict_image(image: torch.Tensor) -> int:
        with torch.inference_mode():
            logits = model(normalizer(image.unsqueeze(0).to(device)))
        return int(logits.argmax(dim=1).item())

    rng = np.random.default_rng(SFA_SELECTION_SEED)
    candidate_indices = rng.permutation(len(cifar10_test))
    selected_examples: list[dict[str, Any]] = []
    used_classes: set[int] = set()

    for raw_index in candidate_indices:
        index = int(raw_index)
        image, target = cifar10_test[index]
        target = int(target)
        if target in used_classes:
            continue
        prediction = predict_image(image)
        if prediction == target:
            selected_examples.append(
                {
                    "index": index,
                    "image": image,
                    "target": target,
                    "clean_prediction": prediction,
                }
            )
            used_classes.add(target)
        if len(selected_examples) == NUM_VISUAL_EXAMPLES:
            break

    if len(selected_examples) < NUM_VISUAL_EXAMPLES:
        selected_indices = {example["index"] for example in selected_examples}
        for raw_index in candidate_indices:
            index = int(raw_index)
            if index in selected_indices:
                continue
            image, target = cifar10_test[index]
            target = int(target)
            prediction = predict_image(image)
            if prediction == target:
                selected_examples.append(
                    {
                        "index": index,
                        "image": image,
                        "target": target,
                        "clean_prediction": prediction,
                    }
                )
                selected_indices.add(index)
            if len(selected_examples) == NUM_VISUAL_EXAMPLES:
                break

    if len(selected_examples) < NUM_VISUAL_EXAMPLES:
        raise RuntimeError("Could not find four correctly classified CIFAR-10 images.")

    epsilons = [parse_epsilon_label(label) for label in SFA_EPSILON_LABELS]
    column_titles = ["Original"] + [f"epsilon = {label}" for label in SFA_EPSILON_LABELS]
    fig, axes = plt.subplots(
        NUM_VISUAL_EXAMPLES,
        len(column_titles),
        figsize=(11.8, 7.4),
        constrained_layout=True,
    )
    metadata_rows: list[dict[str, Any]] = []

    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, fontsize=10)

    for row, example in enumerate(selected_examples):
        image = example["image"]
        variants = [image]
        epsilon_values = [0.0]
        epsilon_labels = ["0"]
        for epsilon_label, epsilon in zip(SFA_EPSILON_LABELS, epsilons):
            noise = single_fourier_noise(
                image.shape[-1],
                SFA_FREQUENCY[0],
                SFA_FREQUENCY[1],
                epsilon,
                channels=image.shape[0],
                device=image.device,
                dtype=image.dtype,
            )
            variants.append(apply_perturbation(image, noise))
            epsilon_values.append(float(epsilon))
            epsilon_labels.append(epsilon_label)

        predictions = [predict_image(variant) for variant in variants]
        target_name = class_names[example["target"]]
        clean_prediction_name = class_names[example["clean_prediction"]]

        for column, (variant, prediction) in enumerate(zip(variants, predictions)):
            ax = axes[row, column]
            ax.imshow(variant.permute(1, 2, 0).cpu().numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"pred. {class_names[prediction]}", fontsize=7, labelpad=2)
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
                spine.set_edgecolor("#444444")

        axes[row, 0].set_ylabel(
            f"vrai: {target_name}\npropre: {clean_prediction_name}",
            fontsize=8,
            rotation=0,
            labelpad=46,
            va="center",
        )
        metadata_rows.append(
            {
                "index": int(example["index"]),
                "true_label": int(example["target"]),
                "true_class": target_name,
                "clean_prediction": int(example["clean_prediction"]),
                "clean_prediction_class": clean_prediction_name,
                "epsilon_labels": epsilon_labels,
                "epsilon_values": epsilon_values,
                "predictions": [int(prediction) for prediction in predictions],
                "prediction_classes": [class_names[prediction] for prediction in predictions],
            }
        )

    fig.suptitle(
        "Progression visuelle de la Single Fourier Attack - VGGSmallBN sur CIFAR-10",
        y=1.02,
        fontsize=12,
    )
    save_png_pdf(fig, figure_dir, "sfa_visual_progression_vgg_cifar10")

    VISUAL_TRACE_DIR.mkdir(parents=True, exist_ok=True)
    trace = {
        "dataset": "cifar10",
        "model_name": model_name,
        "checkpoint": str(OUTPUTS_DIR / "models" / "vgg_small_bn_cifar10_final.pth"),
        "frequency": list(SFA_FREQUENCY),
        "epsilon_labels": SFA_EPSILON_LABELS,
        "epsilon_values": epsilons,
        "selection_seed": SFA_SELECTION_SEED,
        "num_images": NUM_VISUAL_EXAMPLES,
        "selection_rule": (
            "first correctly classified images with distinct true classes from a fixed random "
            "permutation; fallback to correctly classified images if needed"
        ),
        "images": metadata_rows,
    }
    with VISUAL_TRACE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(trace, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"JSON: {VISUAL_TRACE_PATH}")


def plot_mean_clean_spectrum(figure_dir: Path) -> None:
    spectra = {
        "CIFAR-10": np.load(SPECTRUM_DIR / "mean_clean_spectrum_cifar10.npy"),
        "CIFAR-100": np.load(SPECTRUM_DIR / "mean_clean_spectrum_cifar100.npy"),
    }
    global_min = min(float(spectrum.min()) for spectrum in spectra.values())
    global_max = max(float(spectrum.max()) for spectrum in spectra.values())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    image = None
    for ax, (name, spectrum) in zip(axes, spectra.items()):
        image = ax.imshow(
            spectrum,
            origin="lower",
            cmap="magma",
            vmin=global_min,
            vmax=global_max,
        )
        ax.set_title(name)
        ax.set_xlabel("fréquence v")
        ax.set_ylabel("fréquence u")
        ax.set_xticks([0, 16, 31])
        ax.set_yticks([0, 16, 31])
    colorbar = fig.colorbar(image, ax=axes, fraction=0.03, pad=0.04)
    colorbar.set_label(r"$\log(1 + |\mathcal{F}(x)|)$")
    fig.suptitle("Spectre de Fourier moyen des images propres (CIFAR)", fontsize=16)
    save_png_pdf(fig, figure_dir, "mean_clean_spectrum_comparison_cifar", dpi=300)


def load_color_sfa_constraint(constraint: str) -> dict[str, Any]:
    constraint_dir = COLOR_SFA_DIR / constraint
    metadata = load_json(constraint_dir / "metadata.json")
    summary = load_json(constraint_dir / "summary.json")
    best_a_index = np.load(constraint_dir / "best_a_index.npy").astype(int)
    best_fool_ratio = np.load(constraint_dir / "best_fool_ratio.npy").astype(np.float32)
    labels = [str(entry["label"]) for entry in metadata["a_vectors"]]
    return {
        "metadata": metadata,
        "summary": summary,
        "best_a_index": best_a_index,
        "best_fool_ratio": best_fool_ratio,
        "labels": labels,
    }


def color_vector_cmap(num_vectors: int) -> tuple[ListedColormap, BoundaryNorm]:
    colors = plt.get_cmap("tab20", max(num_vectors, 1))(np.arange(num_vectors))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, num_vectors + 0.5, 1.0), num_vectors)
    return cmap, norm


def plot_color_vector_grid(ax: plt.Axes, result: dict[str, Any], title: str) -> None:
    labels = result["labels"]
    best_a_index = result["best_a_index"]
    summary = result["summary"]
    image_size = int(best_a_index.shape[0])
    cmap, norm = color_vector_cmap(len(labels))
    ax.imshow(best_a_index, origin="lower", interpolation="nearest", cmap=cmap, norm=norm)
    best_i, best_j = (int(value) for value in summary["best_frequency"])
    ax.plot(
        best_j,
        best_i,
        marker="o",
        markersize=6.0,
        markerfacecolor="none",
        markeredgecolor="white",
        markeredgewidth=1.2,
    )
    ticks, tick_labels = frequency_ticks(image_size)
    ax.set_xticks(ticks, tick_labels)
    ax.set_yticks(ticks, tick_labels)
    ax.set_xlabel("v")
    ax.set_ylabel("u")
    ax.set_title(title)

    handles = [Patch(facecolor=cmap(index), label=label) for index, label in enumerate(labels)]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=False,
        fontsize=7,
        title="Vecteur a",
        title_fontsize=8,
    )


def plot_color_sfa_composite(results: dict[str, dict[str, Any]], figure_dir: Path) -> None:
    linf = load_color_sfa_constraint("linf")
    l1 = load_color_sfa_constraint("l1")
    cifar10 = results["cifar10"]
    bar_values = [
        float(cifar10["random_mean"]),
        float(cifar10["max_fool_ratio"]),
        float(linf["summary"]["best_fool_ratio_global"]),
        float(l1["summary"]["best_fool_ratio_global"]),
    ]
    bar_labels = ["Random\nnoise", "SFA\nstandard", "Color-SFA\nL_inf", "Color-SFA\nL1"]
    bar_colors = ["#777777", "#3b6ea8", "#6f4aa8", "#b66b2e"]

    fig = plt.figure(figsize=(15.7, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[0.85, 1.0, 1.0])
    bar_ax = fig.add_subplot(grid[0, 0])
    linf_ax = fig.add_subplot(grid[0, 1])
    l1_ax = fig.add_subplot(grid[0, 2])

    x = np.arange(len(bar_values))
    bar_ax.bar(x, bar_values, color=bar_colors, width=0.68)
    bar_ax.set_xticks(x, bar_labels)
    bar_ax.set_ylim(0.0, max(bar_values) * 1.18)
    bar_ax.set_ylabel("Fool ratio")
    bar_ax.set_title("(a) Best fool ratio")
    bar_ax.grid(axis="y", alpha=0.22)
    for index, value in enumerate(bar_values):
        bar_ax.text(index, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    for offset, result in enumerate((linf, l1), start=2):
        summary = result["summary"]
        text = (
            f"FR = {float(summary['best_fool_ratio_global']):.3f}\n"
            f"(u,v)=({summary['best_frequency'][0]}, {summary['best_frequency'][1]})\n"
            f"a={summary['best_a_label']}"
        )
        bar_ax.text(
            offset,
            bar_values[offset] * 0.50,
            text,
            ha="center",
            va="center",
            fontsize=8,
            color="white",
        )

    plot_color_vector_grid(
        linf_ax,
        linf,
        "(b) Best RGB vector a per frequency\nL_inf constraint",
    )
    plot_color_vector_grid(
        l1_ax,
        l1,
        "(c) Best RGB vector a per frequency\nL1 redistributed budget",
    )
    fig.suptitle("Color-SFA - CIFAR-10, VGGSmallBN, epsilon=10/255", fontsize=13)
    save_png_pdf(fig, figure_dir, "color_sfa_foolratio_and_activation_grids", dpi=300)


def plot_layerwise_comparison(
    heatmap_results: dict[str, dict[str, Any]],
    column_ranges: dict[str, dict[str, float | None]],
    *,
    epsilon: float,
    png_path: Path,
    pdf_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 6, figsize=(15.4, 5.7), constrained_layout=True)
    cmap_internal = plt.get_cmap("magma").copy()
    cmap_internal.set_bad(color="#d9d9d9")
    cmap_fool = plt.get_cmap("magma").copy()
    cmap_fool.set_bad(color="#d9d9d9")

    internal_image = None
    fool_image = None
    for col_index, title in enumerate((*LAYER_TITLES, "Fool ratio")):
        axes[0, col_index].set_title(title)

    for row_index, dataset_name in enumerate(DATASET_ORDER):
        result = heatmap_results[dataset_name]
        image_size = int(result["image_size"])
        ticks, labels = frequency_ticks(image_size)
        for col_index, layer_key in enumerate(LAYER_KEYS):
            ax = axes[row_index, col_index]
            display_heatmap = normalize_with_range(
                result["layer_heatmaps"][layer_key],
                column_ranges[layer_key]["min"],
                column_ranges[layer_key]["max"],
            )
            internal_image = ax.imshow(
                np.ma.masked_invalid(display_heatmap),
                origin="lower",
                interpolation="nearest",
                cmap=cmap_internal,
                vmin=0.0,
                vmax=1.0,
            )
            ax.set_xticks(ticks, labels)
            ax.set_yticks(ticks, labels)
            ax.set_xlabel("v")

        fool_ax = axes[row_index, 5]
        fool_image = fool_ax.imshow(
            np.ma.masked_invalid(result["fool_ratio_heatmap"]),
            origin="lower",
            interpolation="nearest",
            cmap=cmap_fool,
            vmin=0.0,
            vmax=1.0,
        )
        max_frequency = result["max_frequency"]
        fool_ax.plot(
            max_frequency[1],
            max_frequency[0],
            marker="o",
            markersize=5.0,
            markerfacecolor="none",
            markeredgecolor="white",
            markeredgewidth=1.1,
        )
        fool_ax.set_xticks(ticks, labels)
        fool_ax.set_yticks(ticks, labels)
        fool_ax.set_xlabel("v")
        axes[row_index, 0].set_ylabel(f"{DATASETS_CFG[dataset_name]['label']}\nu")

    if internal_image is not None:
        colorbar = fig.colorbar(internal_image, ax=axes[:, :5].ravel().tolist(), fraction=0.018, pad=0.01)
        colorbar.set_label("A_l normalisé par colonne")
    if fool_image is not None:
        colorbar = fig.colorbar(fool_image, ax=axes[:, 5].ravel().tolist(), fraction=0.046, pad=0.02)
        colorbar.set_label("Fool ratio")

    fig.suptitle(
        (
            "Heatmaps internes Fourier par couche - VGGSmallBN CIFAR-10 / CIFAR-100, "
            f"epsilon={epsilon:.5f}"
        ),
        fontsize=11,
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_multiseed_figure(
    *,
    layer_names: list[str],
    group_stats: dict[str, dict[str, Any]],
    scatter_rows: list[dict[str, Any]],
    main_frequency: tuple[int, int],
    pearson: float,
    band: str,
    png_path: Path,
    pdf_path: Path,
) -> None:
    display_names = layer_display_names(layer_names)[1:]
    x_layers = np.arange(len(display_names))
    band_label = "SEM" if band == "sem" else "écart-type"

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.6, 5.0),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )
    panel_a, panel_b = axes

    for group in ("high", "medium", "low", "main"):
        stats = group_stats[group]
        mean = np.array(stats["mean"], dtype=np.float64)
        band_values = np.array(stats[band], dtype=np.float64)
        color = GROUP_COLORS[group]
        marker = "o" if group == "main" else None
        linewidth = 2.8 if group == "main" else 2.0
        zorder = 5 if group == "main" else 3
        panel_a.plot(
            x_layers,
            mean,
            marker=marker,
            linewidth=linewidth,
            color=color,
            label=f"{stats['label']} (n={stats['count']})",
            zorder=zorder,
        )
        panel_a.fill_between(
            x_layers,
            mean - band_values,
            mean + band_values,
            color=color,
            alpha=0.13 if group != "main" else 0.18,
            linewidth=0.0,
            zorder=zorder - 1,
        )

    panel_a.set_title(f"A. Amplification multi-seeds par groupe ({band_label})")
    panel_a.set_xticks(x_layers)
    panel_a.set_xticklabels(display_names)
    panel_a.set_xlabel("Couche")
    panel_a.set_ylabel(r"Amplification normalisée $G_l(\delta)$")
    panel_a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=8)

    logits = np.array([row["logits_amplification_mean"] for row in scatter_rows], dtype=np.float64)
    fool_ratios = np.array([row["fool_ratio_mean"] for row in scatter_rows], dtype=np.float64)
    is_main = np.array([bool(row["is_main_frequency"]) for row in scatter_rows])

    scatter = panel_b.scatter(
        logits,
        fool_ratios,
        c=fool_ratios,
        cmap="viridis",
        s=25,
        alpha=0.76,
        edgecolors="none",
    )
    if is_main.any():
        main_index = int(np.argmax(is_main))
        panel_b.scatter(
            [logits[main_index]],
            [fool_ratios[main_index]],
            color=GROUP_COLORS["main"],
            edgecolors="black",
            linewidths=0.8,
            s=96,
            marker="*",
            zorder=6,
        )
        panel_b.annotate(
            f"{main_frequency}",
            xy=(logits[main_index], fool_ratios[main_index]),
            xytext=(8, -10),
            textcoords="offset points",
            color=GROUP_COLORS["main"],
            fontsize=9,
        )

    if len(logits) >= 2 and float(np.std(logits)) > 0.0:
        slope, intercept = np.polyfit(logits, fool_ratios, deg=1)
        line_x = np.linspace(float(logits.min()), float(logits.max()), 120)
        panel_b.plot(line_x, slope * line_x + intercept, color="#222222", linewidth=1.2, linestyle="--", alpha=0.85)

    panel_b.set_title(f"B. Logits et fool ratio (r={format_correlation(pearson)})")
    panel_b.set_xlabel(r"$G_{\mathrm{logits}}(\delta)$ moyen")
    panel_b.set_ylabel("Fool ratio moyen")
    colorbar = fig.colorbar(scatter, ax=panel_b, fraction=0.046, pad=0.04)
    colorbar.set_label("Fool ratio moyen")

    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def copy_file(source: Path, destination: Path) -> None:
    require_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"Copy: {source} -> {destination}")


def copy_stem(source_stem: Path, figure_dir: Path, destination_stem: str, suffixes: tuple[str, ...]) -> None:
    for suffix in suffixes:
        copy_file(source_stem.with_suffix(f".{suffix}"), figure_dir / f"{destination_stem}.{suffix}")


def copy_cached_figures(figure_dir: Path) -> None:
    for dataset_name in DATASET_ORDER:
        aggregate_dir = RUNS_DIR / "stability_vgg_small_bn_retrained" / dataset_name / "aggregate"
        copy_stem(aggregate_dir / "heatmap_mean", figure_dir, f"stability_{dataset_name}_heatmap_mean", ("png", "pdf"))
        copy_stem(aggregate_dir / "heatmap_std", figure_dir, f"stability_{dataset_name}_heatmap_std", ("png", "pdf"))

    copy_stem(
        COLOR_SFA_COMPARISON_DIR / "color_direction_barplot",
        figure_dir,
        "color_sfa_cifar10_barplot",
        ("png", "pdf"),
    )


def plot_layerwise_heatmap_comparison_from_artifacts(figure_dir: Path) -> None:
    metadata = load_json(LAYERWISE_COMPARISON_DIR / "layerwise_fourier_heatmaps_comparison.json")
    arrays = np.load(LAYERWISE_COMPARISON_DIR / "layerwise_fourier_heatmaps_comparison.npz")
    layers = list(metadata["layers"])

    heatmap_results: dict[str, dict[str, Any]] = {}
    for dataset_name in DATASET_ORDER:
        layer_heatmaps = {
            layer: arrays[f"{dataset_name}_{layer}"].astype(np.float32)
            for layer in layers
        }
        fool_ratio = arrays[f"{dataset_name}_fool_ratio"].astype(np.float32)
        heatmap_results[dataset_name] = {
            "image_size": int(fool_ratio.shape[0]),
            "layer_heatmaps": layer_heatmaps,
            "fool_ratio_heatmap": fool_ratio,
            "max_frequency": tuple(metadata["max_frequency_by_dataset"][dataset_name]),
        }

    plot_layerwise_comparison(
        heatmap_results,
        metadata["internal_column_min_max"],
        epsilon=float(metadata["epsilon"]),
        png_path=figure_dir / "layerwise_fourier_heatmaps_comparison_vgg_cifar10_cifar100.png",
        pdf_path=figure_dir / "layerwise_fourier_heatmaps_comparison_vgg_cifar10_cifar100.pdf",
    )
    print(f"Figure: {figure_dir / 'layerwise_fourier_heatmaps_comparison_vgg_cifar10_cifar100.png'}")
    print(f"Figure: {figure_dir / 'layerwise_fourier_heatmaps_comparison_vgg_cifar10_cifar100.pdf'}")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    require_file(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_multiseed_from_artifacts(figure_dir: Path) -> None:
    metadata = load_json(MULTISEED_DIR / "layerwise_frequency_sensitivity_multiseed.json")
    group_rows = read_csv_rows(MULTISEED_DIR / "layerwise_group_summary.csv")
    scatter_rows_raw = read_csv_rows(MULTISEED_DIR / "logits_vs_fool_ratio.csv")

    group_stats: dict[str, dict[str, Any]] = {}
    for row in group_rows:
        group = row["group"]
        stats = group_stats.setdefault(
            group,
            {
                "label": row["group_label"],
                "count": int(row["num_frequencies"]),
                "mean": [],
                "std": [],
                "sem": [],
                "band": row["band"],
            },
        )
        stats["mean"].append(float(row["mean_amplification"]))
        stats["std"].append(float(row["std_across_seed_means"]))
        stats["sem"].append(float(row["sem_across_seed_means"]))

    scatter_rows: list[dict[str, Any]] = []
    for row in scatter_rows_raw:
        scatter_rows.append(
            {
                "frequency_i": int(row["frequency_i"]),
                "frequency_j": int(row["frequency_j"]),
                "frequency": row["frequency"],
                "groups": row["groups"],
                "fool_ratio_mean": float(row["fool_ratio_mean"]),
                "logits_amplification_mean": float(row["logits_amplification_mean"]),
                "logits_amplification_std": float(row["logits_amplification_std"]),
                "logits_amplification_sem": float(row["logits_amplification_sem"]),
                "is_main_frequency": row["is_main_frequency"] == "True",
                "per_seed_logits": row["per_seed_logits"],
            }
        )

    main_frequency = tuple(int(value) for value in metadata["main_frequency"])
    plot_multiseed_figure(
        layer_names=list(metadata["layers"]),
        group_stats=group_stats,
        scatter_rows=scatter_rows,
        main_frequency=main_frequency,
        pearson=float(metadata["pearson_logits_vs_fool_ratio"]),
        band=group_rows[0]["band"],
        png_path=figure_dir / "layerwise_frequency_sensitivity_analysis_vgg_cifar10_multiseed.png",
        pdf_path=figure_dir / "layerwise_frequency_sensitivity_analysis_vgg_cifar10_multiseed.pdf",
    )
    print(f"Figure: {figure_dir / 'layerwise_frequency_sensitivity_analysis_vgg_cifar10_multiseed.png'}")
    print(f"Figure: {figure_dir / 'layerwise_frequency_sensitivity_analysis_vgg_cifar10_multiseed.pdf'}")


def verify_expected_outputs(figure_dir: Path, *, visual_progression: bool) -> None:
    expected = [
        "color_sfa_cifar10_barplot.pdf",
        "color_sfa_foolratio_and_activation_grids.pdf",
        "fourier_fool_ratio_histogram_comparison_vgg_cifar10_cifar100.pdf",
        "fourier_heatmap_comparison_vgg_cifar10_cifar100.pdf",
        "layerwise_fourier_heatmaps_comparison_vgg_cifar10_cifar100.pdf",
        "layerwise_frequency_sensitivity_analysis_vgg_cifar10_multiseed.pdf",
        "mean_clean_spectrum_comparison_cifar.pdf",
        "sfa_epsilon_sweep_with_random_comparison_vgg_cifar10_cifar100.pdf",
        "sfa_spatial_pattern_vgg_cifar10.pdf",
        "sfa_vs_random_comparison_vgg_cifar10_cifar100.pdf",
        "stability_cifar10_heatmap_mean.pdf",
        "stability_cifar10_heatmap_std.pdf",
        "stability_cifar100_heatmap_mean.pdf",
        "stability_cifar100_heatmap_std.pdf",
        "top_frequencies_vgg_cifar10_cifar100.csv",
    ]
    if visual_progression:
        expected.append("sfa_visual_progression_vgg_cifar10.pdf")
    missing = [name for name in expected if not (figure_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Expected output files missing in {figure_dir}: {missing}")


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    figure_dir = args.figure_dir
    figure_dir.mkdir(parents=True, exist_ok=True)
    check_required_inputs(skip_visual_progression=args.skip_visual_progression)
    if args.check_inputs:
        return

    results = load_experiment_results()
    plot_heatmap_comparison(results, figure_dir)
    plot_fool_ratio_histogram(results, figure_dir)
    plot_sfa_vs_random(results, figure_dir)
    plot_random_sweep(results, figure_dir)
    write_top_frequencies(results, figure_dir)
    plot_spatial_pattern(figure_dir)
    plot_mean_clean_spectrum(figure_dir)
    plot_layerwise_heatmap_comparison_from_artifacts(figure_dir)
    plot_multiseed_from_artifacts(figure_dir)
    plot_color_sfa_composite(results, figure_dir)
    copy_cached_figures(figure_dir)
    if not args.skip_visual_progression:
        plot_visual_progression(figure_dir)
    verify_expected_outputs(figure_dir, visual_progression=not args.skip_visual_progression)
    print(f"Report figures written to: {figure_dir}")


if __name__ == "__main__":
    main()
