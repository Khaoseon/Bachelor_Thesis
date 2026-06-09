#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    SUPPORTED_DATASETS,
    build_test_dataset,
    get_dataset_spec,
    make_normalizer,
    normalize_dataset_name,
)
from src.models import get_model
from src.sfa_evaluation import (
    evaluate_fourier_frequency_metrics,
    load_images_and_labels_to_device,
    metrics_to_row,
    predict_all,
    write_csv,
    write_json,
)


def parse_epsilon(value: str) -> float:
    text = value.strip()
    if "/" in text:
        numerator, denominator = text.split("/", maxsplit=1)
        return float(numerator) / float(denominator)
    return float(text)


def to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return {
            field.name: to_serializable(getattr(value, field.name))
            for field in fields(value)
            if hasattr(value, field.name)
        }
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def get_checkpoint_model_name(checkpoint: Any) -> str:
    if not isinstance(checkpoint, dict):
        return "custom_cnn"

    if checkpoint.get("model_name"):
        return str(checkpoint["model_name"])

    config = checkpoint.get("config")
    if isinstance(config, dict):
        return str(config.get("model_name") or "custom_cnn")
    return str(getattr(config, "model_name", "custom_cnn") or "custom_cnn")


def get_checkpoint_dataset(checkpoint: Any) -> str | None:
    if not isinstance(checkpoint, dict):
        return None

    if checkpoint.get("dataset"):
        return str(checkpoint["dataset"])

    config = checkpoint.get("config")
    if isinstance(config, dict):
        dataset = config.get("dataset")
        return None if dataset is None else str(dataset)

    dataset = getattr(config, "dataset", None)
    return None if dataset is None else str(dataset)


def validate_checkpoint_dataset(checkpoint: Any, expected_dataset: str) -> str | None:
    checkpoint_dataset = get_checkpoint_dataset(checkpoint)
    if checkpoint_dataset is None:
        print(
            f"Warning: checkpoint has no dataset metadata; assuming {expected_dataset}.",
            file=sys.stderr,
        )
        return None

    normalized_checkpoint_dataset = normalize_dataset_name(checkpoint_dataset)
    if normalized_checkpoint_dataset != expected_dataset:
        raise ValueError(
            "Checkpoint dataset mismatch: "
            f"checkpoint={normalized_checkpoint_dataset!r}, requested={expected_dataset!r}."
        )
    return normalized_checkpoint_dataset


def build_test_loader(
    dataset_name: str,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    download: bool,
) -> tuple[DataLoader, list[str], int]:
    dataset = build_test_dataset(
        dataset_name,
        download=download,
        transform=transforms.ToTensor(),
    )
    class_names = list(dataset.classes)
    requested_images = len(dataset)

    pin_memory = device.type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return loader, class_names, requested_images


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    num_classes: int,
    *,
    expected_dataset: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = safe_torch_load(checkpoint_path)
    model_name = get_checkpoint_model_name(checkpoint)
    checkpoint_dataset = validate_checkpoint_dataset(checkpoint, expected_dataset)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        checkpoint_info = {
            "keys": sorted(checkpoint.keys()),
            "model_name": model_name,
            "dataset": checkpoint_dataset,
            "config": checkpoint.get("config"),
            "class_names": checkpoint.get("class_names"),
            "history_length": len(checkpoint.get("history", [])),
            "test_metrics": checkpoint.get("test_metrics"),
        }
    else:
        state_dict = checkpoint
        checkpoint_info = {
            "keys": ["state_dict_only"],
            "model_name": model_name,
            "dataset": checkpoint_dataset,
        }

    model = get_model(model_name, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, checkpoint_info


def select_frequencies(size: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(size) for j in range(size)]


def save_heatmap_csv(path: Path, heatmap: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["freq_i", "freq_j", "fool_ratio", "computed"])
        for freq_i in range(heatmap.shape[0]):
            for freq_j in range(heatmap.shape[1]):
                value = heatmap[freq_i, freq_j]
                computed = not np.isnan(value)
                writer.writerow(
                    [
                        freq_i,
                        freq_j,
                        "" if not computed else f"{value:.8f}",
                        int(computed),
                    ]
                )


def plot_heatmap(heatmap: np.ndarray, png_path: Path, pdf_path: Path, *, title: str) -> None:
    masked = np.ma.masked_invalid(heatmap)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="#d9d9d9")

    fig, ax = plt.subplots(figsize=(7.0, 6.0), constrained_layout=True)
    image = ax.imshow(masked, origin="lower", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Frequence j")
    ax.set_ylabel("Frequence i")
    ax.set_xticks([0, heatmap.shape[1] // 2, heatmap.shape[1] - 1])
    ax.set_yticks([0, heatmap.shape[0] // 2, heatmap.shape[0] - 1])
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Fool ratio")
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the Fourier SFA heatmap artifact used by the final report figures."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default="cifar10")
    parser.add_argument("--epsilon", type=parse_epsilon, default=parse_epsilon("10/255"))
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--download", action="store_true", help="Allow dataset download if missing.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.dataset = normalize_dataset_name(args.dataset)
    dataset_spec = get_dataset_spec(args.dataset)

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    loader, class_names, requested_images = build_test_loader(
        args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        download=args.download,
    )
    print(f"Images d'evaluation: {requested_images}")

    model, checkpoint_info = load_model(
        args.checkpoint,
        device,
        num_classes=len(class_names),
        expected_dataset=args.dataset,
    )
    images, labels = load_images_and_labels_to_device(loader, device)
    if images.shape[-1] != images.shape[-2]:
        raise RuntimeError("Only square images are supported for this Fourier heatmap script.")

    normalize = make_normalizer(args.dataset, device, images.dtype)
    print("Calcul des predictions propres...")
    clean_predictions = predict_all(
        model,
        images,
        batch_size=args.batch_size,
        normalize=normalize,
    )
    clean_accuracy = float((clean_predictions == labels).float().mean().item())
    print(f"Clean accuracy: {clean_accuracy:.4f}")

    image_size = images.shape[-1]
    frequencies = select_frequencies(image_size)
    heatmap = np.full((image_size, image_size), np.nan, dtype=np.float32)
    metric_rows: list[dict[str, Any]] = []
    print(f"Frequences evaluees: {len(frequencies)} / {image_size * image_size}")

    start_time = time.perf_counter()
    for index, (freq_i, freq_j) in enumerate(frequencies, start=1):
        metrics = evaluate_fourier_frequency_metrics(
            model,
            images,
            labels,
            clean_predictions,
            freq_i=freq_i,
            freq_j=freq_j,
            epsilon=args.epsilon,
            batch_size=args.batch_size,
            normalize=normalize,
        )
        ratio = metrics.fool_ratio
        heatmap[freq_i, freq_j] = ratio
        metric_rows.append(
            metrics_to_row(
                metrics,
                dataset=args.dataset,
                checkpoint=args.checkpoint,
                frequency=(freq_i, freq_j),
                epsilon=args.epsilon,
            )
        )
        if index == 1 or index == len(frequencies) or index % 32 == 0:
            print(
                f"[{index:4d}/{len(frequencies):4d}] "
                f"freq=({freq_i:2d},{freq_j:2d}) fool_ratio={ratio:.4f} "
                f"attacked_acc={metrics.attacked_accuracy:.4f} "
                f"asr_correct={metrics.attack_success_rate_correct_only:.4f}"
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - start_time

    computed_values = heatmap[~np.isnan(heatmap)]
    best_metric_row = max(metric_rows, key=lambda row: float(row["fool_ratio"]))
    summary = {
        "min": float(computed_values.min()),
        "max": float(computed_values.max()),
        "mean": float(computed_values.mean()),
        "num_computed": int(computed_values.size),
        "best_frequency": [int(best_metric_row["freq_i"]), int(best_metric_row["freq_j"])],
        "best_metrics": best_metric_row,
    }

    np.save(output_dir / "heatmap.npy", heatmap)
    save_heatmap_csv(output_dir / "heatmap.csv", heatmap)
    write_csv(output_dir / "metrics.csv", metric_rows)
    write_json(
        output_dir / "metrics.json",
        {
            "dataset": args.dataset,
            "checkpoint": args.checkpoint,
            "epsilon": args.epsilon,
            "clean_accuracy": clean_accuracy,
            "summary": summary,
            "num_eval_images": int(images.shape[0]),
            "metrics": metric_rows,
        },
    )
    plot_heatmap(
        heatmap,
        output_dir / "heatmap.png",
        output_dir / "heatmap.pdf",
        title=f"SFA heatmap {dataset_spec.display_name}, epsilon={args.epsilon:.6f}",
    )

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "paper": "Tsuzuku and Sato, On the Structural Sensitivity of Deep Convolutional Networks to the Directions of Fourier Basis Functions",
        "reproduction_note": f"Simplified reproduction on a custom {dataset_spec.display_name} CNN checkpoint.",
        "checkpoint": str(args.checkpoint),
        "checkpoint_info": checkpoint_info,
        "dataset": args.dataset,
        "dataset_display_name": dataset_spec.display_name,
        "epsilon": args.epsilon,
        "image_size": image_size,
        "channels": int(images.shape[1]),
        "num_eval_images": int(images.shape[0]),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": str(device),
        "frequencies": frequencies,
        "elapsed_seconds": elapsed_seconds,
        "clean_accuracy": clean_accuracy,
        "summary": summary,
        "outputs": {
            "heatmap_npy": str(output_dir / "heatmap.npy"),
            "heatmap_csv": str(output_dir / "heatmap.csv"),
            "metrics_csv": str(output_dir / "metrics.csv"),
            "metrics_json": str(output_dir / "metrics.json"),
            "metadata_json": str(output_dir / "metadata.json"),
            "heatmap_png": str(output_dir / "heatmap.png"),
            "heatmap_pdf": str(output_dir / "heatmap.pdf"),
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(to_serializable(metadata), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        "Fool ratio computed values: "
        f"min={summary['min']:.4f}, max={summary['max']:.4f}, mean={summary['mean']:.4f}"
    )
    print(f"Temps heatmap: {elapsed_seconds:.1f}s")
    print(f"Resultats sauvegardes dans: {output_dir}")


if __name__ == "__main__":
    main()
