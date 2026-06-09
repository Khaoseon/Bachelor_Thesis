from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.data import build_test_dataset, normalize_dataset_name
from src.fourier import (
    apply_perturbation,
    normalize_linf,
    real_fourier_pattern_2d,
    single_fourier_noise,
)
from src.models import get_model

Normalizer = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class AttackMetrics:
    clean_accuracy: float
    attacked_accuracy: float
    fool_ratio: float
    attack_success_rate_correct_only: float
    correct_to_wrong_count: int
    prediction_changed_count: int
    total_count: int
    clean_correct_count: int
    attacked_correct_count: int


def parse_epsilon(value: str | float) -> float:
    if isinstance(value, float):
        return value
    text = value.strip()
    if "/" in text:
        numerator, denominator = text.split("/", maxsplit=1)
        return float(numerator) / float(denominator)
    return float(text)


def to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        if isinstance(value, type):
            return str(value)
        try:
            return to_serializable(asdict(value))
        except TypeError:
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
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_serializable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: to_csv_value(row.get(key, "")) for key in fieldnames})


def to_csv_value(value: Any) -> Any:
    value = to_serializable(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return value


def safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def requested_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def get_checkpoint_model_name(checkpoint: Any, default_model_name: str = "custom_cnn") -> str:
    if not isinstance(checkpoint, dict):
        return default_model_name
    if checkpoint.get("model_name"):
        return str(checkpoint["model_name"])
    config = checkpoint.get("config")
    if isinstance(config, dict):
        return str(config.get("model_name") or default_model_name)
    return str(getattr(config, "model_name", default_model_name) or default_model_name)


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


def load_model_from_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
    num_classes: int,
    expected_dataset: str,
    default_model_name: str = "custom_cnn",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = safe_torch_load(checkpoint_path)
    model_name = get_checkpoint_model_name(checkpoint, default_model_name=default_model_name)
    checkpoint_dataset = validate_checkpoint_dataset(checkpoint, expected_dataset)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        checkpoint_keys = sorted(str(key) for key in checkpoint.keys())
        checkpoint_config = checkpoint.get("config")
        test_metrics = checkpoint.get("test_metrics")
        history_length = len(checkpoint.get("history", []))
    else:
        state_dict = checkpoint
        checkpoint_keys = ["state_dict_only"]
        checkpoint_config = None
        test_metrics = None
        history_length = 0

    model = get_model(model_name, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    checkpoint_info = {
        "path": checkpoint_path,
        "keys": checkpoint_keys,
        "model_name": model_name,
        "dataset": checkpoint_dataset,
        "config": checkpoint_config,
        "test_metrics": test_metrics,
        "history_length": history_length,
    }
    return model, checkpoint_info


def build_eval_loader(
    dataset_name: str,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    max_images: int | None,
    download: bool,
) -> tuple[DataLoader, list[str], int]:
    dataset = build_test_dataset(
        dataset_name,
        download=download,
        transform=transforms.ToTensor(),
    )
    class_names = list(getattr(dataset, "classes", []))
    requested_images = len(dataset)
    if max_images is not None:
        requested_images = min(requested_images, max_images)
    if requested_images <= 0:
        raise ValueError("At least one evaluation image is required.")
    if requested_images < len(dataset):
        dataset = Subset(dataset, list(range(requested_images)))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    return loader, class_names, requested_images


def load_images_and_labels_to_device(
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    for images, labels in loader:
        image_chunks.append(images)
        label_chunks.append(labels)
    if not image_chunks:
        raise RuntimeError("No evaluation images were loaded.")

    images_cpu = torch.cat(image_chunks, dim=0)
    labels_cpu = torch.cat(label_chunks, dim=0)
    if device.type == "cuda":
        images_cpu = images_cpu.pin_memory()
        labels_cpu = labels_cpu.pin_memory()
    return (
        images_cpu.to(device, non_blocking=device.type == "cuda"),
        labels_cpu.to(device, non_blocking=device.type == "cuda"),
    )


def predict_all(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    batch_size: int,
    normalize: Normalizer,
    use_amp: bool = False,
) -> torch.Tensor:
    predictions: list[torch.Tensor] = []
    device_type = images.device.type
    amp_enabled = use_amp and device_type == "cuda"
    with torch.inference_mode():
        for start in range(0, images.size(0), batch_size):
            batch = images[start : start + batch_size]
            with torch.autocast(device_type=device_type, enabled=amp_enabled):
                logits = model(normalize(batch))
            predictions.append(logits.argmax(dim=1))
    return torch.cat(predictions, dim=0)


def metrics_from_counts(
    *,
    clean_correct_count: int,
    attacked_correct_count: int,
    prediction_changed_count: int,
    correct_to_wrong_count: int,
    total_count: int,
) -> AttackMetrics:
    if total_count <= 0:
        raise ValueError("total_count must be positive.")
    return AttackMetrics(
        clean_accuracy=clean_correct_count / total_count,
        attacked_accuracy=attacked_correct_count / total_count,
        fool_ratio=prediction_changed_count / total_count,
        attack_success_rate_correct_only=(
            correct_to_wrong_count / clean_correct_count if clean_correct_count > 0 else 0.0
        ),
        correct_to_wrong_count=int(correct_to_wrong_count),
        prediction_changed_count=int(prediction_changed_count),
        total_count=int(total_count),
        clean_correct_count=int(clean_correct_count),
        attacked_correct_count=int(attacked_correct_count),
    )


def compute_attack_metrics(
    clean_predictions: torch.Tensor,
    attacked_predictions: torch.Tensor,
    labels: torch.Tensor,
) -> AttackMetrics:
    if clean_predictions.shape != attacked_predictions.shape or clean_predictions.shape != labels.shape:
        raise ValueError("clean_predictions, attacked_predictions and labels must have the same shape.")
    if labels.numel() == 0:
        raise ValueError("Cannot compute metrics on empty tensors.")

    clean_correct = clean_predictions == labels
    attacked_correct = attacked_predictions == labels
    prediction_changed = attacked_predictions != clean_predictions
    correct_to_wrong = clean_correct & ~attacked_correct
    return metrics_from_counts(
        clean_correct_count=int(clean_correct.sum().item()),
        attacked_correct_count=int(attacked_correct.sum().item()),
        prediction_changed_count=int(prediction_changed.sum().item()),
        correct_to_wrong_count=int(correct_to_wrong.sum().item()),
        total_count=int(labels.numel()),
    )


def clean_metrics(clean_predictions: torch.Tensor, labels: torch.Tensor) -> AttackMetrics:
    return compute_attack_metrics(clean_predictions, clean_predictions, labels)


def evaluate_noise_metrics(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    clean_predictions: torch.Tensor,
    noise: torch.Tensor,
    *,
    batch_size: int,
    normalize: Normalizer,
    use_amp: bool = False,
) -> AttackMetrics:
    clean_correct_count = 0
    attacked_correct_count = 0
    prediction_changed_count = 0
    correct_to_wrong_count = 0
    total_count = 0
    device_type = images.device.type
    amp_enabled = use_amp and device_type == "cuda"

    with torch.inference_mode():
        for start in range(0, images.size(0), batch_size):
            batch = images[start : start + batch_size]
            label_batch = labels[start : start + batch.size(0)]
            clean_batch_predictions = clean_predictions[start : start + batch.size(0)]
            perturbed = apply_perturbation(batch, noise)
            with torch.autocast(device_type=device_type, enabled=amp_enabled):
                logits = model(normalize(perturbed))
            attacked_predictions = logits.argmax(dim=1)

            clean_correct = clean_batch_predictions == label_batch
            attacked_correct = attacked_predictions == label_batch
            clean_correct_count += int(clean_correct.sum().item())
            attacked_correct_count += int(attacked_correct.sum().item())
            prediction_changed_count += int((attacked_predictions != clean_batch_predictions).sum().item())
            correct_to_wrong_count += int((clean_correct & ~attacked_correct).sum().item())
            total_count += int(label_batch.numel())

    return metrics_from_counts(
        clean_correct_count=clean_correct_count,
        attacked_correct_count=attacked_correct_count,
        prediction_changed_count=prediction_changed_count,
        correct_to_wrong_count=correct_to_wrong_count,
        total_count=total_count,
    )


def evaluate_fourier_frequency_metrics(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    clean_predictions: torch.Tensor,
    *,
    freq_i: int,
    freq_j: int,
    epsilon: float,
    batch_size: int,
    normalize: Normalizer,
    use_amp: bool = False,
) -> AttackMetrics:
    noise = single_fourier_noise(
        images.shape[-1],
        freq_i,
        freq_j,
        epsilon,
        channels=images.shape[1],
        device=images.device,
        dtype=images.dtype,
    )
    return evaluate_noise_metrics(
        model,
        images,
        labels,
        clean_predictions,
        noise,
        batch_size=batch_size,
        normalize=normalize,
        use_amp=use_amp,
    )


def normalize_color_vector_linf(color_vector: Iterable[float]) -> tuple[float, float, float]:
    values = tuple(float(component) for component in color_vector)
    if len(values) != 3:
        raise ValueError("A color vector must have exactly three components.")
    max_abs = max(abs(component) for component in values)
    if max_abs == 0.0:
        raise ValueError("Cannot normalize a zero color vector.")
    return tuple(component / max_abs for component in values)  # type: ignore[return-value]


def color_fourier_noise(
    size: int,
    freq_i: int,
    freq_j: int,
    epsilon: float,
    color_vector: Iterable[float],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    pattern = real_fourier_pattern_2d(
        size,
        freq_i,
        freq_j,
        device=device,
        dtype=dtype,
    )
    pattern = normalize_linf(pattern, 1.0)
    vector = normalize_color_vector_linf(color_vector)
    a = torch.tensor(vector, device=device, dtype=dtype).view(3, 1, 1)
    noise = epsilon * a * pattern.unsqueeze(0)
    max_abs = float(noise.abs().amax().item())
    if max_abs > epsilon + max(1e-7, epsilon * 1e-6):
        raise RuntimeError(f"Color Fourier noise violates L_inf epsilon: {max_abs} > {epsilon}.")
    return noise


def select_frequencies(
    size: int,
    *,
    smoke_test: bool = False,
    max_frequencies: int | None = None,
) -> list[tuple[int, int]]:
    if smoke_test:
        candidates = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (2, 2),
            (4, 0),
            (0, 4),
            (4, 4),
            (size // 4, size // 4),
            (size // 2, size // 2),
        ]
        frequencies: list[tuple[int, int]] = []
        for freq_i, freq_j in candidates:
            frequency = (freq_i % size, freq_j % size)
            if frequency not in frequencies:
                frequencies.append(frequency)
    else:
        frequencies = [(freq_i, freq_j) for freq_i in range(size) for freq_j in range(size)]

    if max_frequencies is not None:
        frequencies = frequencies[:max_frequencies]
    return frequencies


def top_frequencies(heatmap: np.ndarray, top_k: int) -> list[dict[str, Any]]:
    finite = np.argwhere(np.isfinite(heatmap))
    ranked = sorted(
        ((float(heatmap[int(i), int(j)]), int(i), int(j)) for i, j in finite),
        reverse=True,
    )
    rows = []
    for rank, (value, freq_i, freq_j) in enumerate(ranked[:top_k], start=1):
        rows.append(
            {
                "rank": rank,
                "freq_i": freq_i,
                "freq_j": freq_j,
                "fool_ratio": value,
            }
        )
    return rows


def metrics_to_row(
    metrics: AttackMetrics,
    *,
    dataset: str | None = None,
    seed: int | None = None,
    checkpoint: Path | str | None = None,
    frequency: tuple[int, int] | None = None,
    epsilon: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    if dataset is not None:
        row["dataset"] = dataset
    if seed is not None:
        row["seed"] = seed
    if checkpoint is not None:
        row["checkpoint"] = str(checkpoint)
    if frequency is not None:
        row["freq_i"] = int(frequency[0])
        row["freq_j"] = int(frequency[1])
    if epsilon is not None:
        row["epsilon"] = float(epsilon)
    row.update(asdict(metrics))
    if extra:
        row.update(extra)
    return row
