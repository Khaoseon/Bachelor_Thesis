#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

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
from src.sfa_evaluation import load_images_and_labels_to_device, predict_all, to_serializable


def parse_epsilon(value: str) -> float:
    text = value.strip()
    if "/" in text:
        numerator, denominator = text.split("/", maxsplit=1)
        return float(numerator) / float(denominator)
    return float(text)


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
    eval_transform = transforms.ToTensor()
    dataset = build_test_dataset(
        dataset_name,
        download=download,
        transform=eval_transform,
    )
    class_names = list(dataset.classes)
    requested_images = len(dataset)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    return loader, class_names, requested_images


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    num_classes: int,
    *,
    expected_dataset: str,
) -> tuple[torch.nn.Module, str, Any, str | None]:
    checkpoint = safe_torch_load(checkpoint_path)
    model_name = get_checkpoint_model_name(checkpoint)
    checkpoint_dataset = validate_checkpoint_dataset(checkpoint, expected_dataset)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model = get_model(model_name, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, model_name, checkpoint, checkpoint_dataset


def evaluate_random_noise_trial(
    model: torch.nn.Module,
    images: torch.Tensor,
    clean_predictions: torch.Tensor,
    *,
    epsilon: float,
    batch_size: int,
    normalize,
    generator: torch.Generator,
) -> float:
    changed = 0
    total = 0

    with torch.inference_mode():
        for start in range(0, images.size(0), batch_size):
            batch = images[start : start + batch_size]
            noise = torch.rand(
                batch.shape,
                device=batch.device,
                dtype=batch.dtype,
                generator=generator,
            )
            noise = (2.0 * noise - 1.0) * epsilon
            perturbed = torch.clamp(batch + noise, 0.0, 1.0)
            logits = model(normalize(perturbed))
            perturbed_predictions = logits.argmax(dim=1)
            clean_batch_predictions = clean_predictions[start : start + batch.size(0)]
            changed += (perturbed_predictions != clean_batch_predictions).sum().item()
            total += batch.size(0)

    return changed / total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a uniform random-noise baseline on CIFAR-10 or CIFAR-100."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default="cifar10")
    parser.add_argument("--epsilon", type=parse_epsilon, default=parse_epsilon("10/255"))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
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
    if args.num_trials <= 0:
        raise ValueError("--num-trials must be positive.")

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
    model, model_name, checkpoint, checkpoint_dataset = load_model(
        args.checkpoint,
        device,
        num_classes=len(class_names),
        expected_dataset=args.dataset,
    )
    images, _labels = load_images_and_labels_to_device(loader, device)
    normalize = make_normalizer(args.dataset, device, images.dtype)

    print("Calcul des predictions propres...")
    clean_predictions = predict_all(
        model,
        images,
        batch_size=args.batch_size,
        normalize=normalize,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    fool_ratios = []
    for trial_index in range(1, args.num_trials + 1):
        ratio = evaluate_random_noise_trial(
            model,
            images,
            clean_predictions,
            epsilon=args.epsilon,
            batch_size=args.batch_size,
            normalize=normalize,
            generator=generator,
        )
        fool_ratios.append(ratio)
        print(f"[{trial_index:3d}/{args.num_trials:3d}] fool_ratio={ratio:.4f}")

    values = np.asarray(fool_ratios, dtype=np.float64)
    summary = {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }

    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(args.checkpoint),
        "model_name": model_name,
        "checkpoint_dataset": checkpoint_dataset,
        "checkpoint_config": checkpoint.get("config") if isinstance(checkpoint, dict) else None,
        "dataset": args.dataset,
        "dataset_display_name": dataset_spec.display_name,
        "epsilon": args.epsilon,
        "num_trials": args.num_trials,
        "seed": args.seed,
        "num_eval_images": int(images.size(0)),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": str(device),
        "fool_ratios": fool_ratios,
        **summary,
    }

    print(
        "Random noise fool ratio: "
        f"min={summary['min']:.4f}, max={summary['max']:.4f}, "
        f"mean={summary['mean']:.4f}, std={summary['std']:.4f}"
    )

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / "random_noise_baseline.json"
        output_path.write_text(
            json.dumps(to_serializable(result), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Resultats sauvegardes dans: {output_path}")


if __name__ == "__main__":
    main()
