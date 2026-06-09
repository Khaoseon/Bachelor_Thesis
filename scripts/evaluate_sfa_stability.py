#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import shutil
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig, OUTPUTS_DIR  # noqa: E402
from src.data import SUPPORTED_DATASETS, get_dataset_spec, make_normalizer, normalize_dataset_name  # noqa: E402
from src.sfa_evaluation import (  # noqa: E402
    build_eval_loader,
    evaluate_fourier_frequency_metrics,
    load_images_and_labels_to_device,
    load_model_from_checkpoint,
    metrics_to_row,
    parse_epsilon,
    predict_all,
    requested_device,
    safe_torch_load,
    select_frequencies,
    top_frequencies,
    write_csv,
    write_json,
)
from src.train import train_model  # noqa: E402
from src.utils import set_seed  # noqa: E402


DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "runs" / "stability_vgg_small_bn_retrained"
DEFAULT_CHECKPOINT_DIR = OUTPUTS_DIR / "models" / "stability"
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_MIN_CLEAN_ACCURACY = {
    "cifar10": 0.90,
    "cifar100": 0.65,
}


def plot_heatmap(
    heatmap: np.ndarray,
    png_path: Path,
    pdf_path: Path,
    *,
    title: str,
    colorbar_label: str = "Fool ratio",
    vmin: float = 0.0,
    vmax: float | None = 1.0,
) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    masked = np.ma.masked_invalid(heatmap)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="#d9d9d9")
    fig, ax = plt.subplots(figsize=(7.0, 6.0), constrained_layout=True)
    image = ax.imshow(masked, origin="lower", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Frequence j")
    ax.set_ylabel("Frequence i")
    ax.set_xticks([0, heatmap.shape[1] // 2, heatmap.shape[1] - 1])
    ax.set_yticks([0, heatmap.shape[0] // 2, heatmap.shape[0] - 1])
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)


def plot_mean_std(mean_heatmap: np.ndarray, std_heatmap: np.ndarray, png_path: Path, pdf_path: Path, *, title: str) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), constrained_layout=True)
    panels = [
        (axes[0], mean_heatmap, "Mean fool ratio", "magma", 0.0, 1.0),
        (axes[1], std_heatmap, "Std across seeds", "viridis", 0.0, None),
    ]
    for ax, heatmap, panel_title, cmap_name, vmin, vmax in panels:
        masked = np.ma.masked_invalid(heatmap)
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(color="#d9d9d9")
        image = ax.imshow(masked, origin="lower", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(panel_title)
        ax.set_xlabel("Frequence j")
        ax.set_xticks([0, heatmap.shape[1] // 2, heatmap.shape[1] - 1])
        ax.set_yticks([0, heatmap.shape[0] // 2, heatmap.shape[0] - 1])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    axes[0].set_ylabel("Frequence i")
    fig.suptitle(title)
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)


def checkpoint_for_seed(checkpoint_dir: Path, dataset: str, seed: int, *, kind: str = "final") -> Path:
    return checkpoint_dir / f"vgg_small_bn_{dataset}_seed_{seed}_{kind}.pth"


def build_training_config(args: argparse.Namespace, dataset: str, seed: int, device: torch.device) -> ExperimentConfig:
    return replace(
        ExperimentConfig(),
        model_name="vgg_small_bn",
        dataset=dataset,
        checkpoint_suffix=f"seed_{seed}",
        seed=seed,
        epochs=args.train_epochs,
        batch_size=args.train_batch_size,
        optimizer=args.train_optimizer,
        lr=args.train_lr,
        momentum=args.train_momentum,
        weight_decay=args.train_weight_decay,
        scheduler=args.train_scheduler,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        download=args.download,
        deterministic=args.deterministic,
        device=str(device),
        outputs_dir=args.training_outputs_dir,
    )


def save_planned_training_config(args: argparse.Namespace, dataset: str, seed: int, config: ExperimentConfig) -> None:
    seed_dir = args.output_dir / dataset / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    write_json(seed_dir / "training_config.json", config)


def copy_trained_checkpoints(args: argparse.Namespace, train_result: dict[str, Any], dataset: str, seed: int) -> Path:
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_target = checkpoint_for_seed(args.checkpoint_dir, dataset, seed, kind="final")
    best_target = checkpoint_for_seed(args.checkpoint_dir, dataset, seed, kind="best")
    shutil.copy2(Path(train_result["final_checkpoint"]), final_target)
    shutil.copy2(Path(train_result["best_checkpoint"]), best_target)
    return final_target


def ensure_checkpoint(args: argparse.Namespace, dataset: str, seed: int, device: torch.device) -> Path:
    checkpoint_path = checkpoint_for_seed(args.checkpoint_dir, dataset, seed, kind="final")
    train_config = build_training_config(args, dataset, seed, device)
    save_planned_training_config(args, dataset, seed, train_config)

    if checkpoint_path.exists() and (args.skip_existing_checkpoints or not args.force_train):
        print(f"Checkpoint utilise: {checkpoint_path}")
        return checkpoint_path
    if args.skip_train:
        raise FileNotFoundError(
            f"Checkpoint missing for dataset={dataset}, seed={seed}: {checkpoint_path}. "
            "Remove --skip-train to train it."
        )
    if args.dry_run:
        print(f"Dry run: checkpoint planifie pour {dataset}/seed={seed}: {checkpoint_path}")
        return checkpoint_path

    print(f"Entrainement VGGSmallBN pour {dataset}/seed={seed} avec protocole principal...")
    train_result = train_model(train_config)
    final_checkpoint = copy_trained_checkpoints(args, train_result, dataset, seed)
    seed_dir = args.output_dir / dataset / f"seed_{seed}"
    write_json(
        seed_dir / "training_metrics.json",
        {
            "run_dir": train_result["run_dir"],
            "source_best_checkpoint": train_result["best_checkpoint"],
            "source_final_checkpoint": train_result["final_checkpoint"],
            "stability_checkpoint": final_checkpoint,
            "history": train_result["history"],
            "test_metrics": train_result["test_metrics"],
        },
    )
    write_json(seed_dir / "test_metrics.json", train_result["test_metrics"])
    del train_result["model"]
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"Checkpoint stabilite entraine: {final_checkpoint}")
    return final_checkpoint


def save_heatmap_csv(path: Path, heatmap: np.ndarray) -> None:
    rows = []
    for freq_i in range(heatmap.shape[0]):
        for freq_j in range(heatmap.shape[1]):
            value = heatmap[freq_i, freq_j]
            rows.append(
                {
                    "freq_i": freq_i,
                    "freq_j": freq_j,
                    "fool_ratio": "" if not np.isfinite(value) else float(value),
                    "computed": int(np.isfinite(value)),
                }
            )
    write_csv(path, rows, fieldnames=["freq_i", "freq_j", "fool_ratio", "computed"])


def enrich_top_rows(top_rows: list[dict[str, Any]], metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_frequency = {
        (int(row["freq_i"]), int(row["freq_j"])): row
        for row in metric_rows
    }
    enriched = []
    for row in top_rows:
        key = (int(row["freq_i"]), int(row["freq_j"]))
        merged = dict(row)
        merged.update(by_frequency.get(key, {}))
        merged["rank"] = row["rank"]
        enriched.append(merged)
    return enriched


def config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def training_fields_from_config(config: Any, *, eval_batch_size: int) -> dict[str, Any]:
    return {
        "model_name": config_value(config, "model_name", "vgg_small_bn"),
        "train_epochs": config_value(config, "epochs"),
        "optimizer": config_value(config, "optimizer"),
        "lr": config_value(config, "lr"),
        "momentum": config_value(config, "momentum"),
        "weight_decay": config_value(config, "weight_decay"),
        "scheduler": config_value(config, "scheduler"),
        "train_batch_size": config_value(config, "batch_size"),
        "eval_batch_size": eval_batch_size,
        "val_ratio": config_value(config, "val_ratio"),
        "deterministic": config_value(config, "deterministic"),
    }


def min_clean_accuracy_for_dataset(args: argparse.Namespace, dataset: str) -> float | None:
    if args.no_clean_accuracy_gate:
        return None
    if args.min_clean_accuracy is not None:
        return args.min_clean_accuracy
    return DEFAULT_MIN_CLEAN_ACCURACY[dataset]


def write_checkpoint_training_artifacts(seed_dir: Path, checkpoint_path: Path) -> None:
    checkpoint = safe_torch_load(checkpoint_path)
    if isinstance(checkpoint, dict):
        write_json(
            seed_dir / "training_metrics.json",
            {
                "checkpoint_path": checkpoint_path,
                "history": checkpoint.get("history", []),
                "test_metrics": checkpoint.get("test_metrics"),
            },
        )
        return
    write_json(
        seed_dir / "training_metrics.json",
        {
            "checkpoint_path": checkpoint_path,
            "history": [],
            "test_metrics": None,
            "note": "Checkpoint is a raw state_dict without training metadata.",
        },
    )


def evaluate_seed(
    *,
    args: argparse.Namespace,
    dataset: str,
    seed: int,
    checkpoint_path: Path,
    images: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str],
    device: torch.device,
    frequencies: list[tuple[int, int]],
) -> tuple[np.ndarray, dict[str, Any]]:
    seed_dir = args.output_dir / dataset / f"seed_{seed}"
    heatmap_path = seed_dir / "heatmap.npy"
    metrics_json_path = seed_dir / "metrics.json"
    if heatmap_path.exists() and metrics_json_path.exists() and not args.force and not args.force_train:
        print(f"{dataset}/seed={seed}: reutilisation de {seed_dir}")
        heatmap = np.load(heatmap_path).astype(np.float32)
        import json

        payload = json.loads(metrics_json_path.read_text(encoding="utf-8"))
        return heatmap, payload["summary"]

    seed_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    model, checkpoint_info = load_model_from_checkpoint(
        checkpoint_path,
        device=device,
        num_classes=len(class_names),
        expected_dataset=dataset,
        default_model_name="vgg_small_bn",
    )
    checkpoint_config = checkpoint_info.get("config") or build_training_config(args, dataset, seed, device)
    train_fields = training_fields_from_config(checkpoint_config, eval_batch_size=args.batch_size)
    write_json(seed_dir / "training_config.json", checkpoint_config)
    write_checkpoint_training_artifacts(seed_dir, checkpoint_path)
    write_json(
        seed_dir / "test_metrics.json",
        {
            "checkpoint_test_metrics": checkpoint_info.get("test_metrics"),
            "checkpoint_path": checkpoint_path,
        },
    )
    normalize = make_normalizer(dataset, device, images.dtype)

    print(f"{dataset}/seed={seed}: predictions propres...")
    clean_predictions = predict_all(
        model,
        images,
        batch_size=args.batch_size,
        normalize=normalize,
        use_amp=args.amp,
    )
    clean_accuracy = float((clean_predictions == labels).float().mean().item())
    print(f"{dataset}/seed={seed}: clean_accuracy={clean_accuracy:.4f}")
    write_json(
        seed_dir / "test_metrics.json",
        {
            "checkpoint_test_metrics": checkpoint_info.get("test_metrics"),
            "evaluation_clean_accuracy": clean_accuracy,
            "num_eval_images": int(images.shape[0]),
            "checkpoint_path": checkpoint_path,
        },
    )

    min_clean_accuracy = min_clean_accuracy_for_dataset(args, dataset)
    if min_clean_accuracy is not None and clean_accuracy < min_clean_accuracy:
        message = (
            f"{dataset}/seed={seed}: clean_accuracy={clean_accuracy:.4f} "
            f"< seuil {min_clean_accuracy:.4f}; heatmap non lancee."
        )
        print(message)
        image_size = int(images.shape[-1])
        heatmap = np.full((image_size, image_size), np.nan, dtype=np.float32)
        np.save(seed_dir / "heatmap.npy", heatmap)
        save_heatmap_csv(seed_dir / "heatmap.csv", heatmap)
        write_csv(
            seed_dir / "metrics.csv",
            [],
            fieldnames=[
                "dataset",
                "seed",
                "checkpoint",
                "freq_i",
                "freq_j",
                "epsilon",
                "clean_accuracy",
                "attacked_accuracy",
                "fool_ratio",
                "attack_success_rate_correct_only",
                "correct_to_wrong_count",
                "prediction_changed_count",
                "total_count",
                "clean_correct_count",
                "attacked_correct_count",
            ],
        )
        write_csv(
            seed_dir / "top_frequencies.csv",
            [],
            fieldnames=["rank", "freq_i", "freq_j", "fool_ratio"],
        )
        summary = {
            "dataset": dataset,
            "seed": seed,
            "checkpoint_path": checkpoint_path,
            "checkpoint": checkpoint_path,
            **train_fields,
            "clean_accuracy": clean_accuracy,
            "min_clean_accuracy": min_clean_accuracy,
            "skipped": True,
            "skip_reason": message,
            "best_frequency": None,
            "best_fool_ratio": None,
            "attacked_accuracy": None,
            "attack_success_rate_correct_only": None,
            "top_frequencies": [],
            "top_5_frequencies": [],
            "num_eval_images": int(images.shape[0]),
            "num_frequencies": 0,
            "elapsed_seconds": 0.0,
            "outputs": {
                "heatmap_npy": seed_dir / "heatmap.npy",
                "heatmap_csv": seed_dir / "heatmap.csv",
                "metrics_csv": seed_dir / "metrics.csv",
                "top_frequencies_csv": seed_dir / "top_frequencies.csv",
                "metrics_json": seed_dir / "metrics.json",
                "training_config_json": seed_dir / "training_config.json",
                "training_metrics_json": seed_dir / "training_metrics.json",
                "test_metrics_json": seed_dir / "test_metrics.json",
            },
        }
        write_json(
            metrics_json_path,
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "experiment": "sfa_stability_seed",
                "summary": summary,
                "checkpoint_info": checkpoint_info,
                "metrics": [],
            },
        )
        return heatmap, summary

    image_size = int(images.shape[-1])
    heatmap = np.full((image_size, image_size), np.nan, dtype=np.float32)
    metric_rows: list[dict[str, Any]] = []
    best_ratio = -1.0
    best_frequency = None
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
            use_amp=args.amp,
        )
        heatmap[freq_i, freq_j] = metrics.fool_ratio
        metric_rows.append(
            metrics_to_row(
                metrics,
                dataset=dataset,
                seed=seed,
                checkpoint=checkpoint_path,
                frequency=(freq_i, freq_j),
                epsilon=args.epsilon,
            )
        )
        if metrics.fool_ratio > best_ratio:
            best_ratio = metrics.fool_ratio
            best_frequency = (freq_i, freq_j)
        if index == 1 or index == len(frequencies) or index % 32 == 0:
            print(
                f"{dataset}/seed={seed} [{index:4d}/{len(frequencies):4d}] "
                f"freq=({freq_i:2d},{freq_j:2d}) fool_ratio={metrics.fool_ratio:.4f} "
                f"best={best_ratio:.4f}@{best_frequency}"
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - start_time
    top_rows = enrich_top_rows(top_frequencies(heatmap, args.top_k), metric_rows)
    best_row = max(metric_rows, key=lambda row: float(row["fool_ratio"]))
    summary = {
        "dataset": dataset,
        "seed": seed,
        "checkpoint_path": checkpoint_path,
        "checkpoint": checkpoint_path,
        **train_fields,
        "best_frequency": [int(best_row["freq_i"]), int(best_row["freq_j"])],
        "best_fool_ratio": float(best_row["fool_ratio"]),
        "clean_accuracy": clean_accuracy,
        "attacked_accuracy": float(best_row["attacked_accuracy"]),
        "attack_success_rate_correct_only": float(best_row["attack_success_rate_correct_only"]),
        "skipped": False,
        "top_frequencies": top_rows,
        "top_5_frequencies": [[row["freq_i"], row["freq_j"]] for row in top_rows[:5]],
        "num_eval_images": int(images.shape[0]),
        "num_frequencies": len(frequencies),
        "elapsed_seconds": elapsed_seconds,
        "outputs": {
            "heatmap_npy": seed_dir / "heatmap.npy",
            "heatmap_csv": seed_dir / "heatmap.csv",
            "metrics_csv": seed_dir / "metrics.csv",
            "metrics_json": seed_dir / "metrics.json",
            "training_config_json": seed_dir / "training_config.json",
            "training_metrics_json": seed_dir / "training_metrics.json",
            "test_metrics_json": seed_dir / "test_metrics.json",
            "top_frequencies_csv": seed_dir / "top_frequencies.csv",
            "heatmap_png": seed_dir / "heatmap.png",
            "heatmap_pdf": seed_dir / "heatmap.pdf",
        },
    }

    np.save(seed_dir / "heatmap.npy", heatmap)
    save_heatmap_csv(seed_dir / "heatmap.csv", heatmap)
    write_csv(seed_dir / "metrics.csv", metric_rows)
    write_csv(seed_dir / "top_frequencies.csv", top_rows)
    write_json(
        metrics_json_path,
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "experiment": "sfa_stability_seed",
            "summary": summary,
            "checkpoint_info": checkpoint_info,
            "metrics": metric_rows,
        },
    )
    dataset_label = get_dataset_spec(dataset).display_name
    plot_heatmap(
        heatmap,
        seed_dir / "heatmap.png",
        seed_dir / "heatmap.pdf",
        title=f"SFA heatmap {dataset_label}, seed={seed}, epsilon={args.epsilon:.6f}",
    )
    print(f"{dataset}/seed={seed}: outputs -> {seed_dir}")
    return heatmap, summary


def rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def correlation(x_values: np.ndarray, y_values: np.ndarray) -> tuple[float | None, float | None, int]:
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    x = x_values[mask].astype(np.float64)
    y = y_values[mask].astype(np.float64)
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None, None, int(x.size)
    pearson = float(np.corrcoef(x, y)[0, 1])
    x_rank = rank_values(x)
    y_rank = rank_values(y)
    spearman = float(np.corrcoef(x_rank, y_rank)[0, 1])
    return pearson, spearman, int(x.size)


def aggregate_dataset(args: argparse.Namespace, dataset: str, seed_results: list[tuple[int, np.ndarray, dict[str, Any]]]) -> None:
    aggregate_dir = args.output_dir / dataset / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    seeds = [seed for seed, _heatmap, _summary in seed_results]
    heatmaps = np.stack([heatmap for _seed, heatmap, _summary in seed_results], axis=0)
    valid_count = np.isfinite(heatmaps).sum(axis=0)
    heatmap_sum = np.nansum(heatmaps, axis=0)
    mean_heatmap = np.full(heatmaps.shape[1:], np.nan, dtype=np.float64)
    np.divide(heatmap_sum, valid_count, out=mean_heatmap, where=valid_count > 0)
    centered = heatmaps - mean_heatmap
    squared_sum = np.nansum(centered * centered, axis=0)
    std_heatmap = np.full(heatmaps.shape[1:], np.nan, dtype=np.float64)
    np.divide(squared_sum, valid_count, out=std_heatmap, where=valid_count > 0)
    std_heatmap = np.sqrt(std_heatmap).astype(np.float32)
    mean_heatmap = mean_heatmap.astype(np.float32)
    np.save(aggregate_dir / "heatmap_mean.npy", mean_heatmap)
    np.save(aggregate_dir / "heatmap_std.npy", std_heatmap)

    dataset_label = get_dataset_spec(dataset).display_name
    plot_heatmap(
        mean_heatmap,
        aggregate_dir / "heatmap_mean.png",
        aggregate_dir / "heatmap_mean.pdf",
        title=f"Mean SFA heatmap {dataset_label}, seeds={seeds}",
    )
    plot_heatmap(
        std_heatmap,
        aggregate_dir / "heatmap_std.png",
        aggregate_dir / "heatmap_std.pdf",
        title=f"Std SFA heatmap {dataset_label}, seeds={seeds}",
        colorbar_label="Std fool ratio",
        vmax=None,
    )
    plot_mean_std(
        mean_heatmap,
        std_heatmap,
        aggregate_dir / "heatmap_mean_std.png",
        aggregate_dir / "heatmap_mean_std.pdf",
        title=f"SFA stability {dataset_label}",
    )

    summary_rows = []
    for seed, _heatmap, summary in seed_results:
        summary_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "checkpoint_path": summary.get("checkpoint_path") or summary.get("checkpoint"),
                "model_name": summary.get("model_name"),
                "train_epochs": summary.get("train_epochs"),
                "optimizer": summary.get("optimizer"),
                "lr": summary.get("lr"),
                "momentum": summary.get("momentum"),
                "weight_decay": summary.get("weight_decay"),
                "scheduler": summary.get("scheduler"),
                "train_batch_size": summary.get("train_batch_size"),
                "eval_batch_size": summary.get("eval_batch_size"),
                "val_ratio": summary.get("val_ratio"),
                "deterministic": summary.get("deterministic"),
                "clean_accuracy": summary.get("clean_accuracy"),
                "best_frequency": summary.get("best_frequency"),
                "best_fool_ratio": summary.get("best_fool_ratio"),
                "attacked_accuracy": summary.get("attacked_accuracy"),
                "attack_success_rate_correct_only": summary.get("attack_success_rate_correct_only"),
                "skipped": summary.get("skipped", False),
                "skip_reason": summary.get("skip_reason"),
                "top_5_frequencies": summary.get("top_5_frequencies"),
                "num_eval_images": summary.get("num_eval_images"),
                "num_frequencies": summary.get("num_frequencies"),
                "elapsed_seconds": summary.get("elapsed_seconds"),
            }
        )
    write_csv(aggregate_dir / "summary.csv", summary_rows)

    correlation_rows = []
    for (seed_a, heatmap_a, _summary_a), (seed_b, heatmap_b, _summary_b) in itertools.combinations(seed_results, 2):
        pearson, spearman, overlap = correlation(heatmap_a.ravel(), heatmap_b.ravel())
        correlation_rows.append(
            {
                "dataset": dataset,
                "seed_a": seed_a,
                "seed_b": seed_b,
                "pearson": pearson,
                "spearman": spearman,
                "num_overlapping_frequencies": overlap,
            }
        )
    write_csv(aggregate_dir / "correlations.csv", correlation_rows)
    write_json(
        aggregate_dir / "summary.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "experiment": "sfa_stability_aggregate",
            "dataset": dataset,
            "seeds": seeds,
            "epsilon": args.epsilon,
            "summary": summary_rows,
            "correlations": correlation_rows,
            "outputs": {
                "heatmap_mean_npy": aggregate_dir / "heatmap_mean.npy",
                "heatmap_std_npy": aggregate_dir / "heatmap_std.npy",
                "heatmap_mean_pdf": aggregate_dir / "heatmap_mean.pdf",
                "heatmap_std_pdf": aggregate_dir / "heatmap_std.pdf",
                "heatmap_mean_std_pdf": aggregate_dir / "heatmap_mean_std.pdf",
                "summary_csv": aggregate_dir / "summary.csv",
                "correlations_csv": aggregate_dir / "correlations.csv",
            },
        },
    )
    print(f"{dataset}: aggregate -> {aggregate_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate SFA heatmap stability across VGGSmallBN training seeds.")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=["cifar10", "cifar100"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--epsilon", type=parse_epsilon, default=parse_epsilon("10/255"))
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4096, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-train", action="store_true", help="Require existing stability checkpoints; never train.")
    parser.add_argument(
        "--skip-existing-checkpoints",
        action="store_true",
        help="Reuse existing stability checkpoints even when --force-train is provided.",
    )
    parser.add_argument("--force-train", action="store_true", help="Retrain stability checkpoints even if they exist.")
    parser.add_argument("--force", action="store_true", help="Recompute heatmaps even if seed outputs already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Write planned training configs without training/evaluation.")
    parser.add_argument("--download", action="store_true", help="Allow dataset download if missing.")
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast for model inference.")
    parser.add_argument("--train-epochs", type=int, default=100)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--optimizer", "--train-optimizer", dest="train_optimizer", choices=["adam", "sgd"], default="sgd")
    parser.add_argument("--train-lr", type=float, default=0.1)
    parser.add_argument("--train-momentum", type=float, default=0.9)
    parser.add_argument("--train-weight-decay", type=float, default=5e-4)
    parser.add_argument("--train-scheduler", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", default=True)
    parser.add_argument("--non-deterministic", dest="deterministic", action="store_false")
    parser.add_argument(
        "--min-clean-accuracy",
        type=float,
        default=None,
        help="Override dataset defaults; by default CIFAR-10 uses 0.90 and CIFAR-100 uses 0.65.",
    )
    parser.add_argument("--no-clean-accuracy-gate", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    args.datasets = [normalize_dataset_name(dataset) for dataset in args.datasets]
    args.checkpoint_dir = Path(args.checkpoint_dir)
    args.training_outputs_dir = args.output_dir / "_training"
    if args.seeds is None:
        args.seeds = DEFAULT_SEEDS
    if args.max_images is None:
        args.max_images = 10000
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if args.max_images <= 0:
        raise ValueError("--max-images must be positive.")
    if args.epsilon < 0:
        raise ValueError("--epsilon must be non-negative.")
    if args.train_epochs <= 0:
        raise ValueError("--train-epochs must be positive.")
    if args.train_batch_size <= 0:
        raise ValueError("--train-batch-size must be positive.")
    if args.train_lr <= 0:
        raise ValueError("--train-lr must be positive.")
    if args.train_momentum < 0:
        raise ValueError("--train-momentum must be non-negative.")
    if args.train_weight_decay < 0:
        raise ValueError("--train-weight-decay must be non-negative.")
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be strictly between 0 and 1.")
    if args.min_clean_accuracy is not None and not 0 <= args.min_clean_accuracy <= 1:
        raise ValueError("--min-clean-accuracy must be in [0, 1].")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = requested_device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Datasets: {args.datasets}")
    print(f"Seeds: {args.seeds}")
    print(f"Epsilon: {args.epsilon:.8f}")
    print(f"Output dir: {args.output_dir}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(
        "Train protocol: "
        f"model=vgg_small_bn epochs={args.train_epochs} optimizer={args.train_optimizer} "
        f"lr={args.train_lr} momentum={args.train_momentum} weight_decay={args.train_weight_decay} "
        f"scheduler={args.train_scheduler} batch_size={args.train_batch_size} val_ratio={args.val_ratio}"
    )

    if args.dry_run:
        for dataset in args.datasets:
            for seed in args.seeds:
                config = build_training_config(args, dataset, seed, device)
                save_planned_training_config(args, dataset, seed, config)
        write_json(
            args.output_dir / "dry_run_manifest.json",
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "experiment": "sfa_stability_dry_run",
                "datasets": args.datasets,
                "seeds": args.seeds,
                "checkpoint_dir": args.checkpoint_dir,
                "training_outputs_dir": args.training_outputs_dir,
            },
        )
        print(f"Dry run termine: configs ecrites dans {args.output_dir}")
        return

    for dataset in args.datasets:
        print(f"=== Dataset {dataset} ===")
        loader, class_names, requested_images = build_eval_loader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            max_images=args.max_images,
            download=args.download,
        )
        print(f"{dataset}: images d'evaluation={requested_images}")
        images, labels = load_images_and_labels_to_device(loader, device)
        if images.shape[-1] != images.shape[-2]:
            raise RuntimeError("Only square images are supported for SFA heatmaps.")
        frequencies = select_frequencies(int(images.shape[-1]))
        print(f"{dataset}: frequences evaluees={len(frequencies)} / {images.shape[-1] * images.shape[-1]}")

        seed_results: list[tuple[int, np.ndarray, dict[str, Any]]] = []
        for seed in args.seeds:
            checkpoint_path = ensure_checkpoint(args, dataset, seed, device)
            heatmap, summary = evaluate_seed(
                args=args,
                dataset=dataset,
                seed=seed,
                checkpoint_path=checkpoint_path,
                images=images,
                labels=labels,
                class_names=class_names,
                device=device,
                frequencies=frequencies,
            )
            seed_results.append((seed, heatmap, summary))
            if device.type == "cuda":
                torch.cuda.empty_cache()

        aggregate_dataset(args, dataset, seed_results)


if __name__ == "__main__":
    main()
