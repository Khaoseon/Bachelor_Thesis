import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import ExperimentConfig
from src.data import SUPPORTED_DATASETS, get_dataloaders, normalize_dataset_name
from src.models import get_model
from src.utils import ensure_dir, get_device, save_json, set_seed


def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            predictions = outputs.argmax(dim=1)
            total += batch_size
            correct += (predictions == labels).sum().item()

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


def build_optimizer(model: nn.Module, config: ExperimentConfig) -> optim.Optimizer:
    optimizer_name = config.optimizer.lower()

    if optimizer_name == "adam":
        return optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    if optimizer_name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

    raise ValueError("Optimizer inconnu. Valeurs disponibles: adam, sgd.")


def build_scheduler(optimizer: optim.Optimizer, config: ExperimentConfig):
    scheduler_name = config.scheduler.lower()

    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
        )

    raise ValueError("Scheduler inconnu. Valeurs disponibles: none, cosine.")


def current_learning_rate(optimizer: optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def build_checkpoint_suffix_tag(checkpoint_suffix: str) -> str:
    suffix = checkpoint_suffix.strip()
    if not suffix:
        return ""

    allowed_characters = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    if any(character not in allowed_characters for character in suffix):
        raise ValueError("--checkpoint-suffix doit contenir uniquement lettres, chiffres, '_' ou '-'.")
    return f"_{suffix}"


def train_model(config: ExperimentConfig | None = None) -> dict[str, object]:
    config = config or ExperimentConfig()
    dataset_name = normalize_dataset_name(config.dataset)
    checkpoint_suffix_tag = build_checkpoint_suffix_tag(config.checkpoint_suffix)
    set_seed(config.seed, deterministic=config.deterministic)
    device = get_device(config.device)
    print(f"Device utilise: {device}")
    print(f"Dataset: {dataset_name}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        config=config,
        device=device,
    )
    model = get_model(config.model_name, num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    lr_scheduler = build_scheduler(optimizer, config)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs_dir = Path(config.outputs_dir)
    run_dir = ensure_dir(outputs_dir / "runs" / run_name)
    models_dir = ensure_dir(outputs_dir / "models")
    model_tag = config.model_name.lower()
    best_checkpoint_path = models_dir / f"{model_tag}_{dataset_name}{checkpoint_suffix_tag}_best.pth"
    final_checkpoint_path = models_dir / f"{model_tag}_{dataset_name}{checkpoint_suffix_tag}_final.pth"
    history = []
    best_val_accuracy = -1.0
    best_state_dict = None

    for epoch in range(1, config.epochs + 1): # On itère sur les époques d'entraînement, en évaluant la performance sur le set de validation à chaque fois.
        model.train()
        running_loss = 0.0
        seen_samples = 0
        epoch_lr = current_learning_rate(optimizer)

        for images, labels in train_loader: # On itère sur les batches d'entraînement, en calculant la loss et en mettant à jour les poids du modèle à chaque étape.
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            seen_samples += batch_size

        train_loss = running_loss / seen_samples
        val_metrics = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "lr": epoch_lr,
            }
        )

        print(
            f"Epoch [{epoch}/{config.epochs}] "
            f"- lr: {epoch_lr:.6f} "
            f"- train_loss: {train_loss:.4f} "
            f"- val_loss: {val_metrics['loss']:.4f} "
            f"- val_acc: {val_metrics['accuracy'] * 100:.2f}%"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_state_dict = deepcopy(model.state_dict())
            torch.save(best_state_dict, best_checkpoint_path)

        if lr_scheduler is not None:
            lr_scheduler.step()

    if best_state_dict is None:
        raise RuntimeError("Aucun checkpoint valide n'a ete produit pendant l'entrainement.")

    model.load_state_dict(best_state_dict)
    test_metrics = evaluate(model, test_loader, criterion, device)

    final_checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "dataset": dataset_name,
        "model_name": config.model_name,
        "class_names": class_names,
        "history": history,
        "test_metrics": test_metrics,
    }

    torch.save(final_checkpoint, final_checkpoint_path)
    torch.save(final_checkpoint, run_dir / "checkpoint.pth")
    save_json(
        run_dir / "metrics.json",
        {
            "config": config,
            "dataset": dataset_name,
            "model_name": config.model_name,
            "history": history,
            "test_metrics": test_metrics,
            "best_val_accuracy": best_val_accuracy,
            "best_checkpoint": best_checkpoint_path,
            "final_checkpoint": final_checkpoint_path,
        },
    )

    print(f"Meilleure validation: {best_val_accuracy * 100:.2f}%")
    print(f"Accuracy test finale: {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Checkpoint best: {best_checkpoint_path}")
    print(f"Checkpoint final: {final_checkpoint_path}")
    print(f"Artefacts sauvegardes dans: {run_dir}")

    return {
        "model": model,
        "history": history,
        "test_metrics": test_metrics,
        "run_dir": run_dir,
        "best_checkpoint": best_checkpoint_path,
        "final_checkpoint": final_checkpoint_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a CIFAR-10 or CIFAR-100 model.")
    parser.add_argument("--model-name", default=None, choices=["custom_cnn", "vgg_small_bn"])
    parser.add_argument("--dataset", default="cifar10", choices=SUPPORTED_DATASETS)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--optimizer", default=None, choices=["adam", "sgd"])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--scheduler", default=None, choices=["none", "cosine"])
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-suffix", default=None)
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig()
    updates = {
        "model_name": args.model_name,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "device": args.device,
        "checkpoint_suffix": args.checkpoint_suffix,
    }
    clean_updates = {key: value for key, value in updates.items() if value is not None}
    return replace(config, **clean_updates)


if __name__ == "__main__":
    train_model(config_from_args(build_parser().parse_args()))
