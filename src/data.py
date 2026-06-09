from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.config import DATA_DIR, ExperimentConfig
from src.utils import build_data_loader_generator, seed_worker


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    display_name: str
    dataset_class: type
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    num_classes: int


DATASET_REGISTRY = {
    "cifar10": DatasetSpec(
        name="cifar10",
        display_name="CIFAR-10",
        dataset_class=datasets.CIFAR10,
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
        num_classes=10,
    ),
    "cifar100": DatasetSpec(
        name="cifar100",
        display_name="CIFAR-100",
        dataset_class=datasets.CIFAR100,
        mean=CIFAR100_MEAN,
        std=CIFAR100_STD,
        num_classes=100,
    ),
}
SUPPORTED_DATASETS = tuple(DATASET_REGISTRY.keys())


def normalize_dataset_name(dataset_name: str) -> str:
    normalized_name = dataset_name.lower()
    if normalized_name not in DATASET_REGISTRY:
        available = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(f"Dataset inconnu: {dataset_name!r}. Datasets disponibles: {available}.")
    return normalized_name


def get_dataset_spec(dataset_name: str) -> DatasetSpec:
    return DATASET_REGISTRY[normalize_dataset_name(dataset_name)]


def get_num_classes(dataset_name: str) -> int:
    return get_dataset_spec(dataset_name).num_classes


def get_dataset_class_names(dataset) -> list[str]:
    return list(getattr(dataset, "classes", []))


def build_test_dataset(
    dataset_name: str,
    *,
    root: Path | str | None = None,
    transform=None,
    download: bool = False,
):
    spec = get_dataset_spec(dataset_name)
    data_dir = Path(root or DATA_DIR)
    return spec.dataset_class(
        root=str(data_dir),
        train=False,
        download=download,
        transform=transform,
    )


def make_normalizer(dataset_name: str, device: torch.device, dtype: torch.dtype):
    spec = get_dataset_spec(dataset_name)
    mean = torch.tensor(spec.mean, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(spec.std, device=device, dtype=dtype).view(1, 3, 1, 1)

    def normalize(images: torch.Tensor) -> torch.Tensor:
        return (images - mean) / std

    return normalize


def build_train_transform(dataset_name: str) -> transforms.Compose:
    spec = get_dataset_spec(dataset_name)
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=spec.mean, std=spec.std),
        ]
    )


def build_eval_transform(dataset_name: str) -> transforms.Compose:
    spec = get_dataset_spec(dataset_name)
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=spec.mean, std=spec.std),
        ]
    )


def build_cifar10_train_transform() -> transforms.Compose:
    return build_train_transform("cifar10")


def build_cifar10_eval_transform() -> transforms.Compose:
    return build_eval_transform("cifar10")


def build_cifar10_transform() -> transforms.Compose:
    return build_cifar10_eval_transform()


def get_dataloaders(
    config: ExperimentConfig,
    device: torch.device,
):
    """
    Prépare des DataLoaders déterministes avec un split validation.
    """
    if not 0 < config.val_ratio < 1:
        raise ValueError("config.val_ratio doit etre strictement compris entre 0 et 1.")

    spec = get_dataset_spec(config.dataset)
    data_dir = Path(config.data_dir or DATA_DIR)
    train_transform = build_train_transform(spec.name)
    eval_transform = build_eval_transform(spec.name)

    dataset_train_aug = spec.dataset_class(
        root=str(data_dir),
        train=True,
        download=config.download,
        transform=train_transform,
    )
    dataset_train_eval = spec.dataset_class(
        root=str(data_dir),
        train=True,
        download=config.download,
        transform=eval_transform,
    )
    test_dataset = spec.dataset_class(
        root=str(data_dir),
        train=False,
        download=config.download,
        transform=eval_transform,
    )

    if len(dataset_train_aug) != len(dataset_train_eval):
        raise RuntimeError(f"Les datasets train {spec.display_name} doivent avoir la meme taille.")

    val_size = max(1, int(len(dataset_train_aug) * config.val_ratio))
    train_size = len(dataset_train_aug) - val_size
    split_generator = torch.Generator().manual_seed(config.seed)
    indices = torch.randperm(len(dataset_train_aug), generator=split_generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_subset = Subset(dataset_train_aug, train_indices)
    val_subset = Subset(dataset_train_eval, val_indices)

    use_pin_memory = device.type == "cuda"
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": use_pin_memory,
        "persistent_workers": config.num_workers > 0,
        "worker_init_fn": seed_worker if config.num_workers > 0 else None,
    }

    train_loader = DataLoader(
        train_subset,
        shuffle=True,
        generator=build_data_loader_generator(config.seed),
        **loader_kwargs,
    )
    val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    class_names = get_dataset_class_names(dataset_train_aug)
    if len(class_names) != spec.num_classes:
        raise RuntimeError(
            f"{spec.display_name} devrait avoir {spec.num_classes} classes, "
            f"mais {len(class_names)} noms de classes ont ete trouves."
        )

    return train_loader, val_loader, test_loader, class_names


def get_cifar10_dataloaders(
    config: ExperimentConfig,
    device: torch.device,
):
    """
    Prépare des DataLoaders CIFAR-10 déterministes avec un split validation.
    """
    return get_dataloaders(
        config=replace(config, dataset="cifar10"),
        device=device,
    )
