import json
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Fixe les seeds Python, NumPy et PyTorch pour des runs répétables.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_data_loader_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def get_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_serializable(data: Any) -> Any:
    if is_dataclass(data):
        return _to_serializable(asdict(data))
    if isinstance(data, dict):
        return {key: _to_serializable(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [_to_serializable(value) for value in data]
    if isinstance(data, Path):
        return str(data)
    return data


def save_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_to_serializable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
