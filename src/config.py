from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
RUNS_DIR = OUTPUTS_DIR / "runs"
FIGURES_DIR = OUTPUTS_DIR / "figures"


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str = "custom_cnn"
    dataset: str = "cifar10"
    checkpoint_suffix: str = ""
    optimizer: str = "adam"
    batch_size: int = 128
    epochs: int = 10
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 5e-4
    scheduler: str = "none"
    seed: int = 42
    num_workers: int = 4
    val_ratio: float = 0.1
    download: bool = True
    deterministic: bool = True
    device: str = "auto"
    data_dir: Path = DATA_DIR
    outputs_dir: Path = OUTPUTS_DIR
