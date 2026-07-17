"""Baseline CNN training experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.training.losses import LossType
from src.training.trainer import OptimizerType, SchedulerType, TrainingEngineConfig, get_training_engine_config

EXPERIMENT_ROOT = Path("experiments/baseline_cnn")
SAVE_DIR = Path("saved_models/baseline_cnn")
TENSORBOARD_DIR = EXPERIMENT_ROOT / "tensorboard"
HISTORY_PATH = SAVE_DIR / "training_history.json"
BEST_MODEL_PATH = SAVE_DIR / "best_model.pth"
LAST_MODEL_PATH = SAVE_DIR / "last_model.pth"
REPORT_JSON = Path("reports/baseline_training_report.json")
REPORT_MD = Path("reports/baseline_training_report.md")

DEFAULT_NUM_EPOCHS = 30
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EARLY_STOPPING_PATIENCE = 7
DEFAULT_RANDOM_SEED = 42


@dataclass
class BaselineExperimentConfig:
    """Experiment-level settings for the baseline CNN training run.

    Attributes:
        num_epochs: Maximum training epochs.
        learning_rate: Adam learning rate.
        early_stopping_patience: Early stopping patience.
        random_seed: Deterministic seed.
        resume_checkpoint: Optional checkpoint for resume.
        save_dir: Model and history output directory.
        tensorboard_dir: TensorBoard log directory.
        history_path: Training history JSON path.
    """

    num_epochs: int = DEFAULT_NUM_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE
    random_seed: int = DEFAULT_RANDOM_SEED
    resume_checkpoint: Path | None = None
    save_dir: Path = field(default_factory=lambda: SAVE_DIR)
    tensorboard_dir: Path = field(default_factory=lambda: TENSORBOARD_DIR)
    history_path: Path = field(default_factory=lambda: HISTORY_PATH)

    def __post_init__(self) -> None:
        self.save_dir = Path(self.save_dir)
        self.tensorboard_dir = Path(self.tensorboard_dir)
        self.history_path = Path(self.history_path)
        if self.resume_checkpoint is not None:
            self.resume_checkpoint = Path(self.resume_checkpoint)


def build_engine_config(
    experiment: BaselineExperimentConfig,
    *,
    num_classes: int,
    device: str = "auto",
) -> TrainingEngineConfig:
    """Build a :class:`TrainingEngineConfig` for the baseline CNN experiment.

    Args:
        experiment: Experiment configuration.
        num_classes: Number of dataset classes.
        device: Compute device string.

    Returns:
        Configured training engine settings.
    """
    use_amp = device == "cuda" or (device == "auto" and _cuda_available())

    return get_training_engine_config(
        num_epochs=experiment.num_epochs,
        learning_rate=experiment.learning_rate,
        optimizer=OptimizerType.ADAM,
        scheduler=SchedulerType.PLATEAU,
        loss_type=LossType.CROSS_ENTROPY,
        num_classes=num_classes,
        device=device,
        use_amp=use_amp,
        use_class_weights=True,
        early_stopping_patience=experiment.early_stopping_patience,
        monitor_metric="accuracy",
        monitor_mode="max",
        random_seed=experiment.random_seed,
        checkpoint_dir=experiment.save_dir,
        log_dir=experiment.tensorboard_dir,
        history_path=experiment.history_path,
        resume_checkpoint=experiment.resume_checkpoint,
        save_best_only=False,
        weight_decay=0.0,
    )


def _cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()
