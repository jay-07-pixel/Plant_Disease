"""EfficientNet-B3 transfer-learning experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.training.losses import LossType
from src.training.trainer import OptimizerType, SchedulerType, TrainingEngineConfig, get_training_engine_config

EXPERIMENT_ROOT = Path("experiments/efficientnet_b3")
SAVE_DIR = Path("saved_models/efficientnet_b3")
TENSORBOARD_DIR = EXPERIMENT_ROOT / "tensorboard"
HISTORY_PATH = SAVE_DIR / "training_history.json"
BEST_MODEL_PATH = SAVE_DIR / "best_model.pth"
LAST_MODEL_PATH = SAVE_DIR / "last_model.pth"
REPORT_JSON = Path("reports/efficientnet_b3_training_report.json")
REPORT_MD = Path("reports/efficientnet_b3_training_report.md")

DEFAULT_NUM_EPOCHS = 30
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_STAGE1_EPOCHS = 10
DEFAULT_STAGE2_EPOCHS = 20
DEFAULT_STAGE2_LEARNING_RATE = 0.0001
DEFAULT_EARLY_STOPPING_PATIENCE = 7
DEFAULT_RANDOM_SEED = 42

STAGE1_BEST_MODEL_PATH = SAVE_DIR / "stage1_best_model.pth"


@dataclass
class EfficientNetB3ExperimentConfig:
    """Experiment-level settings for the EfficientNet-B3 transfer-learning run.

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

    @property
    def stage1_epochs(self) -> int:
        """Epochs for Stage 1 feature extraction."""
        return DEFAULT_STAGE1_EPOCHS

    @property
    def stage2_epochs(self) -> int:
        """Epochs for Stage 2 fine-tuning."""
        return DEFAULT_STAGE2_EPOCHS

    @property
    def stage2_learning_rate(self) -> float:
        """Learning rate for Stage 2 fine-tuning."""
        return DEFAULT_STAGE2_LEARNING_RATE


def build_stage_experiment(
    experiment: EfficientNetB3ExperimentConfig,
    *,
    num_epochs: int,
    learning_rate: float,
    resume_checkpoint: Path | None = None,
) -> EfficientNetB3ExperimentConfig:
    """Derive a stage-specific experiment configuration."""
    return EfficientNetB3ExperimentConfig(
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        early_stopping_patience=experiment.early_stopping_patience,
        random_seed=experiment.random_seed,
        resume_checkpoint=resume_checkpoint,
        save_dir=experiment.save_dir,
        tensorboard_dir=experiment.tensorboard_dir,
        history_path=experiment.history_path,
    )


def build_engine_config(
    experiment: EfficientNetB3ExperimentConfig,
    *,
    num_classes: int,
    device: str = "auto",
    num_epochs: int | None = None,
    learning_rate: float | None = None,
    resume_checkpoint: Path | None = None,
) -> TrainingEngineConfig:
    """Build a :class:`TrainingEngineConfig` for the EfficientNet-B3 experiment.

    Args:
        experiment: Experiment configuration.
        num_classes: Number of dataset classes.
        device: Compute device string.

    Returns:
        Configured training engine settings.
    """
    use_amp = device == "cuda" or (device == "auto" and _cuda_available())
    resolved_epochs = num_epochs if num_epochs is not None else experiment.num_epochs
    resolved_lr = learning_rate if learning_rate is not None else experiment.learning_rate
    resolved_resume = resume_checkpoint if resume_checkpoint is not None else experiment.resume_checkpoint

    return get_training_engine_config(
        num_epochs=resolved_epochs,
        learning_rate=resolved_lr,
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
        resume_checkpoint=resolved_resume,
        save_best_only=False,
        weight_decay=0.0,
    )


def _cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()
