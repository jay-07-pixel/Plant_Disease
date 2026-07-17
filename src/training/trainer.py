"""Generic PyTorch training engine for all vision models."""

from __future__ import annotations

import logging
import random
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    ReduceLROnPlateau,
    StepLR,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.callbacks import Callback, CallbackContext, build_default_callbacks
from src.training.checkpoint import CheckpointManager, CheckpointState, DEFAULT_CHECKPOINT_DIR
from src.training.dataloader import PlantDiseaseBatch
from src.training.logger import DEFAULT_HISTORY_PATH, DEFAULT_LOG_DIR, TrainingLogger, config_to_hparams
from src.training.losses import LossType, build_loss_function
from src.training.metrics import EpochMetrics, MetricsTracker, TrainingHistory

logger = logging.getLogger(__name__)


def _create_grad_scaler(enabled: bool) -> Any:
    """Create a GradScaler compatible with PyTorch 2.0 and 2.1+."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        from torch.cuda.amp import GradScaler

        return GradScaler(enabled=enabled)


def _autocast_context(device: torch.device, enabled: bool):
    """Return an autocast context manager for the active device."""
    if enabled and device.type == "cuda":
        try:
            return torch.amp.autocast("cuda", enabled=True)
        except (AttributeError, TypeError):
            from torch.cuda.amp import autocast

            return autocast(enabled=True)
    return torch.amp.autocast("cpu", enabled=False)


class OptimizerType(str, Enum):
    """Supported optimizer types."""

    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"


class SchedulerType(str, Enum):
    """Supported learning-rate scheduler types."""

    COSINE = "cosine"
    PLATEAU = "plateau"
    STEP = "step"
    NONE = "none"


@dataclass
class TrainingEngineConfig:
    """Configuration for the generic training engine.

    Attributes:
        num_epochs: Maximum number of training epochs.
        learning_rate: Initial learning rate.
        weight_decay: Optimizer weight decay.
        optimizer: Optimizer algorithm.
        scheduler: Learning-rate scheduler type.
        loss_type: Loss function type.
        num_classes: Number of output classes.
        device: Compute device (``auto``, ``cpu``, or ``cuda``).
        use_amp: Enable automatic mixed precision.
        gradient_clip_norm: Max gradient norm for clipping (``0`` disables).
        gradient_accumulation_steps: Optimizer steps every N micro-batches.
        use_class_weights: Apply class weights to the loss function.
        label_smoothing: Cross-entropy label smoothing factor.
        focal_gamma: Focal loss gamma (when ``loss_type=focal``).
        early_stopping_patience: Epochs without improvement before stopping.
        monitor_metric: Validation metric for checkpointing / early stopping.
        monitor_mode: ``min`` or ``max`` for the monitored metric.
        random_seed: Seed for reproducibility.
        num_workers: Reserved for DataLoader configuration passthrough.
        checkpoint_dir: Directory for model checkpoints.
        log_dir: Directory for TensorBoard logs.
        history_path: Path for JSON training history.
        experiment_name: Optional TensorBoard run name.
        resume_checkpoint: Optional checkpoint path to resume from.
        cosine_t_max: ``T_max`` for :class:`CosineAnnealingLR`.
        step_size: Step size for :class:`StepLR`.
        step_gamma: Multiplicative factor for :class:`StepLR`.
        plateau_factor: Factor for :class:`ReduceLROnPlateau`.
        plateau_patience: Patience for :class:`ReduceLROnPlateau`.
        sgd_momentum: Momentum for SGD.
        sgd_nesterov: Enable Nesterov momentum for SGD.
        save_best_only: Only persist best-model checkpoints.
    """

    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: OptimizerType = OptimizerType.ADAMW
    scheduler: SchedulerType = SchedulerType.COSINE
    loss_type: LossType = LossType.CROSS_ENTROPY
    num_classes: int = 43
    device: str = "auto"
    use_amp: bool = True
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    use_class_weights: bool = True
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    early_stopping_patience: int = 10
    monitor_metric: str = "loss"
    monitor_mode: str = "min"
    random_seed: int = 42
    num_workers: int = 0
    checkpoint_dir: Path = field(default_factory=lambda: DEFAULT_CHECKPOINT_DIR)
    log_dir: Path = field(default_factory=lambda: DEFAULT_LOG_DIR)
    history_path: Path = field(default_factory=lambda: DEFAULT_HISTORY_PATH)
    experiment_name: str | None = None
    resume_checkpoint: Path | None = None
    cosine_t_max: int | None = None
    step_size: int = 10
    step_gamma: float = 0.1
    plateau_factor: float = 0.5
    plateau_patience: int = 5
    sgd_momentum: float = 0.9
    sgd_nesterov: bool = True
    save_best_only: bool = False

    def __post_init__(self) -> None:
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be >= 1")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if self.monitor_mode not in {"min", "max"}:
            raise ValueError("monitor_mode must be 'min' or 'max'")
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.log_dir = Path(self.log_dir)
        self.history_path = Path(self.history_path)
        if self.resume_checkpoint is not None:
            self.resume_checkpoint = Path(self.resume_checkpoint)
        if self.cosine_t_max is None:
            self.cosine_t_max = self.num_epochs

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a dictionary."""
        payload = asdict(self)
        for key in ("checkpoint_dir", "log_dir", "history_path", "resume_checkpoint"):
            value = payload.get(key)
            if value is not None:
                payload[key] = str(value)
        payload["optimizer"] = self.optimizer.value
        payload["scheduler"] = self.scheduler.value
        payload["loss_type"] = self.loss_type.value
        return payload


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve the compute device.

    Args:
        device: ``auto``, ``cpu``, ``cuda``, or ``cuda:N``.

    Returns:
        Resolved :class:`torch.device`.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, config: TrainingEngineConfig) -> Optimizer:
    """Build an optimizer from configuration.

    Args:
        model: Model whose parameters will be optimized.
        config: Training engine configuration.

    Returns:
        Configured optimizer instance.
    """
    parameters = model.parameters()

    if config.optimizer == OptimizerType.ADAM:
        return torch.optim.Adam(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    if config.optimizer == OptimizerType.ADAMW:
        return torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    if config.optimizer == OptimizerType.SGD:
        return torch.optim.SGD(
            parameters,
            lr=config.learning_rate,
            momentum=config.sgd_momentum,
            weight_decay=config.weight_decay,
            nesterov=config.sgd_nesterov,
        )

    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def build_scheduler(optimizer: Optimizer, config: TrainingEngineConfig) -> LRScheduler | None:
    """Build a learning-rate scheduler from configuration.

    Args:
        optimizer: Optimizer to schedule.
        config: Training engine configuration.

    Returns:
        Scheduler instance, or ``None`` when disabled.
    """
    if config.scheduler == SchedulerType.NONE:
        return None

    if config.scheduler == SchedulerType.COSINE:
        return CosineAnnealingLR(optimizer, T_max=config.cosine_t_max or config.num_epochs)

    if config.scheduler == SchedulerType.PLATEAU:
        return ReduceLROnPlateau(
            optimizer,
            mode=config.monitor_mode,
            factor=config.plateau_factor,
            patience=config.plateau_patience,
        )

    if config.scheduler == SchedulerType.STEP:
        return StepLR(optimizer, step_size=config.step_size, gamma=config.step_gamma)

    raise ValueError(f"Unsupported scheduler: {config.scheduler}")


class Trainer:
    """Generic training engine for PyTorch classification models.

    Works with any ``nn.Module`` that accepts image batches and returns logits.
    Integrates AMP, gradient clipping/accumulation, callbacks, checkpointing,
    TensorBoard logging, and TorchMetrics evaluation.

    Args:
        model: Trainable PyTorch model (architecture-agnostic).
        config: Training engine configuration.
        train_loader: Training :class:`DataLoader`.
        val_loader: Validation :class:`DataLoader`.
        class_weights: Optional per-class loss weights.
        callbacks: Optional callback list (default callbacks used when ``None``).
        checkpoint_manager: Optional checkpoint manager override.
        training_logger: Optional training logger override.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingEngineConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        class_weights: torch.Tensor | None = None,
        callbacks: list[Callback] | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        training_logger: TrainingLogger | None = None,
    ) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        set_seed(config.random_seed)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_weights = class_weights

        self.criterion = build_loss_function(config, class_weights, device=self.device)
        self.optimizer = build_optimizer(self.model, config)
        self.scheduler = build_scheduler(self.optimizer, config)

        self.scaler = _create_grad_scaler(
            enabled=config.use_amp and self.device.type == "cuda"
        )

        self.checkpoint_manager = checkpoint_manager or CheckpointManager(
            config.checkpoint_dir,
            monitor=f"val_{config.monitor_metric}",
            mode=config.monitor_mode,
            save_best_only=config.save_best_only,
        )
        self.training_logger = training_logger or TrainingLogger(
            log_dir=config.log_dir,
            history_path=config.history_path,
            experiment_name=config.experiment_name,
        )

        self.callbacks = callbacks or build_default_callbacks(
            self.checkpoint_manager,
            early_stopping_patience=config.early_stopping_patience,
            monitor=config.monitor_metric,
            mode=config.monitor_mode,
        )

        for callback in self.callbacks:
            if hasattr(callback, "bind_trainer"):
                callback.bind_trainer(self)

        self.history = TrainingHistory(
            best_metric_name=f"val_{config.monitor_metric}",
            best_metric_value=float("inf") if config.monitor_mode == "min" else float("-inf"),
        )
        self.start_epoch = 0
        self.global_step = 0

        if config.resume_checkpoint is not None:
            self.resume(config.resume_checkpoint)

        logger.info("Trainer initialized on device=%s, amp=%s", self.device, self.scaler.is_enabled())

    def resume(self, checkpoint_path: Path | str) -> CheckpointState:
        """Resume training from a checkpoint file.

        Args:
            checkpoint_path: Path to a saved checkpoint.

        Returns:
            Restored :class:`CheckpointState`.
        """
        state = self.checkpoint_manager.load_checkpoint(
            checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            map_location=self.device,
        )
        self.start_epoch = state.epoch + 1
        self.global_step = state.global_step
        self.history = TrainingHistory(
            train=[EpochMetrics(**m) for m in state.history.get("train", [])],
            val=[EpochMetrics(**m) for m in state.history.get("val", [])],
            learning_rates=state.history.get("learning_rates", []),
            best_epoch=state.best_epoch,
            best_metric_name=state.best_metric_name,
            best_metric_value=state.best_metric_value,
        )
        return state

    def train(self) -> TrainingHistory:
        """Run the full training loop.

        Returns:
            Complete :class:`TrainingHistory`.
        """
        self._invoke_callbacks("on_train_begin")

        try:
            for epoch in range(self.start_epoch, self.config.num_epochs):
                context = self._run_epoch(epoch)
                if context.should_stop:
                    logger.info("Training stopped early at epoch %d", epoch + 1)
                    break
        finally:
            self.training_logger.save_history(self.history)
            self.training_logger.log_hparams(
                config_to_hparams(self.config),
                self._final_hparam_metrics(),
            )
            self.training_logger.close()
            self._invoke_callbacks("on_train_end")

        return self.history

    def _run_epoch(self, epoch: int) -> CallbackContext:
        context = CallbackContext(
            epoch=epoch,
            global_step=self.global_step,
            train_metrics=EpochMetrics(
                loss=0.0,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                top1_accuracy=0.0,
                top5_accuracy=0.0,
            ),
            val_metrics=EpochMetrics(
                loss=0.0,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                top1_accuracy=0.0,
                top5_accuracy=0.0,
            ),
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            history=self.history,
        )
        self._invoke_epoch_callbacks("on_epoch_begin", context)

        train_metrics = self._train_epoch(epoch)
        val_metrics = self._validate_epoch(epoch)
        learning_rate = float(self.optimizer.param_groups[0]["lr"])

        self.history.train.append(train_metrics)
        self.history.val.append(val_metrics)
        self.history.learning_rates.append(learning_rate)

        self.training_logger.log_epoch(
            epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            learning_rate=learning_rate,
        )

        context.train_metrics = train_metrics
        context.val_metrics = val_metrics
        context.learning_rate = learning_rate
        context.global_step = self.global_step

        self._invoke_epoch_callbacks("on_epoch_end", context)
        return context

    def _train_epoch(self, epoch: int) -> EpochMetrics:
        self.model.train()
        tracker = MetricsTracker(self.config.num_classes, device=self.device)
        tracker.reset()

        progress = tqdm(
            self.train_loader,
            desc=f"Train {epoch + 1}/{self.config.num_epochs}",
            leave=False,
        )

        self.optimizer.zero_grad(set_to_none=True)

        for batch_index, batch in enumerate(progress):
            loss, logits, targets = self._forward_batch(batch, training=True)
            scaled_loss = loss / self.config.gradient_accumulation_steps

            if self.scaler.is_enabled():
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            is_accumulation_step = (
                (batch_index + 1) % self.config.gradient_accumulation_steps == 0
                or (batch_index + 1) == len(self.train_loader)
            )

            if is_accumulation_step:
                if self.scaler.is_enabled():
                    if self.config.gradient_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clip_norm,
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.config.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clip_norm,
                        )
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

            tracker.update(loss, logits, targets)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        return tracker.compute()

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> EpochMetrics:
        self.model.eval()
        tracker = MetricsTracker(self.config.num_classes, device=self.device)
        tracker.reset()

        progress = tqdm(
            self.val_loader,
            desc=f"Val {epoch + 1}/{self.config.num_epochs}",
            leave=False,
        )

        first_non_finite_logged = False

        for batch_index, batch in enumerate(progress):
            loss, logits, targets = self._forward_batch(batch, training=False)

            if not first_non_finite_logged and (
                torch.isnan(logits).any()
                or torch.isinf(logits).any()
                or torch.isnan(loss)
                or torch.isinf(loss)
            ):
                logger.error(
                    "Non-finite validation values at epoch %d batch %d: "
                    "logits_nan=%s logits_inf=%s loss_nan=%s loss_inf=%s",
                    epoch + 1,
                    batch_index,
                    bool(torch.isnan(logits).any().item()),
                    bool(torch.isinf(logits).any().item()),
                    bool(torch.isnan(loss).item()),
                    bool(torch.isinf(loss).item()),
                )
                first_non_finite_logged = True

            tracker.update(loss, logits, targets)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        return tracker.compute()

    def _forward_batch(
        self,
        batch: PlantDiseaseBatch | Any,
        *,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images, targets = self._extract_batch(batch)
        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        # Validation loss must run in full precision: larger models (e.g. EfficientNet-B3)
        # can produce NaN logits under fp16 autocast while argmax accuracy still looks valid.
        use_amp = self.scaler.is_enabled() and training

        with _autocast_context(self.device, use_amp):
            logits = self.model(images)
            loss = self.criterion(logits, targets)

        return loss, logits, targets

    @staticmethod
    def _extract_batch(batch: PlantDiseaseBatch | Any) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, PlantDiseaseBatch):
            return batch.images, batch.class_indices

        if isinstance(batch, dict):
            return batch["images"], batch["class_indices"]

        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            return batch[0], batch[1]

        raise TypeError(f"Unsupported batch type: {type(batch).__name__}")

    def _invoke_callbacks(self, method_name: str) -> None:
        for callback in self.callbacks:
            method = getattr(callback, method_name, None)
            if callable(method):
                method(self)

    def _invoke_epoch_callbacks(self, method_name: str, context: CallbackContext) -> None:
        for callback in self.callbacks:
            method = getattr(callback, method_name, None)
            if callable(method):
                method(context)

    def _final_hparam_metrics(self) -> dict[str, float]:
        if not self.history.val:
            return {}
        last_val = self.history.val[-1]
        return {
            "val_loss": last_val.loss,
            "val_accuracy": last_val.accuracy,
            "val_f1_score": last_val.f1_score,
        }


def get_training_engine_config(**overrides: Any) -> TrainingEngineConfig:
    """Build a :class:`TrainingEngineConfig` with optional overrides.

    Args:
        **overrides: Field overrides accepted by :class:`TrainingEngineConfig`.

    Returns:
        Configured training engine instance.
    """
    return TrainingEngineConfig(**overrides)
