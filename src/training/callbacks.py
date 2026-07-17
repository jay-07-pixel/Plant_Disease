"""Training callbacks for early stopping, checkpointing, and lifecycle hooks."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.training.metrics import EpochMetrics, TrainingHistory

if TYPE_CHECKING:
    from src.training.checkpoint import CheckpointManager
    from src.training.trainer import Trainer

logger = logging.getLogger(__name__)


@dataclass
class CallbackContext:
    """Shared context passed to callbacks during training.

    Attributes:
        epoch: Current epoch index (0-based).
        global_step: Total optimizer steps completed.
        train_metrics: Training metrics for the current epoch.
        val_metrics: Validation metrics for the current epoch.
        learning_rate: Current learning rate.
        model: Model being trained.
        optimizer: Active optimizer.
        scheduler: Active scheduler, if any.
        history: Training history accumulated so far.
        should_stop: Flag set by callbacks to request early termination.
    """

    epoch: int
    global_step: int
    train_metrics: EpochMetrics
    val_metrics: EpochMetrics
    learning_rate: float
    model: nn.Module
    optimizer: Optimizer
    scheduler: LRScheduler | None
    history: TrainingHistory
    should_stop: bool = False


class Callback(ABC):
    """Base class for training callbacks."""

    def on_train_begin(self, trainer: Trainer) -> None:
        """Called once before the training loop starts."""

    def on_train_end(self, trainer: Trainer) -> None:
        """Called once after the training loop ends."""

    def on_epoch_begin(self, context: CallbackContext) -> None:
        """Called at the start of each epoch."""

    def on_epoch_end(self, context: CallbackContext) -> None:
        """Called at the end of each epoch after validation."""

    def on_batch_end(self, context: CallbackContext) -> None:
        """Called after each optimizer step during training."""


class EarlyStoppingCallback(Callback):
    """Stop training when a monitored validation metric stops improving.

    Args:
        monitor: Metric key on :class:`EpochMetrics` (e.g. ``val_loss`` resolved via val_metrics).
        patience: Epochs to wait without improvement before stopping.
        mode: ``min`` or ``max`` for the monitored metric.
        min_delta: Minimum change to qualify as an improvement.
    """

    def __init__(
        self,
        monitor: str = "loss",
        patience: int = 10,
        mode: str = "min",
        min_delta: float = 0.0,
    ) -> None:
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self._best_value = float("inf") if mode == "min" else float("-inf")
        self._epochs_without_improvement = 0

    def _metric_value(self, context: CallbackContext) -> float:
        return float(getattr(context.val_metrics, self.monitor))

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "min":
            return value < (self._best_value - self.min_delta)
        return value > (self._best_value + self.min_delta)

    def on_epoch_end(self, context: CallbackContext) -> None:
        current = self._metric_value(context)
        if self._is_improvement(current):
            self._best_value = current
            self._epochs_without_improvement = 0
            return

        self._epochs_without_improvement += 1
        if self._epochs_without_improvement >= self.patience:
            context.should_stop = True
            logger.info(
                "Early stopping triggered after %d epochs without %s improvement",
                self.patience,
                self.monitor,
            )


class CheckpointCallback(Callback):
    """Persist full checkpoints through :class:`CheckpointManager`.

    Requires the trainer reference to be set via :meth:`bind_trainer`.
    """

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        monitor: str = "loss",
        mode: str = "min",
    ) -> None:
        self.checkpoint_manager = checkpoint_manager
        self.monitor = monitor
        self.mode = mode
        self._trainer: Trainer | None = None

    def bind_trainer(self, trainer: Trainer) -> None:
        """Attach the trainer instance for checkpoint serialization."""
        self._trainer = trainer

    def on_epoch_end(self, context: CallbackContext) -> None:
        if self._trainer is None:
            raise RuntimeError("CheckpointCallback.bind_trainer() must be called before training.")

        metric_value = float(getattr(context.val_metrics, self.monitor))
        is_best = self.checkpoint_manager.is_improvement(metric_value)

        if is_best:
            self.checkpoint_manager.best_metric_value = metric_value
            self.checkpoint_manager.best_epoch = context.epoch
            context.history.best_epoch = context.epoch
            context.history.best_metric_name = f"val_{self.monitor}"
            context.history.best_metric_value = metric_value

        from src.training.checkpoint import CheckpointState

        state = CheckpointState(
            epoch=context.epoch,
            global_step=context.global_step,
            best_metric_value=self.checkpoint_manager.best_metric_value,
            best_epoch=self.checkpoint_manager.best_epoch,
            best_metric_name=f"val_{self.monitor}",
            history=context.history.to_dict(),
            config=self._trainer.config.to_dict(),
        )

        self.checkpoint_manager.save_checkpoint(
            model=context.model,
            optimizer=context.optimizer,
            scheduler=context.scheduler,
            scaler=self._trainer.scaler,
            state=state,
            is_best=is_best,
        )


class LearningRateSchedulerCallback(Callback):
    """Step the learning-rate scheduler at epoch boundaries.

    Args:
        monitor: Metric for ``ReduceLROnPlateau`` (ignored by step-based schedulers).
        step_on: ``epoch`` or ``batch`` stepping interval.
    """

    def __init__(self, monitor: str = "loss", step_on: str = "epoch") -> None:
        self.monitor = monitor
        self.step_on = step_on

    def on_epoch_end(self, context: CallbackContext) -> None:
        if context.scheduler is None or self.step_on != "epoch":
            return

        if self._is_plateau_scheduler(context.scheduler):
            metric_value = float(getattr(context.val_metrics, self.monitor))
            context.scheduler.step(metric_value)
        else:
            context.scheduler.step()

    def on_batch_end(self, context: CallbackContext) -> None:
        if context.scheduler is None or self.step_on != "batch":
            return
        if not self._is_plateau_scheduler(context.scheduler):
            context.scheduler.step()

    @staticmethod
    def _is_plateau_scheduler(scheduler: LRScheduler) -> bool:
        return scheduler.__class__.__name__ == "ReduceLROnPlateau"


def build_default_callbacks(
    checkpoint_manager: CheckpointManager,
    *,
    early_stopping_patience: int | None = 10,
    monitor: str = "loss",
    mode: str = "min",
) -> list[Callback]:
    """Build the default callback list for training.

    Args:
        checkpoint_manager: Checkpoint manager for model persistence.
        early_stopping_patience: Patience for early stopping (``None`` disables).
        monitor: Validation metric attribute name.
        mode: ``min`` or ``max`` for monitored metric.

    Returns:
        List of configured callbacks.
    """
    callbacks: list[Callback] = [
        CheckpointCallback(checkpoint_manager, monitor=monitor, mode=mode),
        LearningRateSchedulerCallback(monitor=monitor, step_on="epoch"),
    ]

    if early_stopping_patience is not None and early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                monitor=monitor,
                patience=early_stopping_patience,
                mode=mode,
            )
        )

    return callbacks
