"""Experiment callbacks extending the training framework without modifying it."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from src.training.callbacks import Callback, CallbackContext

if TYPE_CHECKING:
    from src.training.trainer import Trainer


class EpochTimingCallback(Callback):
    """Record per-epoch wall-clock duration and log to TensorBoard.

    Args:
        epoch_times: Mutable list populated with epoch durations in seconds.
    """

    def __init__(self, epoch_times: list[float]) -> None:
        self.epoch_times = epoch_times
        self._epoch_start: float | None = None
        self._train_start: float | None = None
        self._trainer: Trainer | None = None

    def bind_trainer(self, trainer: Trainer) -> None:
        """Attach trainer reference for TensorBoard logging."""
        self._trainer = trainer

    def on_train_begin(self, trainer: Trainer) -> None:
        self._trainer = trainer
        self._train_start = time.perf_counter()

    def on_epoch_begin(self, context: CallbackContext) -> None:
        self._epoch_start = time.perf_counter()

    def on_epoch_end(self, context: CallbackContext) -> None:
        if self._epoch_start is None:
            return

        elapsed = time.perf_counter() - self._epoch_start
        self.epoch_times.append(elapsed)

        if self._trainer is not None:
            self._trainer.training_logger.writer.add_scalar(
                "epoch_time_seconds",
                elapsed,
                context.epoch,
            )

    @property
    def total_training_time(self) -> float:
        """Total wall-clock time since training began."""
        if self._train_start is None:
            return sum(self.epoch_times)
        return time.perf_counter() - self._train_start
