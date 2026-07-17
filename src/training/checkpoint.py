"""Model checkpoint saving, best-model tracking, and resume support."""

from __future__ import annotations

import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_DIR = Path("saved_models")
LATEST_CHECKPOINT_NAME = "latest_checkpoint.pt"
BEST_MODEL_NAME = "best_model.pt"


@dataclass
class CheckpointState:
    """Serializable training state for resume support.

    Attributes:
        epoch: Last completed epoch index (0-based).
        global_step: Total optimizer steps taken.
        best_metric_value: Best validation metric observed.
        best_epoch: Epoch index of the best model.
        best_metric_name: Metric name used for model selection.
        history: Serialized training history dictionary.
        config: Serialized training configuration dictionary.
    """

    epoch: int
    global_step: int
    best_metric_value: float
    best_epoch: int
    best_metric_name: str
    history: dict[str, Any]
    config: dict[str, Any]


class CheckpointManager:
    """Manages periodic, best, and latest model checkpoints.

    Args:
        checkpoint_dir: Directory for ``.pt`` checkpoint files.
        monitor: Validation metric to minimize (``val_loss``) or maximize (``val_f1_score``).
        mode: ``min`` or ``max`` for the monitored metric.
        save_best_only: If ``True``, only persist improvements to the best model.
    """

    def __init__(
        self,
        checkpoint_dir: Path | str = DEFAULT_CHECKPOINT_DIR,
        *,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = False,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_metric_value = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = -1

    @property
    def latest_checkpoint_path(self) -> Path:
        """Path to the most recent full checkpoint."""
        return self.checkpoint_dir / LATEST_CHECKPOINT_NAME

    @property
    def best_model_path(self) -> Path:
        """Path to the best-performing model weights."""
        return self.checkpoint_dir / BEST_MODEL_NAME

    def is_improvement(self, metric_value: float) -> bool:
        """Return whether ``metric_value`` improves on the current best."""
        if self.mode == "min":
            return metric_value < self.best_metric_value
        return metric_value > self.best_metric_value

    def save_checkpoint(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        scaler: torch.amp.GradScaler | None,
        state: CheckpointState,
        is_best: bool = False,
        epoch_checkpoint: bool = True,
    ) -> Path:
        """Save a full training checkpoint.

        Args:
            model: Model being trained.
            optimizer: Optimizer instance.
            scheduler: Optional learning-rate scheduler.
            scaler: Optional AMP gradient scaler.
            state: Training state metadata.
            is_best: Whether this checkpoint is the new best model.
            epoch_checkpoint: Whether to write ``latest_checkpoint.pt``.

        Returns:
            Path to the saved checkpoint file.
        """
        payload: dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "state": asdict(state),
        }

        if is_best:
            self.best_metric_value = state.best_metric_value
            self.best_epoch = state.best_epoch
            torch.save(payload, self.best_model_path)
            logger.info(
                "Saved best model to %s (%s=%.6f, epoch=%d)",
                self.best_model_path,
                self.monitor,
                state.best_metric_value,
                state.best_epoch,
            )

        if epoch_checkpoint and not (self.save_best_only and not is_best):
            torch.save(payload, self.latest_checkpoint_path)
            logger.info("Saved latest checkpoint to %s", self.latest_checkpoint_path)

        epoch_path = self.checkpoint_dir / f"checkpoint_epoch_{state.epoch:04d}.pt"
        if not self.save_best_only:
            torch.save(payload, epoch_path)

        return self.latest_checkpoint_path if epoch_checkpoint else self.best_model_path

    def load_checkpoint(
        self,
        checkpoint_path: Path | str,
        *,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        scaler: torch.amp.GradScaler | None = None,
        map_location: str | torch.device = "cpu",
    ) -> CheckpointState:
        """Load a checkpoint and restore training state.

        Args:
            checkpoint_path: Path to a ``.pt`` checkpoint file.
            model: Model to restore weights into.
            optimizer: Optional optimizer to restore.
            scheduler: Optional scheduler to restore.
            scaler: Optional AMP scaler to restore.
            map_location: Device mapping for ``torch.load``.

        Returns:
            Restored :class:`CheckpointState`.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        payload = torch.load(path, map_location=map_location, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])

        if optimizer is not None and payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])

        if scheduler is not None and payload.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(payload["scheduler_state_dict"])

        if scaler is not None and payload.get("scaler_state_dict") is not None:
            scaler.load_state_dict(payload["scaler_state_dict"])

        state_dict = payload["state"]
        state = CheckpointState(**state_dict)
        self.best_metric_value = state.best_metric_value
        self.best_epoch = state.best_epoch

        logger.info("Loaded checkpoint from %s (epoch=%d)", path, state.epoch)
        return state

    def export_best_model_copy(self, destination: Path | str) -> Path:
        """Copy the best model checkpoint to another path.

        Args:
            destination: Target file path.

        Returns:
            Destination path.
        """
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not self.best_model_path.exists():
            raise FileNotFoundError(f"Best model not found: {self.best_model_path}")
        shutil.copy2(self.best_model_path, dest)
        return dest
