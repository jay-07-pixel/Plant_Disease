"""Training logger with TensorBoard and JSON history support."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter

from src.training.metrics import EpochMetrics, TrainingHistory

logger = logging.getLogger(__name__)

DEFAULT_LOG_DIR = Path("logs")
DEFAULT_HISTORY_PATH = Path("logs/training_history.json")


class TrainingLogger:
    """Logs metrics to TensorBoard, Python logging, and JSON history files.

    Args:
        log_dir: Root directory for TensorBoard event files.
        history_path: Path for the JSON training history artifact.
        experiment_name: Optional run name appended to ``log_dir``.
    """

    def __init__(
        self,
        log_dir: Path | str = DEFAULT_LOG_DIR,
        history_path: Path | str = DEFAULT_HISTORY_PATH,
        experiment_name: str | None = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        if experiment_name:
            self.log_dir = self.log_dir / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

        self._writer = SummaryWriter(log_dir=str(self.log_dir))
        logger.info("TensorBoard logging enabled at %s", self.log_dir)

    @property
    def writer(self) -> SummaryWriter:
        """Underlying TensorBoard :class:`SummaryWriter`."""
        return self._writer

    def log_epoch(
        self,
        epoch: int,
        *,
        train_metrics: EpochMetrics,
        val_metrics: EpochMetrics,
        learning_rate: float,
    ) -> None:
        """Log scalar metrics for one epoch.

        Args:
            epoch: Current epoch index (0-based).
            train_metrics: Training epoch metrics.
            val_metrics: Validation epoch metrics.
            learning_rate: Current learning rate.
        """
        self._log_split_metrics("train", epoch, train_metrics)
        self._log_split_metrics("val", epoch, val_metrics)
        self._writer.add_scalar("learning_rate", learning_rate, epoch)

        logger.info(
            "Epoch %03d | train_loss=%.4f val_loss=%.4f val_acc=%.4f val_f1=%.4f lr=%.2e",
            epoch + 1,
            train_metrics.loss,
            val_metrics.loss,
            val_metrics.accuracy,
            val_metrics.f1_score,
            learning_rate,
        )

    def _log_split_metrics(self, split: str, epoch: int, metrics: EpochMetrics) -> None:
        for name, value in metrics.to_dict().items():
            if name == "num_samples":
                continue
            self._writer.add_scalar(f"{split}/{name}", value, epoch)

    def save_history(self, history: TrainingHistory) -> Path:
        """Persist training history to JSON.

        Args:
            history: Complete training history.

        Returns:
            Path to the saved JSON file.
        """
        payload = history.to_dict()
        with self.history_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info("Saved training history to %s", self.history_path)
        return self.history_path

    def log_hparams(self, hparams: dict[str, Any], metrics: dict[str, float]) -> None:
        """Log hyperparameters and final metrics to TensorBoard.

        Args:
            hparams: Flat hyperparameter dictionary (scalar values only).
            metrics: Final metric dictionary.
        """
        flat_hparams = {key: _serialize_hparam(value) for key, value in hparams.items()}
        self._writer.add_hparams(flat_hparams, metrics)

    def close(self) -> None:
        """Flush and close the TensorBoard writer."""
        self._writer.flush()
        self._writer.close()
        logger.info("Closed TensorBoard writer")


def _serialize_hparam(value: Any) -> str | int | float | bool:
    """Convert a hyperparameter value to a TensorBoard-compatible scalar."""
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def config_to_hparams(config: Any) -> dict[str, Any]:
    """Serialize a dataclass config to a flat hparams dictionary.

    Args:
        config: Dataclass configuration instance.

    Returns:
        Dictionary suitable for :meth:`TrainingLogger.log_hparams`.
    """
    return {key: _serialize_hparam(value) for key, value in asdict(config).items()}
