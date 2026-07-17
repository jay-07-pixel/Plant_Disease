"""Epoch metrics using TorchMetrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)


@dataclass
class EpochMetrics:
    """Metrics collected for one training or validation epoch.

    Attributes:
        loss: Average loss for the epoch.
        accuracy: Top-1 accuracy (macro).
        precision: Macro precision.
        recall: Macro recall.
        f1_score: Macro F1 score.
        top1_accuracy: Explicit top-1 accuracy.
        top5_accuracy: Top-5 accuracy (``0.0`` when fewer than five classes).
        num_samples: Number of samples evaluated.
    """

    loss: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    top1_accuracy: float
    top5_accuracy: float
    num_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics to a dictionary."""
        return asdict(self)


@dataclass
class TrainingHistory:
    """Full training history across epochs.

    Attributes:
        train: Per-epoch training metrics.
        val: Per-epoch validation metrics.
        learning_rates: Learning rate recorded after each epoch.
        best_epoch: Epoch index with the best validation score.
        best_metric_name: Metric used for best-model selection.
        best_metric_value: Best validation metric value observed.
    """

    train: list[EpochMetrics] = field(default_factory=list)
    val: list[EpochMetrics] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_metric_name: str = "val_loss"
    best_metric_value: float = float("inf")

    def to_dict(self) -> dict[str, Any]:
        """Serialize history to a JSON-compatible dictionary."""
        return {
            "train": [metrics.to_dict() for metrics in self.train],
            "val": [metrics.to_dict() for metrics in self.val],
            "learning_rates": self.learning_rates,
            "best_epoch": self.best_epoch,
            "best_metric_name": self.best_metric_name,
            "best_metric_value": self.best_metric_value,
        }


def build_metric_collection(
    num_classes: int,
    *,
    top_k: int = 5,
    device: torch.device | None = None,
) -> MetricCollection:
    """Build a TorchMetrics collection for multi-class classification.

    Args:
        num_classes: Number of output classes.
        top_k: ``k`` for top-k accuracy (capped at ``num_classes``).
        device: Device for metric state tensors.

    Returns:
        :class:`MetricCollection` with accuracy, precision, recall, F1, and top-k.
    """
    effective_top_k = min(top_k, num_classes)
    metrics: dict[str, Any] = {
        "accuracy": MulticlassAccuracy(num_classes=num_classes, average="macro"),
        "precision": MulticlassPrecision(num_classes=num_classes, average="macro"),
        "recall": MulticlassRecall(num_classes=num_classes, average="macro"),
        "f1_score": MulticlassF1Score(num_classes=num_classes, average="macro"),
        "top1_accuracy": MulticlassAccuracy(num_classes=num_classes, top_k=1, average="macro"),
    }

    if effective_top_k > 1:
        metrics["top5_accuracy"] = MulticlassAccuracy(
            num_classes=num_classes,
            top_k=effective_top_k,
            average="macro",
        )

    collection = MetricCollection(metrics)
    if device is not None:
        collection = collection.to(device)
    return collection


class MetricsTracker:
    """Accumulates loss and classification metrics for one epoch."""

    def __init__(
        self,
        num_classes: int,
        *,
        top_k: int = 5,
        device: torch.device | None = None,
    ) -> None:
        self._metrics = build_metric_collection(num_classes, top_k=top_k, device=device)
        self._total_loss = 0.0
        self._num_samples = 0
        self._num_batches = 0
        self._top_k_available = "top5_accuracy" in self._metrics

    def reset(self) -> None:
        """Reset accumulated state for a new epoch."""
        self._metrics.reset()
        self._total_loss = 0.0
        self._num_samples = 0
        self._num_batches = 0

    def update(
        self,
        loss: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Update metrics with one batch.

        Args:
            loss: Scalar batch loss tensor.
            logits: Model logits ``(N, C)``.
            targets: Ground-truth labels ``(N,)``.
        """
        batch_size = targets.numel()
        self._total_loss += float(loss.detach().item()) * batch_size
        self._num_samples += batch_size
        self._num_batches += 1
        self._metrics.update(logits.detach(), targets)

    def compute(self) -> EpochMetrics:
        """Compute aggregated epoch metrics.

        Returns:
            :class:`EpochMetrics` for the completed epoch.
        """
        computed = self._metrics.compute()
        avg_loss = self._total_loss / max(self._num_samples, 1)

        top5 = 0.0
        if self._top_k_available:
            top5 = float(computed["top5_accuracy"].item())

        return EpochMetrics(
            loss=avg_loss,
            accuracy=float(computed["accuracy"].item()),
            precision=float(computed["precision"].item()),
            recall=float(computed["recall"].item()),
            f1_score=float(computed["f1_score"].item()),
            top1_accuracy=float(computed["top1_accuracy"].item()),
            top5_accuracy=top5,
            num_samples=self._num_samples,
        )
