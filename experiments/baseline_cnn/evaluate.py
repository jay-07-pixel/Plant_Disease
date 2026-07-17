"""Post-training evaluation: confusion matrix and per-class accuracy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.dataloader import PlantDiseaseBatch
from src.training.dataset import LabelEncoder


@dataclass
class EvaluationResult:
    """Validation evaluation results for reporting.

    Attributes:
        confusion_matrix: Square matrix ``(num_classes, num_classes)``.
        per_class_accuracy: Mapping of canonical label to accuracy.
        class_labels: Ordered canonical labels by class index.
        predictions: All predicted class indices.
        targets: All ground-truth class indices.
        overall_accuracy: Fraction of correct predictions.
    """

    confusion_matrix: list[list[int]]
    per_class_accuracy: dict[str, float]
    class_labels: list[str]
    predictions: list[int]
    targets: list[int]
    overall_accuracy: float


@torch.no_grad()
def evaluate_on_dataloader(
    model: nn.Module,
    dataloader: DataLoader,
    label_encoder: LabelEncoder,
    device: torch.device,
) -> EvaluationResult:
    """Run inference and compute confusion matrix on a dataloader.

    Args:
        model: Trained classification model.
        dataloader: Evaluation dataloader (typically validation).
        label_encoder: Label encoder for canonical class names.
        device: Compute device.

    Returns:
        :class:`EvaluationResult` with confusion matrix and per-class accuracy.
    """
    model.eval()
    num_classes = label_encoder.num_classes

    all_predictions: list[int] = []
    all_targets: list[int] = []

    for batch in dataloader:
        if not isinstance(batch, PlantDiseaseBatch):
            raise TypeError("Expected PlantDiseaseBatch from project DataLoader.")

        images = batch.images.to(device, non_blocking=True)
        targets = batch.class_indices.to(device, non_blocking=True)
        logits = model(images)
        predictions = torch.argmax(logits, dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    confusion = _compute_confusion_matrix(all_targets, all_predictions, num_classes)
    class_labels = [label_encoder.index_to_label[index] for index in range(num_classes)]
    per_class_accuracy = _per_class_accuracy(confusion, class_labels)
    overall_accuracy = float(np.trace(confusion) / max(len(all_targets), 1))

    return EvaluationResult(
        confusion_matrix=confusion.tolist(),
        per_class_accuracy=per_class_accuracy,
        class_labels=class_labels,
        predictions=all_predictions,
        targets=all_targets,
        overall_accuracy=overall_accuracy,
    )


def _compute_confusion_matrix(
    targets: list[int],
    predictions: list[int],
    num_classes: int,
) -> np.ndarray:
    """Build a confusion matrix from prediction lists."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1
    return matrix


def _per_class_accuracy(
    confusion: np.ndarray,
    class_labels: list[str],
) -> dict[str, float]:
    """Compute per-class accuracy from a confusion matrix."""
    per_class: dict[str, float] = {}
    for index, label in enumerate(class_labels):
        support = int(confusion[index].sum())
        if support == 0:
            per_class[label] = 0.0
        else:
            per_class[label] = float(confusion[index, index] / support)
    return per_class


def evaluation_result_to_dict(result: EvaluationResult) -> dict[str, Any]:
    """Serialize an evaluation result to a JSON-compatible dictionary."""
    return {
        "confusion_matrix": result.confusion_matrix,
        "per_class_accuracy": result.per_class_accuracy,
        "class_labels": result.class_labels,
        "overall_accuracy": result.overall_accuracy,
        "num_samples": len(result.targets),
    }
