"""Loss functions for the training engine."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from src.training.trainer import TrainingEngineConfig


class LossType(str, Enum):
    """Supported loss function types."""

    CROSS_ENTROPY = "cross_entropy"
    FOCAL = "focal"


class FocalLoss(nn.Module):
    """Focal loss for imbalanced multi-class classification.

    Reference: Lin et al., "Focal Loss for Dense Object Detection".

    Args:
        gamma: Focusing parameter (higher values down-weight easy examples).
        alpha: Optional per-class weight tensor.
        reduction: Reduction mode (``mean``, ``sum``, or ``none``).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Logits tensor ``(N, C)``.
            targets: Ground-truth class indices ``(N,)``.

        Returns:
            Scalar loss (when ``reduction='mean'``).
        """
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        probabilities = torch.exp(-ce_loss)
        focal_loss = ((1.0 - probabilities) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def build_loss_function(
    config: TrainingEngineConfig,
    class_weights: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> nn.Module:
    """Build a loss function from engine configuration.

    Args:
        config: Training engine configuration.
        class_weights: Optional per-class weights for the loss.
        device: Device to place class weights on.

    Returns:
        Configured loss module.

    Raises:
        ValueError: If the loss type is unsupported.
    """
    weights = None
    if class_weights is not None and config.use_class_weights:
        weights = class_weights.to(device) if device is not None else class_weights

    if config.loss_type == LossType.CROSS_ENTROPY:
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=config.label_smoothing)

    if config.loss_type == LossType.FOCAL:
        return FocalLoss(gamma=config.focal_gamma, alpha=weights)

    raise ValueError(f"Unsupported loss type: {config.loss_type}")
