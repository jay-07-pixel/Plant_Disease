"""DenseNet121 transfer-learning classifier for plant disease detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121

logger = logging.getLogger(__name__)

DEFAULT_NUM_CLASSES = 43


@dataclass(frozen=True)
class ParameterCounts:
    """Trainable and frozen parameter statistics.

    Attributes:
        total: Total parameter count.
        trainable: Parameters with ``requires_grad=True``.
        frozen: Parameters with ``requires_grad=False``.
    """

    total: int
    trainable: int
    frozen: int


class DenseNet121TransferClassifier(nn.Module):
    """ImageNet-pretrained DenseNet121 with staged transfer-learning support.

    Loads ``torchvision.models.densenet121`` with ImageNet weights, replaces
    ``classifier`` with a linear layer for ``num_classes`` plant disease
    categories, and supports two training stages:

    * **Stage 1 (feature extraction):** freeze the backbone, train only
      ``classifier``.
    * **Stage 2 (fine-tuning):** unfreeze ``features.denseblock4`` and
      ``classifier``, keep earlier layers frozen.

    Args:
        num_classes: Number of output classes.
        freeze_backbone: If ``True``, freeze all layers except ``classifier``.
        weights: Pretrained weights enum. Defaults to ImageNet V1 weights.
    """

    def __init__(
        self,
        num_classes: int = DEFAULT_NUM_CLASSES,
        *,
        freeze_backbone: bool = True,
        weights: DenseNet121_Weights | None = DenseNet121_Weights.IMAGENET1K_V1,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")

        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone

        self.backbone = densenet121(weights=weights)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            self._freeze_backbone()

        counts = count_parameters(self)
        logger.info(
            "DenseNet121TransferClassifier ready: classes=%d, trainable=%s, frozen=%s",
            num_classes,
            f"{counts.trainable:,}",
            f"{counts.frozen:,}",
        )

    def _freeze_backbone(self) -> None:
        """Freeze all parameters except the final classifier head."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        for parameter in self.backbone.classifier.parameters():
            parameter.requires_grad = True

    def freeze_feature_extractor(self) -> ParameterCounts:
        """Stage 1: freeze the full backbone and train only the classifier head."""
        self._freeze_backbone()
        self.freeze_backbone = True
        counts = count_parameters(self)
        logger.info(
            "Feature extractor frozen: trainable=%s, frozen=%s",
            f"{counts.trainable:,}",
            f"{counts.frozen:,}",
        )
        return counts

    def unfreeze_denseblock4(self) -> ParameterCounts:
        """Stage 2: unfreeze ``denseblock4`` and ``classifier``; keep earlier layers frozen."""
        trainable_prefixes = (
            "features.denseblock4.",
            "classifier.",
        )

        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad = name.startswith(trainable_prefixes)

        self.freeze_backbone = False
        counts = count_parameters(self)
        logger.info(
            "Denseblock4 unfrozen for fine-tuning: trainable=%s, frozen=%s",
            f"{counts.trainable:,}",
            f"{counts.frozen:,}",
        )
        return counts

    # Alias used by the shared two-stage experiment orchestration.
    def unfreeze_layer4(self) -> ParameterCounts:
        """Compatibility alias for Stage 2 unfreeze (DenseNet denseblock4)."""
        return self.unfreeze_denseblock4()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for an image batch.

        Args:
            x: Input tensor of shape ``(N, 3, H, W)``.

        Returns:
            Logits tensor of shape ``(N, num_classes)``.
        """
        return self.backbone(x)

    @property
    def classifier(self) -> nn.Linear:
        """Final classification layer."""
        return self.backbone.classifier


def count_parameters(model: nn.Module) -> ParameterCounts:
    """Count total, trainable, and frozen parameters.

    Args:
        model: PyTorch model.

    Returns:
        :class:`ParameterCounts` instance.
    """
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count

    return ParameterCounts(total=total, trainable=trainable, frozen=total - trainable)


@torch.no_grad()
def verify_forward_pass(
    model: DenseNet121TransferClassifier,
    *,
    batch_size: int = 2,
    image_size: int = 224,
    device: torch.device | None = None,
) -> dict[str, list[int] | str]:
    """Run a dummy forward pass to verify tensor shapes.

    Args:
        model: DenseNet121 transfer classifier.
        batch_size: Dummy batch size.
        image_size: Dummy spatial size.
        device: Optional device for the forward pass.

    Returns:
        Dictionary with input/output shapes and status.

    Raises:
        RuntimeError: If output shape does not match expectations.
    """
    device = device or torch.device("cpu")
    model = model.to(device)
    model.eval()

    dummy_input = torch.zeros(batch_size, 3, image_size, image_size, device=device)
    logits = model(dummy_input)
    expected_shape = [batch_size, model.num_classes]

    if list(logits.shape) != expected_shape:
        raise RuntimeError(
            f"Forward pass shape mismatch: expected {expected_shape}, got {list(logits.shape)}"
        )

    return {
        "status": "passed",
        "input_shape": list(dummy_input.shape),
        "output_shape": list(logits.shape),
        "dtype": str(logits.dtype),
    }
