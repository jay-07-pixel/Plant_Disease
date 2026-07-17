"""ResNet101 transfer-learning classifier for plant disease detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models import ResNet101_Weights, resnet101

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


class ResNet101TransferClassifier(nn.Module):
    """ImageNet-pretrained ResNet101 with staged transfer-learning support.

    Loads ``torchvision.models.resnet101`` with ImageNet weights, replaces ``fc``
    with a linear layer for ``num_classes`` plant disease categories, and
    supports two training stages:

    * **Stage 1 (feature extraction):** freeze the backbone, train only ``fc``.
    * **Stage 2 (fine-tuning):** unfreeze ``layer4`` and ``fc``, keep earlier
      blocks frozen.

    Args:
        num_classes: Number of output classes.
        freeze_backbone: If ``True``, freeze all layers except ``fc`` on init.
        weights: Pretrained weights enum. Defaults to ImageNet V1 weights.
    """

    def __init__(
        self,
        num_classes: int = DEFAULT_NUM_CLASSES,
        *,
        freeze_backbone: bool = True,
        weights: ResNet101_Weights | None = ResNet101_Weights.IMAGENET1K_V1,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")

        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone

        self.backbone = resnet101(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            self._freeze_backbone()

        counts = count_parameters(self)
        logger.info(
            "ResNet101TransferClassifier ready: classes=%d, trainable=%s, frozen=%s",
            num_classes,
            f"{counts.trainable:,}",
            f"{counts.frozen:,}",
        )

    def _freeze_backbone(self) -> None:
        """Freeze all parameters except the final fully-connected layer."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        for parameter in self.backbone.fc.parameters():
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

    def unfreeze_layer4(self) -> ParameterCounts:
        """Stage 2: unfreeze ``layer4`` and ``fc``; keep earlier blocks frozen."""
        frozen_prefixes = ("conv1.", "bn1.", "layer1.", "layer2.", "layer3.")
        trainable_prefixes = ("layer4.", "fc.")

        for name, parameter in self.backbone.named_parameters():
            if name.startswith(trainable_prefixes):
                parameter.requires_grad = True
            elif name.startswith(frozen_prefixes):
                parameter.requires_grad = False
            else:
                parameter.requires_grad = False

        self.freeze_backbone = False
        counts = count_parameters(self)
        logger.info(
            "Layer4 unfrozen for fine-tuning: trainable=%s, frozen=%s",
            f"{counts.trainable:,}",
            f"{counts.frozen:,}",
        )
        return counts

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
        return self.backbone.fc


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
    model: ResNet101TransferClassifier,
    *,
    batch_size: int = 2,
    image_size: int = 224,
    device: torch.device | None = None,
) -> dict[str, list[int] | str]:
    """Run a dummy forward pass to verify tensor shapes.

    Args:
        model: ResNet101 transfer classifier.
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
