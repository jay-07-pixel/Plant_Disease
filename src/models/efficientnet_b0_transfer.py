"""EfficientNet-B0 transfer-learning classifier for plant disease detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

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


class EfficientNetB0TransferClassifier(nn.Module):
    """ImageNet-pretrained EfficientNet-B0 with staged transfer-learning support.

    Loads ``torchvision.models.efficientnet_b0`` with ImageNet weights, replaces
    the final linear classifier for ``num_classes`` plant disease categories, and
    supports two training stages:

    * **Stage 1 (feature extraction):** freeze the backbone, train only the
      classifier head.
    * **Stage 2 (fine-tuning):** unfreeze the final feature stage
      (``features.7`` last MBConv block + ``features.8`` head conv) and the
      classifier; keep earlier layers frozen.

    Args:
        num_classes: Number of output classes.
        freeze_backbone: If ``True``, freeze all layers except the classifier.
        weights: Pretrained weights enum. Defaults to ImageNet V1 weights.
        dropout: Dropout probability kept in the classifier Sequential.
    """

    def __init__(
        self,
        num_classes: int = DEFAULT_NUM_CLASSES,
        *,
        freeze_backbone: bool = True,
        weights: EfficientNet_B0_Weights | None = EfficientNet_B0_Weights.IMAGENET1K_V1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")

        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone

        self.backbone = efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

        if freeze_backbone:
            self._freeze_backbone()

        counts = count_parameters(self)
        logger.info(
            "EfficientNetB0TransferClassifier ready: classes=%d, trainable=%s, frozen=%s",
            num_classes,
            f"{counts.trainable:,}",
            f"{counts.frozen:,}",
        )

    def _freeze_backbone(self) -> None:
        """Freeze all parameters except the classifier head."""
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

    def unfreeze_last_feature_stage(self) -> ParameterCounts:
        """Stage 2: unfreeze the final feature stage and classifier.

        Unfreezes ``features.7`` (last MBConv stage) and ``features.8`` (final
        1×1 feature head), matching the last-block fine-tuning used by
        ResNet/DenseNet experiments.
        """
        trainable_prefixes = (
            "features.7.",
            "features.8.",
            "classifier.",
        )

        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad = name.startswith(trainable_prefixes)

        self.freeze_backbone = False
        counts = count_parameters(self)
        logger.info(
            "Last feature stage unfrozen for fine-tuning: trainable=%s, frozen=%s",
            f"{counts.trainable:,}",
            f"{counts.frozen:,}",
        )
        return counts

    def unfreeze_denseblock4(self) -> ParameterCounts:
        """Compatibility alias used by DenseNet-style experiment orchestration."""
        return self.unfreeze_last_feature_stage()

    def unfreeze_layer4(self) -> ParameterCounts:
        """Compatibility alias used by ResNet-style experiment orchestration."""
        return self.unfreeze_last_feature_stage()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for an image batch.

        Args:
            x: Input tensor of shape ``(N, 3, H, W)``.

        Returns:
            Logits tensor of shape ``(N, num_classes)``.
        """
        return self.backbone(x)

    @property
    def classifier(self) -> nn.Sequential:
        """Final classification head (Dropout + Linear)."""
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
    model: EfficientNetB0TransferClassifier,
    *,
    batch_size: int = 2,
    image_size: int = 224,
    device: torch.device | None = None,
) -> dict[str, list[int] | str]:
    """Run a dummy forward pass to verify tensor shapes.

    Args:
        model: EfficientNet-B0 transfer classifier.
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
