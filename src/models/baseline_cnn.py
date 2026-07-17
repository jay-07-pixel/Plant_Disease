"""Lightweight baseline CNN for multi-class plant disease classification."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

DEFAULT_NUM_CLASSES = 43
DEFAULT_INPUT_SIZE = 224
DEFAULT_DROPOUT = 0.5
DEFAULT_REPORT_JSON = Path("reports/model_baseline.json")
DEFAULT_REPORT_MD = Path("reports/model_baseline.md")

# Channel widths for each convolutional block.
DEFAULT_CHANNELS: tuple[int, int, int, int] = (32, 64, 128, 256)


@dataclass
class BaselineCNNConfig:
    """Configuration for :class:`BaselineCNN`.

    Attributes:
        num_classes: Number of output classes.
        in_channels: Number of input image channels (RGB = 3).
        input_size: Expected spatial input size (height and width).
        channels: Output channels for conv blocks 1–4.
        dropout: Dropout probability before the classifier head.
        kernel_size: Convolution kernel size for all conv blocks.
    """

    num_classes: int = DEFAULT_NUM_CLASSES
    in_channels: int = 3
    input_size: int = DEFAULT_INPUT_SIZE
    channels: tuple[int, int, int, int] = DEFAULT_CHANNELS
    dropout: float = DEFAULT_DROPOUT
    kernel_size: int = 3

    def __post_init__(self) -> None:
        if self.num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        if len(self.channels) != 4:
            raise ValueError("channels must contain exactly four block widths")


@dataclass
class LayerSummary:
    """Summary of one layer or block in the model.

    Attributes:
        name: Layer or block name.
        layer_type: Module class name.
        output_shape: Output tensor shape as a string (when known).
        parameters: Number of parameters in this module.
        trainable_parameters: Number of trainable parameters.
    """

    name: str
    layer_type: str
    output_shape: str | None
    parameters: int
    trainable_parameters: int


@dataclass
class ModelSummary:
    """Full architecture summary for reporting.

    Attributes:
        model_name: Model identifier.
        input_shape: Expected input tensor shape ``(N, C, H, W)``.
        output_shape: Output logits shape ``(N, num_classes)``.
        num_classes: Number of output classes.
        total_parameters: Total parameter count.
        trainable_parameters: Trainable parameter count.
        non_trainable_parameters: Frozen parameter count.
        layers: Per-layer summaries.
        trainer_compatible: Whether the model works with the training engine.
        trainer_verification_message: Result of trainer compatibility check.
    """

    model_name: str
    input_shape: list[int]
    output_shape: list[int]
    num_classes: int
    total_parameters: int
    trainable_parameters: int
    non_trainable_parameters: int
    layers: list[LayerSummary] = field(default_factory=list)
    trainer_compatible: bool = False
    trainer_verification_message: str = ""


class ConvBlock(nn.Module):
    """Convolutional block: Conv2D → BatchNorm → ReLU → MaxPool.

    Args:
        in_channels: Input feature channels.
        out_channels: Output feature channels.
        kernel_size: Convolution kernel size.
        pool: Whether to apply ``2×2`` max pooling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        pool: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolutional block."""
        return self.block(x)


class BaselineCNN(nn.Module):
    """Lightweight CNN baseline for plant disease classification.

    Architecture::

        Input (3×224×224)
          → Conv Block 1 (Conv → BN → ReLU → MaxPool)
          → Conv Block 2
          → Conv Block 3
          → Conv Block 4
          → Global Average Pooling
          → Dropout
          → Fully Connected
          → num_classes logits

    The forward pass returns raw logits for use with ``CrossEntropyLoss``.
    Apply ``softmax`` at inference time for class probabilities.

    Args:
        config: Model configuration. When ``None``, defaults are used.
        num_classes: Shortcut for ``config.num_classes`` when ``config`` is ``None``.
    """

    def __init__(
        self,
        config: BaselineCNNConfig | None = None,
        *,
        num_classes: int = DEFAULT_NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.config = config or BaselineCNNConfig(num_classes=num_classes)
        channels = self.config.channels

        self.features = nn.Sequential(
            ConvBlock(self.config.in_channels, channels[0], self.config.kernel_size),
            ConvBlock(channels[0], channels[1], self.config.kernel_size),
            ConvBlock(channels[1], channels[2], self.config.kernel_size),
            ConvBlock(channels[2], channels[3], self.config.kernel_size),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=self.config.dropout)
        self.classifier = nn.Linear(channels[3], self.config.num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Kaiming initialization to convolutional and linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning class logits.

        Args:
            x: Input image batch of shape ``(N, 3, H, W)``.

        Returns:
            Logits tensor of shape ``(N, num_classes)``.
        """
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities via softmax (inference use).

        Args:
            x: Input image batch.

        Returns:
            Probability tensor of shape ``(N, num_classes)``.
        """
        return torch.softmax(self.forward(x), dim=1)


def count_parameters(model: nn.Module) -> tuple[int, int, int]:
    """Count total, trainable, and non-trainable parameters.

    Args:
        model: PyTorch model.

    Returns:
        Tuple of ``(total, trainable, non_trainable)``.
    """
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable, total - trainable


def build_model_summary(
    model: BaselineCNN,
    *,
    batch_size: int = 1,
    device: torch.device | None = None,
) -> ModelSummary:
    """Build a layer-wise summary using a dummy forward pass.

    Args:
        model: Baseline CNN instance.
        batch_size: Batch size for shape propagation.
        device: Device for the dummy input tensor.

    Returns:
        :class:`ModelSummary` with layer output shapes and parameter counts.
    """
    device = device or torch.device("cpu")
    model = model.to(device)
    model.eval()

    config = model.config
    dummy_input = torch.zeros(
        batch_size,
        config.in_channels,
        config.input_size,
        config.input_size,
        device=device,
    )

    layer_summaries: list[LayerSummary] = []
    activation_shapes: dict[str, tuple[int, ...]] = {}

    def hook_fn(name: str):
        def _hook(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
            activation_shapes[name] = tuple(output.shape)

        return _hook

    handles: list[Any] = []
    for name, module in model.named_modules():
        if name == "":
            continue
        handles.append(module.register_forward_hook(hook_fn(name)))

    with torch.no_grad():
        output = model(dummy_input)

    for handle in handles:
        handle.remove()

    for name, module in model.named_modules():
        if name == "":
            continue
        module_total = sum(parameter.numel() for parameter in module.parameters())
        module_trainable = sum(
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad
        )
        shape = activation_shapes.get(name)
        layer_summaries.append(
            LayerSummary(
                name=name,
                layer_type=module.__class__.__name__,
                output_shape=str(list(shape)) if shape is not None else None,
                parameters=module_total,
                trainable_parameters=module_trainable,
            )
        )

    total, trainable, non_trainable = count_parameters(model)

    return ModelSummary(
        model_name="BaselineCNN",
        input_shape=[batch_size, config.in_channels, config.input_size, config.input_size],
        output_shape=list(output.shape),
        num_classes=config.num_classes,
        total_parameters=total,
        trainable_parameters=trainable,
        non_trainable_parameters=non_trainable,
        layers=layer_summaries,
    )


def print_model_summary(
    model: BaselineCNN,
    *,
    batch_size: int = 1,
    device: torch.device | None = None,
) -> ModelSummary:
    """Print and return a human-readable model summary.

    Args:
        model: Baseline CNN instance.
        batch_size: Batch size for shape propagation.
        device: Device for the dummy forward pass.

    Returns:
        :class:`ModelSummary` instance.
    """
    summary = build_model_summary(model, batch_size=batch_size, device=device)

    separator = "=" * 72
    print(separator)
    print(f"Model: {summary.model_name}")
    print(f"Input shape:  {summary.input_shape}")
    print(f"Output shape: {summary.output_shape}")
    print(f"Classes:      {summary.num_classes}")
    print(f"Parameters:   {summary.trainable_parameters:,} trainable / {summary.total_parameters:,} total")
    print(separator)
    print(f"{'Layer':<40} {'Type':<18} {'Output Shape':<20} {'Params':>10}")
    print("-" * 72)

    for layer in summary.layers:
        shape_display = layer.output_shape or "—"
        print(
            f"{layer.name:<40} {layer.layer_type:<18} {shape_display:<20} {layer.parameters:>10,}"
        )

    print(separator)
    return summary


def verify_trainer_compatibility(
    model: BaselineCNN,
    *,
    batch_size: int = 4,
) -> tuple[bool, str]:
    """Verify the model integrates with the existing training engine.

    Instantiates a :class:`Trainer` without running ``train()``.

    Args:
        model: Baseline CNN instance.
        batch_size: Batch size for a smoke-test forward pass.

    Returns:
        Tuple of ``(compatible, message)``.
    """
    from src.training.dataloader import build_dataloaders, get_dataloader_config
    from src.training.train import create_trainer
    from src.training.trainer import get_training_engine_config, resolve_device

    try:
        device = resolve_device("auto")
        loaders = build_dataloaders(get_dataloader_config(batch_size=batch_size))
        expected_classes = loaders.label_encoder.num_classes

        if model.config.num_classes != expected_classes:
            return (
                False,
                f"num_classes mismatch: model={model.config.num_classes}, "
                f"dataset={expected_classes}",
            )

        model = model.to(device)
        batch = next(iter(loaders.train.dataloader))
        images = batch.images.to(device)
        targets = batch.class_indices.to(device)

        model.eval()
        with torch.no_grad():
            logits = model(images)

        if logits.shape != (images.shape[0], expected_classes):
            return (
                False,
                f"Output shape mismatch: expected {(images.shape[0], expected_classes)}, "
                f"got {tuple(logits.shape)}",
            )

        trainer = create_trainer(
            model,
            engine_config=get_training_engine_config(use_amp=False, device=str(device)),
        )
        if trainer.model is not model:
            return False, "Trainer model reference mismatch."

        loss_fn = trainer.criterion
        loss = loss_fn(logits, targets)
        if not torch.isfinite(loss):
            return False, "Loss is not finite during compatibility check."

        return True, (
            f"Compatible with Trainer: input={list(images.shape)}, "
            f"output={list(logits.shape)}, loss={loss.item():.4f}"
        )

    except Exception as exc:
        return False, f"Trainer compatibility check failed: {exc}"


def save_model_report_json(
    summary: ModelSummary,
    output_path: Path | str = DEFAULT_REPORT_JSON,
) -> None:
    """Save the model summary report as JSON.

    Args:
        summary: Model summary.
        output_path: Destination file path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = asdict(summary)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["architecture"] = [
        "Input (3×224×224)",
        "Conv Block 1: Conv2D → BatchNorm → ReLU → MaxPool",
        "Conv Block 2: Conv2D → BatchNorm → ReLU → MaxPool",
        "Conv Block 3: Conv2D → BatchNorm → ReLU → MaxPool",
        "Conv Block 4: Conv2D → BatchNorm → ReLU → MaxPool",
        "Global Average Pooling",
        "Dropout",
        "Fully Connected",
        f"{summary.num_classes}-class logits (softmax at inference)",
    ]

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    logger.info("Saved model baseline report JSON to %s", path)


def save_model_report_markdown(
    summary: ModelSummary,
    output_path: Path | str = DEFAULT_REPORT_MD,
) -> None:
    """Save the model summary report as Markdown.

    Args:
        summary: Model summary.
        output_path: Destination file path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Baseline CNN Model Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}  ",
        f"**Model:** `{summary.model_name}`  ",
        "",
        "> Architecture verification only — no training performed.",
        "",
        "## Summary",
        "",
        f"- **Input shape:** `{summary.input_shape}`",
        f"- **Output shape:** `{summary.output_shape}`",
        f"- **Number of classes:** {summary.num_classes}",
        f"- **Total parameters:** {summary.total_parameters:,}",
        f"- **Trainable parameters:** {summary.trainable_parameters:,}",
        f"- **Non-trainable parameters:** {summary.non_trainable_parameters:,}",
        f"- **Trainer compatible:** {summary.trainer_compatible}",
        f"- **Verification:** {summary.trainer_verification_message}",
        "",
        "## Architecture",
        "",
        "```",
        "Input (3×224×224)",
        "  ↓",
        "Conv Block 1: Conv2D → BatchNorm → ReLU → MaxPool",
        "  ↓",
        "Conv Block 2: Conv2D → BatchNorm → ReLU → MaxPool",
        "  ↓",
        "Conv Block 3: Conv2D → BatchNorm → ReLU → MaxPool",
        "  ↓",
        "Conv Block 4: Conv2D → BatchNorm → ReLU → MaxPool",
        "  ↓",
        "Global Average Pooling",
        "  ↓",
        "Dropout",
        "  ↓",
        "Fully Connected",
        "  ↓",
        f"{summary.num_classes}-class logits (softmax at inference)",
        "```",
        "",
        "## Layers",
        "",
        "| Layer | Type | Output Shape | Parameters |",
        "|-------|------|--------------|----------:|",
    ]

    for layer in summary.layers:
        shape_display = f"`{layer.output_shape}`" if layer.output_shape else "—"
        lines.append(
            f"| `{layer.name}` | {layer.layer_type} | {shape_display} | {layer.parameters:,} |"
        )

    lines.extend(
        [
            "",
            "## Initialization",
            "",
            "- **Conv2d / Linear:** Kaiming normal (fan_out, ReLU)",
            "- **BatchNorm2d:** Weight = 1, Bias = 0",
            "",
            "## Integration",
            "",
            "```python",
            "from src.models.baseline_cnn import BaselineCNN",
            "from src.training.train import create_trainer",
            "",
            "model = BaselineCNN(num_classes=43)",
            "trainer = create_trainer(model)",
            "history = trainer.train()  # when ready to train",
            "```",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved model baseline report markdown to %s", path)


def run_baseline_verification(
    *,
    num_classes: int = DEFAULT_NUM_CLASSES,
    report_json_path: Path | str = DEFAULT_REPORT_JSON,
    report_md_path: Path | str = DEFAULT_REPORT_MD,
) -> ModelSummary:
    """Build, summarize, and verify the baseline CNN without training.

    Args:
        num_classes: Number of output classes.
        report_json_path: JSON report output path.
        report_md_path: Markdown report output path.

    Returns:
        Complete :class:`ModelSummary` with trainer compatibility results.
    """
    model = BaselineCNN(num_classes=num_classes)
    summary = print_model_summary(model)

    compatible, message = verify_trainer_compatibility(model)
    summary.trainer_compatible = compatible
    summary.trainer_verification_message = message

    logger.info("Trainer compatibility: %s — %s", compatible, message)

    save_model_report_json(summary, report_json_path)
    save_model_report_markdown(summary, report_md_path)

    if not compatible:
        raise RuntimeError(message)

    return summary


def main() -> None:
    """CLI entry point for baseline CNN verification."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    summary = run_baseline_verification()
    print()
    print(f"Trainer compatible: {summary.trainer_compatible}")
    print(f"Reports: {DEFAULT_REPORT_MD}, {DEFAULT_REPORT_JSON}")


if __name__ == "__main__":
    main()
