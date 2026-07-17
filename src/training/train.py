"""Training framework entry point and verification utilities.

Validates the reusable training engine without implementing model architectures
or running a full training loop. Pass a model to :func:`create_trainer` when
architectures are available.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.training.callbacks import build_default_callbacks
from src.training.checkpoint import CheckpointManager
from src.training.dataloader import build_dataloaders, get_dataloader_config
from src.training.logger import TrainingLogger, config_to_hparams
from src.training.losses import FocalLoss, LossType, build_loss_function
from src.training.metrics import MetricsTracker, build_metric_collection
from src.training.trainer import (
    OptimizerType,
    SchedulerType,
    TrainingEngineConfig,
    build_optimizer,
    build_scheduler,
    get_training_engine_config,
    resolve_device,
)

logger = logging.getLogger(__name__)

DEFAULT_FRAMEWORK_REPORT = Path("logs/training_framework_report.json")


def create_trainer(
    model: nn.Module,
    *,
    engine_config: TrainingEngineConfig | None = None,
    dataloader_config: Any | None = None,
) -> Any:
    """Create a :class:`Trainer` wired to project dataloaders.

    Args:
        model: Any PyTorch classification model returning logits ``(N, C)``.
        engine_config: Training engine configuration.
        dataloader_config: Optional data loader configuration override.

    Returns:
        Configured :class:`Trainer` instance (import deferred to avoid cycles).

    Raises:
        RuntimeError: If ``num_classes`` does not match the dataset encoder.
    """
    from src.training.trainer import Trainer

    engine_config = engine_config or get_training_engine_config()
    loaders = build_dataloaders(dataloader_config or get_dataloader_config())

    num_classes = loaders.label_encoder.num_classes
    if engine_config.num_classes != num_classes:
        engine_config.num_classes = num_classes

    return Trainer(
        model=model,
        config=engine_config,
        train_loader=loaders.train.dataloader,
        val_loader=loaders.val.dataloader,
        class_weights=loaders.class_weights,
    )


def verify_training_framework(
    *,
    report_path: Path | str = DEFAULT_FRAMEWORK_REPORT,
) -> dict[str, Any]:
    """Verify training framework components without training any model.

    Instantiates optimizers, schedulers, loss functions, metrics, callbacks,
    checkpoint manager, and TensorBoard logger. Writes a JSON verification
    report to ``logs/``.

    Args:
        report_path: Output path for the verification report.

    Returns:
        Verification report dictionary.
    """
    device = resolve_device("auto")
    loaders = build_dataloaders(get_dataloader_config(batch_size=4))
    num_classes = loaders.label_encoder.num_classes

    config = get_training_engine_config(
        num_classes=num_classes,
        num_epochs=50,
        device=str(device),
        use_amp=device.type == "cuda",
        experiment_name="framework_verification",
    )

    class _StubModel(nn.Module):
        """Minimal stub for framework verification only — not a real architecture."""

        def __init__(self, num_classes: int) -> None:
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(3, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.pool(x).flatten(1)
            return self.fc(x)

    stub_model = _StubModel(num_classes).to(device)
    optimizer = build_optimizer(stub_model, config)
    scheduler = build_scheduler(optimizer, config)
    criterion = build_loss_function(config, loaders.class_weights, device=device)
    metrics = build_metric_collection(num_classes, device=device)

    checkpoint_manager = CheckpointManager(
        config.checkpoint_dir,
        monitor=f"val_{config.monitor_metric}",
        mode=config.monitor_mode,
    )
    callbacks = build_default_callbacks(
        checkpoint_manager,
        early_stopping_patience=config.early_stopping_patience,
        monitor=config.monitor_metric,
        mode=config.monitor_mode,
    )
    training_logger = TrainingLogger(
        log_dir=config.log_dir,
        history_path=config.history_path,
        experiment_name="framework_verification",
    )

    batch = next(iter(loaders.val.dataloader))
    images = batch.images.to(device)
    targets = batch.class_indices.to(device)
    logits = stub_model(images)
    loss = criterion(logits, targets)

    metrics_tracker = MetricsTracker(num_classes, device=device)
    metrics_tracker.update(loss, logits, targets)
    epoch_metrics = metrics_tracker.compute()

    focal_available = issubclass(FocalLoss, nn.Module)

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "verified",
        "message": "Training framework verified. No model architecture trained.",
        "device": str(device),
        "num_classes": num_classes,
        "dataset_samples": {
            "train": len(loaders.train.dataset),
            "val": len(loaders.val.dataset),
            "test": len(loaders.test.dataset),
        },
        "engine_config": config.to_dict(),
        "supported_optimizers": [item.value for item in OptimizerType],
        "supported_schedulers": [item.value for item in SchedulerType],
        "supported_losses": [item.value for item in LossType],
        "focal_loss_available": focal_available,
        "sample_batch_shape": list(images.shape),
        "sample_metrics": epoch_metrics.to_dict(),
        "callbacks": [callback.__class__.__name__ for callback in callbacks],
        "checkpoint_dir": str(config.checkpoint_dir),
        "log_dir": str(training_logger.log_dir),
        "history_path": str(config.history_path),
        "tensorboard_enabled": True,
        "hparams_sample": config_to_hparams(config),
        "trainer_features": [
            "gpu_cpu_selection",
            "mixed_precision_amp",
            "gradient_clipping",
            "gradient_accumulation",
            "early_stopping",
            "learning_rate_scheduler",
            "model_checkpoint_saving",
            "resume_training",
            "best_model_saving",
            "tensorboard_logging",
            "tqdm_progress_bars",
            "torchmetrics_evaluation",
        ],
    }

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    training_logger.close()
    del stub_model, optimizer, scheduler, criterion

    logger.info("Training framework verification complete: %s", report_path)
    return report


def main() -> None:
    """CLI entry point — verifies the training framework without training."""
    parser = argparse.ArgumentParser(
        description="PlantDiseaseAI training framework verification (no model training).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_FRAMEWORK_REPORT,
        help="Path for the JSON verification report.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    report = verify_training_framework(report_path=args.report)

    print("Training framework verified successfully.")
    print(f"  Device:          {report['device']}")
    print(f"  Classes:         {report['num_classes']}")
    print(f"  Train samples:   {report['dataset_samples']['train']:,}")
    print(f"  Val samples:     {report['dataset_samples']['val']:,}")
    print(f"  Report:          {args.report}")
    print()
    print("To train a model once architectures are implemented:")
    print("  from src.training.train import create_trainer")
    print("  trainer = create_trainer(model)")
    print("  history = trainer.train()")


if __name__ == "__main__":
    main()
