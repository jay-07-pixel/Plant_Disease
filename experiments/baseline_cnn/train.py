"""Baseline CNN training experiment orchestration."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from experiments.baseline_cnn.callbacks import EpochTimingCallback
from experiments.baseline_cnn.config import (
    BEST_MODEL_PATH,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_RANDOM_SEED,
    LAST_MODEL_PATH,
    REPORT_JSON,
    REPORT_MD,
    BaselineExperimentConfig,
    SAVE_DIR,
    TENSORBOARD_DIR,
    build_engine_config,
)
from experiments.baseline_cnn.evaluate import evaluate_on_dataloader
from experiments.baseline_cnn.report import (
    build_epoch_records,
    build_training_report,
    save_training_report_json,
    save_training_report_markdown,
)
from src.models.baseline_cnn import BaselineCNN
from src.training.callbacks import build_default_callbacks
from src.training.checkpoint import CheckpointManager
from src.training.dataloader import build_dataloaders, get_dataloader_config
from src.training.logger import TrainingLogger
from src.training.trainer import Trainer, resolve_device, set_seed

logger = logging.getLogger(__name__)

ARTIFACT_PATTERNS = (
    "best_model.pth",
    "best_model.pt",
    "last_model.pth",
    "latest_checkpoint.pt",
    "training_history.json",
    "checkpoint_epoch_*.pt",
)


def cleanup_previous_artifacts() -> list[str]:
    """Remove artifacts from the previous CPU verification run.

    Returns:
        List of deleted file paths.
    """
    deleted: list[str] = []

    if SAVE_DIR.exists():
        for pattern in ARTIFACT_PATTERNS:
            for path in SAVE_DIR.glob(pattern):
                path.unlink(missing_ok=True)
                deleted.append(str(path))

    for report in (REPORT_MD, REPORT_JSON):
        if report.exists():
            report.unlink()
            deleted.append(str(report))

    if TENSORBOARD_DIR.exists():
        shutil.rmtree(TENSORBOARD_DIR)
        deleted.append(str(TENSORBOARD_DIR))

    return deleted


def verify_gpu_environment(
    loaders: Any,
    *,
    device: torch.device,
    num_epochs: int,
    use_amp: bool,
) -> dict[str, Any]:
    """Verify CUDA and dataset readiness before training.

    Args:
        loaders: Built project dataloaders.
        device: Resolved compute device.
        num_epochs: Planned epoch count.
        use_amp: Whether AMP is enabled.

    Returns:
        Verification summary dictionary.

    Raises:
        RuntimeError: If CUDA is required but unavailable.
    """
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        raise RuntimeError(
            "CUDA is not available. This experiment requires a GPU. "
            "Install a CUDA-enabled PyTorch build and verify drivers."
        )

    gpu_name = torch.cuda.get_device_name(0)
    gpu_props = torch.cuda.get_device_properties(0)
    gpu_memory_gb = gpu_props.total_memory / (1024**3)

    return {
        "cuda_available": cuda_available,
        "device": str(device),
        "device_label": "CUDA" if device.type == "cuda" else device.type.upper(),
        "gpu_name": gpu_name,
        "gpu_memory_gb": round(gpu_memory_gb, 2),
        "epochs": num_epochs,
        "batch_size": loaders.config.batch_size,
        "train_samples": len(loaders.train.dataset),
        "val_samples": len(loaders.val.dataset),
        "test_samples": len(loaders.test.dataset),
        "num_classes": loaders.label_encoder.num_classes,
        "amp_enabled": use_amp,
    }


def print_training_summary(summary: dict[str, Any]) -> None:
    """Print the pre-training configuration summary."""
    print()
    print("=" * 56)
    print("Baseline CNN — GPU Training Run")
    print("=" * 56)
    print(f"Device:              {summary['device_label']}")
    print(f"GPU:                 {summary['gpu_name']}")
    print(f"GPU Memory:          {summary['gpu_memory_gb']:.1f} GB")
    print(f"Epochs:              {summary['epochs']}")
    print(f"Batch Size:          {summary['batch_size']}")
    print(f"Train Samples:       {summary['train_samples']:,}")
    print(f"Validation Samples:  {summary['val_samples']:,}")
    print(f"Test Samples:        {summary['test_samples']:,}")
    print(f"Classes:             {summary['num_classes']}")
    print(f"AMP:                 {'Enabled' if summary['amp_enabled'] else 'Disabled'}")
    print("=" * 56)
    print()


def set_deterministic_seed(seed: int) -> None:
    """Configure deterministic behavior for reproducible training."""
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_experiment_callbacks(
    checkpoint_manager: CheckpointManager,
    timing_callback: EpochTimingCallback,
    *,
    early_stopping_patience: int,
) -> list:
    """Assemble framework and experiment callbacks."""
    callbacks = build_default_callbacks(
        checkpoint_manager,
        early_stopping_patience=early_stopping_patience,
        monitor="accuracy",
        mode="max",
    )
    callbacks.append(timing_callback)
    return callbacks


def save_extended_history(
    history_path: Path,
    history_dict: dict[str, Any],
    epoch_times: list[float],
    total_training_time: float,
) -> None:
    """Persist enriched training history with epoch timing."""
    payload = {
        **history_dict,
        "epoch_times_seconds": epoch_times,
        "total_training_time_seconds": total_training_time,
        "epochs": build_epoch_records_from_dict(history_dict, epoch_times),
    }
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Saved extended training history to %s", history_path)


def build_epoch_records_from_dict(
    history_dict: dict[str, Any],
    epoch_times: list[float],
) -> list[dict[str, Any]]:
    """Build epoch records from a serialized history dictionary."""
    from src.training.metrics import EpochMetrics, TrainingHistory

    history = TrainingHistory(
        train=[EpochMetrics(**item) for item in history_dict.get("train", [])],
        val=[EpochMetrics(**item) for item in history_dict.get("val", [])],
        learning_rates=history_dict.get("learning_rates", []),
        best_epoch=history_dict.get("best_epoch", -1),
        best_metric_name=history_dict.get("best_metric_name", "val_accuracy"),
        best_metric_value=history_dict.get("best_metric_value", 0.0),
    )
    from experiments.baseline_cnn.report import build_epoch_records

    return build_epoch_records(history, epoch_times)


def export_model_artifacts(
    checkpoint_manager: CheckpointManager,
    *,
    best_model_path: Path = BEST_MODEL_PATH,
    last_model_path: Path = LAST_MODEL_PATH,
) -> None:
    """Export best and last checkpoints to experiment ``.pth`` paths."""
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_manager.best_model_path.exists():
        shutil.copy2(checkpoint_manager.best_model_path, best_model_path)
        logger.info("Exported best model to %s", best_model_path)
    else:
        logger.warning("Best model checkpoint not found at %s", checkpoint_manager.best_model_path)

    if checkpoint_manager.latest_checkpoint_path.exists():
        shutil.copy2(checkpoint_manager.latest_checkpoint_path, last_model_path)
        logger.info("Exported last model to %s", last_model_path)
    else:
        logger.warning("Last checkpoint not found at %s", checkpoint_manager.latest_checkpoint_path)


def run_baseline_experiment(
    experiment: BaselineExperimentConfig | None = None,
) -> dict[str, Any]:
    """Train the baseline CNN and generate experiment artifacts.

    Args:
        experiment: Optional experiment configuration overrides.

    Returns:
        Complete training report dictionary.
    """
    experiment = experiment or BaselineExperimentConfig()
    if experiment.resume_checkpoint is not None:
        raise ValueError("Use run_baseline_experiment_resume() to resume training.")

    experiment.save_dir.mkdir(parents=True, exist_ok=True)
    experiment.tensorboard_dir.mkdir(parents=True, exist_ok=True)

    set_deterministic_seed(experiment.random_seed)

    loaders = build_dataloaders(get_dataloader_config())
    num_classes = loaders.label_encoder.num_classes
    device = resolve_device("auto")

    engine_config = build_engine_config(experiment, num_classes=num_classes, device="auto")
    verification = verify_gpu_environment(
        loaders,
        device=device,
        num_epochs=experiment.num_epochs,
        use_amp=engine_config.use_amp,
    )
    print_training_summary(verification)

    model = BaselineCNN(num_classes=num_classes)

    checkpoint_manager = CheckpointManager(
        experiment.save_dir,
        monitor="val_accuracy",
        mode="max",
        save_best_only=False,
    )
    timing_callback = EpochTimingCallback(epoch_times=[])
    callbacks = build_experiment_callbacks(
        checkpoint_manager,
        timing_callback,
        early_stopping_patience=experiment.early_stopping_patience,
    )

    training_logger = TrainingLogger(
        log_dir=experiment.tensorboard_dir,
        history_path=experiment.history_path,
    )

    trainer = Trainer(
        model=model,
        config=engine_config,
        train_loader=loaders.train.dataloader,
        val_loader=loaders.val.dataloader,
        class_weights=loaders.class_weights,
        callbacks=callbacks,
        checkpoint_manager=checkpoint_manager,
        training_logger=training_logger,
    )

    for callback in callbacks:
        if hasattr(callback, "bind_trainer"):
            callback.bind_trainer(trainer)

    logger.info(
        "Starting baseline CNN training: epochs=%d, device=%s, batch_size=%d",
        experiment.num_epochs,
        device,
        loaders.config.batch_size,
    )

    history = trainer.train()
    total_training_time = timing_callback.total_training_time

    save_extended_history(
        experiment.history_path,
        history.to_dict(),
        timing_callback.epoch_times,
        total_training_time,
    )
    export_model_artifacts(checkpoint_manager)

    best_checkpoint = (
        checkpoint_manager.best_model_path
        if checkpoint_manager.best_model_path.exists()
        else checkpoint_manager.latest_checkpoint_path
    )
    if best_checkpoint.exists():
        trainer.checkpoint_manager.load_checkpoint(
            best_checkpoint,
            model=trainer.model,
            map_location=device,
        )

    evaluation = evaluate_on_dataloader(
        trainer.model,
        loaders.val.dataloader,
        loaders.label_encoder,
        device,
    )

    experiment_config = asdict(experiment)
    for key, value in list(experiment_config.items()):
        if isinstance(value, Path):
            experiment_config[key] = str(value)

    report = build_training_report(
        history=history,
        epoch_times=timing_callback.epoch_times,
        total_training_time=total_training_time,
        evaluation=evaluation,
        engine_config=engine_config.to_dict(),
        experiment_config=experiment_config,
        device=str(device),
    )

    save_training_report_json(report, REPORT_JSON)
    save_training_report_markdown(report, REPORT_MD)

    logger.info("Baseline CNN experiment complete. Reports saved to %s and %s", REPORT_MD, REPORT_JSON)
    return report


def run_baseline_experiment_resume(
    experiment: BaselineExperimentConfig,
) -> dict[str, Any]:
    """Resume an interrupted baseline CNN training run from a checkpoint."""
    if experiment.resume_checkpoint is None:
        raise ValueError("resume_checkpoint is required to resume training.")

    experiment.save_dir.mkdir(parents=True, exist_ok=True)
    experiment.tensorboard_dir.mkdir(parents=True, exist_ok=True)

    set_deterministic_seed(experiment.random_seed)

    loaders = build_dataloaders(get_dataloader_config())
    num_classes = loaders.label_encoder.num_classes
    device = resolve_device("auto")

    engine_config = build_engine_config(experiment, num_classes=num_classes, device="auto")
    verification = verify_gpu_environment(
        loaders,
        device=device,
        num_epochs=experiment.num_epochs,
        use_amp=engine_config.use_amp,
    )
    print_training_summary(verification)
    print(f"Resuming from: {experiment.resume_checkpoint}")
    print()

    model = BaselineCNN(num_classes=num_classes)

    checkpoint_manager = CheckpointManager(
        experiment.save_dir,
        monitor="val_accuracy",
        mode="max",
        save_best_only=False,
    )
    timing_callback = EpochTimingCallback(epoch_times=[])
    callbacks = build_experiment_callbacks(
        checkpoint_manager,
        timing_callback,
        early_stopping_patience=experiment.early_stopping_patience,
    )

    training_logger = TrainingLogger(
        log_dir=experiment.tensorboard_dir,
        history_path=experiment.history_path,
    )

    trainer = Trainer(
        model=model,
        config=engine_config,
        train_loader=loaders.train.dataloader,
        val_loader=loaders.val.dataloader,
        class_weights=loaders.class_weights,
        callbacks=callbacks,
        checkpoint_manager=checkpoint_manager,
        training_logger=training_logger,
    )

    for callback in callbacks:
        if hasattr(callback, "bind_trainer"):
            callback.bind_trainer(trainer)

    logger.info(
        "Resuming baseline CNN training from epoch %d: epochs=%d, device=%s",
        trainer.start_epoch + 1,
        experiment.num_epochs,
        device,
    )

    history = trainer.train()
    total_training_time = timing_callback.total_training_time

    save_extended_history(
        experiment.history_path,
        history.to_dict(),
        timing_callback.epoch_times,
        total_training_time,
    )
    export_model_artifacts(checkpoint_manager)

    best_checkpoint = (
        checkpoint_manager.best_model_path
        if checkpoint_manager.best_model_path.exists()
        else checkpoint_manager.latest_checkpoint_path
    )
    if best_checkpoint.exists():
        trainer.checkpoint_manager.load_checkpoint(
            best_checkpoint,
            model=trainer.model,
            map_location=device,
        )

    evaluation = evaluate_on_dataloader(
        trainer.model,
        loaders.val.dataloader,
        loaders.label_encoder,
        device,
    )

    experiment_config = asdict(experiment)
    for key, value in list(experiment_config.items()):
        if isinstance(value, Path):
            experiment_config[key] = str(value)

    report = build_training_report(
        history=history,
        epoch_times=timing_callback.epoch_times,
        total_training_time=total_training_time,
        evaluation=evaluation,
        engine_config=engine_config.to_dict(),
        experiment_config=experiment_config,
        device=str(device),
    )

    save_training_report_json(report, REPORT_JSON)
    save_training_report_markdown(report, REPORT_MD)

    logger.info("Baseline CNN experiment complete. Reports saved to %s and %s", REPORT_MD, REPORT_JSON)
    return report


def main() -> None:
    """CLI entry point for the baseline CNN training experiment."""
    parser = argparse.ArgumentParser(description="Train the Baseline CNN experiment.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a checkpoint (.pt or .pth). Disabled for fresh GPU runs.",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip deletion of previous experiment artifacts.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.resume is not None:
        experiment = BaselineExperimentConfig(
            num_epochs=args.epochs,
            learning_rate=args.lr,
            random_seed=args.seed,
            resume_checkpoint=args.resume,
        )
        report = run_baseline_experiment_resume(experiment)
    else:
        if not args.skip_cleanup:
            deleted = cleanup_previous_artifacts()
            if deleted:
                logger.info("Removed %d previous artifact(s)", len(deleted))
            else:
                logger.info("No previous artifacts found to remove")

        experiment = BaselineExperimentConfig(
            num_epochs=args.epochs,
            learning_rate=args.lr,
            random_seed=args.seed,
            resume_checkpoint=None,
        )
        report = run_baseline_experiment(experiment)
    summary = report["summary"]

    print("Baseline CNN training complete.")
    print(f"  Best epoch:              {summary['best_epoch']}")
    print(f"  Best validation accuracy: {summary['best_validation_accuracy']:.4f}")
    print(f"  Final training accuracy:  {summary['final_training_accuracy']:.4f}")
    print(f"  Final validation accuracy:{summary['final_validation_accuracy']:.4f}")
    print(f"  Training time:            {summary['total_training_time_seconds']:.1f}s")
    print(f"  Best model:               {BEST_MODEL_PATH}")
    print(f"  Report:                   {REPORT_MD}")


if __name__ == "__main__":
    main()
