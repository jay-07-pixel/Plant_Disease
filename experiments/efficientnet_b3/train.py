"""EfficientNet-B3 transfer-learning experiment orchestration."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from experiments.efficientnet_b3.callbacks import EpochTimingCallback
from experiments.efficientnet_b3.config import (
    BEST_MODEL_PATH,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_STAGE1_EPOCHS,
    DEFAULT_STAGE2_EPOCHS,
    DEFAULT_STAGE2_LEARNING_RATE,
    LAST_MODEL_PATH,
    REPORT_JSON,
    REPORT_MD,
    STAGE1_BEST_MODEL_PATH,
    EfficientNetB3ExperimentConfig,
    SAVE_DIR,
    TENSORBOARD_DIR,
    build_engine_config,
    build_stage_experiment,
)
from experiments.efficientnet_b3.evaluate import evaluate_on_dataloader
from experiments.efficientnet_b3.report import (
    build_epoch_records,
    build_training_report,
    save_training_report_json,
    save_training_report_markdown,
)
from src.models.efficientnet_b3_transfer import (
    EfficientNetB3TransferClassifier,
    count_parameters,
    verify_forward_pass,
)
from src.training.callbacks import build_default_callbacks
from src.training.checkpoint import CheckpointManager
from src.training.dataloader import build_dataloaders, get_dataloader_config
from src.training.logger import TrainingLogger
from src.training.train import create_trainer
from src.training.trainer import Trainer, resolve_device, set_seed
from src.training.metrics import EpochMetrics, TrainingHistory

logger = logging.getLogger(__name__)

ARTIFACT_PATTERNS = (
    "best_model.pth",
    "best_model.pt",
    "last_model.pth",
    "latest_checkpoint.pt",
    "stage1_best_model.pth",
    "training_history.json",
    "checkpoint_epoch_*.pt",
)


def cleanup_previous_artifacts() -> list[str]:
    """Remove artifacts from a previous EfficientNet-B3 training run."""
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


def set_deterministic_seed(seed: int) -> None:
    """Configure deterministic behavior for reproducible training."""
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_environment_summary(
    loaders: Any,
    model: EfficientNetB3TransferClassifier,
    *,
    device: torch.device,
    num_epochs: int,
    use_amp: bool,
) -> dict[str, Any]:
    """Collect device, dataset, and model statistics for display."""
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    gpu_memory_gb = None
    if cuda_available:
        gpu_memory_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)

    param_counts = count_parameters(model)

    return {
        "cuda_available": cuda_available,
        "device": str(device),
        "device_label": "CUDA" if device.type == "cuda" else device.type.upper(),
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_memory_gb,
        "epochs": num_epochs,
        "batch_size": loaders.config.batch_size,
        "train_samples": len(loaders.train.dataset),
        "val_samples": len(loaders.val.dataset),
        "test_samples": len(loaders.test.dataset),
        "num_classes": loaders.label_encoder.num_classes,
        "amp_enabled": use_amp and device.type == "cuda",
        "trainable_parameters": param_counts.trainable,
        "frozen_parameters": param_counts.frozen,
        "total_parameters": param_counts.total,
    }


def print_pre_training_summary(summary: dict[str, Any], *, title: str) -> None:
    """Print the pre-training configuration summary."""
    print()
    print("=" * 56)
    print(title)
    print("=" * 56)
    print(f"Device:                {summary['device_label']} ({summary['device']})")
    print(f"CUDA Available:        {summary['cuda_available']}")
    print(f"GPU:                   {summary['gpu_name']}")
    if summary.get("gpu_memory_gb") is not None:
        print(f"GPU Memory:            {summary['gpu_memory_gb']:.1f} GB")
    print(f"AMP:                   {'Enabled' if summary['amp_enabled'] else 'Disabled'}")
    print(f"Trainable Parameters:  {summary['trainable_parameters']:,}")
    print(f"Frozen Parameters:     {summary['frozen_parameters']:,}")
    print(f"Total Parameters:      {summary['total_parameters']:,}")
    print(f"Classes:               {summary['num_classes']}")
    print(f"Train Samples:         {summary['train_samples']:,}")
    print(f"Validation Samples:    {summary['val_samples']:,}")
    print(f"Test Samples:          {summary['test_samples']:,}")
    if "stage1_epochs" in summary:
        print(f"Stage 1 Epochs:        {summary['stage1_epochs']} (LR {summary['stage1_learning_rate']})")
        print(f"Stage 2 Epochs:        {summary['stage2_epochs']} (LR {summary['stage2_learning_rate']})")
        print(f"Total Epochs:          {summary['epochs']}")
    else:
        print(f"Epochs:                {summary['epochs']}")
    print(f"Batch Size:            {summary['batch_size']}")
    print("=" * 56)
    print()


def log_stage_boundary(
    *,
    event: str,
    stage: int,
    details: dict[str, Any] | None = None,
) -> None:
    """Log a clear boundary between training stages."""
    banner = "=" * 60
    logger.info(banner)
    logger.info(event)
    logger.info("Stage: %d", stage)
    if details:
        for key, value in details.items():
            logger.info("  %s: %s", key, value)
    logger.info(banner)

    print()
    print(banner)
    print(event)
    if details:
        for key, value in details.items():
            print(f"  {key}: {value}")
    print(banner)
    print()


def print_resume_banner(
    *,
    stage: int,
    start_epoch_1based: int,
    total_epochs: int,
    checkpoint_path: Path,
) -> None:
    """Print the resume banner (never labels a resume as a fresh Stage 1 start)."""
    print()
    print("=" * 60)
    print(f"Resuming Stage {stage}")
    print(f"Starting from epoch {start_epoch_1based}/{total_epochs}")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 60)
    print()
    logger.info(
        "Resuming Stage %d from epoch %d/%d (checkpoint=%s)",
        stage,
        start_epoch_1based,
        total_epochs,
        checkpoint_path,
    )


def read_checkpoint_state(checkpoint_path: Path) -> dict[str, Any]:
    """Read checkpoint metadata without restoring model weights."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["state"]


def resolve_resume_start_epoch(checkpoint_path: Path) -> int:
    """Return the next epoch index (0-based) for a checkpoint.

    Checkpoint ``state.epoch`` is the last *completed* epoch (0-based).
    Training must continue at ``epoch + 1``.
    """
    state = read_checkpoint_state(checkpoint_path)
    return int(state["epoch"]) + 1


def history_from_checkpoint_state(state: dict[str, Any]) -> TrainingHistory:
    """Rebuild :class:`TrainingHistory` from a checkpoint state dictionary."""
    history_data = state.get("history", {})
    return TrainingHistory(
        train=[EpochMetrics(**item) for item in history_data.get("train", [])],
        val=[EpochMetrics(**item) for item in history_data.get("val", [])],
        learning_rates=list(history_data.get("learning_rates", [])),
        best_epoch=int(state.get("best_epoch", history_data.get("best_epoch", -1))),
        best_metric_name=str(
            state.get("best_metric_name", history_data.get("best_metric_name", "val_accuracy"))
        ),
        best_metric_value=float(
            state.get("best_metric_value", history_data.get("best_metric_value", 0.0))
        ),
    )


def detect_resume_stage(start_epoch: int, stage1_epochs: int) -> int:
    """Return ``1`` or ``2`` based on the next epoch to train."""
    return 1 if start_epoch < stage1_epochs else 2


def verify_trainer_compatibility(
    model: EfficientNetB3TransferClassifier,
    experiment: EfficientNetB3ExperimentConfig,
    *,
    batch_size: int = 4,
) -> tuple[bool, str]:
    """Verify the EfficientNet-B3 model integrates with the existing Trainer."""
    try:
        device = resolve_device("auto")
        loaders = build_dataloaders(get_dataloader_config(batch_size=batch_size))
        expected_classes = loaders.label_encoder.num_classes

        if model.num_classes != expected_classes:
            return (
                False,
                f"num_classes mismatch: model={model.num_classes}, dataset={expected_classes}",
            )

        engine_config = build_engine_config(
            experiment,
            num_classes=expected_classes,
            device=str(device),
        )
        model_on_device = model.to(device)
        batch = next(iter(loaders.train.dataloader))
        images = batch.images.to(device)
        targets = batch.class_indices.to(device)

        model_on_device.eval()
        with torch.no_grad():
            logits = model_on_device(images)

        if logits.shape != (images.shape[0], expected_classes):
            return (
                False,
                f"Output shape mismatch: expected {(images.shape[0], expected_classes)}, "
                f"got {tuple(logits.shape)}",
            )

        trainer = create_trainer(model_on_device, engine_config=engine_config)
        if trainer.model is not model_on_device:
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


def verify_experiment_readiness(
    experiment: EfficientNetB3ExperimentConfig | None = None,
) -> dict[str, Any]:
    """Verify EfficientNet-B3 experiment readiness without starting training."""
    experiment = experiment or EfficientNetB3ExperimentConfig()
    set_deterministic_seed(experiment.random_seed)

    loaders = build_dataloaders(get_dataloader_config())
    num_classes = loaders.label_encoder.num_classes
    device = resolve_device("auto")

    engine_config = build_engine_config(experiment, num_classes=num_classes, device="auto")
    model = EfficientNetB3TransferClassifier(num_classes=num_classes, freeze_backbone=True)

    summary = build_environment_summary(
        loaders,
        model,
        device=device,
        num_epochs=experiment.num_epochs,
        use_amp=engine_config.use_amp,
    )
    summary.update(
        {
            "stage1_epochs": experiment.stage1_epochs,
            "stage2_epochs": experiment.stage2_epochs,
            "stage1_learning_rate": experiment.learning_rate,
            "stage2_learning_rate": experiment.stage2_learning_rate,
        }
    )
    print_pre_training_summary(summary, title="EfficientNet-B3 Transfer Learning — Readiness Check")

    forward_result = verify_forward_pass(model, device=device)
    print("Dummy forward pass (Stage 1 — frozen backbone):")
    print(f"  Status:       {forward_result['status']}")
    print(f"  Input shape:  {forward_result['input_shape']}")
    print(f"  Output shape: {forward_result['output_shape']}")
    print(f"  Dtype:        {forward_result['dtype']}")
    print()

    model.unfreeze_last_feature_stage()
    stage2_counts = count_parameters(model)
    print("Stage 2 parameter check (last feature stage + classifier unfrozen):")
    print(f"  Trainable Parameters:  {stage2_counts.trainable:,}")
    print(f"  Frozen Parameters:     {stage2_counts.frozen:,}")
    print()
    model.freeze_feature_extractor()

    compatible, message = verify_trainer_compatibility(model, experiment)
    print(f"Trainer compatibility: {'PASS' if compatible else 'FAIL'}")
    print(f"  {message}")
    print()

    if not compatible:
        raise RuntimeError(message)

    print("Experiment is ready. Start training with:")
    print("  python -m experiments.efficientnet_b3.train --train")
    print()

    return {
        "ready": True,
        "environment": summary,
        "forward_pass": forward_result,
        "trainer_compatible": compatible,
        "trainer_message": message,
        "output_paths": {
            "save_dir": str(SAVE_DIR),
            "tensorboard_dir": str(TENSORBOARD_DIR),
            "best_model": str(BEST_MODEL_PATH),
            "last_model": str(LAST_MODEL_PATH),
            "report_md": str(REPORT_MD),
            "report_json": str(REPORT_JSON),
        },
    }


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
    *,
    stage_config: dict[str, Any] | None = None,
) -> None:
    """Persist enriched training history with epoch timing and stage metadata."""
    payload = {
        **history_dict,
        "epoch_times_seconds": epoch_times,
        "total_training_time_seconds": total_training_time,
        "epochs": build_epoch_records_from_dict(history_dict, epoch_times),
        "stages": stage_config or {
            "stage1_epochs": DEFAULT_STAGE1_EPOCHS,
            "stage2_epochs": DEFAULT_STAGE2_EPOCHS,
            "stage1_learning_rate": DEFAULT_LEARNING_RATE,
            "stage2_learning_rate": DEFAULT_STAGE2_LEARNING_RATE,
            "stage_boundary_epoch": DEFAULT_STAGE1_EPOCHS,
        },
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


def _build_trainer(
    experiment: EfficientNetB3ExperimentConfig,
    model: EfficientNetB3TransferClassifier,
    loaders: Any,
    device: torch.device,
    *,
    num_epochs: int | None = None,
    learning_rate: float | None = None,
    resume_checkpoint: Path | None = None,
    checkpoint_manager: CheckpointManager | None = None,
    timing_callback: EpochTimingCallback | None = None,
    callbacks: list | None = None,
) -> tuple[Trainer, CheckpointManager, EpochTimingCallback, Any]:
    """Construct Trainer and supporting objects for a training stage."""
    num_classes = loaders.label_encoder.num_classes
    engine_config = build_engine_config(
        experiment,
        num_classes=num_classes,
        device="auto",
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        resume_checkpoint=resume_checkpoint,
    )

    checkpoint_manager = checkpoint_manager or CheckpointManager(
        experiment.save_dir,
        monitor="val_accuracy",
        mode="max",
        save_best_only=False,
    )
    timing_callback = timing_callback or EpochTimingCallback(epoch_times=[])
    callbacks = callbacks or build_experiment_callbacks(
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

    return trainer, checkpoint_manager, timing_callback, engine_config


def _prepare_stage2_trainer(
    *,
    experiment: EfficientNetB3ExperimentConfig,
    model: EfficientNetB3TransferClassifier,
    loaders: Any,
    device: torch.device,
    stage1_history: TrainingHistory,
    stage1_global_step: int,
    checkpoint_manager: CheckpointManager,
    timing_callback: EpochTimingCallback,
    callbacks: list,
    stage1_best_path: Path | None,
    resume_checkpoint: Path | None = None,
) -> tuple[Trainer, Any]:
    """Create a Stage 2 trainer continuing from Stage 1 history.

    When ``resume_checkpoint`` is mid-Stage-2 (next epoch > stage1_epochs), a
    full Trainer.resume() restores model, optimizer, scheduler, scaler, epoch,
    and history. When entering Stage 2 for the first time, only model weights
    are loaded from the Stage 1 best checkpoint and a fresh Stage 2 optimizer
    is created at the Stage 2 learning rate.
    """
    model.unfreeze_last_feature_stage()
    stage2_experiment = build_stage_experiment(
        experiment,
        num_epochs=experiment.num_epochs,
        learning_rate=experiment.stage2_learning_rate,
        resume_checkpoint=None,
    )

    trainer, _, _, engine_config = _build_trainer(
        stage2_experiment,
        model,
        loaders,
        device,
        num_epochs=experiment.num_epochs,
        learning_rate=experiment.stage2_learning_rate,
        resume_checkpoint=None,
        checkpoint_manager=checkpoint_manager,
        timing_callback=timing_callback,
        callbacks=callbacks,
    )

    mid_stage2_resume = (
        resume_checkpoint is not None
        and resolve_resume_start_epoch(resume_checkpoint) > experiment.stage1_epochs
    )

    if mid_stage2_resume:
        trainer.resume(resume_checkpoint)
    else:
        trainer.history = TrainingHistory(
            train=list(stage1_history.train),
            val=list(stage1_history.val),
            learning_rates=list(stage1_history.learning_rates),
            best_epoch=stage1_history.best_epoch,
            best_metric_name=stage1_history.best_metric_name,
            best_metric_value=stage1_history.best_metric_value,
        )
        trainer.start_epoch = experiment.stage1_epochs
        trainer.global_step = stage1_global_step

        weight_source = stage1_best_path
        if weight_source is None or not Path(weight_source).exists():
            if checkpoint_manager.best_model_path.exists():
                weight_source = checkpoint_manager.best_model_path
            elif resume_checkpoint is not None and Path(resume_checkpoint).exists():
                weight_source = resume_checkpoint
            else:
                raise FileNotFoundError(
                    "Stage 2 requires Stage 1 weights but no Stage 1 best/resume "
                    "checkpoint was found."
                )

        checkpoint_manager.load_checkpoint(
            weight_source,
            model=trainer.model,
            map_location=device,
        )

    return trainer, engine_config


def _export_stage1_best(checkpoint_manager: CheckpointManager) -> Path:
    """Copy the best Stage 1 checkpoint to a dedicated artifact path."""
    STAGE1_BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint_manager.best_model_path.exists():
        raise FileNotFoundError(
            f"Stage 1 best checkpoint not found at {checkpoint_manager.best_model_path}"
        )
    shutil.copy2(checkpoint_manager.best_model_path, STAGE1_BEST_MODEL_PATH)
    logger.info("Exported Stage 1 best model to %s", STAGE1_BEST_MODEL_PATH)
    return STAGE1_BEST_MODEL_PATH


def _run_stage1(
    *,
    experiment: EfficientNetB3ExperimentConfig,
    model: EfficientNetB3TransferClassifier,
    loaders: Any,
    device: torch.device,
    checkpoint_manager: CheckpointManager,
    timing_callback: EpochTimingCallback,
    callbacks: list,
    resume_checkpoint: Path | None = None,
) -> tuple[Trainer, Any, TrainingHistory]:
    """Run Stage 1 feature-extraction training."""
    model.freeze_feature_extractor()
    stage1_experiment = build_stage_experiment(
        experiment,
        num_epochs=experiment.stage1_epochs,
        learning_rate=experiment.learning_rate,
        resume_checkpoint=resume_checkpoint,
    )

    if resume_checkpoint is not None:
        start_epoch = resolve_resume_start_epoch(resume_checkpoint)
        logger.info(
            "Stage 1 resume confirmed: next_epoch=%d/%d",
            start_epoch + 1,
            experiment.num_epochs,
        )
    else:
        log_stage_boundary(
            event="STAGE 1 BEGINNING — Feature extraction (frozen backbone, classifier head only)",
            stage=1,
            details={
                "epochs": experiment.stage1_epochs,
                "learning_rate": experiment.learning_rate,
                "trainable_parameters": f"{count_parameters(model).trainable:,}",
            },
        )

    trainer, _, _, engine_config = _build_trainer(
        stage1_experiment,
        model,
        loaders,
        device,
        num_epochs=experiment.stage1_epochs,
        learning_rate=experiment.learning_rate,
        resume_checkpoint=resume_checkpoint,
        checkpoint_manager=checkpoint_manager,
        timing_callback=timing_callback,
        callbacks=callbacks,
    )

    history = trainer.train()

    log_stage_boundary(
        event="STAGE 1 COMPLETE — Feature extraction finished",
        stage=1,
        details={
            "epochs_completed": len(history.train),
            "best_epoch": history.best_epoch + 1 if history.best_epoch >= 0 else "N/A",
            "best_val_accuracy": f"{history.best_metric_value:.4f}"
            if history.best_metric_name == "val_accuracy"
            else "N/A",
        },
    )

    return trainer, engine_config, history


def _run_stage2(
    *,
    experiment: EfficientNetB3ExperimentConfig,
    model: EfficientNetB3TransferClassifier,
    loaders: Any,
    device: torch.device,
    stage1_history: TrainingHistory,
    stage1_global_step: int,
    checkpoint_manager: CheckpointManager,
    timing_callback: EpochTimingCallback,
    callbacks: list,
    stage1_best_path: Path | None,
    resume_checkpoint: Path | None = None,
) -> tuple[Trainer, Any, TrainingHistory]:
    """Run Stage 2 fine-tuning on last feature stage and the classifier head."""
    mid_stage2_resume = (
        resume_checkpoint is not None
        and resolve_resume_start_epoch(resume_checkpoint) > experiment.stage1_epochs
    )
    entering_stage2_from_resume = (
        resume_checkpoint is not None
        and resolve_resume_start_epoch(resume_checkpoint) == experiment.stage1_epochs
    )

    if mid_stage2_resume or entering_stage2_from_resume:
        start_epoch = resolve_resume_start_epoch(resume_checkpoint)
        logger.info(
            "Stage 2 resume confirmed: next_epoch=%d/%d lr=%s",
            start_epoch + 1,
            experiment.num_epochs,
            experiment.stage2_learning_rate,
        )
    else:
        log_stage_boundary(
            event="STAGE 2 BEGINNING — Fine-tuning (last feature stage + classifier, lower learning rate)",
            stage=2,
            details={
                "epochs": experiment.stage2_epochs,
                "learning_rate": experiment.stage2_learning_rate,
                "stage1_checkpoint": str(stage1_best_path) if stage1_best_path else "N/A",
                "trainable_parameters": "pending (after unfreeze)",
            },
        )

    trainer, engine_config = _prepare_stage2_trainer(
        experiment=experiment,
        model=model,
        loaders=loaders,
        device=device,
        stage1_history=stage1_history,
        stage1_global_step=stage1_global_step,
        checkpoint_manager=checkpoint_manager,
        timing_callback=timing_callback,
        callbacks=callbacks,
        stage1_best_path=stage1_best_path,
        resume_checkpoint=resume_checkpoint,
    )

    stage2_counts = count_parameters(trainer.model)
    logger.info(
        "Stage 2 trainer ready: start_epoch=%d, total_epochs=%d, lr=%s, trainable=%s, frozen=%s",
        trainer.start_epoch + 1,
        experiment.num_epochs,
        experiment.stage2_learning_rate,
        f"{stage2_counts.trainable:,}",
        f"{stage2_counts.frozen:,}",
    )
    print(
        f"Stage 2 trainable={stage2_counts.trainable:,} "
        f"frozen={stage2_counts.frozen:,} "
        f"lr={experiment.stage2_learning_rate}"
    )
    print(f"Continuing from epoch {trainer.start_epoch + 1}/{experiment.num_epochs}")
    print()

    history = trainer.train()

    log_stage_boundary(
        event="STAGE 2 COMPLETE — Fine-tuning finished",
        stage=2,
        details={
            "epochs_completed": len(history.train),
            "best_epoch": history.best_epoch + 1 if history.best_epoch >= 0 else "N/A",
            "best_val_accuracy": f"{history.best_metric_value:.4f}"
            if history.best_metric_name == "val_accuracy"
            else "N/A",
        },
    )

    return trainer, engine_config, history


def run_two_stage_efficientnet_b3_experiment(
    experiment: EfficientNetB3ExperimentConfig,
    *,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Run the full two-stage EfficientNet-B3 transfer-learning experiment."""
    experiment.save_dir.mkdir(parents=True, exist_ok=True)
    experiment.tensorboard_dir.mkdir(parents=True, exist_ok=True)

    set_deterministic_seed(experiment.random_seed)

    loaders = build_dataloaders(get_dataloader_config())
    num_classes = loaders.label_encoder.num_classes
    device = resolve_device("auto")

    model = EfficientNetB3TransferClassifier(num_classes=num_classes, freeze_backbone=True)
    model.freeze_feature_extractor()

    stage1_engine_config = build_engine_config(
        experiment,
        num_classes=num_classes,
        device="auto",
        num_epochs=experiment.stage1_epochs,
        learning_rate=experiment.learning_rate,
    )
    summary = build_environment_summary(
        loaders,
        model,
        device=device,
        num_epochs=experiment.num_epochs,
        use_amp=stage1_engine_config.use_amp,
    )
    summary.update(
        {
            "stage1_epochs": experiment.stage1_epochs,
            "stage2_epochs": experiment.stage2_epochs,
            "stage1_learning_rate": experiment.learning_rate,
            "stage2_learning_rate": experiment.stage2_learning_rate,
        }
    )

    start_epoch = 0
    resume_stage = None
    if resume_checkpoint is not None:
        resume_checkpoint = Path(resume_checkpoint)
        if not resume_checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")
        start_epoch = resolve_resume_start_epoch(resume_checkpoint)
        resume_stage = detect_resume_stage(start_epoch, experiment.stage1_epochs)
        if start_epoch >= experiment.num_epochs:
            raise RuntimeError(
                f"Checkpoint already completed all {experiment.num_epochs} epochs "
                f"(next epoch would be {start_epoch + 1})."
            )
        print_pre_training_summary(
            summary,
            title="EfficientNet-B3 Transfer Learning — Resume Training",
        )
        print_resume_banner(
            stage=resume_stage,
            start_epoch_1based=start_epoch + 1,
            total_epochs=experiment.num_epochs,
            checkpoint_path=resume_checkpoint,
        )
    else:
        print_pre_training_summary(
            summary,
            title="EfficientNet-B3 Transfer Learning — Two-Stage Training Run",
        )
        forward_result = verify_forward_pass(model, device=device)
        logger.info(
            "Dummy forward pass verified: input=%s output=%s",
            forward_result["input_shape"],
            forward_result["output_shape"],
        )

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

    stage1_history: TrainingHistory
    stage1_global_step = 0
    stage1_best_path: Path | None = None

    if start_epoch < experiment.stage1_epochs:
        trainer1, _, stage1_history = _run_stage1(
            experiment=experiment,
            model=model,
            loaders=loaders,
            device=device,
            checkpoint_manager=checkpoint_manager,
            timing_callback=timing_callback,
            callbacks=callbacks,
            resume_checkpoint=resume_checkpoint,
        )
        stage1_best_path = _export_stage1_best(checkpoint_manager)
        stage1_global_step = trainer1.global_step
        # Stage 1 just finished in this process — Stage 2 is a fresh transition.
        stage2_resume_checkpoint = None
    else:
        # Resume directly into Stage 2: restore freeze state, history, and weights.
        model.unfreeze_last_feature_stage()
        state = read_checkpoint_state(resume_checkpoint)
        full_history = history_from_checkpoint_state(state)
        stage1_history = TrainingHistory(
            train=list(full_history.train[: experiment.stage1_epochs]),
            val=list(full_history.val[: experiment.stage1_epochs]),
            learning_rates=list(full_history.learning_rates[: experiment.stage1_epochs]),
            best_epoch=full_history.best_epoch,
            best_metric_name=full_history.best_metric_name,
            best_metric_value=full_history.best_metric_value,
        )
        stage1_global_step = int(state.get("global_step", 0))

        if STAGE1_BEST_MODEL_PATH.exists():
            stage1_best_path = STAGE1_BEST_MODEL_PATH
        elif checkpoint_manager.best_model_path.exists():
            stage1_best_path = checkpoint_manager.best_model_path
        else:
            stage1_best_path = resume_checkpoint

        stage2_resume_checkpoint = resume_checkpoint

    trainer, engine_config, combined_history = _run_stage2(
        experiment=experiment,
        model=model,
        loaders=loaders,
        device=device,
        stage1_history=stage1_history,
        stage1_global_step=stage1_global_step,
        checkpoint_manager=checkpoint_manager,
        timing_callback=timing_callback,
        callbacks=callbacks,
        stage1_best_path=stage1_best_path,
        resume_checkpoint=stage2_resume_checkpoint,
    )

    total_training_time = sum(timing_callback.epoch_times)
    stage_config = {
        "stage1_epochs": experiment.stage1_epochs,
        "stage2_epochs": experiment.stage2_epochs,
        "stage1_learning_rate": experiment.learning_rate,
        "stage2_learning_rate": experiment.stage2_learning_rate,
        "stage_boundary_epoch": experiment.stage1_epochs,
    }

    return _finalize_experiment(
        trainer=trainer,
        checkpoint_manager=checkpoint_manager,
        timing_callback=timing_callback,
        experiment=experiment,
        loaders=loaders,
        device=device,
        engine_config=engine_config,
        history=combined_history,
        total_training_time=total_training_time,
        stage_config=stage_config,
    )


def _finalize_experiment(
    *,
    trainer: Trainer,
    checkpoint_manager: CheckpointManager,
    timing_callback: EpochTimingCallback,
    experiment: EfficientNetB3ExperimentConfig,
    loaders: Any,
    device: torch.device,
    engine_config: Any,
    history: Any,
    total_training_time: float | None = None,
    stage_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save artifacts, evaluate, and build the training report."""
    if total_training_time is None:
        total_training_time = sum(timing_callback.epoch_times) or timing_callback.total_training_time

    save_extended_history(
        experiment.history_path,
        history.to_dict(),
        timing_callback.epoch_times,
        total_training_time,
        stage_config=stage_config,
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

    param_counts = count_parameters(trainer.model)
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
        parameter_counts={
            "trainable": param_counts.trainable,
            "frozen": param_counts.frozen,
            "total": param_counts.total,
        },
        stage_config=stage_config,
    )

    save_training_report_json(report, REPORT_JSON)
    save_training_report_markdown(report, REPORT_MD)

    logger.info("EfficientNet-B3 experiment complete. Reports saved to %s and %s", REPORT_MD, REPORT_JSON)
    return report


def run_efficientnet_b3_experiment(
    experiment: EfficientNetB3ExperimentConfig | None = None,
) -> dict[str, Any]:
    """Train the EfficientNet-B3 transfer classifier using two-stage transfer learning."""
    experiment = experiment or EfficientNetB3ExperimentConfig()
    if experiment.resume_checkpoint is not None:
        raise ValueError("Use run_efficientnet_b3_experiment_resume() to resume training.")
    return run_two_stage_efficientnet_b3_experiment(experiment)


def run_efficientnet_b3_experiment_resume(
    experiment: EfficientNetB3ExperimentConfig,
) -> dict[str, Any]:
    """Resume an interrupted two-stage EfficientNet-B3 training run from a checkpoint."""
    if experiment.resume_checkpoint is None:
        raise ValueError("resume_checkpoint is required to resume training.")
    return run_two_stage_efficientnet_b3_experiment(
        experiment,
        resume_checkpoint=experiment.resume_checkpoint,
    )


def main() -> None:
    """CLI entry point for the EfficientNet-B3 transfer-learning experiment."""
    parser = argparse.ArgumentParser(description="EfficientNet-B3 transfer-learning experiment.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a checkpoint (.pt or .pth). Never deletes artifacts.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run training. Without this flag, only readiness verification is performed.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Start a brand-new run and delete previous EfficientNet-B3 artifacts. "
            "Required if a checkpoint already exists and --resume is not used."
        ),
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Deprecated alias kept for compatibility; prefer omitting --fresh.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    experiment = EfficientNetB3ExperimentConfig(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        random_seed=args.seed,
        resume_checkpoint=args.resume,
    )

    if not args.train:
        verify_experiment_readiness(experiment)
        return

    latest_checkpoint = SAVE_DIR / "latest_checkpoint.pt"

    if args.resume is not None:
        # Resume must never delete checkpoints, history, or TensorBoard logs.
        logger.info("Resume mode: preserving all artifacts under %s and %s", SAVE_DIR, TENSORBOARD_DIR)
        print(f"Resuming from: {args.resume}")
        print("Artifacts will NOT be deleted.")
        print()
        report = run_efficientnet_b3_experiment_resume(experiment)
    else:
        if latest_checkpoint.exists() and not args.fresh:
            message = (
                f"Found existing checkpoint at {latest_checkpoint}.\n"
                "Refusing to start a fresh run that would overwrite resume state.\n\n"
                "To continue training:\n"
                f"  python -m experiments.efficientnet_b3.train --train --resume {latest_checkpoint.as_posix()}\n\n"
                "To intentionally discard artifacts and restart:\n"
                "  python -m experiments.efficientnet_b3.train --train --fresh"
            )
            print(message)
            raise SystemExit(1)

        if args.fresh and not args.skip_cleanup:
            deleted = cleanup_previous_artifacts()
            if deleted:
                logger.info("Removed %d previous artifact(s) (--fresh)", len(deleted))
            else:
                logger.info("No previous artifacts found to remove")
        elif not latest_checkpoint.exists() and not args.skip_cleanup and not args.fresh:
            # First-time run with leftover partial/report-only artifacts is safe to tidy.
            deleted = cleanup_previous_artifacts()
            if deleted:
                logger.info("Removed %d leftover artifact(s) before fresh start", len(deleted))

        experiment.resume_checkpoint = None
        report = run_efficientnet_b3_experiment(experiment)

    summary = report["summary"]

    print("EfficientNet-B3 training complete.")
    print(f"  Best epoch:               {summary['best_epoch']}")
    print(f"  Best validation accuracy: {summary['best_validation_accuracy']:.4f}")
    print(f"  Final training accuracy:  {summary['final_training_accuracy']:.4f}")
    print(f"  Final validation accuracy:{summary['final_validation_accuracy']:.4f}")
    print(f"  Training time:            {summary['total_training_time_seconds']:.1f}s")
    print(f"  Best model:               {BEST_MODEL_PATH}")
    print(f"  Report:                   {REPORT_MD}")


if __name__ == "__main__":
    main()