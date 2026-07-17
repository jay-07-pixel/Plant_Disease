"""Training report generation for the DenseNet121 transfer-learning experiment."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.densenet121.config import DEFAULT_STAGE1_EPOCHS, REPORT_JSON, REPORT_MD
from experiments.densenet121.evaluate import EvaluationResult, evaluation_result_to_dict
from src.training.metrics import TrainingHistory


def build_epoch_records(
    history: TrainingHistory,
    epoch_times: list[float],
    *,
    stage_boundary_epoch: int = DEFAULT_STAGE1_EPOCHS,
) -> list[dict[str, Any]]:
    """Merge per-epoch metrics with timing information."""
    records: list[dict[str, Any]] = []
    num_epochs = len(history.train)

    for index in range(num_epochs):
        train_metrics = history.train[index]
        val_metrics = history.val[index]
        learning_rate = history.learning_rates[index] if index < len(history.learning_rates) else 0.0
        epoch_time = epoch_times[index] if index < len(epoch_times) else 0.0
        display_epoch = index + 1

        records.append(
            {
                "epoch": display_epoch,
                "stage": 1 if display_epoch <= stage_boundary_epoch else 2,
                "train_loss": train_metrics.loss,
                "val_loss": val_metrics.loss,
                "train_accuracy": train_metrics.accuracy,
                "val_accuracy": val_metrics.accuracy,
                "train_precision": train_metrics.precision,
                "val_precision": val_metrics.precision,
                "train_recall": train_metrics.recall,
                "val_recall": val_metrics.recall,
                "train_f1_score": train_metrics.f1_score,
                "val_f1_score": val_metrics.f1_score,
                "train_top1_accuracy": train_metrics.top1_accuracy,
                "val_top1_accuracy": val_metrics.top1_accuracy,
                "train_top5_accuracy": train_metrics.top5_accuracy,
                "val_top5_accuracy": val_metrics.top5_accuracy,
                "learning_rate": learning_rate,
                "epoch_time_seconds": epoch_time,
            }
        )

    return records


def build_training_report(
    *,
    history: TrainingHistory,
    epoch_times: list[float],
    total_training_time: float,
    evaluation: EvaluationResult,
    engine_config: dict[str, Any],
    experiment_config: dict[str, Any],
    device: str,
    parameter_counts: dict[str, int] | None = None,
    stage_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the complete DenseNet121 training report payload."""
    stages = stage_config or {
        "stage1_epochs": DEFAULT_STAGE1_EPOCHS,
        "stage2_epochs": 20,
        "stage1_learning_rate": 0.001,
        "stage2_learning_rate": 0.0001,
        "stage_boundary_epoch": DEFAULT_STAGE1_EPOCHS,
    }
    epoch_records = build_epoch_records(
        history,
        epoch_times,
        stage_boundary_epoch=int(stages.get("stage_boundary_epoch", DEFAULT_STAGE1_EPOCHS)),
    )
    final_train = history.train[-1] if history.train else None
    final_val = history.val[-1] if history.val else None

    best_epoch_display = history.best_epoch + 1 if history.best_epoch >= 0 else None
    best_val_accuracy = (
        history.best_metric_value if history.best_metric_name == "val_accuracy" else None
    )

    if history.val:
        accuracies = [metrics.accuracy for metrics in history.val]
        accuracy_best = max(accuracies)
        accuracy_best_epoch = accuracies.index(accuracy_best) + 1
        if best_val_accuracy is None or accuracy_best >= best_val_accuracy:
            best_val_accuracy = accuracy_best
            best_epoch_display = accuracy_best_epoch

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "densenet121",
        "device": device,
        "engine_config": engine_config,
        "experiment_config": experiment_config,
        "parameter_counts": parameter_counts or {},
        "stages": stages,
        "summary": {
            "best_epoch": best_epoch_display,
            "best_validation_accuracy": best_val_accuracy,
            "final_training_accuracy": final_train.accuracy if final_train else None,
            "final_validation_accuracy": final_val.accuracy if final_val else None,
            "final_training_loss": final_train.loss if final_train else None,
            "final_validation_loss": final_val.loss if final_val else None,
            "final_validation_precision": final_val.precision if final_val else None,
            "final_validation_recall": final_val.recall if final_val else None,
            "final_validation_f1_score": final_val.f1_score if final_val else None,
            "final_validation_top1_accuracy": final_val.top1_accuracy if final_val else None,
            "final_validation_top5_accuracy": final_val.top5_accuracy if final_val else None,
            "total_training_time_seconds": total_training_time,
            "epochs_completed": len(history.train),
        },
        "epochs": epoch_records,
        "training_history": history.to_dict(),
        "evaluation": evaluation_result_to_dict(evaluation),
    }


def save_training_report_json(report: dict[str, Any], output_path: Path | str = REPORT_JSON) -> Path:
    """Save the training report as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return path


def save_training_report_markdown(report: dict[str, Any], output_path: Path | str = REPORT_MD) -> Path:
    """Save the training report as Markdown."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    summary = report["summary"]
    evaluation = report["evaluation"]
    params = report.get("parameter_counts", {})
    stages = report.get("stages", {})

    lines = [
        "# DenseNet121 Transfer Learning Training Report",
        "",
        f"**Generated:** {report['generated_at']}  ",
        f"**Experiment:** `{report['experiment']}`  ",
        f"**Device:** `{report['device']}`  ",
        "",
        "## Summary",
        "",
        f"- **Best epoch:** {summary['best_epoch']}",
        f"- **Best validation accuracy:** {summary['best_validation_accuracy']:.4f}"
        if summary["best_validation_accuracy"] is not None
        else "- **Best validation accuracy:** N/A",
        f"- **Final training accuracy:** {summary['final_training_accuracy']:.4f}"
        if summary["final_training_accuracy"] is not None
        else "- **Final training accuracy:** N/A",
        f"- **Final validation accuracy:** {summary['final_validation_accuracy']:.4f}"
        if summary["final_validation_accuracy"] is not None
        else "- **Final validation accuracy:** N/A",
        f"- **Final validation precision:** {summary['final_validation_precision']:.4f}"
        if summary.get("final_validation_precision") is not None
        else "- **Final validation precision:** N/A",
        f"- **Final validation recall:** {summary['final_validation_recall']:.4f}"
        if summary.get("final_validation_recall") is not None
        else "- **Final validation recall:** N/A",
        f"- **Final validation F1 score:** {summary['final_validation_f1_score']:.4f}"
        if summary.get("final_validation_f1_score") is not None
        else "- **Final validation F1 score:** N/A",
        f"- **Final validation top-1 accuracy:** {summary['final_validation_top1_accuracy']:.4f}"
        if summary.get("final_validation_top1_accuracy") is not None
        else "- **Final validation top-1 accuracy:** N/A",
        f"- **Final validation top-5 accuracy:** {summary['final_validation_top5_accuracy']:.4f}"
        if summary.get("final_validation_top5_accuracy") is not None
        else "- **Final validation top-5 accuracy:** N/A",
        f"- **Training time:** {summary['total_training_time_seconds']:.1f}s",
        f"- **Epochs completed:** {summary['epochs_completed']}",
        "",
        "## Model",
        "",
        f"- **Trainable parameters:** {params.get('trainable', 'N/A'):,}"
        if params.get("trainable") is not None
        else "- **Trainable parameters:** N/A",
        f"- **Frozen parameters:** {params.get('frozen', 'N/A'):,}"
        if params.get("frozen") is not None
        else "- **Frozen parameters:** N/A",
        "",
        "## Configuration",
        "",
        "### Two-Stage Transfer Learning",
        "",
        "| Stage | Epochs | Learning Rate | Trainable Layers |",
        "|------:|-------:|--------------:|------------------|",
        f"| 1 — Feature extraction | {stages.get('stage1_epochs', 'N/A')} | {stages.get('stage1_learning_rate', 'N/A')} | Classifier head only |",
        f"| 2 — Fine-tuning | {stages.get('stage2_epochs', 'N/A')} | {stages.get('stage2_learning_rate', 'N/A')} | denseblock4 + classifier |",
        "",
        "| Parameter | Value |",
        "|-----------|------:|",
        f"| Total epochs | {report['experiment_config'].get('num_epochs')} |",
        f"| Optimizer | Adam |",
        f"| Stage 1 learning rate | {stages.get('stage1_learning_rate', report['experiment_config'].get('learning_rate'))} |",
        f"| Stage 2 learning rate | {stages.get('stage2_learning_rate', 'N/A')} |",
        f"| Loss | CrossEntropyLoss |",
        f"| Scheduler | ReduceLROnPlateau |",
        f"| Early stopping patience | {report['experiment_config'].get('early_stopping_patience')} |",
        f"| Random seed | {report['experiment_config'].get('random_seed')} |",
        "",
        "## Per-Epoch Metrics",
        "",
        "| Epoch | Stage | Train Loss | Val Loss | Train Acc | Val Acc | Val F1 | LR | Time (s) |",
        "|------:|------:|-----------:|---------:|----------:|--------:|-------:|---:|---------:|",
    ]

    for epoch in report["epochs"]:
        lines.append(
            f"| {epoch['epoch']} | {epoch.get('stage', '—')} | {epoch['train_loss']:.4f} | {epoch['val_loss']:.4f} | "
            f"{epoch['train_accuracy']:.4f} | {epoch['val_accuracy']:.4f} | "
            f"{epoch['val_f1_score']:.4f} | {epoch['learning_rate']:.2e} | "
            f"{epoch['epoch_time_seconds']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Per-Class Validation Accuracy",
            "",
            "| Class | Accuracy |",
            "|-------|--------:|",
        ]
    )

    for label, accuracy in sorted(
        evaluation["per_class_accuracy"].items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        lines.append(f"| `{label}` | {accuracy:.4f} |")

    lines.extend(
        [
            "",
            "## Confusion Matrix",
            "",
            "Rows = true class, columns = predicted class. See JSON report for the full matrix.",
            "",
            f"- **Overall validation accuracy:** {evaluation['overall_accuracy']:.4f}",
            f"- **Classes:** {len(evaluation['class_labels'])}",
            f"- **Samples evaluated:** {evaluation['num_samples']:,}",
            "",
            "## Artifacts",
            "",
            "| File | Description |",
            "|------|-------------|",
            "| `saved_models/densenet121/best_model.pth` | Best validation checkpoint |",
            "| `saved_models/densenet121/last_model.pth` | Last epoch checkpoint |",
            "| `saved_models/densenet121/stage1_best_model.pth` | Best Stage 1 checkpoint |",
            "| `saved_models/densenet121/training_history.json` | Combined training history |",
            "| `experiments/densenet121/tensorboard/` | TensorBoard event files |",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
