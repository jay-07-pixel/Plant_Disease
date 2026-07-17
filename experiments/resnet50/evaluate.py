"""Post-training evaluation reused from the baseline experiment."""

from experiments.baseline_cnn.evaluate import (
    EvaluationResult,
    evaluate_on_dataloader,
    evaluation_result_to_dict,
)

__all__ = [
    "EvaluationResult",
    "evaluate_on_dataloader",
    "evaluation_result_to_dict",
]
