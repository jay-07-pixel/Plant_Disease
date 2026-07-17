"""Weighted sampling and class-weight utilities for training."""

from __future__ import annotations

import logging

import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

from src.training.dataset import LabelEncoder, PlantDiseaseDataset

logger = logging.getLogger(__name__)


def build_per_sample_weights(
    dataset: PlantDiseaseDataset,
    balancing_plan: pd.DataFrame,
    *,
    weight_column: str = "sampling_weight",
) -> list[float]:
    """Assign a sampling weight to every sample from the balancing plan.

    Args:
        dataset: Training dataset.
        balancing_plan: Training balancing plan indexed by canonical label.
        weight_column: Column containing per-class sampling weights.

    Returns:
        Per-sample weights aligned with dataset indices.

    Raises:
        KeyError: If a class in the dataset is missing from the balancing plan.
        ValueError: If the weight column is absent from the plan.
    """
    if weight_column not in balancing_plan.columns:
        raise ValueError(
            f"Balancing plan missing column '{weight_column}'. "
            f"Available: {list(balancing_plan.columns)}"
        )

    weights: list[float] = []
    for index in range(len(dataset)):
        label = dataset.get_canonical_label(index)
        if label not in balancing_plan.index:
            raise KeyError(f"Balancing plan missing class: {label}")
        weights.append(float(balancing_plan.loc[label, weight_column]))

    return weights


def build_weighted_random_sampler(
    dataset: PlantDiseaseDataset,
    balancing_plan: pd.DataFrame,
    *,
    num_samples: int | None = None,
    replacement: bool = True,
    generator: torch.Generator | None = None,
) -> WeightedRandomSampler:
    """Build a :class:`WeightedRandomSampler` for the training split.

    Uses per-class ``sampling_weight`` values from the balancing plan. No
    images are duplicated on disk — oversampling happens only at load time.

    Args:
        dataset: Training dataset.
        balancing_plan: Training balancing plan indexed by canonical label.
        num_samples: Draws per epoch (defaults to dataset length).
        replacement: Whether samples are drawn with replacement.
        generator: Optional PyTorch random generator for reproducibility.

    Returns:
        Configured :class:`WeightedRandomSampler`.
    """
    sample_weights = build_per_sample_weights(dataset, balancing_plan)
    weight_tensor = torch.tensor(sample_weights, dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights=weight_tensor,
        num_samples=num_samples or len(dataset),
        replacement=replacement,
        generator=generator,
    )

    logger.info(
        "Built WeightedRandomSampler: %d samples, weight range [%.4f, %.4f]",
        sampler.num_samples,
        min(sample_weights),
        max(sample_weights),
    )
    return sampler


def build_class_weights_tensor(
    label_encoder: LabelEncoder,
    balancing_plan: pd.DataFrame,
    *,
    weight_column: str = "sampling_weight",
    normalize_mean: bool = True,
) -> torch.Tensor:
    """Build a per-class weight tensor for loss functions (e.g. ``CrossEntropyLoss``).

    Weights are ordered by class index. Rare classes receive higher values.

    Args:
        label_encoder: Shared label encoder.
        balancing_plan: Training balancing plan indexed by canonical label.
        weight_column: Column containing per-class weights.
        normalize_mean: If ``True``, scale weights so their mean is ``1.0``.

    Returns:
        Float tensor of shape ``(num_classes,)``.
    """
    if weight_column not in balancing_plan.columns:
        raise ValueError(f"Balancing plan missing column '{weight_column}'.")

    weights: list[float] = []
    for index in range(label_encoder.num_classes):
        label = label_encoder.index_to_label[index]
        if label not in balancing_plan.index:
            raise KeyError(f"Balancing plan missing class: {label}")
        weights.append(float(balancing_plan.loc[label, weight_column]))

    tensor = torch.tensor(weights, dtype=torch.float32)
    if normalize_mean and tensor.numel() > 0:
        mean = tensor.mean()
        if mean > 0:
            tensor = tensor / mean

    logger.info(
        "Built class weights tensor: %d classes, range [%.4f, %.4f]",
        len(weights),
        float(tensor.min()),
        float(tensor.max()),
    )
    return tensor


def summarize_sampler(
    dataset: PlantDiseaseDataset,
    balancing_plan: pd.DataFrame,
) -> dict:
    """Summarize weighted-sampling configuration for reporting.

    Args:
        dataset: Training dataset.
        balancing_plan: Training balancing plan.

    Returns:
        JSON-serializable sampler summary dictionary.
    """
    per_sample_weights = build_per_sample_weights(dataset, balancing_plan)
    sampler_classes = balancing_plan[balancing_plan["uses_weighted_sampler"] == True]  # noqa: E712

    return {
        "weighted_random_sampler_enabled": True,
        "num_samples_per_epoch": len(dataset),
        "replacement": True,
        "weight_column": "sampling_weight",
        "per_sample_weight_min": min(per_sample_weights),
        "per_sample_weight_max": max(per_sample_weights),
        "per_sample_weight_mean": sum(per_sample_weights) / len(per_sample_weights),
        "classes_using_weighted_sampler": int(len(sampler_classes)),
        "classes_using_augmentation": int(
            balancing_plan["uses_augmentation"].sum()
        ),
    }
