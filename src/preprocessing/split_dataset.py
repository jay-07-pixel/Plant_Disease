"""Stratified dataset splitting for preprocessing."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.preprocessing.preprocessing_config import SplitConfig

logger = logging.getLogger(__name__)

SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"


@dataclass
class SplitStatistics:
    """Statistics for a stratified dataset split.

    Attributes:
        total_images: Total images assigned to splits.
        train_count: Images in the training split.
        val_count: Images in the validation split.
        test_count: Images in the test split.
        train_ratio: Achieved training fraction.
        val_ratio: Achieved validation fraction.
        test_ratio: Achieved test fraction.
        classes_with_small_splits: Classes where a split has fewer than 2 samples.
    """

    total_images: int
    train_count: int
    val_count: int
    test_count: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    classes_with_small_splits: int


def _allocate_split_counts(
    total: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[int, int, int]:
    """Allocate per-class sample counts to train, val, and test.

    Args:
        total: Number of samples in the class.
        train_ratio: Target training fraction.
        val_ratio: Target validation fraction.

    Returns:
        Tuple of ``(train_count, val_count, test_count)``.
    """
    if total <= 0:
        return 0, 0, 0

    if total == 1:
        return 1, 0, 0

    if total == 2:
        return 1, 0, 1

    train_count = int(round(total * train_ratio))
    val_count = int(round(total * val_ratio))
    test_count = total - train_count - val_count

    train_count = max(1, train_count)
    val_count = max(0, val_count)
    test_count = max(1, test_count)

    while train_count + val_count + test_count > total:
        if train_count > 1:
            train_count -= 1
        elif val_count > 0:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            break

    while train_count + val_count + test_count < total:
        train_count += 1

    return train_count, val_count, test_count


def assign_stratified_splits(
    df: pd.DataFrame,
    config: SplitConfig,
    *,
    label_column: str = "canonical_label",
    split_column: str = "processed_split",
) -> pd.DataFrame:
    """Assign stratified train/val/test splits based on canonical labels.

    Args:
        df: Metadata DataFrame.
        config: Split ratio configuration.
        label_column: Column used for stratification.
        split_column: Output column name for assigned splits.

    Returns:
        Copy of ``df`` with ``split_column`` populated.
    """
    if label_column not in df.columns:
        raise ValueError(f"Missing label column: {label_column}")

    result = df.copy()
    rng = np.random.default_rng(config.random_seed)
    split_assignments: dict[int, str] = {}

    for label, group in result.groupby(label_column):
        indices = group.index.to_numpy()
        rng.shuffle(indices)

        train_n, val_n, test_n = _allocate_split_counts(
            len(indices),
            config.train_ratio,
            config.val_ratio,
        )

        train_idx = indices[:train_n]
        val_idx = indices[train_n : train_n + val_n]
        test_idx = indices[train_n + val_n : train_n + val_n + test_n]

        for idx in train_idx:
            split_assignments[int(idx)] = SPLIT_TRAIN
        for idx in val_idx:
            split_assignments[int(idx)] = SPLIT_VAL
        for idx in test_idx:
            split_assignments[int(idx)] = SPLIT_TEST

    result[split_column] = result.index.map(split_assignments)
    logger.info(
        "Assigned stratified splits: train=%d, val=%d, test=%d",
        (result[split_column] == SPLIT_TRAIN).sum(),
        (result[split_column] == SPLIT_VAL).sum(),
        (result[split_column] == SPLIT_TEST).sum(),
    )
    return result


def compute_split_statistics(
    df: pd.DataFrame,
    *,
    split_column: str = "processed_split",
    label_column: str = "canonical_label",
) -> SplitStatistics:
    """Compute statistics for assigned dataset splits.

    Args:
        df: Metadata DataFrame with split assignments.
        split_column: Column containing split labels.
        label_column: Canonical label column.

    Returns:
        A :class:`SplitStatistics` summary.
    """
    total = len(df)
    train_count = int((df[split_column] == SPLIT_TRAIN).sum())
    val_count = int((df[split_column] == SPLIT_VAL).sum())
    test_count = int((df[split_column] == SPLIT_TEST).sum())

    classes_with_small_splits = 0
    for _, group in df.groupby(label_column):
        counts = group[split_column].value_counts()
        if (counts < 2).any() and len(group) > 2:
            classes_with_small_splits += 1

    return SplitStatistics(
        total_images=total,
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
        train_ratio=train_count / total if total else 0.0,
        val_ratio=val_count / total if total else 0.0,
        test_ratio=test_count / total if total else 0.0,
        classes_with_small_splits=classes_with_small_splits,
    )


def split_label_distribution(
    df: pd.DataFrame,
    *,
    split_column: str = "processed_split",
    label_column: str = "canonical_label",
) -> dict[str, dict[str, int]]:
    """Compute per-split class distributions.

    Args:
        df: Metadata DataFrame.
        split_column: Split assignment column.
        label_column: Canonical label column.

    Returns:
        Nested dict ``{split: {label: count}}``.
    """
    distribution: dict[str, dict[str, int]] = {}
    for split_name in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST):
        split_df = df[df[split_column] == split_name]
        distribution[split_name] = {
            str(label): int(count)
            for label, count in split_df[label_column].value_counts().items()
        }
    return distribution
