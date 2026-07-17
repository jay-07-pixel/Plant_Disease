"""Training dataset balancing plan — read-only hybrid strategy design.

Analyzes the TRAIN split of processed metadata and produces a balancing plan
using WeightedRandomSampler, class weights, and future data augmentation.
Does not modify source data, processed images, validation, or test splits.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.balancing_strategy import (
    BalancingAction,
    ClassCategory,
    categorize_class,
    recommend_balancing_action,
)
from src.data.prepare_dataset import DEFAULT_PROCESSED_DIR
from src.preprocessing.split_dataset import SPLIT_TRAIN

logger = logging.getLogger(__name__)

DEFAULT_PROCESSED_METADATA_CSV = DEFAULT_PROCESSED_DIR / "processed_metadata.csv"
DEFAULT_CLASS_WEIGHTS_CSV = DEFAULT_PROCESSED_DIR / "class_weights.csv"
DEFAULT_BALANCING_PLAN_CSV = DEFAULT_PROCESSED_DIR / "training_balancing_plan.csv"
DEFAULT_BALANCING_PLAN_JSON = DEFAULT_PROCESSED_DIR / "training_balancing_plan.json"
DEFAULT_REPORT_JSON = Path("reports/training_balancing_report.json")
DEFAULT_REPORT_MD = Path("reports/training_balancing_report.md")

TRAIN_SPLIT = SPLIT_TRAIN


@dataclass
class ClassBalancingPlan:
    """Balancing plan for one canonical class in the training split.

    Attributes:
        canonical_label: Stable cross-dataset label key.
        plant: Canonical plant name.
        disease: Canonical disease name.
        is_healthy: Whether the class is healthy.
        category: Imbalance tier within the training split.
        recommended_action: Hybrid balancing recommendation.
        current_image_count: Images in the training split.
        target_image_count: Target effective samples per epoch.
        required_augmentation_count: Additional virtual samples via augmentation.
        augmentation_factor: Ratio of target to current count.
        sampling_weight: Per-sample weight for ``WeightedRandomSampler``.
        normalized_sampling_weight: Sampling weight normalized across classes.
        uses_weighted_sampler: Whether weighted sampling is recommended.
        uses_augmentation: Whether training-time augmentation is recommended.
        percentage_of_train: Class share within the training split (%).
    """

    canonical_label: str
    plant: str
    disease: str
    is_healthy: bool
    category: ClassCategory
    recommended_action: BalancingAction
    current_image_count: int
    target_image_count: int
    required_augmentation_count: int
    augmentation_factor: float
    sampling_weight: float
    normalized_sampling_weight: float
    uses_weighted_sampler: bool
    uses_augmentation: bool
    percentage_of_train: float


@dataclass
class TrainingBalancingReport:
    """Full training balancing plan and analysis report.

    Attributes:
        generated_at: UTC ISO-8601 timestamp.
        metadata_source: Path to processed metadata CSV.
        class_weights_source: Path to class weights CSV.
        train_image_count: Total training images analyzed.
        train_class_count: Distinct canonical classes in training.
        target_samples_per_class: Median class count used as augmentation target.
        class_plans: Per-class balancing plans.
        distribution_before: Class counts before balancing.
        distribution_target: Target effective distribution.
        action_summary: Count of classes per recommended action.
        category_summary: Count of classes per imbalance tier.
    """

    generated_at: str
    metadata_source: str
    class_weights_source: str
    train_image_count: int
    train_class_count: int
    target_samples_per_class: int
    class_plans: list[ClassBalancingPlan]
    distribution_before: dict[str, int]
    distribution_target: dict[str, int]
    action_summary: dict[str, int] = field(default_factory=dict)
    category_summary: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_processed_metadata(
    metadata_path: Path | str = DEFAULT_PROCESSED_METADATA_CSV,
) -> pd.DataFrame:
    """Load processed metadata CSV.

    Args:
        metadata_path: Path to ``processed_metadata.csv``.

    Returns:
        Processed metadata DataFrame.
    """
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Processed metadata not found: {path}")

    df = pd.read_csv(path, encoding="utf-8")
    logger.info("Loaded processed metadata from %s (%d rows)", path, len(df))
    return df


def load_class_weights_reference(
    class_weights_path: Path | str = DEFAULT_CLASS_WEIGHTS_CSV,
) -> pd.DataFrame:
    """Load global class weights reference table.

    Args:
        class_weights_path: Path to ``class_weights.csv``.

    Returns:
        Class weights DataFrame.
    """
    path = Path(class_weights_path)
    if not path.exists():
        raise FileNotFoundError(f"Class weights not found: {path}")

    df = pd.read_csv(path, encoding="utf-8")
    logger.info("Loaded class weights reference from %s (%d classes)", path, len(df))
    return df


def filter_train_split(
    df: pd.DataFrame,
    *,
    split_column: str = "processed_split",
) -> pd.DataFrame:
    """Filter metadata to the training split only.

    Args:
        df: Processed metadata DataFrame.
        split_column: Column containing split labels.

    Returns:
        Training-split DataFrame. Validation and test rows are excluded.
    """
    train_df = df[df[split_column] == TRAIN_SPLIT].copy()
    logger.info(
        "Filtered training split: %d/%d images (val/test excluded)",
        len(train_df),
        len(df),
    )
    return train_df


# ---------------------------------------------------------------------------
# Balancing plan logic
# ---------------------------------------------------------------------------


def compute_train_sampling_weights(train_df: pd.DataFrame) -> dict[str, float]:
    """Compute inverse-frequency sampling weights for the training split.

    Uses::

        weight[class] = total_train / (num_classes * class_count)

    Args:
        train_df: Training metadata DataFrame.

    Returns:
        Mapping of canonical label to sampling weight.
    """
    counts = train_df["canonical_label"].value_counts()
    total = len(train_df)
    num_classes = len(counts)

    return {
        str(label): float(total / (num_classes * count))
        for label, count in counts.items()
    }


def normalize_sampling_weights(weights: dict[str, float]) -> dict[str, float]:
    """Normalize sampling weights to sum to 1.0."""
    total = sum(weights.values())
    if total == 0:
        return {label: 0.0 for label in weights}
    return {label: weight / total for label, weight in weights.items()}


def _compute_target_count(
    current_count: int,
    category: ClassCategory,
    train_median: int,
) -> int:
    """Compute target effective sample count for a class.

    Args:
        current_count: Current training images for the class.
        category: Imbalance tier.
        train_median: Median class count in the training split.

    Returns:
        Target effective sample count per epoch.
    """
    if category in {ClassCategory.RARE, ClassCategory.EXTREMELY_RARE}:
        return max(current_count, train_median)

    return current_count


def _action_flags(action: BalancingAction) -> tuple[bool, bool]:
    """Return ``(uses_weighted_sampler, uses_augmentation)`` for an action."""
    mapping = {
        BalancingAction.NO_ACTION: (False, False),
        BalancingAction.WEIGHTED_SAMPLING: (True, False),
        BalancingAction.DATA_AUGMENTATION: (False, True),
        BalancingAction.WEIGHTED_SAMPLING_AND_AUGMENTATION: (True, True),
    }
    return mapping[action]


def build_class_balancing_plan(
    canonical_label: str,
    current_count: int,
    train_total: int,
    train_median: int,
    sampling_weight: float,
    normalized_weight: float,
    label_info: dict,
) -> ClassBalancingPlan:
    """Build a balancing plan for one canonical class.

    Args:
        canonical_label: Class label key.
        current_count: Images in the training split.
        train_total: Total training images.
        train_median: Median training class count.
        sampling_weight: Raw inverse-frequency weight.
        normalized_weight: Normalized sampling weight.
        label_info: Plant, disease, and health metadata.

    Returns:
        A :class:`ClassBalancingPlan` instance.
    """
    percentage = (current_count / train_total * 100.0) if train_total else 0.0
    category = categorize_class(percentage)
    action = recommend_balancing_action(category)
    uses_sampler, uses_augmentation = _action_flags(action)

    target_count = _compute_target_count(current_count, category, train_median)
    required_augmentation = max(0, target_count - current_count) if uses_augmentation else 0
    augmentation_factor = (
        float(target_count / current_count) if current_count > 0 else 1.0
    )

    return ClassBalancingPlan(
        canonical_label=canonical_label,
        plant=str(label_info.get("plant", "")),
        disease=str(label_info.get("disease", "")),
        is_healthy=bool(label_info.get("is_healthy", False)),
        category=category,
        recommended_action=action,
        current_image_count=current_count,
        target_image_count=target_count,
        required_augmentation_count=required_augmentation,
        augmentation_factor=augmentation_factor,
        sampling_weight=sampling_weight,
        normalized_sampling_weight=normalized_weight,
        uses_weighted_sampler=uses_sampler,
        uses_augmentation=uses_augmentation,
        percentage_of_train=percentage,
    )


def build_training_balancing_plan(
    train_df: pd.DataFrame,
    class_weights_df: pd.DataFrame | None = None,
) -> TrainingBalancingReport:
    """Build the full hybrid training balancing plan.

    Args:
        train_df: Training-split metadata only.
        class_weights_df: Optional global class weights for label metadata.

    Returns:
        A :class:`TrainingBalancingReport` with per-class plans.
    """
    if train_df.empty:
        raise ValueError("Training split is empty; cannot build balancing plan.")

    counts = train_df["canonical_label"].value_counts()
    train_median = int(counts.median())
    sampling_weights = compute_train_sampling_weights(train_df)
    normalized_weights = normalize_sampling_weights(sampling_weights)

    label_lookup: dict[str, dict] = {}
    if class_weights_df is not None:
        for _, row in class_weights_df.iterrows():
            label_lookup[str(row["canonical_label"])] = {
                "plant": row.get("plant", ""),
                "disease": row.get("disease", ""),
                "is_healthy": row.get("is_healthy", False),
            }

    train_grouped = (
        train_df.groupby("canonical_label")
        .agg(
            plant=("plant", "first"),
            disease=("disease", "first"),
            is_healthy=("is_healthy", "first"),
        )
        .to_dict(orient="index")
    )

    class_plans: list[ClassBalancingPlan] = []

    for canonical_label, current_count in counts.items():
        label = str(canonical_label)
        info = label_lookup.get(label, train_grouped.get(label, {}))

        plan = build_class_balancing_plan(
            canonical_label=label,
            current_count=int(current_count),
            train_total=len(train_df),
            train_median=train_median,
            sampling_weight=sampling_weights[label],
            normalized_weight=normalized_weights[label],
            label_info=info,
        )
        class_plans.append(plan)

    class_plans.sort(key=lambda plan: plan.current_image_count)

    distribution_before = {plan.canonical_label: plan.current_image_count for plan in class_plans}
    distribution_target = {plan.canonical_label: plan.target_image_count for plan in class_plans}

    action_summary: dict[str, int] = {}
    category_summary: dict[str, int] = {}
    for plan in class_plans:
        action_summary[plan.recommended_action.value] = (
            action_summary.get(plan.recommended_action.value, 0) + 1
        )
        category_summary[plan.category.value] = (
            category_summary.get(plan.category.value, 0) + 1
        )

    logger.info(
        "Built training balancing plan: %d classes, median target=%d",
        len(class_plans),
        train_median,
    )

    return TrainingBalancingReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        metadata_source=str(DEFAULT_PROCESSED_METADATA_CSV),
        class_weights_source=str(DEFAULT_CLASS_WEIGHTS_CSV),
        train_image_count=len(train_df),
        train_class_count=len(class_plans),
        target_samples_per_class=train_median,
        class_plans=class_plans,
        distribution_before=distribution_before,
        distribution_target=distribution_target,
        action_summary=action_summary,
        category_summary=category_summary,
    )


# ---------------------------------------------------------------------------
# Export and reporting
# ---------------------------------------------------------------------------


def class_plans_to_dataframe(plans: list[ClassBalancingPlan]) -> pd.DataFrame:
    """Convert class balancing plans to a DataFrame."""
    rows = []
    for plan in plans:
        row = asdict(plan)
        row["category"] = plan.category.value
        row["recommended_action"] = plan.recommended_action.value
        rows.append(row)
    return pd.DataFrame(rows)


def save_balancing_plan_csv(
    report: TrainingBalancingReport,
    output_path: Path | str = DEFAULT_BALANCING_PLAN_CSV,
) -> Path:
    """Save the training balancing plan as CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = class_plans_to_dataframe(report.class_plans)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("Saved training balancing plan CSV to %s", path)
    return path


def serialize_balancing_plan(report: TrainingBalancingReport) -> dict:
    """Serialize balancing plan to a JSON-serializable dictionary."""
    return {
        "generated_at": report.generated_at,
        "metadata_source": report.metadata_source,
        "class_weights_source": report.class_weights_source,
        "train_image_count": report.train_image_count,
        "train_class_count": report.train_class_count,
        "target_samples_per_class": report.target_samples_per_class,
        "distribution_before": report.distribution_before,
        "distribution_target": report.distribution_target,
        "action_summary": report.action_summary,
        "category_summary": report.category_summary,
        "hybrid_strategy": {
            "weighted_random_sampler": True,
            "class_weights": True,
            "data_augmentation": "training_split_only",
            "validation_balanced": False,
            "test_balanced": False,
            "duplicate_images": False,
            "modify_source_images": False,
        },
        "class_plans": [
            {
                **asdict(plan),
                "category": plan.category.value,
                "recommended_action": plan.recommended_action.value,
            }
            for plan in report.class_plans
        ],
    }


def save_balancing_plan_json(
    report: TrainingBalancingReport,
    output_path: Path | str = DEFAULT_BALANCING_PLAN_JSON,
) -> Path:
    """Save the training balancing plan as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(serialize_balancing_plan(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved training balancing plan JSON to %s", path)
    return path


def save_training_balancing_report_json(
    report: TrainingBalancingReport,
    output_path: Path | str = DEFAULT_REPORT_JSON,
) -> Path:
    """Save the training balancing report as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(serialize_balancing_plan(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved training balancing report JSON to %s", path)
    return path


def generate_training_balancing_report_markdown(
    report: TrainingBalancingReport,
    output_path: Path | str = DEFAULT_REPORT_MD,
) -> Path:
    """Write a human-readable training balancing markdown report."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    total_aug_required = sum(plan.required_augmentation_count for plan in report.class_plans)

    lines = [
        "# Training Balancing Report",
        "",
        f"**Generated:** {report.generated_at}  ",
        f"**Metadata source:** `{report.metadata_source}`  ",
        f"**Class weights source:** `{report.class_weights_source}`  ",
        "",
        "> Read-only balancing plan. Only the TRAIN split is analyzed. "
        "Val/test are never balanced. No images were modified or augmented.",
        "",
        "## Summary",
        "",
        f"- **Training images:** {report.train_image_count:,}",
        f"- **Training classes:** {report.train_class_count}",
        f"- **Target samples per class (median):** {report.target_samples_per_class:,}",
        f"- **Total virtual augmentations required:** {total_aug_required:,}",
        "",
        "## Hybrid Strategy",
        "",
        "| Component | Applied to |",
        "|-----------|------------|",
        "| WeightedRandomSampler | Training split only |",
        "| Class weights | Training split only |",
        "| Data augmentation | Training split only (future) |",
        "| Validation / Test | Never balanced |",
        "",
        "## Action Summary",
        "",
    ]

    for action, count in sorted(report.action_summary.items()):
        lines.append(f"- **{action}:** {count} classes")
    lines.append("")

    lines.extend(["## Category Summary", ""])
    for category, count in sorted(report.category_summary.items()):
        lines.append(f"- **{category}:** {count} classes")
    lines.append("")

    lines.extend(
        [
            "## Per-Class Balancing Plan",
            "",
            "| Canonical Label | Current | Target | Aug. Needed | Aug. Factor | "
            "Sampling Weight | Category | Action |",
            "|-------------------|--------:|-------:|------------:|------------:|"
            "----------------:|----------|--------|",
        ]
    )

    for plan in sorted(report.class_plans, key=lambda p: p.current_image_count):
        lines.append(
            f"| `{plan.canonical_label}` | {plan.current_image_count:,} | "
            f"{plan.target_image_count:,} | {plan.required_augmentation_count:,} | "
            f"{plan.augmentation_factor:.2f} | {plan.sampling_weight:.4f} | "
            f"{plan.category.value} | {plan.recommended_action.value} |"
        )

    lines.extend(
        [
            "",
            "## Distribution Comparison (Top 15 Smallest Classes)",
            "",
            "| Canonical Label | Before | Target |",
            "|-------------------|-------:|-------:|",
        ]
    )

    smallest = sorted(report.class_plans, key=lambda p: p.current_image_count)[:15]
    for plan in smallest:
        before = report.distribution_before[plan.canonical_label]
        target = report.distribution_target[plan.canonical_label]
        lines.append(f"| `{plan.canonical_label}` | {before:,} | {target:,} |")

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "| File | Description |",
            "|------|-------------|",
            "| `datasets/processed/training_balancing_plan.csv` | Per-class balancing plan |",
            "| `datasets/processed/training_balancing_plan.json` | Plan data (JSON) |",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved training balancing report markdown to %s", path)
    return path


def run_training_balancing_plan(
    metadata_path: Path | str = DEFAULT_PROCESSED_METADATA_CSV,
    class_weights_path: Path | str = DEFAULT_CLASS_WEIGHTS_CSV,
    plan_csv_path: Path | str = DEFAULT_BALANCING_PLAN_CSV,
    plan_json_path: Path | str = DEFAULT_BALANCING_PLAN_JSON,
    report_json_path: Path | str = DEFAULT_REPORT_JSON,
    report_md_path: Path | str = DEFAULT_REPORT_MD,
) -> TrainingBalancingReport:
    """Run the full read-only training balancing plan pipeline.

    Reads processed metadata and class weights, analyzes only the training
    split, and exports balancing plans and reports. No images are modified,
    duplicated, or augmented.

    Args:
        metadata_path: Path to processed metadata CSV.
        class_weights_path: Path to global class weights CSV.
        plan_csv_path: Output balancing plan CSV path.
        plan_json_path: Output balancing plan JSON path.
        report_json_path: Output report JSON path.
        report_md_path: Output report markdown path.

    Returns:
        The completed :class:`TrainingBalancingReport`.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    logger.info("Starting training balancing plan (TRAIN split only)")

    metadata = load_processed_metadata(metadata_path)
    class_weights = load_class_weights_reference(class_weights_path)
    train_df = filter_train_split(metadata)

    report = build_training_balancing_plan(train_df, class_weights)
    report.metadata_source = str(Path(metadata_path))
    report.class_weights_source = str(Path(class_weights_path))

    save_balancing_plan_csv(report, plan_csv_path)
    save_balancing_plan_json(report, plan_json_path)
    save_training_balancing_report_json(report, report_json_path)
    generate_training_balancing_report_markdown(report, report_md_path)

    logger.info(
        "Training balancing plan complete: %d classes, %d training images",
        report.train_class_count,
        report.train_image_count,
    )
    return report


if __name__ == "__main__":
    run_training_balancing_plan()
