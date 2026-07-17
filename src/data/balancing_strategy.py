"""Dataset balancing strategy — read-only imbalance analysis and recommendations.

Analyzes class distribution from prepared metadata and designs a balancing
strategy for future preprocessing and training. Does not modify, copy, delete,
augment, or resample any images.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import pandas as pd

from src.data.prepare_dataset import DEFAULT_METADATA_CSV, DEFAULT_PROCESSED_DIR

logger = logging.getLogger(__name__)

DEFAULT_CLASS_WEIGHTS_CSV = DEFAULT_PROCESSED_DIR / "class_weights.csv"
DEFAULT_CLASS_WEIGHTS_JSON = DEFAULT_PROCESSED_DIR / "class_weights.json"
DEFAULT_STRATEGY_JSON = Path("reports/balancing_strategy.json")
DEFAULT_STRATEGY_MD = Path("reports/balancing_strategy.md")

# Percentage-of-dataset thresholds for class categorization.
CATEGORY_THRESHOLDS: dict[str, float] = {
    "dominant": 7.0,
    "common": 3.0,
    "moderate": 1.0,
    "rare": 0.2,
}


class ClassCategory(str, Enum):
    """Imbalance tier for a canonical class."""

    EXTREMELY_RARE = "Extremely Rare"
    RARE = "Rare"
    MODERATE = "Moderate"
    COMMON = "Common"
    DOMINANT = "Dominant"


class BalancingAction(str, Enum):
    """Recommended balancing action for a canonical class."""

    NO_ACTION = "No Action"
    WEIGHTED_SAMPLING = "Weighted Sampling"
    DATA_AUGMENTATION = "Data Augmentation"
    WEIGHTED_SAMPLING_AND_AUGMENTATION = "Weighted Sampling + Augmentation"


@dataclass
class ClassBalanceAnalysis:
    """Imbalance metrics and strategy for one canonical class.

    Attributes:
        canonical_label: Stable cross-dataset label key.
        plant: Canonical plant name (representative).
        disease: Canonical disease name (representative).
        is_healthy: Whether the class represents healthy plants.
        total_images: Number of images in the class.
        percentage_of_dataset: Class share as a percentage of all images.
        class_frequency: Class share as a proportion in ``[0, 1]``.
        imbalance_ratio: Ratio of majority class count to this class count.
        inverse_frequency_weight: Raw inverse-frequency sampling weight.
        normalized_class_weight: Weight normalized to sum to 1.0 across classes.
        category: Imbalance tier classification.
        recommended_action: Suggested balancing approach.
    """

    canonical_label: str
    plant: str
    disease: str
    is_healthy: bool
    total_images: int
    percentage_of_dataset: float
    class_frequency: float
    imbalance_ratio: float
    inverse_frequency_weight: float
    normalized_class_weight: float
    category: ClassCategory
    recommended_action: BalancingAction


@dataclass
class GlobalImbalanceMetrics:
    """Dataset-wide imbalance summary statistics.

    Attributes:
        total_images: Total images in the metadata table.
        total_classes: Number of canonical classes.
        majority_class: Class with the highest image count.
        majority_count: Image count of the majority class.
        minority_class: Class with the lowest image count.
        minority_count: Image count of the minority class.
        global_imbalance_ratio: Ratio of majority to minority class counts.
        coefficient_of_variation: Std dev / mean of per-class counts.
        gini_coefficient: Gini coefficient of class counts (0 = balanced).
    """

    total_images: int
    total_classes: int
    majority_class: str
    majority_count: int
    minority_class: str
    minority_count: int
    global_imbalance_ratio: float
    coefficient_of_variation: float
    gini_coefficient: float


@dataclass
class BalancingStrategyReport:
    """Full balancing strategy analysis report.

    Attributes:
        generated_at: UTC ISO-8601 timestamp.
        metadata_source: Path to the metadata CSV used.
        global_metrics: Dataset-wide imbalance metrics.
        class_analyses: Per-class balance analyses keyed by canonical label.
    """

    generated_at: str
    metadata_source: str
    global_metrics: GlobalImbalanceMetrics
    class_analyses: list[ClassBalanceAnalysis]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_metadata_csv(metadata_path: Path | str = DEFAULT_METADATA_CSV) -> pd.DataFrame:
    """Load the prepared dataset metadata table.

    Args:
        metadata_path: Path to ``dataset_metadata.csv``.

    Returns:
        Metadata DataFrame.

    Raises:
        FileNotFoundError: If the metadata file does not exist.
    """
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")

    df = pd.read_csv(path, encoding="utf-8")
    logger.info("Loaded metadata from %s (%d rows)", path, len(df))
    return df


# ---------------------------------------------------------------------------
# Imbalance analysis
# ---------------------------------------------------------------------------


def compute_gini_coefficient(counts: pd.Series) -> float:
    """Compute the Gini coefficient for a distribution of class counts.

    Args:
        counts: Per-class image counts.

    Returns:
        Gini coefficient in ``[0, 1]`` where 0 is perfectly balanced.
    """
    if counts.empty:
        return 0.0

    values = counts.sort_values().to_numpy(dtype="float64")
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0

    index = pd.Series(range(1, n + 1))
    return float(
        (2 * (index * values).sum()) / (n * values.sum()) - (n + 1) / n
    )


def compute_global_imbalance_metrics(df: pd.DataFrame) -> GlobalImbalanceMetrics:
    """Compute dataset-wide imbalance summary statistics.

    Args:
        df: Metadata DataFrame with a ``canonical_label`` column.

    Returns:
        A :class:`GlobalImbalanceMetrics` instance.
    """
    if df.empty:
        return GlobalImbalanceMetrics(
            total_images=0,
            total_classes=0,
            majority_class="",
            majority_count=0,
            minority_class="",
            minority_count=0,
            global_imbalance_ratio=0.0,
            coefficient_of_variation=0.0,
            gini_coefficient=0.0,
        )

    counts = df["canonical_label"].value_counts()
    majority_class = str(counts.idxmax())
    minority_class = str(counts.idxmin())
    majority_count = int(counts.max())
    minority_count = int(counts.min())

    mean_count = float(counts.mean())
    std_count = float(counts.std(ddof=0))
    coefficient_of_variation = std_count / mean_count if mean_count > 0 else 0.0

    global_ratio = (
        float(majority_count / minority_count) if minority_count > 0 else float("inf")
    )

    return GlobalImbalanceMetrics(
        total_images=len(df),
        total_classes=len(counts),
        majority_class=majority_class,
        majority_count=majority_count,
        minority_class=minority_class,
        minority_count=minority_count,
        global_imbalance_ratio=global_ratio,
        coefficient_of_variation=coefficient_of_variation,
        gini_coefficient=compute_gini_coefficient(counts),
    )


def categorize_class(percentage_of_dataset: float) -> ClassCategory:
    """Assign an imbalance tier based on class percentage of the dataset.

    Args:
        percentage_of_dataset: Class share as a percentage.

    Returns:
        A :class:`ClassCategory` tier.
    """
    if percentage_of_dataset >= CATEGORY_THRESHOLDS["dominant"]:
        return ClassCategory.DOMINANT
    if percentage_of_dataset >= CATEGORY_THRESHOLDS["common"]:
        return ClassCategory.COMMON
    if percentage_of_dataset >= CATEGORY_THRESHOLDS["moderate"]:
        return ClassCategory.MODERATE
    if percentage_of_dataset >= CATEGORY_THRESHOLDS["rare"]:
        return ClassCategory.RARE
    return ClassCategory.EXTREMELY_RARE


def recommend_balancing_action(category: ClassCategory) -> BalancingAction:
    """Recommend a balancing action for an imbalance tier.

    Args:
        category: Class imbalance tier.

    Returns:
        A :class:`BalancingAction` recommendation.
    """
    mapping = {
        ClassCategory.DOMINANT: BalancingAction.NO_ACTION,
        ClassCategory.COMMON: BalancingAction.NO_ACTION,
        ClassCategory.MODERATE: BalancingAction.WEIGHTED_SAMPLING,
        ClassCategory.RARE: BalancingAction.DATA_AUGMENTATION,
        ClassCategory.EXTREMELY_RARE: BalancingAction.WEIGHTED_SAMPLING_AND_AUGMENTATION,
    }
    return mapping[category]


def compute_inverse_frequency_weights(df: pd.DataFrame) -> dict[str, float]:
    """Compute raw inverse-frequency weights per canonical class.

    Uses::

        weight[class] = total_samples / (num_classes * class_count)

    Args:
        df: Metadata DataFrame.

    Returns:
        Mapping of canonical label to inverse-frequency weight.
    """
    if df.empty:
        return {}

    counts = df["canonical_label"].value_counts()
    total_samples = len(df)
    num_classes = len(counts)

    return {
        str(label): float(total_samples / (num_classes * count))
        for label, count in counts.items()
    }


def normalize_class_weights(weights: dict[str, float]) -> dict[str, float]:
    """Normalize class weights so they sum to 1.0.

    Args:
        weights: Raw per-class weights.

    Returns:
        Normalized weights summing to 1.0.
    """
    if not weights:
        return {}

    total = sum(weights.values())
    if total == 0:
        return {label: 0.0 for label in weights}

    return {label: weight / total for label, weight in weights.items()}


def analyze_class_imbalance(df: pd.DataFrame) -> list[ClassBalanceAnalysis]:
    """Analyze imbalance and build per-class balancing strategy entries.

    Args:
        df: Metadata DataFrame.

    Returns:
        List of :class:`ClassBalanceAnalysis` sorted by count ascending.
    """
    if df.empty:
        return []

    total_images = len(df)
    counts = df["canonical_label"].value_counts()
    majority_count = int(counts.max())
    inverse_weights = compute_inverse_frequency_weights(df)
    normalized_weights = normalize_class_weights(inverse_weights)

    label_metadata = (
        df.groupby("canonical_label")
        .agg(
            plant=("plant", "first"),
            disease=("disease", "first"),
            is_healthy=("is_healthy", "first"),
        )
        .to_dict(orient="index")
    )

    analyses: list[ClassBalanceAnalysis] = []

    for canonical_label, count in counts.items():
        label = str(canonical_label)
        count_int = int(count)
        frequency = count_int / total_images
        percentage = frequency * 100.0
        category = categorize_class(percentage)
        meta = label_metadata.get(label, {})

        analyses.append(
            ClassBalanceAnalysis(
                canonical_label=label,
                plant=str(meta.get("plant", "")),
                disease=str(meta.get("disease", "")),
                is_healthy=bool(meta.get("is_healthy", False)),
                total_images=count_int,
                percentage_of_dataset=percentage,
                class_frequency=frequency,
                imbalance_ratio=(
                    float(majority_count / count_int) if count_int > 0 else float("inf")
                ),
                inverse_frequency_weight=inverse_weights[label],
                normalized_class_weight=normalized_weights[label],
                category=category,
                recommended_action=recommend_balancing_action(category),
            )
        )

    analyses.sort(key=lambda item: item.total_images)
    logger.info("Analyzed imbalance for %d canonical classes", len(analyses))
    return analyses


def build_balancing_strategy_report(
    df: pd.DataFrame,
    *,
    metadata_source: str,
) -> BalancingStrategyReport:
    """Build a full balancing strategy report from metadata.

    Args:
        df: Metadata DataFrame.
        metadata_source: Path description for the metadata source file.

    Returns:
        A :class:`BalancingStrategyReport`.
    """
    return BalancingStrategyReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        metadata_source=metadata_source,
        global_metrics=compute_global_imbalance_metrics(df),
        class_analyses=analyze_class_imbalance(df),
    )


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def class_analyses_to_dataframe(analyses: list[ClassBalanceAnalysis]) -> pd.DataFrame:
    """Convert class balance analyses to a pandas DataFrame.

    Args:
        analyses: Per-class balance analyses.

    Returns:
        DataFrame suitable for CSV export.
    """
    if not analyses:
        return pd.DataFrame()

    rows = []
    for analysis in analyses:
        row = asdict(analysis)
        row["category"] = analysis.category.value
        row["recommended_action"] = analysis.recommended_action.value
        rows.append(row)

    return pd.DataFrame(rows)


def save_class_weights_csv(
    analyses: list[ClassBalanceAnalysis],
    output_path: Path | str = DEFAULT_CLASS_WEIGHTS_CSV,
) -> Path:
    """Export class weights and strategy to CSV.

    Args:
        analyses: Per-class balance analyses.
        output_path: Destination CSV path.

    Returns:
        Path the file was written to.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = class_analyses_to_dataframe(analyses)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("Saved class weights CSV to %s (%d classes)", path, len(df))
    return path


def save_class_weights_json(
    analyses: list[ClassBalanceAnalysis],
    output_path: Path | str = DEFAULT_CLASS_WEIGHTS_JSON,
) -> Path:
    """Export class weights and strategy to JSON.

    Args:
        analyses: Per-class balance analyses.
        output_path: Destination JSON path.

    Returns:
        Path the file was written to.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_classes": len(analyses),
        "classes": [
            {
                **asdict(analysis),
                "category": analysis.category.value,
                "recommended_action": analysis.recommended_action.value,
            }
            for analysis in analyses
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved class weights JSON to %s", path)
    return path


def serialize_balancing_strategy_report(report: BalancingStrategyReport) -> dict:
    """Convert a balancing strategy report to a JSON-serializable dictionary."""
    global_metrics = asdict(report.global_metrics)
    category_summary: dict[str, int] = {}
    action_summary: dict[str, int] = {}

    for analysis in report.class_analyses:
        category_summary[analysis.category.value] = (
            category_summary.get(analysis.category.value, 0) + 1
        )
        action_summary[analysis.recommended_action.value] = (
            action_summary.get(analysis.recommended_action.value, 0) + 1
        )

    return {
        "generated_at": report.generated_at,
        "metadata_source": report.metadata_source,
        "global_metrics": global_metrics,
        "category_summary": category_summary,
        "action_summary": action_summary,
        "class_analyses": [
            {
                **asdict(analysis),
                "category": analysis.category.value,
                "recommended_action": analysis.recommended_action.value,
            }
            for analysis in report.class_analyses
        ],
    }


def save_balancing_strategy_json(
    report: BalancingStrategyReport,
    output_path: Path | str = DEFAULT_STRATEGY_JSON,
) -> Path:
    """Save the balancing strategy report as JSON.

    Args:
        report: Balancing strategy report.
        output_path: Destination JSON path.

    Returns:
        Path the file was written to.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(serialize_balancing_strategy_report(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved balancing strategy JSON to %s", path)
    return path


def generate_balancing_strategy_markdown(
    report: BalancingStrategyReport,
    output_path: Path | str = DEFAULT_STRATEGY_MD,
) -> Path:
    """Write a human-readable balancing strategy markdown report.

    Args:
        report: Balancing strategy report.
        output_path: Destination markdown path.

    Returns:
        Path the file was written to.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = report.global_metrics

    lines = [
        "# Dataset Balancing Strategy Report",
        "",
        f"**Generated:** {report.generated_at}  ",
        f"**Metadata source:** `{report.metadata_source}`  ",
        "",
        "> Read-only strategy design. No images were modified, copied, or balanced.",
        "",
        "## Global Imbalance Metrics",
        "",
        f"- **Total images:** {metrics.total_images:,}",
        f"- **Total classes:** {metrics.total_classes}",
        f"- **Majority class:** `{metrics.majority_class}` ({metrics.majority_count:,} images)",
        f"- **Minority class:** `{metrics.minority_class}` ({metrics.minority_count:,} images)",
        f"- **Global imbalance ratio:** {metrics.global_imbalance_ratio:.2f}",
        f"- **Coefficient of variation:** {metrics.coefficient_of_variation:.3f}",
        f"- **Gini coefficient:** {metrics.gini_coefficient:.3f}",
        "",
        "## Categorization Thresholds",
        "",
        "| Tier | Minimum % of Dataset |",
        "|------|---------------------:|",
        f"| Dominant | ≥ {CATEGORY_THRESHOLDS['dominant']:.1f}% |",
        f"| Common | ≥ {CATEGORY_THRESHOLDS['common']:.1f}% |",
        f"| Moderate | ≥ {CATEGORY_THRESHOLDS['moderate']:.1f}% |",
        f"| Rare | ≥ {CATEGORY_THRESHOLDS['rare']:.1f}% |",
        f"| Extremely Rare | < {CATEGORY_THRESHOLDS['rare']:.1f}% |",
        "",
        "## Recommended Actions by Tier",
        "",
        "| Tier | Action |",
        "|------|--------|",
        f"| Dominant | {BalancingAction.NO_ACTION.value} |",
        f"| Common | {BalancingAction.NO_ACTION.value} |",
        f"| Moderate | {BalancingAction.WEIGHTED_SAMPLING.value} |",
        f"| Rare | {BalancingAction.DATA_AUGMENTATION.value} |",
        f"| Extremely Rare | {BalancingAction.WEIGHTED_SAMPLING_AND_AUGMENTATION.value} |",
        "",
        "## Action Summary",
        "",
    ]

    payload = serialize_balancing_strategy_report(report)
    for action, count in sorted(payload["action_summary"].items()):
        lines.append(f"- **{action}:** {count} classes")
    lines.append("")

    lines.extend(["## Category Summary", ""])
    for category, count in sorted(payload["category_summary"].items()):
        lines.append(f"- **{category}:** {count} classes")
    lines.append("")

    lines.extend(
        [
            "## Per-Class Balancing Strategy",
            "",
            "| Canonical Label | Images | % | Imbalance Ratio | Category | Action | Norm. Weight |",
            "|-----------------|-------:|--:|----------------:|----------|--------|-------------:|",
        ]
    )

    for analysis in sorted(
        report.class_analyses, key=lambda item: item.total_images, reverse=True
    ):
        lines.append(
            f"| `{analysis.canonical_label}` | {analysis.total_images:,} | "
            f"{analysis.percentage_of_dataset:.2f} | {analysis.imbalance_ratio:.1f} | "
            f"{analysis.category.value} | {analysis.recommended_action.value} | "
            f"{analysis.normalized_class_weight:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Weight Formulas",
            "",
            "- **Inverse frequency weight:** `total_samples / (num_classes × class_count)`",
            "- **Normalized class weight:** `inverse_weight / sum(all inverse_weights)`",
            "- **Imbalance ratio:** `majority_class_count / class_count`",
            "",
            "## Output Files",
            "",
            "| File | Description |",
            "|------|-------------|",
            "| `datasets/processed/class_weights.csv` | Per-class weights and recommendations |",
            "| `datasets/processed/class_weights.json` | Same data in JSON format |",
            "| `reports/balancing_strategy.json` | Full strategy report |",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved balancing strategy markdown to %s", path)
    return path


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run_balancing_strategy(
    metadata_path: Path | str = DEFAULT_METADATA_CSV,
    class_weights_csv_path: Path | str = DEFAULT_CLASS_WEIGHTS_CSV,
    class_weights_json_path: Path | str = DEFAULT_CLASS_WEIGHTS_JSON,
    strategy_json_path: Path | str = DEFAULT_STRATEGY_JSON,
    strategy_md_path: Path | str = DEFAULT_STRATEGY_MD,
) -> BalancingStrategyReport:
    """Run the full read-only balancing strategy pipeline.

    Reads prepared metadata, analyzes class imbalance, categorizes classes,
    recommends balancing actions, and exports strategy artifacts. No images
    are modified or resampled.

    Args:
        metadata_path: Path to ``dataset_metadata.csv``.
        class_weights_csv_path: Output class weights CSV path.
        class_weights_json_path: Output class weights JSON path.
        strategy_json_path: Output strategy report JSON path.
        strategy_md_path: Output strategy markdown report path.

    Returns:
        The completed :class:`BalancingStrategyReport`.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    logger.info("Starting balancing strategy analysis")
    df = load_metadata_csv(metadata_path)

    report = build_balancing_strategy_report(
        df,
        metadata_source=str(Path(metadata_path)),
    )

    save_class_weights_csv(report.class_analyses, class_weights_csv_path)
    save_class_weights_json(report.class_analyses, class_weights_json_path)
    save_balancing_strategy_json(report, strategy_json_path)
    generate_balancing_strategy_markdown(report, strategy_md_path)

    logger.info(
        "Balancing strategy complete: %d classes, global imbalance ratio %.2f",
        report.global_metrics.total_classes,
        report.global_metrics.global_imbalance_ratio,
    )
    return report


if __name__ == "__main__":
    run_balancing_strategy()
