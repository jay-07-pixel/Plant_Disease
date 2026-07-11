"""Cross-dataset comparison for read-only dataset audits.

Builds comparison reports and statistics from per-dataset audit results.
No images are modified, merged, or preprocessed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.data.dataset_audit import AuditSummary, DatasetAuditResult

logger = logging.getLogger(__name__)

DEFAULT_COMPARISON_DIR = Path("reports/comparison")


@dataclass
class DatasetComparisonEntry:
    """Summary metrics for one dataset in a cross-dataset comparison.

    Attributes:
        dataset_name: Canonical dataset identifier.
        num_classes: Number of distinct inferred class labels.
        num_images: Total image count.
        average_width: Mean image width in pixels, or ``None``.
        average_height: Mean image height in pixels, or ``None``.
        average_megapixels: Mean megapixels per image, or ``None``.
        image_formats: Count of images per detected format.
        class_names: Sorted inferred class labels in this dataset.
        corrupted_images: Number of corrupted image files.
        empty_folders: Number of empty directories.
        duplicate_filename_groups: Number of duplicate filename groups.
        max_directory_depth: Maximum directory depth for any image.
    """

    dataset_name: str
    num_classes: int
    num_images: int
    average_width: float | None
    average_height: float | None
    average_megapixels: float | None
    image_formats: dict[str, int]
    class_names: list[str]
    corrupted_images: int
    empty_folders: int
    duplicate_filename_groups: int
    max_directory_depth: int


@dataclass
class ClassOverlapAnalysis:
    """Class label overlap analysis across datasets.

    Class labels are normalized to their leaf folder name (lowercased) before
    comparison so datasets with different directory layouts can be compared.

    Attributes:
        normalized_classes_by_dataset: Normalized label sets per dataset.
        common_classes: Labels present in every dataset.
        unique_classes: Labels present in only one dataset, keyed by dataset.
        missing_classes: For each dataset, labels present in other datasets but
            absent from this one, keyed by the reference dataset name.
    """

    normalized_classes_by_dataset: dict[str, list[str]]
    common_classes: list[str]
    unique_classes: dict[str, list[str]]
    missing_classes: dict[str, dict[str, list[str]]]


@dataclass
class DatasetComparisonReport:
    """Full cross-dataset comparison report.

    Attributes:
        generated_at: UTC ISO-8601 timestamp.
        source_dir: Root directory that was audited.
        datasets: Per-dataset comparison entries keyed by dataset name.
        class_overlap: Cross-dataset class overlap analysis.
    """

    generated_at: str
    source_dir: Path
    datasets: dict[str, DatasetComparisonEntry]
    class_overlap: ClassOverlapAnalysis


def normalize_class_label(class_name: str) -> str:
    """Normalize a class label for cross-dataset comparison.

    Uses the leaf folder component of the inferred class path so datasets with
    different directory hierarchies can be compared read-only.

    Args:
        class_name: Inferred class label from the audit module.

    Returns:
        Lowercased leaf class label.
    """
    return class_name.replace("\\", "/").split("/")[-1].lower().strip()


def _normalized_class_set(result: DatasetAuditResult) -> set[str]:
    """Return the set of normalized class labels for a dataset."""
    return {normalize_class_label(name) for name in result.class_names}


def build_comparison_entry(result: DatasetAuditResult) -> DatasetComparisonEntry:
    """Build a comparison entry from a single dataset audit result.

    Args:
        result: Completed per-dataset audit.

    Returns:
        A :class:`DatasetComparisonEntry` for cross-dataset reporting.
    """
    stats = result.resolution_stats
    return DatasetComparisonEntry(
        dataset_name=result.dataset_name,
        num_classes=result.total_classes,
        num_images=result.total_images,
        average_width=stats.width_mean if stats else None,
        average_height=stats.height_mean if stats else None,
        average_megapixels=stats.megapixels_mean if stats else None,
        image_formats=dict(result.format_stats),
        class_names=list(result.class_names),
        corrupted_images=len(result.corrupted_images),
        empty_folders=len(result.empty_folders),
        duplicate_filename_groups=len(result.duplicate_filenames),
        max_directory_depth=result.max_directory_depth,
    )


def analyze_class_overlap(
    datasets: dict[str, DatasetAuditResult],
) -> ClassOverlapAnalysis:
    """Analyze class label overlap across multiple datasets.

    Args:
        datasets: Per-dataset audit results keyed by dataset name.

    Returns:
        A :class:`ClassOverlapAnalysis` with common, unique, and missing labels.
    """
    normalized_by_dataset = {
        name: sorted(_normalized_class_set(result))
        for name, result in datasets.items()
    }
    sets_by_dataset = {
        name: set(labels) for name, labels in normalized_by_dataset.items()
    }

    if not sets_by_dataset:
        return ClassOverlapAnalysis(
            normalized_classes_by_dataset={},
            common_classes=[],
            unique_classes={},
            missing_classes={},
        )

    all_sets = list(sets_by_dataset.values())
    common = set.intersection(*all_sets) if len(all_sets) > 1 else set(all_sets[0])

    unique: dict[str, list[str]] = {}
    for name, class_set in sets_by_dataset.items():
        other_union = set.union(*(s for n, s in sets_by_dataset.items() if n != name))
        unique[name] = sorted(class_set - other_union)

    missing: dict[str, dict[str, list[str]]] = {}
    for target_name, target_set in sets_by_dataset.items():
        missing[target_name] = {}
        for reference_name, reference_set in sets_by_dataset.items():
            if target_name == reference_name:
                continue
            missing[target_name][reference_name] = sorted(
                reference_set - target_set
            )

    return ClassOverlapAnalysis(
        normalized_classes_by_dataset=normalized_by_dataset,
        common_classes=sorted(common),
        unique_classes=unique,
        missing_classes=missing,
    )


def build_comparison_report(summary: AuditSummary) -> DatasetComparisonReport:
    """Build a cross-dataset comparison report from an audit summary.

    Args:
        summary: Completed audit summary for multiple datasets.

    Returns:
        A :class:`DatasetComparisonReport`.
    """
    entries = {
        name: build_comparison_entry(result)
        for name, result in summary.datasets.items()
    }
    overlap = analyze_class_overlap(summary.datasets)

    logger.info(
        "Built comparison report for %d dataset(s); %d common class(es)",
        len(entries),
        len(overlap.common_classes),
    )

    return DatasetComparisonReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        source_dir=summary.source_dir,
        datasets=entries,
        class_overlap=overlap,
    )


def serialize_comparison_report(report: DatasetComparisonReport) -> dict:
    """Convert a comparison report to a JSON-serializable dictionary.

    Args:
        report: Comparison report to serialize.

    Returns:
        A dictionary suitable for JSON export.
    """
    return {
        "generated_at": report.generated_at,
        "source_dir": str(report.source_dir),
        "datasets": {
            name: asdict(entry) for name, entry in sorted(report.datasets.items())
        },
        "class_overlap": {
            "normalized_classes_by_dataset": report.class_overlap.normalized_classes_by_dataset,
            "common_classes": report.class_overlap.common_classes,
            "unique_classes": report.class_overlap.unique_classes,
            "missing_classes": report.class_overlap.missing_classes,
        },
    }


def save_comparison_json(
    report: DatasetComparisonReport,
    output_path: Path | str = DEFAULT_COMPARISON_DIR / "dataset_comparison.json",
) -> Path:
    """Save the comparison report as JSON.

    Args:
        report: Comparison report to export.
        output_path: Destination JSON file path.

    Returns:
        Path the report was written to.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(serialize_comparison_report(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved comparison JSON to %s", path)
    return path
