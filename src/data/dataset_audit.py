"""Read-only dataset audit for images under project dataset directories.

This module inspects datasets without modifying, renaming, resizing, deleting,
or preprocessing any files. All operations are limited to metadata reads and
in-memory analysis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image, UnidentifiedImageError

from src.config.dataset_schema import VALID_IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path("datasets/raw")
DEFAULT_EXTERNAL_DIR = Path("datasets/external")
DEFAULT_REPORTS_DIR = Path("reports")
DEFAULT_COMPARISON_DIR = Path("reports/comparison")

SUPPORTED_EXTENSIONS = VALID_IMAGE_EXTENSIONS
IGNORED_FILENAMES = frozenset({".gitkeep", ".ds_store", "thumbs.db", "desktop.ini"})


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ImageAuditInfo:
    """Read-only metadata collected for a single image file.

    Attributes:
        path: Absolute path to the image file.
        relative_path: Path relative to the dataset root.
        dataset_name: Name of the parent dataset directory.
        class_name: Inferred class label from parent folder structure.
        width: Image width in pixels, or ``None`` if unreadable.
        height: Image height in pixels, or ``None`` if unreadable.
        image_format: Detected format extension (lowercase, no dot).
        is_corrupted: Whether the file could not be opened as a valid image.
        perceptual_hash: Perceptual hash string, or ``None`` if unavailable.
        directory_depth: Folder depth relative to the dataset root.
    """

    path: Path
    relative_path: Path
    dataset_name: str
    class_name: str
    width: int | None
    height: int | None
    image_format: str
    is_corrupted: bool
    perceptual_hash: str | None
    directory_depth: int


@dataclass
class ResolutionStats:
    """Summary statistics for image resolutions within a dataset.

    Attributes:
        count: Number of successfully read images included in the stats.
        width_min: Minimum width in pixels.
        width_max: Maximum width in pixels.
        width_mean: Mean width in pixels.
        width_median: Median width in pixels.
        height_min: Minimum height in pixels.
        height_max: Maximum height in pixels.
        height_mean: Mean height in pixels.
        height_median: Median height in pixels.
        megapixels_min: Minimum total megapixels (width * height / 1e6).
        megapixels_max: Maximum total megapixels.
        megapixels_mean: Mean total megapixels.
        megapixels_median: Median total megapixels.
    """

    count: int
    width_min: int
    width_max: int
    width_mean: float
    width_median: float
    height_min: int
    height_max: int
    height_mean: float
    height_median: float
    megapixels_min: float
    megapixels_max: float
    megapixels_mean: float
    megapixels_median: float


@dataclass
class ClassImbalanceStats:
    """Class distribution imbalance metrics.

    Attributes:
        majority_class: Class with the highest image count.
        majority_count: Image count of the majority class.
        minority_class: Class with the lowest image count (among non-empty).
        minority_count: Image count of the minority class.
        imbalance_ratio: Ratio of majority to minority class counts.
        coefficient_of_variation: Std dev / mean of per-class counts.
    """

    majority_class: str
    majority_count: int
    minority_class: str
    minority_count: int
    imbalance_ratio: float
    coefficient_of_variation: float


@dataclass
class DatasetAuditResult:
    """Complete read-only audit result for one dataset.

    Attributes:
        dataset_name: Directory name of the dataset under the audited source root.
        dataset_path: Absolute path to the dataset root.
        total_images: Count of discovered image files.
        total_classes: Number of distinct inferred class labels.
        class_names: Sorted list of inferred class labels.
        images_per_class: Mapping of class label to image count.
        class_imbalance: Imbalance metrics for the class distribution.
        empty_folders: Relative paths of directories containing no files.
        non_image_files: Relative paths of non-image files (excluding ignored).
        corrupted_images: Relative paths of files that failed image reads.
        resolution_stats: Resolution summary, or ``None`` if no valid images.
        format_stats: Count of images per file format extension.
        duplicate_hash_groups: Groups of paths sharing the same perceptual hash.
        duplicate_filenames: Filename to list of relative paths with that name.
        max_directory_depth: Maximum folder depth observed for any image.
        images: Per-image audit records (excluded from JSON export by default).
    """

    dataset_name: str
    dataset_path: Path
    total_images: int
    total_classes: int
    class_names: list[str]
    images_per_class: dict[str, int]
    class_imbalance: ClassImbalanceStats | None
    empty_folders: list[str]
    non_image_files: list[str]
    corrupted_images: list[str]
    resolution_stats: ResolutionStats | None
    format_stats: dict[str, int]
    duplicate_hash_groups: list[list[str]]
    duplicate_filenames: dict[str, list[str]]
    max_directory_depth: int
    images: list[ImageAuditInfo] = field(repr=False, default_factory=list)


@dataclass
class AuditSummary:
    """Aggregate audit results across all discovered datasets.

    Attributes:
        source_dir: Path that was audited (e.g. ``datasets/external``).
        generated_at: UTC timestamp of the audit run.
        datasets: Per-dataset audit results keyed by dataset name.
        aggregate_images_per_class: Combined class counts across datasets.
        aggregate_format_stats: Combined format counts across datasets.
        total_images: Total images across all datasets.
        total_classes: Total distinct class labels across all datasets.
    """

    source_dir: Path
    generated_at: str
    datasets: dict[str, DatasetAuditResult]
    aggregate_images_per_class: dict[str, int]
    aggregate_format_stats: dict[str, int]
    total_images: int
    total_classes: int

    @property
    def raw_dir(self) -> Path:
        """Backward-compatible alias for :attr:`source_dir`."""
        return self.source_dir


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_datasets(
    data_dir: Path | str,
    *,
    include_root_if_images: bool = False,
) -> list[Path]:
    """Discover dataset directories inside a data root.

    Each immediate subdirectory of ``data_dir`` that is not hidden is treated
    as one dataset.

    Args:
        data_dir: Root directory containing dataset folders.
        include_root_if_images: When ``True``, also audit ``data_dir`` itself
            if it contains image files directly (used for ``datasets/raw``).

    Returns:
        A list of dataset root paths to audit.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning("Data directory does not exist: %s", data_path)
        return []

    datasets: list[Path] = []
    for entry in sorted(data_path.iterdir()):
        if entry.is_dir() and not entry.name.startswith("."):
            datasets.append(entry)

    if include_root_if_images and _directory_contains_images(data_path):
        datasets.insert(0, data_path)

    return datasets


def discover_external_datasets(
    external_dir: Path | str = DEFAULT_EXTERNAL_DIR,
) -> list[Path]:
    """Discover dataset directories inside ``datasets/external/``.

    Args:
        external_dir: Root directory containing ingested datasets.

    Returns:
        A list of dataset root paths to audit.
    """
    datasets = discover_datasets(external_dir, include_root_if_images=False)
    logger.info(
        "Discovered %d external dataset(s) in %s",
        len(datasets),
        external_dir,
    )
    return datasets


def _directory_contains_images(directory: Path) -> bool:
    """Return whether any supported image exists directly under ``directory``."""
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower().lstrip(".") in SUPPORTED_EXTENSIONS:
            return True
    return False


# ---------------------------------------------------------------------------
# Read-only image inspection
# ---------------------------------------------------------------------------


def _infer_class_name(relative_path: Path) -> str:
    """Infer a class label from the parent folder of an image."""
    if len(relative_path.parts) <= 1:
        return "_root_"
    return str(Path(*relative_path.parts[:-1]))


def _read_image_metadata(
    image_path: Path,
    dataset_name: str,
    dataset_root: Path,
) -> ImageAuditInfo:
    """Read image metadata without modifying the file.

    Args:
        image_path: Absolute path to the image.
        dataset_name: Name of the dataset being audited.
        dataset_root: Root directory of the dataset.

    Returns:
        An :class:`ImageAuditInfo` record for the image.
    """
    relative_path = image_path.relative_to(dataset_root)
    image_format = image_path.suffix.lower().lstrip(".")
    class_name = _infer_class_name(relative_path)
    directory_depth = max(len(relative_path.parts) - 1, 0)

    width: int | None = None
    height: int | None = None
    is_corrupted = False
    perceptual_hash: str | None = None

    try:
        with Image.open(image_path) as img:
            img.load()
            width, height = img.size
            if img.format:
                image_format = img.format.lower()
            perceptual_hash = str(imagehash.phash(img))
    except (UnidentifiedImageError, OSError, ValueError, imagehash.ImageHashError) as exc:
        logger.debug("Corrupted or unreadable image %s: %s", image_path, exc)
        is_corrupted = True

    return ImageAuditInfo(
        path=image_path,
        relative_path=relative_path,
        dataset_name=dataset_name,
        class_name=class_name,
        width=width,
        height=height,
        image_format=image_format,
        is_corrupted=is_corrupted,
        perceptual_hash=perceptual_hash,
        directory_depth=directory_depth,
    )


def _is_ignored_file(path: Path) -> bool:
    """Return whether a file should be excluded from non-image reporting."""
    return path.name.lower() in IGNORED_FILENAMES or path.name.startswith(".")


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def find_empty_folders(dataset_root: Path) -> list[str]:
    """Find directories that contain no files at any depth.

    Args:
        dataset_root: Root directory of the dataset.

    Returns:
        Sorted relative paths of empty directories.
    """
    empty: list[str] = []
    for directory in sorted(dataset_root.rglob("*")):
        if not directory.is_dir():
            continue
        if not any(directory.iterdir()):
            empty.append(str(directory.relative_to(dataset_root)))
    return empty


def find_non_image_files(dataset_root: Path) -> list[str]:
    """Find files that are not supported image formats.

    Args:
        dataset_root: Root directory of the dataset.

    Returns:
        Sorted relative paths of non-image files.
    """
    non_images: list[str] = []
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file() or _is_ignored_file(path):
            continue
        if path.suffix.lower().lstrip(".") not in SUPPORTED_EXTENSIONS:
            non_images.append(str(path.relative_to(dataset_root)))
    return non_images


def calculate_resolution_stats(images: list[ImageAuditInfo]) -> ResolutionStats | None:
    """Calculate resolution statistics from successfully read images.

    Args:
        images: Audited image records.

    Returns:
        A :class:`ResolutionStats` instance, or ``None`` if no valid images.
    """
    valid = [img for img in images if not img.is_corrupted and img.width and img.height]
    if not valid:
        return None

    widths = pd.Series([img.width for img in valid], dtype="float64")
    heights = pd.Series([img.height for img in valid], dtype="float64")
    megapixels = (widths * heights) / 1_000_000

    return ResolutionStats(
        count=len(valid),
        width_min=int(widths.min()),
        width_max=int(widths.max()),
        width_mean=float(widths.mean()),
        width_median=float(widths.median()),
        height_min=int(heights.min()),
        height_max=int(heights.max()),
        height_mean=float(heights.mean()),
        height_median=float(heights.median()),
        megapixels_min=float(megapixels.min()),
        megapixels_max=float(megapixels.max()),
        megapixels_mean=float(megapixels.mean()),
        megapixels_median=float(megapixels.median()),
    )


def calculate_format_stats(images: list[ImageAuditInfo]) -> dict[str, int]:
    """Count images per detected format extension.

    Args:
        images: Audited image records.

    Returns:
        Mapping of format name to image count.
    """
    counts: dict[str, int] = {}
    for img in images:
        counts[img.image_format] = counts.get(img.image_format, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def calculate_class_imbalance(images_per_class: dict[str, int]) -> ClassImbalanceStats | None:
    """Calculate class imbalance metrics from per-class counts.

    Args:
        images_per_class: Mapping of class label to image count.

    Returns:
        A :class:`ClassImbalanceStats` instance, or ``None`` if no classes.
    """
    if not images_per_class:
        return None

    series = pd.Series(images_per_class)
    majority_class = str(series.idxmax())
    minority_class = str(series.idxmin())

    mean_count = float(series.mean())
    std_count = float(series.std(ddof=0))
    coefficient_of_variation = std_count / mean_count if mean_count > 0 else 0.0

    return ClassImbalanceStats(
        majority_class=majority_class,
        majority_count=int(series.max()),
        minority_class=minority_class,
        minority_count=int(series.min()),
        imbalance_ratio=float(series.max() / series.min()) if series.min() > 0 else float("inf"),
        coefficient_of_variation=coefficient_of_variation,
    )


def detect_duplicate_hashes(images: list[ImageAuditInfo]) -> list[list[str]]:
    """Detect perceptually duplicate images using perceptual hashing.

    Images are grouped by hash value. Only groups with two or more members are
    returned. No files are removed or modified.

    Args:
        images: Audited image records.

    Returns:
        A list of path groups that share the same perceptual hash.
    """
    hash_map: dict[str, list[str]] = {}
    for img in images:
        if img.perceptual_hash is None:
            continue
        hash_map.setdefault(img.perceptual_hash, []).append(str(img.relative_path))

    return [sorted(paths) for paths in hash_map.values() if len(paths) > 1]


def detect_duplicate_filenames(images: list[ImageAuditInfo]) -> dict[str, list[str]]:
    """Detect filenames that appear more than once within a dataset.

    Args:
        images: Audited image records.

    Returns:
        Mapping of filename to relative paths (only duplicates included).
    """
    name_map: dict[str, list[str]] = {}
    for img in images:
        name_map.setdefault(img.path.name, []).append(str(img.relative_path))

    return {
        filename: sorted(paths)
        for filename, paths in sorted(name_map.items())
        if len(paths) > 1
    }


# ---------------------------------------------------------------------------
# Per-dataset and full audit
# ---------------------------------------------------------------------------


def audit_dataset(dataset_path: Path, dataset_name: str | None = None) -> DatasetAuditResult:
    """Run a read-only audit on a single dataset directory.

    Args:
        dataset_path: Root path of the dataset.
        dataset_name: Optional display name; defaults to directory name.

    Returns:
        A :class:`DatasetAuditResult` with full audit details.
    """
    resolved = dataset_path.resolve()
    name = dataset_name or (
        "_root_"
        if resolved.name in {Path(DEFAULT_RAW_DIR).name, Path(DEFAULT_EXTERNAL_DIR).name}
        else resolved.name
    )
    logger.info("Auditing dataset '%s' at %s", name, resolved)

    images: list[ImageAuditInfo] = []
    for path in sorted(resolved.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower().lstrip(".") not in SUPPORTED_EXTENSIONS:
            continue
        images.append(_read_image_metadata(path, name, resolved))

    images_per_class: dict[str, int] = {}
    for img in images:
        images_per_class[img.class_name] = images_per_class.get(img.class_name, 0) + 1

    class_names = sorted(images_per_class.keys())
    corrupted_images = sorted(
        str(img.relative_path) for img in images if img.is_corrupted
    )
    max_depth = max((img.directory_depth for img in images), default=0)

    result = DatasetAuditResult(
        dataset_name=name,
        dataset_path=resolved,
        total_images=len(images),
        total_classes=len(class_names),
        class_names=class_names,
        images_per_class=images_per_class,
        class_imbalance=calculate_class_imbalance(images_per_class),
        empty_folders=find_empty_folders(resolved),
        non_image_files=find_non_image_files(resolved),
        corrupted_images=corrupted_images,
        resolution_stats=calculate_resolution_stats(images),
        format_stats=calculate_format_stats(images),
        duplicate_hash_groups=detect_duplicate_hashes(images),
        duplicate_filenames=detect_duplicate_filenames(images),
        max_directory_depth=max_depth,
        images=images,
    )

    logger.info(
        "Dataset '%s': %d images, %d classes, %d corrupted",
        name,
        result.total_images,
        result.total_classes,
        len(result.corrupted_images),
    )
    return result


def audit_all_datasets(
    data_dir: Path | str = DEFAULT_EXTERNAL_DIR,
    *,
    include_root_if_images: bool = False,
) -> AuditSummary:
    """Audit every dataset discovered under a data root directory.

    Args:
        data_dir: Root directory containing dataset folders.
        include_root_if_images: Whether to treat ``data_dir`` itself as a
            dataset when it contains images directly.

    Returns:
        An :class:`AuditSummary` aggregating all per-dataset results.
    """
    source_path = Path(data_dir).resolve()
    dataset_paths = discover_datasets(
        source_path,
        include_root_if_images=include_root_if_images,
    )

    datasets: dict[str, DatasetAuditResult] = {}
    for dataset_path in dataset_paths:
        if dataset_path.resolve() == source_path:
            result = audit_dataset(dataset_path, dataset_name="_root_")
        else:
            result = audit_dataset(dataset_path)
        datasets[result.dataset_name] = result

    aggregate_images_per_class: dict[str, int] = {}
    aggregate_format_stats: dict[str, int] = {}

    for result in datasets.values():
        for class_name, count in result.images_per_class.items():
            key = f"{result.dataset_name}/{class_name}"
            aggregate_images_per_class[key] = count
        for fmt, count in result.format_stats.items():
            aggregate_format_stats[fmt] = aggregate_format_stats.get(fmt, 0) + count

    return AuditSummary(
        source_dir=source_path,
        generated_at=datetime.now(timezone.utc).isoformat(),
        datasets=datasets,
        aggregate_images_per_class=aggregate_images_per_class,
        aggregate_format_stats=aggregate_format_stats,
        total_images=sum(result.total_images for result in datasets.values()),
        total_classes=len(aggregate_images_per_class),
    )


def _serialize_audit_summary(summary: AuditSummary) -> dict:
    """Convert an audit summary to a JSON-serializable dictionary."""

    def dataset_to_dict(result: DatasetAuditResult) -> dict:
        payload = asdict(result)
        payload.pop("images", None)
        payload["dataset_path"] = str(result.dataset_path)
        return payload

    return {
        "source_dir": str(summary.source_dir),
        "generated_at": summary.generated_at,
        "total_images": summary.total_images,
        "total_classes": summary.total_classes,
        "aggregate_images_per_class": summary.aggregate_images_per_class,
        "aggregate_format_stats": summary.aggregate_format_stats,
        "datasets": {
            name: dataset_to_dict(result) for name, result in summary.datasets.items()
        },
    }


def run_dataset_audit(
    data_dir: Path | str = DEFAULT_RAW_DIR,
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
    *,
    include_root_if_images: bool = True,
) -> AuditSummary:
    """Run the full read-only audit pipeline and write per-dataset reports.

    Discovers datasets, performs analysis, generates visualizations, and saves
    markdown, JSON, and PNG outputs under ``reports/``.

    Args:
        data_dir: Root directory containing datasets to audit.
        reports_dir: Output directory for audit artifacts.
        include_root_if_images: Whether to audit ``data_dir`` when it contains
            images directly (typically enabled for ``datasets/raw``).

    Returns:
        The completed :class:`AuditSummary`.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    logger.info("Starting read-only dataset audit for %s", data_dir)
    summary = audit_all_datasets(
        data_dir,
        include_root_if_images=include_root_if_images,
    )

    statistics = _serialize_audit_summary(summary)
    statistics_path = reports_path / "dataset_statistics.json"
    statistics_path.write_text(
        json.dumps(statistics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Wrote %s", statistics_path)

    from src.data.eda import generate_audit_outputs

    generate_audit_outputs(summary, reports_path)
    logger.info("Dataset audit complete. Reports saved to %s", reports_path)
    return summary


def run_external_dataset_audit(
    external_dir: Path | str = DEFAULT_EXTERNAL_DIR,
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
    comparison_dir: Path | str = DEFAULT_COMPARISON_DIR,
) -> AuditSummary:
    """Audit all external datasets and generate cross-dataset comparison reports.

    Performs read-only per-dataset audits for every folder in
    ``datasets/external/``, writes standard audit artifacts under ``reports/``,
    and writes comparison outputs under ``reports/comparison/``.

    Args:
        external_dir: Root directory containing ingested datasets.
        reports_dir: Output directory for per-dataset audit artifacts.
        comparison_dir: Output directory for cross-dataset comparison artifacts.

    Returns:
        The completed :class:`AuditSummary`.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    logger.info("Starting read-only external dataset audit for %s", external_dir)
    summary = audit_all_datasets(external_dir, include_root_if_images=False)

    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    statistics = _serialize_audit_summary(summary)
    statistics_path = reports_path / "dataset_statistics.json"
    statistics_path.write_text(
        json.dumps(statistics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Wrote %s", statistics_path)

    from src.data.dataset_comparison import build_comparison_report, save_comparison_json
    from src.data.eda import generate_audit_outputs, generate_comparison_outputs

    generate_audit_outputs(summary, reports_path)

    comparison_report = build_comparison_report(summary)
    comparison_path = Path(comparison_dir)
    save_comparison_json(comparison_report, comparison_path / "dataset_comparison.json")
    generate_comparison_outputs(summary, comparison_report, comparison_path)

    logger.info(
        "External dataset audit complete. Reports: %s | Comparison: %s",
        reports_path,
        comparison_path,
    )
    return summary


if __name__ == "__main__":
    run_external_dataset_audit()
