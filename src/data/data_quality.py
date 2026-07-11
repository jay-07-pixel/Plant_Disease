"""Read-only dataset cleaning and quality control.

Scans ingested datasets under ``datasets/external/`` to assess image quality
before preprocessing or merging. No files are deleted, renamed, resized, or
modified.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import cv2
import imagehash
import pandas as pd
from PIL import Image, UnidentifiedImageError

from src.config.dataset_schema import VALID_IMAGE_EXTENSIONS
from src.data.dataset_audit import DEFAULT_EXTERNAL_DIR, discover_external_datasets

logger = logging.getLogger(__name__)

DEFAULT_REPORTS_DIR = Path("reports")
DEFAULT_JSON_PATH = DEFAULT_REPORTS_DIR / "data_quality.json"
DEFAULT_MARKDOWN_PATH = DEFAULT_REPORTS_DIR / "data_quality.md"

SUPPORTED_EXTENSIONS = VALID_IMAGE_EXTENSIONS
IGNORED_FILENAMES = frozenset({".gitkeep", ".ds_store", "thumbs.db", "desktop.ini"})

# Default quality thresholds (read-only inspection parameters).
DEFAULT_MIN_WIDTH = 32
DEFAULT_MIN_HEIGHT = 32
DEFAULT_BLUR_LAPLACIAN_THRESHOLD = 100.0
DEFAULT_MIN_ASPECT_RATIO = 0.2
DEFAULT_MAX_ASPECT_RATIO = 5.0
MAX_ISSUE_PATHS_IN_JSON = 500


class QualityIssueType(str, Enum):
    """Categories of detected image quality issues."""

    CORRUPTED = "corrupted"
    CANNOT_OPEN = "cannot_open"
    UNSUPPORTED_FORMAT = "unsupported_format"
    TOO_SMALL = "too_small"
    INVALID_ASPECT_RATIO = "invalid_aspect_ratio"
    BLURRY = "blurry"
    DUPLICATE = "duplicate"


@dataclass(frozen=True)
class QualityThresholds:
    """Configurable thresholds for quality inspection.

    Attributes:
        min_width: Minimum acceptable image width in pixels.
        min_height: Minimum acceptable image height in pixels.
        blur_laplacian_threshold: Images with variance below this are flagged blurry.
        min_aspect_ratio: Minimum width/height ratio (portrait limit).
        max_aspect_ratio: Maximum width/height ratio (landscape limit).
    """

    min_width: int = DEFAULT_MIN_WIDTH
    min_height: int = DEFAULT_MIN_HEIGHT
    blur_laplacian_threshold: float = DEFAULT_BLUR_LAPLACIAN_THRESHOLD
    min_aspect_ratio: float = DEFAULT_MIN_ASPECT_RATIO
    max_aspect_ratio: float = DEFAULT_MAX_ASPECT_RATIO


@dataclass
class ImageQualityRecord:
    """Read-only quality assessment for a single image.

    Attributes:
        path: Absolute path to the image file.
        relative_path: Path relative to the dataset root.
        dataset_name: Parent dataset directory name.
        width: Image width in pixels, or ``None`` if unreadable.
        height: Image height in pixels, or ``None`` if unreadable.
        aspect_ratio: Width divided by height, or ``None`` if unavailable.
        image_format: Detected or inferred format extension.
        laplacian_variance: OpenCV Laplacian variance blur metric.
        perceptual_hash: Perceptual hash string when computable.
        issues: Detected quality issue types.
        is_good: Whether the image passed all quality checks.
    """

    path: Path
    relative_path: Path
    dataset_name: str
    width: int | None
    height: int | None
    aspect_ratio: float | None
    image_format: str
    laplacian_variance: float | None
    perceptual_hash: str | None
    issues: list[QualityIssueType] = field(default_factory=list)
    is_good: bool = True


@dataclass
class QualityStatistics:
    """Aggregate quality statistics for a dataset or full scan.

    Attributes:
        total_images: Total image files scanned.
        good_images: Images with no detected quality issues.
        suspected_blurry: Images flagged as blurry.
        duplicate_images: Images participating in duplicate hash groups.
        corrupted_images: Corrupted or unreadable images.
        small_images: Images below minimum width or height.
        invalid_aspect_ratio_images: Images outside acceptable aspect ratios.
        unsupported_format_files: Files with unsupported extensions.
        empty_folders: Count of empty directories.
        cannot_open_images: Images that could not be opened.
    """

    total_images: int = 0
    good_images: int = 0
    suspected_blurry: int = 0
    duplicate_images: int = 0
    corrupted_images: int = 0
    small_images: int = 0
    invalid_aspect_ratio_images: int = 0
    unsupported_format_files: int = 0
    empty_folders: int = 0
    cannot_open_images: int = 0


@dataclass
class DatasetQualityResult:
    """Quality assessment results for one dataset.

    Attributes:
        dataset_name: Dataset directory name.
        dataset_path: Absolute path to the dataset root.
        statistics: Aggregate quality statistics.
        records: Per-image quality records.
        duplicate_hash_groups: Groups of perceptually duplicate images.
        empty_folders: Relative paths of empty directories.
        unsupported_format_files: Relative paths of unsupported files.
    """

    dataset_name: str
    dataset_path: Path
    statistics: QualityStatistics
    records: list[ImageQualityRecord]
    duplicate_hash_groups: list[list[str]] = field(default_factory=list)
    empty_folders: list[str] = field(default_factory=list)
    unsupported_format_files: list[str] = field(default_factory=list)


@dataclass
class DataQualityReport:
    """Full read-only quality report across external datasets.

    Attributes:
        generated_at: UTC ISO-8601 timestamp.
        source_dir: Root directory scanned.
        thresholds: Thresholds used for inspection.
        statistics: Aggregate statistics across all datasets.
        datasets: Per-dataset quality results keyed by dataset name.
    """

    generated_at: str
    source_dir: Path
    thresholds: QualityThresholds
    statistics: QualityStatistics
    datasets: dict[str, DatasetQualityResult]


# ---------------------------------------------------------------------------
# Discovery helpers
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


def find_unsupported_format_files(dataset_root: Path) -> list[str]:
    """Find non-image files that are not ignored system artifacts.

    Args:
        dataset_root: Root directory of the dataset.

    Returns:
        Sorted relative paths of unsupported files.
    """
    unsupported: list[str] = []
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.lower() in IGNORED_FILENAMES or path.name.startswith("."):
            continue
        if path.suffix.lower().lstrip(".") not in SUPPORTED_EXTENSIONS:
            unsupported.append(str(path.relative_to(dataset_root)))
    return unsupported


def iter_image_files(dataset_root: Path) -> list[Path]:
    """List supported image files under a dataset root.

    Args:
        dataset_root: Root directory of the dataset.

    Returns:
        Sorted list of image file paths.
    """
    images: list[Path] = []
    for path in sorted(dataset_root.rglob("*")):
        if path.is_file() and path.suffix.lower().lstrip(".") in SUPPORTED_EXTENSIONS:
            images.append(path)
    return images


# ---------------------------------------------------------------------------
# Read-only quality checks
# ---------------------------------------------------------------------------


def compute_aspect_ratio(width: int, height: int) -> float:
    """Compute width-to-height aspect ratio."""
    return width / height if height > 0 else 0.0


def compute_laplacian_variance(image_path: Path) -> float | None:
    """Compute Laplacian variance as a blur metric using OpenCV.

    Lower values typically indicate blurrier images. The file is read only.

    Args:
        image_path: Path to the image file.

    Returns:
        Laplacian variance, or ``None`` if the image could not be read.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _check_size(
    width: int | None,
    height: int | None,
    thresholds: QualityThresholds,
) -> QualityIssueType | None:
    """Return a size issue type when dimensions are too small."""
    if width is None or height is None:
        return None
    if width < thresholds.min_width or height < thresholds.min_height:
        return QualityIssueType.TOO_SMALL
    return None


def _check_aspect_ratio(
    width: int | None,
    height: int | None,
    thresholds: QualityThresholds,
) -> QualityIssueType | None:
    """Return an aspect-ratio issue when outside configured bounds."""
    if width is None or height is None or height == 0:
        return None
    ratio = compute_aspect_ratio(width, height)
    if ratio < thresholds.min_aspect_ratio or ratio > thresholds.max_aspect_ratio:
        return QualityIssueType.INVALID_ASPECT_RATIO
    return None


def _check_blur(
    laplacian_variance: float | None,
    thresholds: QualityThresholds,
) -> QualityIssueType | None:
    """Return a blur issue when Laplacian variance is below threshold."""
    if laplacian_variance is None:
        return None
    if laplacian_variance < thresholds.blur_laplacian_threshold:
        return QualityIssueType.BLURRY
    return None


def inspect_image_quality(
    image_path: Path,
    dataset_name: str,
    dataset_root: Path,
    thresholds: QualityThresholds,
) -> ImageQualityRecord:
    """Run read-only quality checks on a single image.

    Args:
        image_path: Absolute path to the image.
        dataset_name: Name of the parent dataset.
        dataset_root: Root directory of the dataset.
        thresholds: Quality thresholds to apply.

    Returns:
        An :class:`ImageQualityRecord` with detected issues.
    """
    relative_path = image_path.relative_to(dataset_root)
    image_format = image_path.suffix.lower().lstrip(".")
    issues: list[QualityIssueType] = []

    width: int | None = None
    height: int | None = None
    aspect_ratio: float | None = None
    perceptual_hash: str | None = None
    laplacian_variance: float | None = None

    try:
        with Image.open(image_path) as img:
            img.load()
            width, height = img.size
            aspect_ratio = compute_aspect_ratio(width, height)
            if img.format:
                image_format = img.format.lower()
            perceptual_hash = str(imagehash.phash(img))
    except (UnidentifiedImageError, OSError, ValueError, TypeError):
        issues.extend([QualityIssueType.CORRUPTED, QualityIssueType.CANNOT_OPEN])
        return ImageQualityRecord(
            path=image_path,
            relative_path=relative_path,
            dataset_name=dataset_name,
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
            image_format=image_format,
            laplacian_variance=None,
            perceptual_hash=None,
            issues=_dedupe_issues(issues),
            is_good=False,
        )

    laplacian_variance = compute_laplacian_variance(image_path)
    if laplacian_variance is None:
        issues.append(QualityIssueType.CANNOT_OPEN)

    for check in (
        _check_size(width, height, thresholds),
        _check_aspect_ratio(width, height, thresholds),
        _check_blur(laplacian_variance, thresholds),
    ):
        if check is not None:
            issues.append(check)

    issues = _dedupe_issues(issues)
    return ImageQualityRecord(
        path=image_path,
        relative_path=relative_path,
        dataset_name=dataset_name,
        width=width,
        height=height,
        aspect_ratio=aspect_ratio,
        image_format=image_format,
        laplacian_variance=laplacian_variance,
        perceptual_hash=perceptual_hash,
        issues=issues,
        is_good=len(issues) == 0,
    )


def _dedupe_issues(issues: list[QualityIssueType]) -> list[QualityIssueType]:
    """Preserve issue order while removing duplicates."""
    seen: set[QualityIssueType] = set()
    unique: list[QualityIssueType] = []
    for issue in issues:
        if issue not in seen:
            seen.add(issue)
            unique.append(issue)
    return unique


def detect_duplicate_hash_groups(
    records: list[ImageQualityRecord],
) -> tuple[list[list[str]], set[str]]:
    """Detect perceptual duplicate groups within a dataset.

    Args:
        records: Image quality records with perceptual hashes.

    Returns:
        Tuple of duplicate path groups and the set of paths flagged as duplicates.
    """
    hash_map: dict[str, list[str]] = {}
    for record in records:
        if record.perceptual_hash is None:
            continue
        hash_map.setdefault(record.perceptual_hash, []).append(str(record.relative_path))

    groups = [sorted(paths) for paths in hash_map.values() if len(paths) > 1]
    duplicate_paths: set[str] = set()
    for group in groups:
        duplicate_paths.update(group)

    return groups, duplicate_paths


def apply_duplicate_flags(
    records: list[ImageQualityRecord],
    duplicate_paths: set[str],
) -> None:
    """Mark duplicate images in quality records (in place).

    Args:
        records: Image quality records to update.
        duplicate_paths: Relative paths belonging to duplicate hash groups.
    """
    for record in records:
        if str(record.relative_path) not in duplicate_paths:
            continue
        if QualityIssueType.DUPLICATE not in record.issues:
            record.issues.append(QualityIssueType.DUPLICATE)
            record.is_good = False


# ---------------------------------------------------------------------------
# Dataset and aggregate scanning
# ---------------------------------------------------------------------------


def _count_issue(records: list[ImageQualityRecord], issue: QualityIssueType) -> int:
    """Count records containing a specific issue type."""
    return sum(1 for record in records if issue in record.issues)


def compute_blur_variance_summary(records: list[ImageQualityRecord]) -> dict[str, float]:
    """Compute Laplacian variance summary statistics using pandas.

    Args:
        records: Image quality records with blur metrics.

    Returns:
        Summary statistics dictionary, or empty when no metrics exist.
    """
    values = [
        record.laplacian_variance
        for record in records
        if record.laplacian_variance is not None
    ]
    if not values:
        return {}

    series = pd.Series(values, dtype="float64")
    return {
        "count": float(series.count()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def compute_quality_statistics(
    records: list[ImageQualityRecord],
    *,
    empty_folders: int = 0,
    unsupported_format_files: int = 0,
) -> QualityStatistics:
    """Compute aggregate statistics from image quality records.

    Args:
        records: Per-image quality records.
        empty_folders: Number of empty directories in the dataset.
        unsupported_format_files: Number of unsupported non-image files.

    Returns:
        A :class:`QualityStatistics` summary.
    """
    return QualityStatistics(
        total_images=len(records),
        good_images=sum(1 for record in records if record.is_good),
        suspected_blurry=_count_issue(records, QualityIssueType.BLURRY),
        duplicate_images=_count_issue(records, QualityIssueType.DUPLICATE),
        corrupted_images=_count_issue(records, QualityIssueType.CORRUPTED),
        small_images=_count_issue(records, QualityIssueType.TOO_SMALL),
        invalid_aspect_ratio_images=_count_issue(
            records, QualityIssueType.INVALID_ASPECT_RATIO
        ),
        unsupported_format_files=unsupported_format_files,
        empty_folders=empty_folders,
        cannot_open_images=_count_issue(records, QualityIssueType.CANNOT_OPEN),
    )


def scan_dataset_quality(
    dataset_path: Path,
    thresholds: QualityThresholds,
    dataset_name: str | None = None,
) -> DatasetQualityResult:
    """Run read-only quality control on one external dataset.

    Args:
        dataset_path: Root path of the dataset.
        thresholds: Quality thresholds to apply.
        dataset_name: Optional display name; defaults to directory name.

    Returns:
        A :class:`DatasetQualityResult` for the dataset.
    """
    resolved = dataset_path.resolve()
    name = dataset_name or resolved.name
    logger.info("Scanning quality for dataset '%s' at %s", name, resolved)

    empty_folders = find_empty_folders(resolved)
    unsupported_files = find_unsupported_format_files(resolved)
    image_paths = iter_image_files(resolved)

    records: list[ImageQualityRecord] = []
    for index, image_path in enumerate(image_paths, start=1):
        records.append(inspect_image_quality(image_path, name, resolved, thresholds))
        if index % 5000 == 0:
            logger.info("Dataset '%s': inspected %d/%d images", name, index, len(image_paths))

    duplicate_groups, duplicate_paths = detect_duplicate_hash_groups(records)
    apply_duplicate_flags(records, duplicate_paths)

    statistics = compute_quality_statistics(
        records,
        empty_folders=len(empty_folders),
        unsupported_format_files=len(unsupported_files),
    )

    logger.info(
        "Dataset '%s': %d images, %d good, %d blurry, %d duplicates, %d corrupted",
        name,
        statistics.total_images,
        statistics.good_images,
        statistics.suspected_blurry,
        statistics.duplicate_images,
        statistics.corrupted_images,
    )

    return DatasetQualityResult(
        dataset_name=name,
        dataset_path=resolved,
        statistics=statistics,
        records=records,
        duplicate_hash_groups=duplicate_groups,
        empty_folders=empty_folders,
        unsupported_format_files=unsupported_files,
    )


def _merge_statistics(parts: list[QualityStatistics]) -> QualityStatistics:
    """Merge multiple statistics objects by summing counts."""
    if not parts:
        return QualityStatistics()

    totals = QualityStatistics()
    for part in parts:
        totals.total_images += part.total_images
        totals.good_images += part.good_images
        totals.suspected_blurry += part.suspected_blurry
        totals.duplicate_images += part.duplicate_images
        totals.corrupted_images += part.corrupted_images
        totals.small_images += part.small_images
        totals.invalid_aspect_ratio_images += part.invalid_aspect_ratio_images
        totals.unsupported_format_files += part.unsupported_format_files
        totals.empty_folders += part.empty_folders
        totals.cannot_open_images += part.cannot_open_images
    return totals


def scan_all_dataset_quality(
    external_dir: Path | str = DEFAULT_EXTERNAL_DIR,
    thresholds: QualityThresholds | None = None,
) -> DataQualityReport:
    """Scan quality for every dataset under ``datasets/external/``.

    Args:
        external_dir: Root directory containing ingested datasets.
        thresholds: Optional quality thresholds; defaults apply when omitted.

    Returns:
        A :class:`DataQualityReport` aggregating all dataset scans.
    """
    thresholds = thresholds or QualityThresholds()
    source_path = Path(external_dir).resolve()
    dataset_paths = discover_external_datasets(source_path)

    datasets: dict[str, DatasetQualityResult] = {}
    for dataset_path in dataset_paths:
        result = scan_dataset_quality(dataset_path, thresholds)
        datasets[result.dataset_name] = result

    aggregate = _merge_statistics([result.statistics for result in datasets.values()])

    return DataQualityReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        source_dir=source_path,
        thresholds=thresholds,
        statistics=aggregate,
        datasets=datasets,
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _issue_paths(
    records: list[ImageQualityRecord],
    issue: QualityIssueType,
    *,
    limit: int = MAX_ISSUE_PATHS_IN_JSON,
) -> dict[str, int | list[str]]:
    """Build a JSON-friendly issue path listing with total count."""
    paths = sorted(str(record.relative_path) for record in records if issue in record.issues)
    return {
        "total_count": len(paths),
        "sample_paths": paths[:limit],
    }


def serialize_quality_report(report: DataQualityReport) -> dict:
    """Convert a quality report to a JSON-serializable dictionary.

    Args:
        report: Quality report to serialize.

    Returns:
        Dictionary suitable for JSON export.
    """
    datasets_payload: dict[str, dict] = {}

    all_records: list[ImageQualityRecord] = []
    for name, result in sorted(report.datasets.items()):
        all_records.extend(result.records)
        records = result.records
        datasets_payload[name] = {
            "dataset_name": result.dataset_name,
            "dataset_path": str(result.dataset_path),
            "statistics": asdict(result.statistics),
            "blur_variance_summary": compute_blur_variance_summary(records),
            "empty_folders": {
                "total_count": len(result.empty_folders),
                "sample_paths": result.empty_folders[:MAX_ISSUE_PATHS_IN_JSON],
            },
            "unsupported_format_files": {
                "total_count": len(result.unsupported_format_files),
                "sample_paths": result.unsupported_format_files[:MAX_ISSUE_PATHS_IN_JSON],
            },
            "duplicate_hash_groups": {
                "total_groups": len(result.duplicate_hash_groups),
                "groups": result.duplicate_hash_groups[:100],
            },
            "issues": {
                "corrupted": _issue_paths(records, QualityIssueType.CORRUPTED),
                "cannot_open": _issue_paths(records, QualityIssueType.CANNOT_OPEN),
                "blurry": _issue_paths(records, QualityIssueType.BLURRY),
                "too_small": _issue_paths(records, QualityIssueType.TOO_SMALL),
                "invalid_aspect_ratio": _issue_paths(
                    records, QualityIssueType.INVALID_ASPECT_RATIO
                ),
                "duplicate": _issue_paths(records, QualityIssueType.DUPLICATE),
            },
        }

    return {
        "generated_at": report.generated_at,
        "source_dir": str(report.source_dir),
        "thresholds": asdict(report.thresholds),
        "statistics": asdict(report.statistics),
        "blur_variance_summary": compute_blur_variance_summary(all_records),
        "datasets": datasets_payload,
    }


def save_quality_json(
    report: DataQualityReport,
    output_path: Path | str = DEFAULT_JSON_PATH,
) -> Path:
    """Save the quality report as JSON.

    Args:
        report: Quality report to export.
        output_path: Destination JSON path.

    Returns:
        Path the file was written to.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(serialize_quality_report(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved data quality JSON to %s", path)
    return path


def _format_statistics_markdown(stats: QualityStatistics) -> list[str]:
    """Format quality statistics as markdown bullet points."""
    return [
        f"- **Total images:** {stats.total_images}",
        f"- **Good images:** {stats.good_images}",
        f"- **Suspected blurry images:** {stats.suspected_blurry}",
        f"- **Duplicate images:** {stats.duplicate_images}",
        f"- **Corrupted images:** {stats.corrupted_images}",
        f"- **Small images:** {stats.small_images}",
        f"- **Invalid aspect ratio images:** {stats.invalid_aspect_ratio_images}",
        f"- **Cannot open images:** {stats.cannot_open_images}",
        f"- **Unsupported format files:** {stats.unsupported_format_files}",
        f"- **Empty folders:** {stats.empty_folders}",
    ]


def generate_quality_markdown(
    report: DataQualityReport,
    output_path: Path | str = DEFAULT_MARKDOWN_PATH,
) -> Path:
    """Write a human-readable data quality markdown report.

    Args:
        report: Quality report to document.
        output_path: Destination markdown path.

    Returns:
        Path the file was written to.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Data Quality Report",
        "",
        f"**Generated:** {report.generated_at}  ",
        f"**Source directory:** `{report.source_dir}`  ",
        "",
        "> Read-only quality inspection. No images were deleted, renamed, or modified.",
        "",
        "## Inspection Thresholds",
        "",
        f"- Minimum size: {report.thresholds.min_width} × {report.thresholds.min_height} px",
        f"- Blur Laplacian threshold: {report.thresholds.blur_laplacian_threshold}",
        f"- Aspect ratio range: {report.thresholds.min_aspect_ratio} – "
        f"{report.thresholds.max_aspect_ratio}",
        "",
        "## Aggregate Quality Statistics",
        "",
        *_format_statistics_markdown(report.statistics),
        "",
    ]

    if not report.datasets:
        lines.extend(
            [
                "## No Datasets Found",
                "",
                "No datasets were discovered under `datasets/external/`.",
                "",
            ]
        )
    else:
        for name, result in sorted(report.datasets.items()):
            lines.extend(
                [
                    f"## Dataset: `{name}`",
                    "",
                    f"**Path:** `{result.dataset_path}`  ",
                    "",
                    "### Quality Statistics",
                    "",
                    *_format_statistics_markdown(result.statistics),
                    "",
                    "### Duplicate Groups",
                    "",
                    f"- Perceptual duplicate groups: {len(result.duplicate_hash_groups)}",
                    "",
                ]
            )

            if result.duplicate_hash_groups:
                lines.append("**Sample duplicate groups (max 5):**\n")
                for group in result.duplicate_hash_groups[:5]:
                    lines.append(f"- {len(group)} images: {', '.join(group[:3])}"
                                 f"{'...' if len(group) > 3 else ''}")
                lines.append("")

            if result.empty_folders:
                lines.extend(["### Empty Folders", ""])
                for folder in result.empty_folders[:10]:
                    lines.append(f"- `{folder}`")
                if len(result.empty_folders) > 10:
                    lines.append(f"- ... and {len(result.empty_folders) - 10} more")
                lines.append("")

            if result.unsupported_format_files:
                lines.extend(["### Unsupported Format Files", ""])
                for file_path in result.unsupported_format_files[:10]:
                    lines.append(f"- `{file_path}`")
                if len(result.unsupported_format_files) > 10:
                    lines.append(
                        f"- ... and {len(result.unsupported_format_files) - 10} more"
                    )
                lines.append("")

            lines.extend(["### Issue Samples", ""])
            for issue_type, label in (
                (QualityIssueType.CORRUPTED, "Corrupted"),
                (QualityIssueType.BLURRY, "Blurry"),
                (QualityIssueType.TOO_SMALL, "Too small"),
                (QualityIssueType.INVALID_ASPECT_RATIO, "Invalid aspect ratio"),
                (QualityIssueType.DUPLICATE, "Duplicate"),
            ):
                sample = [
                    str(record.relative_path)
                    for record in result.records
                    if issue_type in record.issues
                ][:5]
                count = _count_issue(result.records, issue_type)
                lines.append(f"**{label}:** {count}")
                for sample_path in sample:
                    lines.append(f"- `{sample_path}`")
                lines.append("")

            lines.extend(["---", ""])

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved data quality markdown to %s", path)
    return path


def run_data_quality_scan(
    external_dir: Path | str = DEFAULT_EXTERNAL_DIR,
    *,
    thresholds: QualityThresholds | None = None,
    json_path: Path | str = DEFAULT_JSON_PATH,
    markdown_path: Path | str = DEFAULT_MARKDOWN_PATH,
) -> DataQualityReport:
    """Run the full read-only data quality pipeline.

    Scans all external datasets, computes quality statistics, and writes JSON
    and markdown reports under ``reports/``.

    Args:
        external_dir: Root directory containing ingested datasets.
        thresholds: Optional quality thresholds.
        json_path: Output JSON report path.
        markdown_path: Output markdown report path.

    Returns:
        The completed :class:`DataQualityReport`.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    logger.info("Starting read-only data quality scan for %s", external_dir)
    report = scan_all_dataset_quality(external_dir, thresholds=thresholds)
    save_quality_json(report, json_path)
    generate_quality_markdown(report, markdown_path)

    logger.info(
        "Data quality scan complete: %d images, %d good, %d issues flagged",
        report.statistics.total_images,
        report.statistics.good_images,
        report.statistics.total_images - report.statistics.good_images,
    )
    return report


if __name__ == "__main__":
    run_data_quality_scan()
