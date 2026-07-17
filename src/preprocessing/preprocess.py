"""Main image preprocessing pipeline orchestrator."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.prepare_dataset import DEFAULT_METADATA_CSV
from src.preprocessing.augmentation import (
    AugmentationConfig,
    augmentation_config_to_dict,
    get_default_augmentation_config,
)
from src.preprocessing.image_transform import (
    TransformResult,
    save_processed_image,
    transform_image,
    validate_image,
)
from src.preprocessing.preprocessing_config import PreprocessingConfig
from src.preprocessing.split_dataset import (
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    assign_stratified_splits,
    compute_split_statistics,
    split_label_distribution,
)

logger = logging.getLogger(__name__)


@dataclass
class ResizeStatistics:
    """Resize statistics collected during preprocessing.

    Attributes:
        mean_scale_factor: Mean resize scale before padding.
        min_scale_factor: Minimum resize scale observed.
        max_scale_factor: Maximum resize scale observed.
        padded_image_count: Images that required letterbox padding.
        mean_original_width: Mean original image width.
        mean_original_height: Mean original image height.
    """

    mean_scale_factor: float
    min_scale_factor: float
    max_scale_factor: float
    padded_image_count: int
    mean_original_width: float
    mean_original_height: float


@dataclass
class PreprocessingReport:
    """Summary report for a preprocessing pipeline run.

    Attributes:
        generated_at: UTC ISO-8601 timestamp.
        config: Preprocessing configuration used.
        images_processed: Successfully processed image count.
        images_skipped: Images skipped due to errors or existing outputs.
        images_failed: Images that failed validation or processing.
        resize_statistics: Aggregated resize metrics.
        split_statistics: Stratified split statistics.
        augmentation_config: Training augmentation configuration.
        skip_reasons: Counts of skip/failure reasons.
    """

    generated_at: str
    config: dict
    images_processed: int
    images_skipped: int
    images_failed: int
    resize_statistics: ResizeStatistics
    split_statistics: dict
    augmentation_config: dict
    skip_reasons: dict[str, int] = field(default_factory=dict)


def sanitize_label_for_path(canonical_label: str) -> str:
    """Convert a canonical label to a filesystem-safe directory name.

    Args:
        canonical_label: Canonical label key (e.g. ``tomato|early_blight``).

    Returns:
        Safe directory name (e.g. ``tomato_early_blight``).
    """
    return canonical_label.replace("|", "_").replace("/", "_").replace("\\", "_")


def build_processed_image_path(
    config: PreprocessingConfig,
    processed_split: str,
    canonical_label: str,
    source_image_path: str,
) -> Path:
    """Build the output path for a processed image.

    Layout: ``{images_dir}/{split}/{canonical_label}/{hash}.jpg``

    Args:
        config: Preprocessing configuration.
        processed_split: Assigned split (train/val/test).
        canonical_label: Canonical class label.
        source_image_path: Original image path used for deterministic naming.

    Returns:
        Absolute output path for the processed image.
    """
    label_dir = sanitize_label_for_path(canonical_label)
    file_hash = hashlib.md5(source_image_path.encode("utf-8")).hexdigest()[:16]
    extension = config.output.image_format.lower().lstrip(".")

    return (
        config.project_root
        / config.output.images_dir
        / processed_split
        / label_dir
        / f"{file_hash}.{extension}"
    )


def load_source_metadata(config: PreprocessingConfig) -> pd.DataFrame:
    """Load the source metadata CSV.

    Args:
        config: Preprocessing configuration.

    Returns:
        Source metadata DataFrame.
    """
    path = config.project_root / config.source_metadata_path
    if not path.exists():
        raise FileNotFoundError(f"Source metadata not found: {path}")

    df = pd.read_csv(path, encoding="utf-8")
    logger.info("Loaded source metadata from %s (%d rows)", path, len(df))
    return df


def _resolve_source_path(config: PreprocessingConfig, image_path: str) -> Path:
    """Resolve a project-relative source image path to an absolute path."""
    path = Path(image_path)
    if path.is_absolute():
        return path
    return config.project_root / path


def _to_project_relative(config: PreprocessingConfig, path: Path) -> str:
    """Convert an absolute path to a project-relative POSIX path."""
    return path.resolve().relative_to(config.project_root.resolve()).as_posix()


def process_single_image(
    row: pd.Series,
    config: PreprocessingConfig,
) -> dict | None:
    """Process one image and return a processed metadata record.

    Args:
        row: Source metadata row.
        config: Preprocessing configuration.

    Returns:
        Processed metadata dictionary, or ``None`` if skipped/failed.
    """
    source_path = _resolve_source_path(config, str(row["image_path"]))
    processed_split = str(row["processed_split"])
    canonical_label = str(row["canonical_label"])

    output_path = build_processed_image_path(
        config,
        processed_split,
        canonical_label,
        str(row["image_path"]),
    )

    if config.skip_existing and output_path.exists():
        return {
            "source_image_path": str(row["image_path"]),
            "processed_image_path": _to_project_relative(config, output_path),
            "dataset_name": row["dataset_name"],
            "processed_split": processed_split,
            "plant": row["plant"],
            "disease": row["disease"],
            "canonical_label": canonical_label,
            "is_healthy": bool(row["is_healthy"]),
            "original_width": row.get("image_width"),
            "original_height": row.get("image_height"),
            "processed_width": config.target_size.width,
            "processed_height": config.target_size.height,
            "image_format": config.output.image_format,
            "sample_weight": row.get("sample_weight"),
            "status": "skipped_existing",
        }

    validation = validate_image(source_path)
    if not validation.is_valid:
        logger.debug("Validation failed for %s: %s", source_path, validation.error)
        return None

    try:
        transform: TransformResult = transform_image(
            source_path,
            config.target_size,
            padding_color=config.padding_color,
        )
        save_processed_image(
            transform.image,
            output_path,
            image_format=config.output.image_format,
            jpeg_quality=config.output.jpeg_quality,
        )

        return {
            "source_image_path": str(row["image_path"]),
            "processed_image_path": _to_project_relative(config, output_path),
            "dataset_name": row["dataset_name"],
            "processed_split": processed_split,
            "plant": row["plant"],
            "disease": row["disease"],
            "canonical_label": canonical_label,
            "is_healthy": bool(row["is_healthy"]),
            "original_width": transform.original_width,
            "original_height": transform.original_height,
            "processed_width": config.target_size.width,
            "processed_height": config.target_size.height,
            "image_format": config.output.image_format,
            "sample_weight": row.get("sample_weight"),
            "scale_factor": transform.scale_factor,
            "padded": transform.padded,
            "status": "processed",
        }

    except (OSError, ValueError) as exc:
        logger.debug("Processing failed for %s: %s", source_path, exc)
        return None


def compute_resize_statistics(records: list[dict]) -> ResizeStatistics:
    """Compute resize statistics from processed records.

    Args:
        records: Processed metadata records with scale information.

    Returns:
        A :class:`ResizeStatistics` summary.
    """
    processed = [r for r in records if r.get("status") == "processed" and "scale_factor" in r]

    if not processed:
        return ResizeStatistics(
            mean_scale_factor=0.0,
            min_scale_factor=0.0,
            max_scale_factor=0.0,
            padded_image_count=0,
            mean_original_width=0.0,
            mean_original_height=0.0,
        )

    scales = [float(r["scale_factor"]) for r in processed]
    widths = [float(r["original_width"]) for r in processed]
    heights = [float(r["original_height"]) for r in processed]
    padded = sum(1 for r in processed if r.get("padded"))

    return ResizeStatistics(
        mean_scale_factor=float(np.mean(scales)),
        min_scale_factor=float(np.min(scales)),
        max_scale_factor=float(np.max(scales)),
        padded_image_count=padded,
        mean_original_width=float(np.mean(widths)),
        mean_original_height=float(np.mean(heights)),
    )


def save_processed_metadata(
    records: list[dict],
    config: PreprocessingConfig,
) -> Path:
    """Save processed metadata to CSV.

    Args:
        records: Processed metadata records.
        config: Preprocessing configuration.

    Returns:
        Path the CSV was written to.
    """
    output_path = config.project_root / config.output.metadata_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    columns = [
        "source_image_path",
        "processed_image_path",
        "dataset_name",
        "processed_split",
        "plant",
        "disease",
        "canonical_label",
        "is_healthy",
        "original_width",
        "original_height",
        "processed_width",
        "processed_height",
        "image_format",
        "sample_weight",
        "status",
    ]
    for column in columns:
        if column not in df.columns:
            df[column] = None

    df[columns].to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Saved processed metadata to %s (%d rows)", output_path, len(df))
    return output_path


def serialize_preprocessing_report(report: PreprocessingReport) -> dict:
    """Convert a preprocessing report to a JSON-serializable dictionary."""
    return asdict(report)


def save_preprocessing_report_json(
    report: PreprocessingReport,
    output_path: Path,
) -> Path:
    """Save preprocessing report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(serialize_preprocessing_report(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved preprocessing report JSON to %s", output_path)
    return output_path


def generate_preprocessing_report_markdown(
    report: PreprocessingReport,
    output_path: Path,
) -> Path:
    """Write a human-readable preprocessing markdown report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resize = report.resize_statistics
    split = report.split_statistics

    lines = [
        "# Preprocessing Report",
        "",
        f"**Generated:** {report.generated_at}  ",
        "",
        "> Source images in `datasets/external/` were not modified.",
        "",
        "## Summary",
        "",
        f"- **Images processed:** {report.images_processed:,}",
        f"- **Images skipped:** {report.images_skipped:,}",
        f"- **Images failed:** {report.images_failed:,}",
        "",
        "## Configuration",
        "",
        f"- **Target size:** {report.config['target_size']['width']}×"
        f"{report.config['target_size']['height']}",
        f"- **Output format:** {report.config['output']['image_format']}",
        f"- **Split ratios:** train={report.config['split']['train_ratio']:.0%}, "
        f"val={report.config['split']['val_ratio']:.0%}, "
        f"test={report.config['split']['test_ratio']:.0%}",
        f"- **Normalization enabled:** {report.config['normalization']['enabled']}",
        "",
        "## Resize Statistics",
        "",
        f"- **Mean scale factor:** {resize.mean_scale_factor:.4f}",
        f"- **Scale range:** {resize.min_scale_factor:.4f} – {resize.max_scale_factor:.4f}",
        f"- **Images with padding:** {resize.padded_image_count:,}",
        f"- **Mean original size:** {resize.mean_original_width:.0f} × "
        f"{resize.mean_original_height:.0f} px",
        "",
        "## Split Statistics",
        "",
        f"- **Train:** {split['train_count']:,} ({split['train_ratio']:.1%})",
        f"- **Validation:** {split['val_count']:,} ({split['val_ratio']:.1%})",
        f"- **Test:** {split['test_count']:,} ({split['test_ratio']:.1%})",
        f"- **Classes with small splits:** {split['classes_with_small_splits']}",
        "",
        "## Augmentation Configuration (Training Only)",
        "",
        "Augmentation is configured but **not applied** during preprocessing.",
        "",
        f"- **Enabled:** {report.augmentation_config['enabled']}",
        f"- **Target split:** {report.augmentation_config['target_split']}",
        f"- **Random flip:** {report.augmentation_config['random_flip']['enabled']}",
        f"- **Random rotation:** {report.augmentation_config['random_rotation']['enabled']}",
        f"- **Random crop:** {report.augmentation_config['random_crop']['enabled']}",
        f"- **Color jitter:** {report.augmentation_config['color_jitter']['enabled']}",
        f"- **Gaussian noise:** {report.augmentation_config['gaussian_noise']['enabled']}",
        "",
        "## Output Files",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `datasets/processed/images/` | Processed images by split and class |",
        "| `datasets/processed/processed_metadata.csv` | Processed metadata table |",
        "",
    ]

    if report.skip_reasons:
        lines.extend(["## Skip / Failure Reasons", ""])
        for reason, count in sorted(report.skip_reasons.items()):
            lines.append(f"- **{reason}:** {count:,}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved preprocessing report markdown to %s", output_path)
    return output_path


def _config_to_dict(config: PreprocessingConfig) -> dict:
    """Serialize preprocessing config for reports."""
    return {
        "target_size": {
            "width": config.target_size.width,
            "height": config.target_size.height,
        },
        "normalization": asdict(config.normalization),
        "split": asdict(config.split),
        "output": {
            "images_dir": str(config.output.images_dir),
            "metadata_csv": str(config.output.metadata_csv),
            "image_format": config.output.image_format,
            "jpeg_quality": config.output.jpeg_quality,
        },
        "source_metadata_path": str(config.source_metadata_path),
        "padding_color": config.padding_color,
        "skip_existing": config.skip_existing,
    }


def run_preprocessing(
    config: PreprocessingConfig | None = None,
    augmentation_config: AugmentationConfig | None = None,
) -> PreprocessingReport:
    """Run the full image preprocessing pipeline.

    Reads source metadata, assigns stratified splits, processes images into
  ``datasets/processed/images/``, and writes processed metadata and reports.
    Original images in ``datasets/external/`` are never modified.

    Args:
        config: Preprocessing configuration; uses defaults when omitted.
        augmentation_config: Training augmentation configuration for reports.

    Returns:
        A :class:`PreprocessingReport` summarizing the pipeline run.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    config = config or PreprocessingConfig()
    augmentation_config = augmentation_config or get_default_augmentation_config()

    logger.info(
        "Starting preprocessing pipeline (target %dx%d)",
        config.target_size.width,
        config.target_size.height,
    )

    df = load_source_metadata(config)
    df = assign_stratified_splits(df, config.split)
    split_stats = compute_split_statistics(df)

    processed_records: list[dict] = []
    images_processed = 0
    images_skipped = 0
    images_failed = 0
    skip_reasons: dict[str, int] = {}

    for index, row in df.iterrows():
        record = process_single_image(row, config)

        if record is None:
            images_failed += 1
            skip_reasons["processing_failed"] = skip_reasons.get("processing_failed", 0) + 1
            continue

        if record.get("status") == "skipped_existing":
            images_skipped += 1
            skip_reasons["skipped_existing"] = skip_reasons.get("skipped_existing", 0) + 1
        else:
            images_processed += 1

        processed_records.append(record)

        if (index + 1) % 5000 == 0:
            logger.info("Processed %d/%d images", index + 1, len(df))

    save_processed_metadata(processed_records, config)

    resize_stats = compute_resize_statistics(processed_records)
    report = PreprocessingReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        config=_config_to_dict(config),
        images_processed=images_processed,
        images_skipped=images_skipped,
        images_failed=images_failed,
        resize_statistics=resize_stats,
        split_statistics=asdict(split_stats),
        augmentation_config=augmentation_config_to_dict(augmentation_config),
        skip_reasons=skip_reasons,
    )

    save_preprocessing_report_json(
        report,
        config.project_root / config.report_json_path,
    )
    generate_preprocessing_report_markdown(
        report,
        config.project_root / config.report_md_path,
    )

    logger.info(
        "Preprocessing complete: %d processed, %d skipped, %d failed",
        images_processed,
        images_skipped,
        images_failed,
    )
    return report


if __name__ == "__main__":
    run_preprocessing()
