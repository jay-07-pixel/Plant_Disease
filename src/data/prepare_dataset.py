"""Dataset preparation — canonical metadata for preprocessing and training.

Builds a unified metadata table from standardized labels and read-only image
inspection. Does not modify, copy, resize, augment, balance, or split data.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

from src.config.dataset_schema import VALID_IMAGE_EXTENSIONS
from src.data.dataset_audit import DEFAULT_EXTERNAL_DIR, discover_external_datasets
from src.data.label_standardizer import DEFAULT_MAPPING_PATH

logger = logging.getLogger(__name__)

DEFAULT_PROCESSED_DIR = Path("datasets/processed")
DEFAULT_METADATA_CSV = DEFAULT_PROCESSED_DIR / "dataset_metadata.csv"
DEFAULT_METADATA_JSON = DEFAULT_PROCESSED_DIR / "dataset_metadata.json"
DEFAULT_PREPARATION_JSON = Path("reports/dataset_preparation.json")
DEFAULT_PREPARATION_MD = Path("reports/dataset_preparation.md")

SUPPORTED_EXTENSIONS = VALID_IMAGE_EXTENSIONS
SPLIT_ALIASES: dict[str, str] = {
    "train": "train",
    "val": "val",
    "validation": "val",
    "test": "test",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelMappingInfo:
    """Canonical label fields resolved from label standardization.

    Attributes:
        plant: Canonical crop name.
        disease: Canonical disease name.
        canonical_label: Stable cross-dataset label key.
        is_healthy: Whether the sample is disease-free.
    """

    plant: str
    disease: str
    canonical_label: str
    is_healthy: bool


@dataclass
class DatasetMetadataRecord:
    """One row in the canonical dataset metadata table.

    Attributes:
        image_path: Project-relative path to the image file.
        dataset_name: Source dataset identifier.
        split: Train/val/test partition when inferable from the path.
        plant: Canonical plant name.
        disease: Canonical disease name.
        canonical_label: Stable label key for training.
        is_healthy: Whether the plant is disease-free.
        image_width: Image width in pixels, or ``None`` if unreadable.
        image_height: Image height in pixels, or ``None`` if unreadable.
        image_format: Detected format extension (lowercase).
        sample_weight: Per-sample weight for ``WeightedRandomSampler``.
    """

    image_path: str
    dataset_name: str
    split: str | None
    plant: str
    disease: str
    canonical_label: str
    is_healthy: bool
    image_width: int | None
    image_height: int | None
    image_format: str
    sample_weight: float = 1.0


@dataclass
class ClassWeightInfo:
    """Class frequency and sampling weight for one canonical label.

    Attributes:
        canonical_label: Stable label key.
        count: Number of images in the class.
        frequency: Proportion of the full dataset.
        weight: Inverse-frequency weight for ``WeightedRandomSampler``.
    """

    canonical_label: str
    count: int
    frequency: float
    weight: float


@dataclass
class PreparationStatistics:
    """Aggregate statistics for dataset preparation.

    Attributes:
        total_images: Total images included in metadata.
        total_canonical_classes: Distinct canonical labels.
        images_per_class: Image count per canonical label.
        images_per_plant: Image count per plant.
        healthy_count: Number of healthy images.
        diseased_count: Number of diseased images.
        dataset_contribution: Image count per source dataset.
        split_distribution: Image count per split partition.
        class_weights: Per-class sampling weights.
        unmapped_images: Images skipped due to missing label mapping.
        unreadable_images: Images with metadata read failures.
    """

    total_images: int
    total_canonical_classes: int
    images_per_class: dict[str, int]
    images_per_plant: dict[str, int]
    healthy_count: int
    diseased_count: int
    dataset_contribution: dict[str, int]
    split_distribution: dict[str, int]
    class_weights: dict[str, float]
    unmapped_images: int = 0
    unreadable_images: int = 0


@dataclass
class DatasetPreparationReport:
    """Full dataset preparation output.

    Attributes:
        generated_at: UTC ISO-8601 timestamp.
        label_mapping_source: Path to the label mapping JSON used.
        external_dir: Root directory scanned for images.
        metadata_csv_path: Path to exported CSV metadata.
        metadata_json_path: Path to exported JSON metadata.
        records: All metadata records.
        statistics: Aggregate preparation statistics.
    """

    generated_at: str
    label_mapping_source: str
    external_dir: str
    metadata_csv_path: str
    metadata_json_path: str
    records: list[DatasetMetadataRecord]
    statistics: PreparationStatistics


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------


def load_label_lookup(
    mapping_path: Path | str = DEFAULT_MAPPING_PATH,
) -> dict[tuple[str, str], LabelMappingInfo]:
    """Load standardized label mappings keyed by dataset and raw class folder.

    Args:
        mapping_path: Path to ``label_mapping.json``.

    Returns:
        Mapping of ``(dataset_name, raw_label)`` to :class:`LabelMappingInfo`.

    Raises:
        FileNotFoundError: If the mapping file does not exist.
    """
    path = Path(mapping_path)
    if not path.exists():
        raise FileNotFoundError(f"Label mapping not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    lookup: dict[tuple[str, str], LabelMappingInfo] = {}

    for dataset_name, dataset_data in payload.get("datasets", {}).items():
        for mapping in dataset_data.get("mappings", []):
            if mapping.get("status") != "mapped":
                continue
            standard = mapping.get("standard_label")
            if not standard:
                continue

            key = (dataset_name.lower(), mapping["raw_label"])
            lookup[key] = LabelMappingInfo(
                plant=standard["plant"],
                disease=standard["disease"],
                canonical_label=standard["canonical_key"],
                is_healthy=bool(standard["is_healthy"]),
            )

    logger.info("Loaded %d label mappings from %s", len(lookup), path)
    return lookup


# ---------------------------------------------------------------------------
# Path parsing (read-only discovery)
# ---------------------------------------------------------------------------


def infer_split_from_path(relative_path: Path) -> str | None:
    """Infer train/val/test split from a relative image path.

    Args:
        relative_path: Path relative to the dataset root.

    Returns:
        Split name (``train``, ``val``, ``test``) or ``None``.
    """
    for part in relative_path.parts:
        normalized = part.lower()
        if normalized in SPLIT_ALIASES:
            return SPLIT_ALIASES[normalized]
    return None


def extract_raw_class_label(relative_path: Path) -> str:
    """Extract the raw class folder name from an image relative path.

    Args:
        relative_path: Path relative to the dataset root.

    Returns:
        Parent folder name used as the raw class label.
    """
    if len(relative_path.parts) <= 1:
        return "_root_"
    return relative_path.parts[-2]


def iter_dataset_images(dataset_root: Path) -> list[Path]:
    """List supported image files under a dataset root.

    Args:
        dataset_root: Root directory of one dataset.

    Returns:
        Sorted list of absolute image paths.
    """
    return sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower().lstrip(".") in SUPPORTED_EXTENSIONS
    )


def read_image_metadata_readonly(image_path: Path) -> tuple[int | None, int | None, str]:
    """Read image dimensions and format without modifying the file.

    Args:
        image_path: Absolute path to the image.

    Returns:
        Tuple of width, height, and format extension.
    """
    image_format = image_path.suffix.lower().lstrip(".")
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            if img.format:
                image_format = img.format.lower()
            return width, height, image_format
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.debug("Could not read metadata for %s: %s", image_path, exc)
        return None, None, image_format


def _to_project_relative_path(image_path: Path, project_root: Path) -> str:
    """Convert an absolute image path to a POSIX project-relative path."""
    return image_path.resolve().relative_to(project_root.resolve()).as_posix()


# ---------------------------------------------------------------------------
# Metadata table construction
# ---------------------------------------------------------------------------


def build_metadata_records(
    external_dir: Path | str = DEFAULT_EXTERNAL_DIR,
    label_lookup: dict[tuple[str, str], LabelMappingInfo] | None = None,
    mapping_path: Path | str = DEFAULT_MAPPING_PATH,
    project_root: Path | None = None,
) -> tuple[list[DatasetMetadataRecord], int, int]:
    """Build canonical metadata records by scanning external datasets.

    Args:
        external_dir: Root directory containing ingested datasets.
        label_lookup: Pre-loaded label mapping; loaded from ``mapping_path`` if omitted.
        mapping_path: Path to label mapping JSON.
        project_root: Project root for relative paths; defaults to cwd.

    Returns:
        Tuple of records, unmapped image count, and unreadable image count.
    """
    external_path = Path(external_dir).resolve()
    project_root = project_root or Path.cwd()
    lookup = label_lookup or load_label_lookup(mapping_path)

    records: list[DatasetMetadataRecord] = []
    unmapped = 0
    unreadable = 0

    dataset_paths = discover_external_datasets(external_path)
    if not dataset_paths:
        logger.warning("No datasets found under %s", external_path)

    for dataset_path in dataset_paths:
        dataset_name = dataset_path.name.lower()
        image_paths = iter_dataset_images(dataset_path)
        logger.info(
            "Building metadata for dataset '%s' (%d images)",
            dataset_name,
            len(image_paths),
        )

        for index, image_path in enumerate(image_paths, start=1):
            relative_path = image_path.relative_to(dataset_path)
            raw_label = extract_raw_class_label(relative_path)
            mapping = lookup.get((dataset_name, raw_label))

            if mapping is None:
                unmapped += 1
                logger.debug(
                    "No mapping for %s/%s at %s",
                    dataset_name,
                    raw_label,
                    relative_path,
                )
                continue

            width, height, image_format = read_image_metadata_readonly(image_path)
            if width is None or height is None:
                unreadable += 1

            records.append(
                DatasetMetadataRecord(
                    image_path=_to_project_relative_path(image_path, project_root),
                    dataset_name=dataset_name,
                    split=infer_split_from_path(relative_path),
                    plant=mapping.plant,
                    disease=mapping.disease,
                    canonical_label=mapping.canonical_label,
                    is_healthy=mapping.is_healthy,
                    image_width=width,
                    image_height=height,
                    image_format=image_format,
                )
            )

            if index % 5000 == 0:
                logger.info(
                    "Dataset '%s': processed %d/%d images",
                    dataset_name,
                    index,
                    len(image_paths),
                )

    logger.info(
        "Built %d metadata records (%d unmapped, %d unreadable)",
        len(records),
        unmapped,
        unreadable,
    )
    return records, unmapped, unreadable


def records_to_dataframe(records: list[DatasetMetadataRecord]) -> pd.DataFrame:
    """Convert metadata records to a pandas DataFrame.

    Args:
        records: Metadata records to convert.

    Returns:
        A DataFrame with one row per image.
    """
    if not records:
        return pd.DataFrame(
            columns=[
                "image_path",
                "dataset_name",
                "split",
                "plant",
                "disease",
                "canonical_label",
                "is_healthy",
                "image_width",
                "image_height",
                "image_format",
                "sample_weight",
            ]
        )
    return pd.DataFrame([asdict(record) for record in records])


# ---------------------------------------------------------------------------
# Statistics and class weights
# ---------------------------------------------------------------------------


def compute_class_frequencies(df: pd.DataFrame) -> dict[str, int]:
    """Compute image counts per canonical class label.

    Args:
        df: Metadata DataFrame with a ``canonical_label`` column.

    Returns:
        Mapping of canonical label to image count.
    """
    if df.empty:
        return {}
    counts = df["canonical_label"].value_counts()
    return {str(label): int(count) for label, count in counts.items()}


def compute_class_weights(df: pd.DataFrame) -> dict[str, float]:
    """Compute per-class weights for ``WeightedRandomSampler``.

    Uses inverse-frequency weighting::

        weight[class] = total_samples / (num_classes * class_count)

    Args:
        df: Metadata DataFrame with a ``canonical_label`` column.

    Returns:
        Mapping of canonical label to sampling weight.
    """
    if df.empty:
        return {}

    class_counts = df["canonical_label"].value_counts()
    total_samples = len(df)
    num_classes = len(class_counts)

    return {
        str(label): float(total_samples / (num_classes * count))
        for label, count in class_counts.items()
    }


def build_class_weight_table(df: pd.DataFrame) -> list[ClassWeightInfo]:
    """Build detailed class weight information.

    Args:
        df: Metadata DataFrame.

    Returns:
        List of :class:`ClassWeightInfo` entries sorted by count descending.
    """
    if df.empty:
        return []

    total = len(df)
    frequencies = compute_class_frequencies(df)
    weights = compute_class_weights(df)

    entries = [
        ClassWeightInfo(
            canonical_label=label,
            count=count,
            frequency=count / total,
            weight=weights[label],
        )
        for label, count in frequencies.items()
    ]
    return sorted(entries, key=lambda item: item.count, reverse=True)


def apply_sample_weights(
    records: list[DatasetMetadataRecord],
    class_weights: dict[str, float],
) -> list[DatasetMetadataRecord]:
    """Attach per-sample weights to metadata records.

    Args:
        records: Metadata records to update.
        class_weights: Mapping of canonical label to weight.

    Returns:
        New list of records with ``sample_weight`` populated.
    """
    updated: list[DatasetMetadataRecord] = []
    for record in records:
        weight = class_weights.get(record.canonical_label, 1.0)
        updated.append(
            DatasetMetadataRecord(
                image_path=record.image_path,
                dataset_name=record.dataset_name,
                split=record.split,
                plant=record.plant,
                disease=record.disease,
                canonical_label=record.canonical_label,
                is_healthy=record.is_healthy,
                image_width=record.image_width,
                image_height=record.image_height,
                image_format=record.image_format,
                sample_weight=weight,
            )
        )
    return updated


def compute_preparation_statistics(
    df: pd.DataFrame,
    *,
    unmapped_images: int = 0,
    unreadable_images: int = 0,
) -> PreparationStatistics:
    """Compute aggregate preparation statistics from metadata.

    Args:
        df: Metadata DataFrame.
        unmapped_images: Count of images skipped due to missing mappings.
        unreadable_images: Count of images with unreadable metadata.

    Returns:
        A :class:`PreparationStatistics` summary.
    """
    images_per_class = compute_class_frequencies(df)
    images_per_plant = (
        {}
        if df.empty
        else {
            str(plant): int(count)
            for plant, count in df["plant"].value_counts().items()
        }
    )

    healthy_count = int(df["is_healthy"].sum()) if not df.empty else 0
    diseased_count = len(df) - healthy_count

    dataset_contribution = (
        {}
        if df.empty
        else {
            str(name): int(count)
            for name, count in df["dataset_name"].value_counts().items()
        }
    )

    split_distribution: dict[str, int] = {}
    if not df.empty:
        split_series = df["split"].fillna("unknown")
        split_distribution = {
            str(split): int(count) for split, count in split_series.value_counts().items()
        }

    class_weights = compute_class_weights(df)

    return PreparationStatistics(
        total_images=len(df),
        total_canonical_classes=df["canonical_label"].nunique() if not df.empty else 0,
        images_per_class=images_per_class,
        images_per_plant=images_per_plant,
        healthy_count=healthy_count,
        diseased_count=diseased_count,
        dataset_contribution=dataset_contribution,
        split_distribution=split_distribution,
        class_weights=class_weights,
        unmapped_images=unmapped_images,
        unreadable_images=unreadable_images,
    )


# ---------------------------------------------------------------------------
# Export and reporting
# ---------------------------------------------------------------------------


def save_metadata_csv(df: pd.DataFrame, output_path: Path | str = DEFAULT_METADATA_CSV) -> Path:
    """Save metadata table to CSV.

    Args:
        df: Metadata DataFrame.
        output_path: Destination CSV path.

    Returns:
        Path the file was written to.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("Saved metadata CSV to %s (%d rows)", path, len(df))
    return path


def save_metadata_json(df: pd.DataFrame, output_path: Path | str = DEFAULT_METADATA_JSON) -> Path:
    """Save metadata table to JSON.

    Args:
        df: Metadata DataFrame.
        output_path: Destination JSON path.

    Returns:
        Path the file was written to.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_records": len(df),
        "records": df.to_dict(orient="records"),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved metadata JSON to %s (%d rows)", path, len(df))
    return path


def serialize_preparation_report(report: DatasetPreparationReport) -> dict:
    """Convert a preparation report to a JSON-serializable dictionary."""
    stats = report.statistics
    return {
        "generated_at": report.generated_at,
        "label_mapping_source": report.label_mapping_source,
        "external_dir": report.external_dir,
        "metadata_csv_path": report.metadata_csv_path,
        "metadata_json_path": report.metadata_json_path,
        "statistics": {
            "total_images": stats.total_images,
            "total_canonical_classes": stats.total_canonical_classes,
            "healthy_count": stats.healthy_count,
            "diseased_count": stats.diseased_count,
            "unmapped_images": stats.unmapped_images,
            "unreadable_images": stats.unreadable_images,
            "images_per_class": stats.images_per_class,
            "images_per_plant": stats.images_per_plant,
            "dataset_contribution": stats.dataset_contribution,
            "split_distribution": stats.split_distribution,
            "class_weights": stats.class_weights,
        },
        "class_weight_table": [
            asdict(entry) for entry in build_class_weight_table(records_to_dataframe(report.records))
        ],
    }


def save_preparation_json(
    report: DatasetPreparationReport,
    output_path: Path | str = DEFAULT_PREPARATION_JSON,
) -> Path:
    """Save preparation statistics as JSON.

    Args:
        report: Preparation report to export.
        output_path: Destination JSON path.

    Returns:
        Path the file was written to.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(serialize_preparation_report(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved preparation JSON to %s", path)
    return path


def generate_preparation_markdown(
    report: DatasetPreparationReport,
    output_path: Path | str = DEFAULT_PREPARATION_MD,
) -> Path:
    """Write a human-readable dataset preparation markdown report.

    Args:
        report: Preparation report to document.
        output_path: Destination markdown path.

    Returns:
        Path the file was written to.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = report.statistics
    weight_table = build_class_weight_table(records_to_dataframe(report.records))

    lines = [
        "# Dataset Preparation Report",
        "",
        f"**Generated:** {report.generated_at}  ",
        f"**Label mapping source:** `{report.label_mapping_source}`  ",
        f"**External directory:** `{report.external_dir}`  ",
        f"**Metadata CSV:** `{report.metadata_csv_path}`  ",
        f"**Metadata JSON:** `{report.metadata_json_path}`  ",
        "",
        "> Read-only metadata preparation. No images were modified, copied, or balanced.",
        "",
        "## Summary",
        "",
        f"- **Total images:** {stats.total_images}",
        f"- **Canonical classes:** {stats.total_canonical_classes}",
        f"- **Healthy images:** {stats.healthy_count}",
        f"- **Diseased images:** {stats.diseased_count}",
        f"- **Unmapped images (skipped):** {stats.unmapped_images}",
        f"- **Unreadable metadata:** {stats.unreadable_images}",
        "",
        "## Dataset Contribution",
        "",
    ]

    for dataset_name, count in sorted(stats.dataset_contribution.items()):
        pct = (count / stats.total_images * 100) if stats.total_images else 0
        lines.append(f"- **{dataset_name}:** {count:,} images ({pct:.1f}%)")
    lines.append("")

    lines.extend(["## Healthy vs Diseased", ""])
    if stats.total_images:
        healthy_pct = stats.healthy_count / stats.total_images * 100
        diseased_pct = stats.diseased_count / stats.total_images * 100
        lines.append(f"- **Healthy:** {stats.healthy_count:,} ({healthy_pct:.1f}%)")
        lines.append(f"- **Diseased:** {stats.diseased_count:,} ({diseased_pct:.1f}%)")
    else:
        lines.append("- No images in metadata table.")
    lines.append("")

    lines.extend(["## Split Distribution", ""])
    if stats.split_distribution:
        for split_name, count in sorted(stats.split_distribution.items()):
            lines.append(f"- **{split_name}:** {count:,}")
    else:
        lines.append("- No split information available.")
    lines.append("")

    lines.extend(["## Images per Plant", ""])
    for plant, count in sorted(stats.images_per_plant.items(), key=lambda x: -x[1]):
        lines.append(f"- **{plant}:** {count:,}")
    lines.append("")

    lines.extend(["## Images per Class (Top 20)", ""])
    sorted_classes = sorted(
        stats.images_per_class.items(), key=lambda item: item[1], reverse=True
    )
    for label, count in sorted_classes[:20]:
        lines.append(f"- `{label}`: {count:,}")
    if len(sorted_classes) > 20:
        lines.append(f"- ... and {len(sorted_classes) - 20} more classes")
    lines.append("")

    lines.extend(
        [
            "## Class Weights (for WeightedRandomSampler)",
            "",
            "Formula: `weight = total_samples / (num_classes × class_count)`",
            "",
            "| Canonical Label | Count | Frequency | Weight |",
            "|-----------------|------:|----------:|-------:|",
        ]
    )
    for entry in weight_table[:25]:
        lines.append(
            f"| `{entry.canonical_label}` | {entry.count:,} | "
            f"{entry.frequency:.4f} | {entry.weight:.6f} |"
        )
    if len(weight_table) > 25:
        lines.append(f"| ... | | | {len(weight_table) - 25} more classes |")
    lines.append("")

    lines.extend(
        [
            "## Metadata Schema",
            "",
            "Each record in `datasets/processed/dataset_metadata.csv` contains:",
            "",
            "| Column | Description |",
            "|--------|-------------|",
            "| `image_path` | Project-relative path to the image |",
            "| `dataset_name` | Source dataset (`plantvillage`, `plantdoc`) |",
            "| `split` | `train`, `val`, `test`, or empty |",
            "| `plant` | Canonical plant name |",
            "| `disease` | Canonical disease name |",
            "| `canonical_label` | Stable cross-dataset label key |",
            "| `is_healthy` | Healthy sample flag |",
            "| `image_width` | Width in pixels |",
            "| `image_height` | Height in pixels |",
            "| `image_format` | File format extension |",
            "| `sample_weight` | Per-sample weight for balanced sampling |",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved preparation markdown to %s", path)
    return path


def run_dataset_preparation(
    external_dir: Path | str = DEFAULT_EXTERNAL_DIR,
    mapping_path: Path | str = DEFAULT_MAPPING_PATH,
    metadata_csv_path: Path | str = DEFAULT_METADATA_CSV,
    metadata_json_path: Path | str = DEFAULT_METADATA_JSON,
    preparation_json_path: Path | str = DEFAULT_PREPARATION_JSON,
    preparation_md_path: Path | str = DEFAULT_PREPARATION_MD,
    project_root: Path | None = None,
) -> DatasetPreparationReport:
    """Run the full dataset preparation pipeline.

    Scans external datasets, applies standardized labels, builds the canonical
    metadata table, computes class weights, and writes outputs. No balancing,
    splitting, or image modification is performed.

    Args:
        external_dir: Root directory containing ingested datasets.
        mapping_path: Path to standardized label mapping JSON.
        metadata_csv_path: Output CSV metadata path.
        metadata_json_path: Output JSON metadata path.
        preparation_json_path: Output preparation statistics JSON path.
        preparation_md_path: Output preparation markdown report path.
        project_root: Project root for relative image paths.

    Returns:
        The completed :class:`DatasetPreparationReport`.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    project_root = project_root or Path.cwd()
    logger.info("Starting dataset preparation pipeline")
    logger.info("External directory: %s", external_dir)
    logger.info("Label mapping: %s", mapping_path)

    label_lookup = load_label_lookup(mapping_path)
    records, unmapped, unreadable = build_metadata_records(
        external_dir=external_dir,
        label_lookup=label_lookup,
        project_root=project_root,
    )

    df = records_to_dataframe(records)
    class_weights = compute_class_weights(df)
    records = apply_sample_weights(records, class_weights)
    df = records_to_dataframe(records)

    statistics = compute_preparation_statistics(
        df,
        unmapped_images=unmapped,
        unreadable_images=unreadable,
    )

    csv_path = save_metadata_csv(df, metadata_csv_path)
    json_path = save_metadata_json(df, metadata_json_path)

    report = DatasetPreparationReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        label_mapping_source=str(Path(mapping_path)),
        external_dir=str(Path(external_dir).resolve()),
        metadata_csv_path=str(csv_path),
        metadata_json_path=str(json_path),
        records=records,
        statistics=statistics,
    )

    save_preparation_json(report, preparation_json_path)
    generate_preparation_markdown(report, preparation_md_path)

    logger.info(
        "Dataset preparation complete: %d records, %d classes",
        statistics.total_images,
        statistics.total_canonical_classes,
    )
    return report


if __name__ == "__main__":
    run_dataset_preparation()
