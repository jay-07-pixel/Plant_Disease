"""Universal dataset schema for PlantDiseaseAI.

Defines the canonical metadata structure, validation rules, and placeholder I/O
functions used across all datasets (PlantVillage, PDDB, PlantDoc, and others).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DatasetName(str, Enum):
    """Registered source datasets."""

    PLANTVILLAGE = "plantvillage"
    PDDB = "pddb"
    PLANTDOC = "plantdoc"
    OTHER = "other"


class DatasetSplit(str, Enum):
    """Train / validation / test partition."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class DiseaseCategory(str, Enum):
    """High-level disease classification."""

    FUNGAL = "fungal"
    BACTERIAL = "bacterial"
    VIRAL = "viral"
    PEST = "pest"
    NUTRITIONAL = "nutritional"
    ENVIRONMENTAL = "environmental"
    HEALTHY = "healthy"
    UNKNOWN = "unknown"


class ImageSource(str, Enum):
    """Origin of the image capture."""

    FIELD = "field"
    LAB = "lab"
    WEB = "web"
    SYNTHETIC = "synthetic"
    UNKNOWN = "unknown"


class AnnotationStatus(str, Enum):
    """Label quality and completeness."""

    ANNOTATED = "annotated"
    UNANNOTATED = "unannotated"
    PARTIAL = "partial"
    VERIFIED = "verified"


class SeverityLevel(str, Enum):
    """Optional disease severity estimate."""

    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEALTHY_LABEL: str = "healthy"

VALID_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {"jpg", "jpeg", "png", "bmp", "tiff", "webp"}
)

VALID_PLANT_NAMES: frozenset[str] = frozenset(
    {
        "apple",
        "blueberry",
        "cherry",
        "corn",
        "grape",
        "orange",
        "peach",
        "pepper",
        "potato",
        "raspberry",
        "soybean",
        "squash",
        "strawberry",
        "tomato",
    }
)

REQUIRED_FIELDS: frozenset[str] = frozenset(
    {
        "image_id",
        "image_path",
        "dataset_name",
        "plant_name",
        "disease_name",
        "is_healthy",
        "disease_category",
        "image_source",
        "annotation_status",
        "split",
        "image_width",
        "image_height",
        "image_format",
    }
)

OPTIONAL_FIELDS: frozenset[str] = frozenset({"severity", "notes"})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Outcome of validating a single :class:`PlantDiseaseRecord`.

    Attributes:
        is_valid: ``True`` when no errors were found.
        errors: Hard failures that must be resolved before use.
        warnings: Non-blocking issues such as missing optional metadata.
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class PlantDiseaseRecord:
    """Canonical metadata record for a single plant-disease image.

    This dataclass is the single source of truth for image metadata across
    ingestion, preprocessing, training, evaluation, and deployment.

    Attributes:
        image_id: Globally unique identifier for the image.
        image_path: Path to the image file (relative or absolute).
        dataset_name: Source dataset identifier.
        plant_name: Canonical plant or crop name.
        disease_name: Canonical disease label; ``healthy`` when disease-free.
        is_healthy: Whether the plant shows no disease.
        disease_category: High-level disease type classification.
        severity: Optional severity estimate.
        image_source: How the image was captured.
        annotation_status: Label quality indicator.
        split: Data partition assignment.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        image_format: File extension without leading dot (e.g. ``jpg``).
        notes: Optional free-text remarks.
    """

    image_id: str
    image_path: str
    dataset_name: DatasetName
    plant_name: str
    disease_name: str
    is_healthy: bool
    disease_category: DiseaseCategory
    image_source: ImageSource
    annotation_status: AnnotationStatus
    split: DatasetSplit
    image_width: int
    image_height: int
    image_format: str
    severity: SeverityLevel | None = None
    notes: str | None = None


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_required_fields(record: PlantDiseaseRecord) -> list[str]:
    """Verify that all required fields are present and non-empty.

    Args:
        record: The record to inspect.

    Returns:
        A list of error messages for any missing or empty required fields.
    """
    errors: list[str] = []

    for field_info in fields(record):
        if field_info.name not in REQUIRED_FIELDS:
            continue

        value = getattr(record, field_info.name)

        if value is None:
            errors.append(f"Required field '{field_info.name}' is missing (None).")
            continue

        if isinstance(value, str) and not value.strip():
            errors.append(f"Required field '{field_info.name}' is empty.")

        if isinstance(value, int) and field_info.name in {"image_width", "image_height"}:
            if value <= 0:
                errors.append(
                    f"Required field '{field_info.name}' must be a positive integer."
                )

    return errors


def validate_plant_name(record: PlantDiseaseRecord) -> list[str]:
    """Verify that ``plant_name`` is a registered canonical name.

    Args:
        record: The record to inspect.

    Returns:
        A list of error messages for invalid plant names.
    """
    normalized = record.plant_name.strip().lower()
    if normalized not in VALID_PLANT_NAMES:
        return [
            f"Invalid plant name '{record.plant_name}'. "
            f"Must be one of the registered canonical names."
        ]
    return []


def validate_healthy_disease_consistency(record: PlantDiseaseRecord) -> list[str]:
    """Verify logical consistency between health status and disease labels.

    Healthy images must use the ``healthy`` disease label and category.
    Diseased images must not use the ``healthy`` disease label.

    Args:
        record: The record to inspect.

    Returns:
        A list of error messages for inconsistent health/disease fields.
    """
    errors: list[str] = []
    normalized_disease = record.disease_name.strip().lower()

    if record.is_healthy:
        if normalized_disease != HEALTHY_LABEL:
            errors.append(
                f"Healthy image (is_healthy=True) cannot have "
                f"disease_name='{record.disease_name}'. "
                f"Expected '{HEALTHY_LABEL}'."
            )
        if record.disease_category != DiseaseCategory.HEALTHY:
            errors.append(
                f"Healthy image must have disease_category='{DiseaseCategory.HEALTHY.value}', "
                f"got '{record.disease_category.value}'."
            )
    else:
        if normalized_disease == HEALTHY_LABEL:
            errors.append(
                f"Diseased image (is_healthy=False) cannot have "
                f"disease_name='{HEALTHY_LABEL}'."
            )

    return errors


def validate_image_extension(record: PlantDiseaseRecord) -> list[str]:
    """Verify that ``image_format`` is a supported image extension.

    Args:
        record: The record to inspect.

    Returns:
        A list of error messages for unsupported extensions.
    """
    normalized = record.image_format.strip().lower().lstrip(".")
    if normalized not in VALID_IMAGE_EXTENSIONS:
        return [
            f"Invalid image format '{record.image_format}'. "
            f"Supported formats: {', '.join(sorted(VALID_IMAGE_EXTENSIONS))}."
        ]

    path_suffix = Path(record.image_path).suffix.lower().lstrip(".")
    if path_suffix and path_suffix != normalized:
        return [
            f"image_format '{record.image_format}' does not match "
            f"image_path extension '.{path_suffix}'."
        ]

    return []


def report_missing_metadata(record: PlantDiseaseRecord) -> list[str]:
    """Report absent optional metadata as warnings.

    Args:
        record: The record to inspect.

    Returns:
        A list of warning messages for missing optional fields.
    """
    warnings: list[str] = []

    for field_name in OPTIONAL_FIELDS:
        value = getattr(record, field_name)
        if value is None:
            warnings.append(f"Optional field '{field_name}' is not provided.")

    return warnings


# ---------------------------------------------------------------------------
# Public validation API
# ---------------------------------------------------------------------------


def validate_record(record: PlantDiseaseRecord) -> ValidationResult:
    """Run all validation checks on a single record.

    Aggregates required-field, plant-name, health-consistency, extension,
    and missing-metadata checks into a single :class:`ValidationResult`.

    Args:
        record: The :class:`PlantDiseaseRecord` to validate.

    Returns:
        A :class:`ValidationResult` with ``is_valid``, ``errors``, and
        ``warnings`` populated.
    """
    errors: list[str] = []
    warnings: list[str] = []

    errors.extend(validate_required_fields(record))
    errors.extend(validate_plant_name(record))
    errors.extend(validate_healthy_disease_consistency(record))
    errors.extend(validate_image_extension(record))
    warnings.extend(report_missing_metadata(record))

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Placeholder I/O functions
# ---------------------------------------------------------------------------


def load_dataset_metadata(path: str | Path) -> list[PlantDiseaseRecord]:
    """Load dataset metadata from a file into canonical records.

  Placeholder — not yet implemented. Will support CSV and Parquet formats
  and return a list of validated :class:`PlantDiseaseRecord` instances.

    Args:
        path: Path to the metadata file.

    Returns:
        A list of :class:`PlantDiseaseRecord` objects.

    Raises:
        NotImplementedError: Always, until I/O is implemented.
    """
    raise NotImplementedError(
        f"load_dataset_metadata() is not yet implemented. "
        f"Requested path: {path}"
    )


def export_metadata(
    records: list[PlantDiseaseRecord],
    path: str | Path,
    *,
    format: str = "csv",
) -> None:
    """Export canonical records to a metadata file.

    Placeholder — not yet implemented. Will write CSV or Parquet files
    conforming to the universal schema.

    Args:
        records: Records to export.
        path: Destination file path.
        format: Output format (``csv`` or ``parquet``).

    Raises:
        NotImplementedError: Always, until I/O is implemented.
    """
    raise NotImplementedError(
        f"export_metadata() is not yet implemented. "
        f"Requested path: {path}, format: {format}, "
        f"record count: {len(records)}"
    )
