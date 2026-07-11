"""Dataset ingestion pipeline for ZIP archives in ``datasets/raw/``.

Discovers archives, validates integrity, extracts them to
``datasets/external/<dataset_name>/``, verifies extraction, and updates the
dataset registry. No preprocessing, deduplication, balancing, or splitting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.data.extract import (
    DEFAULT_EXTERNAL_DIR,
    extract_zip_archive,
    get_extraction_path,
)
from src.data.registry import (
    DEFAULT_REGISTRY_PATH,
    DatasetRegistry,
    DatasetRegistryEntry,
    build_version_fingerprint,
    load_registry,
    save_registry,
    upsert_entry,
)
from src.data.verify import validate_zip_archive, verify_extraction

logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path("datasets/raw")


@dataclass
class IngestionResult:
    """Outcome of ingesting a single ZIP archive.

    Attributes:
        dataset_name: Canonical dataset identifier.
        zip_path: Source archive path.
        extraction_path: Destination directory for extracted files.
        skipped: Whether extraction was skipped.
        validation_passed: Whether pre-extraction ZIP validation succeeded.
        verification_passed: Whether post-extraction verification succeeded.
        files_in_archive: Expected file count from the archive.
        files_on_disk: Actual file count after extraction or skip.
        success: Whether the full ingest pipeline succeeded for this archive.
        errors: Accumulated error messages.
    """

    dataset_name: str
    zip_path: Path
    extraction_path: Path
    skipped: bool
    validation_passed: bool
    verification_passed: bool
    files_in_archive: int
    files_on_disk: int
    success: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class IngestionSummary:
    """Aggregate outcome of an ingestion pipeline run.

    Attributes:
        raw_dir: Directory scanned for ZIP archives.
        external_dir: Root directory for extracted datasets.
        generated_at: UTC ISO-8601 timestamp of the pipeline run.
        results: Per-archive ingestion results.
        registry_path: Path where the dataset registry was saved.
    """

    raw_dir: Path
    external_dir: Path
    generated_at: str
    results: list[IngestionResult]
    registry_path: Path


def discover_zip_archives(raw_dir: Path | str = DEFAULT_RAW_DIR) -> list[Path]:
    """Discover every ZIP archive inside ``datasets/raw/``.

    Matches both ``*.zip`` and ``*.ZIP`` via case-insensitive suffix check.

    Args:
        raw_dir: Directory to scan for archives.

    Returns:
        A sorted list of ZIP archive paths.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        logger.warning("Raw directory does not exist: %s", raw_path)
        return []

    archives = sorted(
        path
        for path in raw_path.iterdir()
        if path.is_file() and path.suffix.lower() == ".zip"
    )
    logger.info("Discovered %d ZIP archive(s) in %s", len(archives), raw_path)
    return archives


def ingest_archive(
    zip_path: Path,
    external_dir: Path | str = DEFAULT_EXTERNAL_DIR,
) -> IngestionResult:
    """Ingest a single ZIP archive through validate → extract → verify.

    Args:
        zip_path: Path to the ZIP archive.
        external_dir: Root directory for extracted datasets.

    Returns:
        An :class:`IngestionResult` for the archive.
    """
    errors: list[str] = []
    extraction_path = get_extraction_path(
        zip_path.stem.lower(),
        external_dir,
    )

    validation = validate_zip_archive(zip_path, extraction_path)
    if not validation.is_valid:
        return IngestionResult(
            dataset_name=validation.dataset_name,
            zip_path=zip_path,
            extraction_path=extraction_path,
            skipped=False,
            validation_passed=False,
            verification_passed=False,
            files_in_archive=validation.file_count,
            files_on_disk=0,
            success=False,
            errors=list(validation.errors),
        )

    extraction = extract_zip_archive(validation, external_dir)
    errors.extend(extraction.errors)

    verification = verify_extraction(extraction)
    errors.extend(verification.errors)

    success = extraction.success and verification.is_valid
    if success:
        action = "skipped" if extraction.skipped else "extracted"
        logger.info(
            "Successfully %s dataset '%s' (%d files)",
            action,
            extraction.dataset_name,
            verification.files_on_disk,
        )
    else:
        logger.error(
            "Ingestion failed for dataset '%s': %s",
            extraction.dataset_name,
            "; ".join(errors),
        )

    return IngestionResult(
        dataset_name=extraction.dataset_name,
        zip_path=zip_path,
        extraction_path=extraction.extraction_path,
        skipped=extraction.skipped,
        validation_passed=True,
        verification_passed=verification.is_valid,
        files_in_archive=extraction.files_in_archive,
        files_on_disk=verification.files_on_disk,
        success=success,
        errors=errors,
    )


def _build_registry_entry(
    zip_path: Path,
    result: IngestionResult,
) -> DatasetRegistryEntry:
    """Create a registry entry from a successful ingestion result."""
    return DatasetRegistryEntry(
        dataset_name=result.dataset_name,
        version=build_version_fingerprint(zip_path),
        number_of_files=result.files_on_disk,
        extraction_date=datetime.now(timezone.utc).isoformat(),
        extraction_path=str(result.extraction_path).replace("\\", "/"),
    )


def run_ingestion_pipeline(
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    external_dir: Path | str = DEFAULT_EXTERNAL_DIR,
    registry_path: Path | str = DEFAULT_REGISTRY_PATH,
) -> IngestionSummary:
    """Run the full dataset ingestion pipeline.

    Discovers ZIP archives, validates, extracts, verifies, and updates the
    dataset registry at ``reports/dataset_registry.json``.

    Args:
        raw_dir: Directory containing ZIP archives.
        external_dir: Root directory for extracted datasets.
        registry_path: Path for the dataset registry JSON file.

    Returns:
        An :class:`IngestionSummary` describing the pipeline run.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    raw_path = Path(raw_dir)
    external_path = Path(external_dir)
    registry_file = Path(registry_path)
    external_path.mkdir(parents=True, exist_ok=True)
    registry_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting dataset ingestion pipeline")
    logger.info("Raw directory: %s", raw_path)
    logger.info("External directory: %s", external_path)

    archives = discover_zip_archives(raw_path)
    registry = load_registry(registry_file)
    results: list[IngestionResult] = []

    for zip_path in archives:
        logger.info("Processing archive: %s", zip_path.name)
        result = ingest_archive(zip_path, external_path)
        results.append(result)

        if result.success:
            upsert_entry(
                registry,
                _build_registry_entry(zip_path, result),
            )

    save_registry(registry, registry_file)

    summary = IngestionSummary(
        raw_dir=raw_path.resolve(),
        external_dir=external_path.resolve(),
        generated_at=datetime.now(timezone.utc).isoformat(),
        results=results,
        registry_path=registry_file.resolve(),
    )

    succeeded = sum(1 for result in results if result.success)
    logger.info(
        "Ingestion pipeline complete: %d/%d archive(s) succeeded",
        succeeded,
        len(results),
    )
    return summary


if __name__ == "__main__":
    run_ingestion_pipeline()
