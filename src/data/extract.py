"""ZIP archive extraction for dataset ingestion.

Extracts validated archives into ``datasets/external/<dataset_name>/``.
No image preprocessing, deduplication, or dataset merging is performed.
"""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.data.verify import ZipValidationResult

logger = logging.getLogger(__name__)

DEFAULT_EXTERNAL_DIR = Path("datasets/external")


@dataclass
class ExtractionResult:
    """Outcome of extracting one ZIP archive.

    Attributes:
        dataset_name: Canonical dataset identifier.
        zip_path: Source archive path.
        extraction_path: Destination directory for extracted files.
        skipped: Whether extraction was skipped because data already exists.
        files_in_archive: Number of file entries expected from the archive.
        files_on_disk: Number of files found on disk after extraction.
        success: Whether extraction or skip verification succeeded.
        extraction_date: UTC ISO-8601 timestamp of the operation.
        errors: Error messages encountered during extraction.
    """

    dataset_name: str
    zip_path: Path
    extraction_path: Path
    skipped: bool
    files_in_archive: int
    files_on_disk: int
    success: bool
    extraction_date: str
    errors: list[str] = field(default_factory=list)


def count_files_on_disk(directory: Path) -> int:
    """Count all files under a directory recursively.

    Args:
        directory: Root directory to scan.

    Returns:
        Total number of files found.
    """
    if not directory.exists():
        return 0
    return sum(1 for path in directory.rglob("*") if path.is_file())


def get_extraction_path(
    dataset_name: str,
    external_dir: Path | str = DEFAULT_EXTERNAL_DIR,
) -> Path:
    """Resolve the extraction directory for a dataset.

    Args:
        dataset_name: Canonical dataset name.
        external_dir: Root directory for external datasets.

    Returns:
        Path to ``<external_dir>/<dataset_name>/``.
    """
    return Path(external_dir) / dataset_name


def is_already_extracted(
    extraction_path: Path,
    expected_file_count: int,
) -> bool:
    """Determine whether a dataset has already been extracted.

    Extraction is considered complete when the destination directory exists,
    is non-empty, and the on-disk file count meets or exceeds the archive
    file count.

    Args:
        extraction_path: Target extraction directory.
        expected_file_count: Number of file entries in the source archive.

    Returns:
        ``True`` if extraction can be safely skipped.
    """
    if not extraction_path.exists():
        return False

    on_disk = count_files_on_disk(extraction_path)
    if on_disk == 0:
        return False

    if on_disk < expected_file_count:
        logger.warning(
            "Extraction path '%s' exists but has fewer files than the archive "
            "(%d on disk vs %d expected). Re-extracting.",
            extraction_path,
            on_disk,
            expected_file_count,
        )
        return False

    logger.info(
        "Skipping extraction for '%s': %d files already present at %s",
        extraction_path.name,
        on_disk,
        extraction_path,
    )
    return True


def extract_zip_archive(
    validation: ZipValidationResult,
    external_dir: Path | str = DEFAULT_EXTERNAL_DIR,
) -> ExtractionResult:
    """Extract a validated ZIP archive to the external datasets directory.

    Args:
        validation: Successful validation result for the archive.
        external_dir: Root directory for extracted datasets.

    Returns:
        An :class:`ExtractionResult` describing the extraction outcome.
    """
    extraction_date = datetime.now(timezone.utc).isoformat()
    extraction_path = get_extraction_path(validation.dataset_name, external_dir)
    errors: list[str] = []

    if not validation.is_valid:
        return ExtractionResult(
            dataset_name=validation.dataset_name,
            zip_path=validation.zip_path,
            extraction_path=extraction_path,
            skipped=False,
            files_in_archive=validation.file_count,
            files_on_disk=0,
            success=False,
            extraction_date=extraction_date,
            errors=list(validation.errors),
        )

    if is_already_extracted(extraction_path, validation.file_count):
        files_on_disk = count_files_on_disk(extraction_path)
        return ExtractionResult(
            dataset_name=validation.dataset_name,
            zip_path=validation.zip_path,
            extraction_path=extraction_path,
            skipped=True,
            files_in_archive=validation.file_count,
            files_on_disk=files_on_disk,
            success=True,
            extraction_date=extraction_date,
        )

    extraction_path.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Extracting '%s' to %s",
        validation.zip_path.name,
        extraction_path,
    )

    try:
        with zipfile.ZipFile(validation.zip_path, mode="r") as archive:
            archive.extractall(extraction_path)
    except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
        errors.append(f"Extraction failed: {exc}")
        logger.error(
            "Extraction failed for '%s': %s",
            validation.zip_path.name,
            exc,
        )
        return ExtractionResult(
            dataset_name=validation.dataset_name,
            zip_path=validation.zip_path,
            extraction_path=extraction_path,
            skipped=False,
            files_in_archive=validation.file_count,
            files_on_disk=count_files_on_disk(extraction_path),
            success=False,
            extraction_date=extraction_date,
            errors=errors,
        )

    files_on_disk = count_files_on_disk(extraction_path)
    logger.info(
        "Extracted '%s': %d files on disk",
        validation.dataset_name,
        files_on_disk,
    )

    return ExtractionResult(
        dataset_name=validation.dataset_name,
        zip_path=validation.zip_path,
        extraction_path=extraction_path,
        skipped=False,
        files_in_archive=validation.file_count,
        files_on_disk=files_on_disk,
        success=True,
        extraction_date=extraction_date,
    )
