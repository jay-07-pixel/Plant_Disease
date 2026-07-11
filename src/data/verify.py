"""ZIP archive validation for dataset ingestion.

Validates archive integrity and safety before extraction. Read-only with
respect to dataset contents — no preprocessing is performed.
"""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.extract import ExtractionResult

logger = logging.getLogger(__name__)


@dataclass
class ZipValidationResult:
    """Outcome of validating a ZIP archive before extraction.

    Attributes:
        zip_path: Path to the validated archive.
        dataset_name: Derived dataset name from the archive filename.
        is_valid: Whether the archive passed all validation checks.
        file_count: Number of file entries (excluding directories) in the archive.
        total_entries: Total ZIP entries including directories.
        errors: Validation error messages.
    """

    zip_path: Path
    dataset_name: str
    is_valid: bool
    file_count: int
    total_entries: int
    errors: list[str] = field(default_factory=list)


def derive_dataset_name(zip_path: Path) -> str:
    """Derive a canonical dataset name from a ZIP filename.

    Args:
        zip_path: Path to the ZIP archive.

    Returns:
        Lowercase stem of the archive filename (e.g. ``plantvillage``).
    """
    return zip_path.stem.lower()


def count_zip_files(zip_path: Path) -> tuple[int, int]:
    """Count file and total entries inside a ZIP archive.

    Args:
        zip_path: Path to the ZIP archive.

    Returns:
        A tuple of ``(file_count, total_entries)``.

    Raises:
        zipfile.BadZipFile: If the archive cannot be opened.
    """
    file_count = 0
    total_entries = 0
    with zipfile.ZipFile(zip_path, mode="r") as archive:
        for info in archive.infolist():
            total_entries += 1
            if not info.is_dir():
                file_count += 1
    return file_count, total_entries


def _check_zip_slip(archive: zipfile.ZipFile, destination: Path) -> list[str]:
    """Detect path-traversal members that would escape the destination directory.

    Args:
        archive: Open ZIP archive.
        destination: Intended extraction root directory.

    Returns:
        A list of error messages for unsafe member paths.
    """
    errors: list[str] = []
    destination_root = destination.resolve()

    for member in archive.namelist():
        target = (destination_root / member).resolve()
        if not str(target).startswith(str(destination_root)):
            errors.append(f"Unsafe ZIP member path detected: {member}")

    return errors


def validate_zip_archive(
    zip_path: Path,
    extraction_root: Path,
) -> ZipValidationResult:
    """Validate a ZIP archive before extraction.

    Checks that the file is a valid ZIP, passes integrity tests, contains
    at least one file entry, and has no path-traversal members.

    Args:
        zip_path: Path to the ZIP archive.
        extraction_root: Planned extraction destination for slip checks.

    Returns:
        A :class:`ZipValidationResult` describing validation outcome.
    """
    dataset_name = derive_dataset_name(zip_path)
    errors: list[str] = []
    file_count = 0
    total_entries = 0

    if not zip_path.exists():
        errors.append(f"Archive does not exist: {zip_path}")
        return ZipValidationResult(
            zip_path=zip_path,
            dataset_name=dataset_name,
            is_valid=False,
            file_count=0,
            total_entries=0,
            errors=errors,
        )

    if not zipfile.is_zipfile(zip_path):
        errors.append(f"File is not a valid ZIP archive: {zip_path}")
        return ZipValidationResult(
            zip_path=zip_path,
            dataset_name=dataset_name,
            is_valid=False,
            file_count=0,
            total_entries=0,
            errors=errors,
        )

    try:
        with zipfile.ZipFile(zip_path, mode="r") as archive:
            corrupt_member = archive.testzip()
            if corrupt_member is not None:
                errors.append(
                    f"Corrupt member detected in archive: {corrupt_member}"
                )

            for info in archive.infolist():
                total_entries += 1
                if not info.is_dir():
                    file_count += 1

            errors.extend(_check_zip_slip(archive, extraction_root))

    except zipfile.BadZipFile as exc:
        errors.append(f"Bad ZIP archive: {exc}")
    except OSError as exc:
        errors.append(f"Failed to read archive: {exc}")

    if file_count == 0 and not errors:
        errors.append("Archive contains no file entries.")

    is_valid = len(errors) == 0
    if is_valid:
        logger.info(
            "Validated archive '%s': %d files, %d total entries",
            zip_path.name,
            file_count,
            total_entries,
        )
    else:
        logger.error(
            "Validation failed for archive '%s': %s",
            zip_path.name,
            "; ".join(errors),
        )

    return ZipValidationResult(
        zip_path=zip_path,
        dataset_name=dataset_name,
        is_valid=is_valid,
        file_count=file_count,
        total_entries=total_entries,
        errors=errors,
    )


@dataclass
class ExtractionVerificationResult:
    """Outcome of verifying an extracted dataset on disk.

    Attributes:
        dataset_name: Canonical dataset identifier.
        extraction_path: Directory that was verified.
        is_valid: Whether verification passed.
        files_in_archive: Expected file count from the source archive.
        files_on_disk: Actual file count found on disk.
        errors: Verification error messages.
    """

    dataset_name: str
    extraction_path: Path
    is_valid: bool
    files_in_archive: int
    files_on_disk: int
    errors: list[str] = field(default_factory=list)


def verify_extraction(extraction: ExtractionResult) -> ExtractionVerificationResult:
    """Verify that extraction completed successfully.

    Checks that the destination directory exists, is non-empty, and contains
    at least as many files as the source archive.

    Args:
        extraction: Extraction result to verify. Accepts any object with
            ``dataset_name``, ``extraction_path``, ``files_in_archive``,
            ``files_on_disk``, and ``success`` attributes.

    Returns:
        An :class:`ExtractionVerificationResult` describing verification outcome.
    """
    errors: list[str] = []
    extraction_path = Path(extraction.extraction_path)
    files_on_disk = int(extraction.files_on_disk)

    if not extraction_path.exists():
        errors.append(f"Extraction directory does not exist: {extraction_path}")
    elif files_on_disk == 0:
        errors.append(f"Extraction directory is empty: {extraction_path}")
    elif files_on_disk < extraction.files_in_archive:
        errors.append(
            f"File count mismatch: {files_on_disk} on disk, "
            f"{extraction.files_in_archive} expected from archive."
        )

    if not extraction.success:
        errors.append("Extraction step reported failure.")

    is_valid = len(errors) == 0
    if is_valid:
        logger.info(
            "Verified extraction for '%s': %d files at %s",
            extraction.dataset_name,
            files_on_disk,
            extraction_path,
        )
    else:
        logger.error(
            "Verification failed for '%s': %s",
            extraction.dataset_name,
            "; ".join(errors),
        )

    return ExtractionVerificationResult(
        dataset_name=extraction.dataset_name,
        extraction_path=extraction_path,
        is_valid=is_valid,
        files_in_archive=extraction.files_in_archive,
        files_on_disk=files_on_disk,
        errors=errors,
    )
