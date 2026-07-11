"""Dataset registry for ingested archives.

Tracks metadata for every dataset extracted from ZIP archives under
``datasets/raw/`` into ``datasets/external/``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = Path("reports/dataset_registry.json")


@dataclass
class DatasetRegistryEntry:
    """Metadata record for one ingested dataset.

    Attributes:
        dataset_name: Canonical dataset identifier (ZIP stem, normalized).
        version: Content fingerprint of the source archive.
        number_of_files: Total files present after extraction.
        extraction_date: UTC ISO-8601 timestamp of the extraction run.
        extraction_path: Relative path to the extracted dataset directory.
    """

    dataset_name: str
    version: str
    number_of_files: int
    extraction_date: str
    extraction_path: str


@dataclass
class DatasetRegistry:
    """Collection of registry entries with generation metadata.

    Attributes:
        generated_at: UTC ISO-8601 timestamp when the registry was saved.
        datasets: Ingested dataset records keyed by dataset name.
    """

    generated_at: str
    datasets: dict[str, DatasetRegistryEntry] = field(default_factory=dict)


def build_version_fingerprint(zip_path: Path) -> str:
    """Build a version string from archive file metadata.

    Uses modification time and file size as a lightweight content fingerprint
    without reading the full archive.

    Args:
        zip_path: Path to the source ZIP archive.

    Returns:
        A version string in ``mtime_iso|size_bytes`` format.
    """
    stat = zip_path.stat()
    mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    return f"{mtime}|{stat.st_size}"


def load_registry(path: Path | str = DEFAULT_REGISTRY_PATH) -> DatasetRegistry:
    """Load an existing dataset registry from disk.

    Args:
        path: Path to the registry JSON file.

    Returns:
        A :class:`DatasetRegistry`. Returns an empty registry if the file
        does not exist.
    """
    registry_path = Path(path)
    if not registry_path.exists():
        logger.info("No existing registry found at %s", registry_path)
        return DatasetRegistry(
            generated_at=datetime.now(timezone.utc).isoformat(),
            datasets={},
        )

    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    datasets = {
        name: DatasetRegistryEntry(**entry)
        for name, entry in payload.get("datasets", {}).items()
    }
    logger.info("Loaded registry with %d dataset(s) from %s", len(datasets), registry_path)
    return DatasetRegistry(
        generated_at=payload.get("generated_at", ""),
        datasets=datasets,
    )


def save_registry(
    registry: DatasetRegistry,
    path: Path | str = DEFAULT_REGISTRY_PATH,
) -> Path:
    """Persist the dataset registry to disk.

    Args:
        registry: Registry to serialize.
        path: Destination JSON file path.

    Returns:
        The path the registry was written to.
    """
    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    registry.generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "generated_at": registry.generated_at,
        "datasets": {
            name: asdict(entry) for name, entry in sorted(registry.datasets.items())
        },
    }
    registry_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved dataset registry to %s", registry_path)
    return registry_path


def upsert_entry(
    registry: DatasetRegistry,
    entry: DatasetRegistryEntry,
) -> None:
    """Insert or update a dataset entry in the registry.

    Args:
        registry: Registry to modify in place.
        entry: Dataset metadata to record.
    """
    registry.datasets[entry.dataset_name] = entry
    logger.debug("Updated registry entry for dataset '%s'", entry.dataset_name)
