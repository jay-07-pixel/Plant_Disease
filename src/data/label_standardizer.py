"""Universal label standardization for multi-dataset plant disease labels.

Converts dataset-specific class labels (PlantVillage, PlantDoc, etc.) into a
canonical label schema. This module is read-only — it does not modify images,
datasets, or folder structures.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from src.config.dataset_schema import DatasetName, VALID_PLANT_NAMES

logger = logging.getLogger(__name__)

DEFAULT_EXTERNAL_DIR = Path("datasets/external")
DEFAULT_STATISTICS_PATH = Path("reports/dataset_statistics.json")
DEFAULT_MAPPING_PATH = Path("reports/label_mapping.json")
DEFAULT_REPORT_PATH = Path("reports/label_standardization.md")

PLANTVILLAGE_SEPARATOR = "___"
HEALTHY_TOKEN = "healthy"

# Ordered longest-first so multi-word aliases match before shorter prefixes.
PLANT_ALIASES: tuple[tuple[str, str], ...] = (
    ("bell_pepper", "Pepper"),
    ("bell pepper", "Pepper"),
    ("cherry (including sour)", "Cherry"),
    ("cherry_(including_sour)", "Cherry"),
    ("corn (maize)", "Corn"),
    ("corn_(maize)", "Corn"),
    ("pepper, bell", "Pepper"),
    ("pepper,_bell", "Pepper"),
    ("soyabean", "Soybean"),
    ("blueberry", "Blueberry"),
    ("strawberry", "Strawberry"),
    ("raspberry", "Raspberry"),
    ("squash", "Squash"),
    ("potato", "Potato"),
    ("tomato", "Tomato"),
    ("cherry", "Cherry"),
    ("grape", "Grape"),
    ("apple", "Apple"),
    ("peach", "Peach"),
    ("corn", "Corn"),
    ("orange", "Orange"),
    ("pepper", "Pepper"),
    ("soybean", "Soybean"),
)

# Disease alias normalizations applied after token cleanup.
DISEASE_ALIASES: dict[str, str] = {
    "apple scab": "Apple Scab",
    "scab": "Apple Scab",
    "black rot": "Black Rot",
    "cedar apple rust": "Cedar Apple Rust",
    "rust": "Rust",
    "powdery mildew": "Powdery Mildew",
    "early blight": "Early Blight",
    "late blight": "Late Blight",
    "bacterial spot": "Bacterial Spot",
    "leaf spot": "Leaf Spot",
    "leaf mold": "Leaf Mold",
    "gray leaf spot": "Gray Leaf Spot",
    "grey leaf spot": "Gray Leaf Spot",
    "septoria leaf spot": "Septoria Leaf Spot",
    "target spot": "Target Spot",
    "leaf scorch": "Leaf Scorch",
    "leaf blight": "Leaf Blight",
    "common rust": "Common Rust",
    "northern leaf blight": "Northern Leaf Blight",
    "cercospora leaf spot gray leaf spot": "Cercospora Leaf Spot Gray Leaf Spot",
    "esca (black measles)": "Esca (Black Measles)",
    "leaf blight (isariopsis leaf spot)": "Leaf Blight (Isariopsis Leaf Spot)",
    "haunglongbing (citrus greening)": "Huanglongbing (Citrus Greening)",
    "spider mites two-spotted spider mite": "Spider Mites Two-Spotted Spider Mite",
    "tomato mosaic virus": "Tomato Mosaic Virus",
    "tomato yellow leaf curl virus": "Tomato Yellow Leaf Curl Virus",
    "mosaic virus": "Tomato Mosaic Virus",
    "yellow virus": "Tomato Yellow Leaf Curl Virus",
    "mold": "Leaf Mold",
    "spot": "Leaf Spot",
    "scab leaf": "Apple Scab",
}


class LabelParseStatus(str, Enum):
    """Outcome status for a label mapping attempt."""

    MAPPED = "mapped"
    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"
    UNMAPPED = "unmapped"


class LabelParserType(str, Enum):
    """Supported dataset-specific label parsers."""

    PLANTVILLAGE = "plantvillage"
    PLANTDOC = "plantdoc"
    GENERIC = "generic"


@dataclass(frozen=True)
class StandardLabel:
    """Universal canonical label schema.

    Attributes:
        plant: Canonical crop name (title case, e.g. ``Tomato``).
        disease: Canonical disease name (title case, ``Healthy`` when disease-free).
        is_healthy: Whether the sample represents a healthy plant.
        canonical_key: Stable cross-dataset identifier (``plant|disease`` slug).
    """

    plant: str
    disease: str
    is_healthy: bool
    canonical_key: str


@dataclass
class LabelMappingEntry:
    """Mapping from one raw dataset label to the universal schema.

    Attributes:
        raw_label: Original class label from the dataset.
        dataset_name: Source dataset identifier.
        image_count: Number of images observed for this raw label (if known).
        standard_label: Canonical label when successfully mapped.
        status: Mapping outcome status.
        parser: Parser used for this label.
        notes: Optional parsing remarks or warnings.
    """

    raw_label: str
    dataset_name: str
    image_count: int | None
    standard_label: StandardLabel | None
    status: LabelParseStatus
    parser: LabelParserType
    notes: str | None = None


@dataclass
class LabelIssue:
    """A label mapping issue requiring review.

    Attributes:
        issue_type: Category of issue (unknown, ambiguous, duplicate, unmapped).
        raw_label: Original label that triggered the issue.
        dataset_name: Source dataset identifier.
        message: Human-readable explanation.
        candidates: Optional alternative standard labels for ambiguous cases.
    """

    issue_type: str
    raw_label: str
    dataset_name: str
    message: str
    candidates: list[str] = field(default_factory=list)


@dataclass
class DatasetLabelReport:
    """Label standardization results for one dataset.

    Attributes:
        dataset_name: Source dataset identifier.
        total_raw_labels: Number of distinct raw labels discovered.
        mapped_labels: Number of successfully mapped labels.
        mappings: All label mapping entries for this dataset.
    """

    dataset_name: str
    total_raw_labels: int
    mapped_labels: int
    mappings: list[LabelMappingEntry]


@dataclass
class LabelStandardizationReport:
    """Full label standardization report across datasets.

    Attributes:
        generated_at: UTC ISO-8601 timestamp.
        source: Description of where raw labels were discovered.
        datasets: Per-dataset label reports keyed by dataset name.
        canonical_labels: Sorted list of unique canonical labels.
        unknown_labels: Labels that could not be parsed.
        ambiguous_labels: Labels with uncertain or multiple interpretations.
        duplicate_mappings: Raw labels sharing the same canonical target.
        unmapped_labels: Labels discovered but not mapped to the universal schema.
    """

    generated_at: str
    source: str
    datasets: dict[str, DatasetLabelReport]
    canonical_labels: list[StandardLabel]
    unknown_labels: list[LabelIssue]
    ambiguous_labels: list[LabelIssue]
    duplicate_mappings: list[LabelIssue]
    unmapped_labels: list[LabelIssue]


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _collapse_whitespace(text: str) -> str:
    """Collapse repeated whitespace to a single space."""
    return re.sub(r"\s+", " ", text.strip())


def _replace_separators(text: str) -> str:
    """Replace underscores and hyphens with spaces."""
    return re.sub(r"[_\-]+", " ", text)


def normalize_text(text: str) -> str:
    """Normalize separators and whitespace in a label fragment."""
    return _collapse_whitespace(_replace_separators(text))


def title_case_label(text: str) -> str:
    """Convert a normalized fragment to title case."""
    normalized = normalize_text(text)
    if not normalized:
        return normalized
    return " ".join(word.capitalize() for word in normalized.split(" "))


def normalize_plant_name(raw_plant: str) -> str | None:
    """Normalize a raw plant name to a canonical crop label.

    Args:
        raw_plant: Raw plant fragment from a dataset label.

    Returns:
        Canonical plant name, or ``None`` if unrecognized.
    """
    cleaned = normalize_text(raw_plant).lower()
    cleaned = cleaned.replace(",", " ,").replace("(", " (").replace("  ", " ")

    for alias, canonical in PLANT_ALIASES:
        if cleaned == alias or cleaned.startswith(alias):
            return canonical

    for alias, canonical in PLANT_ALIASES:
        alias_norm = normalize_text(alias).lower()
        if cleaned == alias_norm:
            return canonical

    # Direct match against registered plant names.
    simple = cleaned.split("(")[0].strip()
    if simple in VALID_PLANT_NAMES:
        return title_case_label(simple)

    return None


def normalize_disease_name(raw_disease: str) -> str:
    """Normalize a raw disease fragment to a canonical disease label.

    Args:
        raw_disease: Raw disease fragment from a dataset label.

    Returns:
        Canonical disease name in title case.
    """
    cleaned = normalize_text(raw_disease).lower().strip(" _")
    if cleaned in DISEASE_ALIASES:
        return DISEASE_ALIASES[cleaned]

    titled = title_case_label(raw_disease)
    alias_key = titled.lower()
    return DISEASE_ALIASES.get(alias_key, titled)


def build_canonical_key(plant: str, disease: str, is_healthy: bool) -> str:
    """Build a stable cross-dataset label key.

    Args:
        plant: Canonical plant name.
        disease: Canonical disease name.
        is_healthy: Whether the label is healthy.

    Returns:
        A lowercase slug key (``plant|disease``).
    """
    plant_slug = plant.lower().replace(" ", "_")
    disease_slug = "healthy" if is_healthy else disease.lower().replace(" ", "_")
    return f"{plant_slug}|{disease_slug}"


def build_standard_label(plant: str, disease: str, is_healthy: bool) -> StandardLabel:
    """Construct a :class:`StandardLabel` with canonical key."""
    canonical_disease = "Healthy" if is_healthy else disease
    return StandardLabel(
        plant=plant,
        disease=canonical_disease,
        is_healthy=is_healthy,
        canonical_key=build_canonical_key(plant, canonical_disease, is_healthy),
    )


# ---------------------------------------------------------------------------
# Dataset-specific parsers
# ---------------------------------------------------------------------------


def parse_plantvillage_label(raw_label: str) -> tuple[StandardLabel | None, LabelParseStatus, str | None]:
    """Parse a PlantVillage folder label (``Plant___Disease``).

    Args:
        raw_label: Raw PlantVillage class folder name.

    Returns:
        Tuple of standard label, parse status, and optional notes.
    """
    if PLANTVILLAGE_SEPARATOR not in raw_label:
        return None, LabelParseStatus.UNKNOWN, "Missing PlantVillage separator '___'."

    plant_part, disease_part = raw_label.split(PLANTVILLAGE_SEPARATOR, 1)
    plant = normalize_plant_name(plant_part)
    if plant is None:
        return None, LabelParseStatus.UNKNOWN, f"Unrecognized plant: '{plant_part}'."

    disease_token = disease_part.strip().rstrip("_")
    is_healthy = disease_token.lower() == HEALTHY_TOKEN

    if is_healthy:
        return (
            build_standard_label(plant, "Healthy", True),
            LabelParseStatus.MAPPED,
            None,
        )

    disease = normalize_disease_name(disease_part)
    if not disease:
        return None, LabelParseStatus.UNMAPPED, "Empty disease fragment."

    return (
        build_standard_label(plant, disease, False),
        LabelParseStatus.MAPPED,
        None,
    )


def _clean_disease_remainder(remainder: str) -> str:
    """Remove redundant ``leaf`` tokens from a PlantDoc disease fragment."""
    cleaned = normalize_text(remainder)
    cleaned = re.sub(r"^leaf\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _strip_plantdoc_suffix(text: str) -> str:
    """Remove trailing ``leaf`` tokens from a PlantDoc label fragment."""
    cleaned = text.strip()
    cleaned = re.sub(r"\s+leaf\s*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _match_plant_prefix(text: str) -> tuple[str | None, str, list[str]]:
    """Match a known plant prefix at the start of a PlantDoc label.

    Args:
        text: Label text after suffix stripping.

    Returns:
        Tuple of canonical plant, remaining text, and candidate plants if ambiguous.
    """
    lowered = normalize_text(text).lower()
    matches: list[tuple[str, str, int]] = []

    for alias, canonical in PLANT_ALIASES:
        alias_norm = normalize_text(alias).lower()
        if lowered == alias_norm:
            return canonical, "", []
        if lowered.startswith(alias_norm + " "):
            remainder = text[len(alias_norm) :].strip()
            matches.append((canonical, remainder, len(alias_norm)))

    if not matches:
        return None, text, []

    matches.sort(key=lambda item: item[2], reverse=True)
    best = matches[0]
    candidates = sorted({match[0] for match in matches if match[2] == best[2]})
    if len(candidates) > 1:
        return None, text, candidates

    return best[0], best[1], []


def parse_plantdoc_label(raw_label: str) -> tuple[StandardLabel | None, LabelParseStatus, str | None]:
    """Parse a PlantDoc folder label (``Plant [disease] leaf``).

    Args:
        raw_label: Raw PlantDoc class folder name.

    Returns:
        Tuple of standard label, parse status, and optional notes.
    """
    stripped = _strip_plantdoc_suffix(raw_label)
    plant, remainder, candidates = _match_plant_prefix(stripped)

    if candidates:
        return (
            None,
            LabelParseStatus.AMBIGUOUS,
            f"Multiple plant matches: {', '.join(candidates)}.",
        )

    if plant is None:
        return None, LabelParseStatus.UNKNOWN, f"Unrecognized plant in '{raw_label}'."

    remainder = _strip_plantdoc_suffix(remainder)
    remainder = _clean_disease_remainder(remainder)

    if not remainder:
        return (
            build_standard_label(plant, "Healthy", True),
            LabelParseStatus.MAPPED,
            None,
        )

    if remainder.lower() in {"leaf", "leaves"}:
        return (
            build_standard_label(plant, "Healthy", True),
            LabelParseStatus.MAPPED,
            None,
        )

    disease = normalize_disease_name(remainder)
    return (
        build_standard_label(plant, disease, False),
        LabelParseStatus.MAPPED,
        None,
    )


def parse_generic_label(raw_label: str) -> tuple[StandardLabel | None, LabelParseStatus, str | None]:
    """Attempt generic parsing for unsupported datasets.

    Args:
        raw_label: Raw class label.

    Returns:
        Tuple of standard label, parse status, and optional notes.
    """
    if PLANTVILLAGE_SEPARATOR in raw_label:
        return parse_plantvillage_label(raw_label)

    if raw_label.lower().endswith("leaf") or raw_label.lower().endswith(" leaf"):
        return parse_plantdoc_label(raw_label)

    return None, LabelParseStatus.UNKNOWN, "No compatible parser for this label format."


def get_parser_for_dataset(dataset_name: str) -> LabelParserType:
    """Resolve the parser type for a dataset name.

    Args:
        dataset_name: Dataset directory name.

    Returns:
        Parser enum value for the dataset.
    """
    normalized = dataset_name.lower()
    if normalized == DatasetName.PLANTVILLAGE.value:
        return LabelParserType.PLANTVILLAGE
    if normalized == DatasetName.PLANTDOC.value:
        return LabelParserType.PLANTDOC
    return LabelParserType.GENERIC


def parse_raw_label(
    raw_label: str,
    dataset_name: str,
) -> tuple[StandardLabel | None, LabelParseStatus, LabelParserType, str | None]:
    """Parse a raw label using the appropriate dataset parser.

    Args:
        raw_label: Original class label.
        dataset_name: Source dataset name.

    Returns:
        Tuple of standard label, status, parser type, and optional notes.
    """
    parser = get_parser_for_dataset(dataset_name)

    if parser == LabelParserType.PLANTVILLAGE:
        standard, status, notes = parse_plantvillage_label(raw_label)
    elif parser == LabelParserType.PLANTDOC:
        standard, status, notes = parse_plantdoc_label(raw_label)
    else:
        standard, status, notes = parse_generic_label(raw_label)

    return standard, status, parser, notes


# ---------------------------------------------------------------------------
# Label discovery (read-only)
# ---------------------------------------------------------------------------


def _extract_raw_label_from_path(relative_class_path: str) -> str:
    """Extract the leaf class folder name from an audited class path."""
    normalized = relative_class_path.replace("\\", "/")
    return normalized.split("/")[-1]


def discover_labels_from_statistics(
    statistics_path: Path | str = DEFAULT_STATISTICS_PATH,
) -> dict[str, dict[str, int]]:
    """Discover raw labels from a dataset audit statistics JSON file.

    Args:
        statistics_path: Path to ``dataset_statistics.json``.

    Returns:
        Mapping of dataset name to ``{raw_label: image_count}``.
    """
    path = Path(statistics_path)
    if not path.exists():
        logger.warning("Statistics file not found: %s", path)
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    aggregate: dict[str, int] = payload.get("aggregate_images_per_class", {})

    labels: dict[str, dict[str, int]] = {}
    for composite_key, count in aggregate.items():
        if "/" not in composite_key:
            continue
        dataset_name, class_path = composite_key.split("/", 1)
        raw_label = _extract_raw_label_from_path(class_path)
        labels.setdefault(dataset_name, {})
        labels[dataset_name][raw_label] = labels[dataset_name].get(raw_label, 0) + count

    logger.info(
        "Discovered labels for %d dataset(s) from %s",
        len(labels),
        path,
    )
    return labels


def discover_labels_from_external_dir(
    external_dir: Path | str = DEFAULT_EXTERNAL_DIR,
) -> dict[str, dict[str, int]]:
    """Discover raw class labels by scanning external dataset directories.

    Uses folder structure only — no images are read or modified.

    Args:
        external_dir: Root directory containing ingested datasets.

    Returns:
        Mapping of dataset name to ``{raw_label: 0}`` (counts unknown).
    """
    root = Path(external_dir)
    if not root.exists():
        logger.warning("External directory not found: %s", root)
        return {}

    labels: dict[str, dict[str, int]] = {}
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
            continue

        dataset_labels: set[str] = set()
        for path in dataset_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower().lstrip(".") not in {"jpg", "jpeg", "png", "bmp", "tiff", "webp"}:
                continue
            relative = path.relative_to(dataset_dir)
            if len(relative.parts) >= 2:
                dataset_labels.add(relative.parts[-2])

        if dataset_labels:
            labels[dataset_dir.name] = {label: 0 for label in sorted(dataset_labels)}

    logger.info("Discovered labels for %d dataset(s) from %s", len(labels), root)
    return labels


def discover_raw_labels(
    *,
    statistics_path: Path | str = DEFAULT_STATISTICS_PATH,
    external_dir: Path | str = DEFAULT_EXTERNAL_DIR,
) -> tuple[dict[str, dict[str, int]], str]:
    """Discover raw labels from audit statistics or external directories.

    Prefers audit statistics when available for accurate image counts.

    Args:
        statistics_path: Path to audit statistics JSON.
        external_dir: Fallback scan directory.

    Returns:
        Tuple of label mapping and source description string.
    """
    labels = discover_labels_from_statistics(statistics_path)
    if labels:
        return labels, str(Path(statistics_path))

    labels = discover_labels_from_external_dir(external_dir)
    return labels, str(Path(external_dir))


# ---------------------------------------------------------------------------
# Report building
# ---------------------------------------------------------------------------


def standardize_dataset_labels(
    dataset_name: str,
    raw_labels: dict[str, int],
) -> DatasetLabelReport:
    """Standardize all raw labels for one dataset.

    Args:
        dataset_name: Source dataset identifier.
        raw_labels: Mapping of raw label to image count.

    Returns:
        A :class:`DatasetLabelReport` for the dataset.
    """
    mappings: list[LabelMappingEntry] = []

    for raw_label in sorted(raw_labels.keys()):
        standard, status, parser, notes = parse_raw_label(raw_label, dataset_name)
        mappings.append(
            LabelMappingEntry(
                raw_label=raw_label,
                dataset_name=dataset_name,
                image_count=raw_labels[raw_label] or None,
                standard_label=standard,
                status=status,
                parser=parser,
                notes=notes,
            )
        )

    mapped = sum(1 for entry in mappings if entry.status == LabelParseStatus.MAPPED)
    logger.info(
        "Standardized %d/%d labels for dataset '%s'",
        mapped,
        len(mappings),
        dataset_name,
    )

    return DatasetLabelReport(
        dataset_name=dataset_name,
        total_raw_labels=len(mappings),
        mapped_labels=mapped,
        mappings=mappings,
    )


def _collect_issues(report: LabelStandardizationReport) -> None:
    """Populate issue lists on a standardization report."""
    canonical_to_raw: dict[str, list[tuple[str, str]]] = {}

    for dataset_report in report.datasets.values():
        for entry in dataset_report.mappings:
            if entry.status == LabelParseStatus.UNKNOWN:
                report.unknown_labels.append(
                    LabelIssue(
                        issue_type="unknown",
                        raw_label=entry.raw_label,
                        dataset_name=entry.dataset_name,
                        message=entry.notes or "Label format not recognized.",
                    )
                )
            elif entry.status == LabelParseStatus.AMBIGUOUS:
                report.ambiguous_labels.append(
                    LabelIssue(
                        issue_type="ambiguous",
                        raw_label=entry.raw_label,
                        dataset_name=entry.dataset_name,
                        message=entry.notes or "Ambiguous label interpretation.",
                    )
                )
            elif entry.status == LabelParseStatus.UNMAPPED:
                report.unmapped_labels.append(
                    LabelIssue(
                        issue_type="unmapped",
                        raw_label=entry.raw_label,
                        dataset_name=entry.dataset_name,
                        message=entry.notes or "Label could not be mapped.",
                    )
                )
            elif entry.standard_label is not None:
                key = entry.standard_label.canonical_key
                canonical_to_raw.setdefault(key, []).append(
                    (entry.dataset_name, entry.raw_label)
                )

    for canonical_key, raw_entries in sorted(canonical_to_raw.items()):
        if len(raw_entries) < 2:
            continue
        raw_descriptions = [f"{dataset}:{label}" for dataset, label in raw_entries]
        report.duplicate_mappings.append(
            LabelIssue(
                issue_type="duplicate_mapping",
                raw_label=canonical_key,
                dataset_name="cross-dataset",
                message=(
                    f"{len(raw_entries)} raw labels map to canonical key "
                    f"'{canonical_key}'."
                ),
                candidates=raw_descriptions,
            )
        )


def build_standardization_report(
    raw_labels_by_dataset: dict[str, dict[str, int]],
    *,
    source: str,
) -> LabelStandardizationReport:
    """Build a full label standardization report.

    Args:
        raw_labels_by_dataset: Mapping of dataset name to raw labels and counts.
        source: Description of label discovery source.

    Returns:
        A completed :class:`LabelStandardizationReport`.
    """
    datasets = {
        name: standardize_dataset_labels(name, labels)
        for name, labels in sorted(raw_labels_by_dataset.items())
    }

    canonical_map: dict[str, StandardLabel] = {}
    for dataset_report in datasets.values():
        for entry in dataset_report.mappings:
            if entry.standard_label is not None:
                canonical_map[entry.standard_label.canonical_key] = entry.standard_label

    report = LabelStandardizationReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        source=source,
        datasets=datasets,
        canonical_labels=sorted(
            canonical_map.values(),
            key=lambda label: (label.plant, label.disease),
        ),
        unknown_labels=[],
        ambiguous_labels=[],
        duplicate_mappings=[],
        unmapped_labels=[],
    )
    _collect_issues(report)
    return report


# ---------------------------------------------------------------------------
# Serialization and report output
# ---------------------------------------------------------------------------


def _standard_label_to_dict(label: StandardLabel) -> dict:
    """Serialize a standard label to a dictionary."""
    return asdict(label)


def _mapping_entry_to_dict(entry: LabelMappingEntry) -> dict:
    """Serialize a mapping entry to a dictionary."""
    return {
        "raw_label": entry.raw_label,
        "dataset_name": entry.dataset_name,
        "image_count": entry.image_count,
        "standard_label": (
            _standard_label_to_dict(entry.standard_label)
            if entry.standard_label
            else None
        ),
        "status": entry.status.value,
        "parser": entry.parser.value,
        "notes": entry.notes,
    }


def serialize_standardization_report(report: LabelStandardizationReport) -> dict:
    """Convert a standardization report to a JSON-serializable dictionary.

    Args:
        report: Report to serialize.

    Returns:
        Dictionary suitable for JSON export.
    """
    return {
        "generated_at": report.generated_at,
        "source": report.source,
        "summary": {
            "datasets": len(report.datasets),
            "total_raw_labels": sum(
                dataset.total_raw_labels for dataset in report.datasets.values()
            ),
            "mapped_labels": sum(
                dataset.mapped_labels for dataset in report.datasets.values()
            ),
            "canonical_labels": len(report.canonical_labels),
            "unknown_labels": len(report.unknown_labels),
            "ambiguous_labels": len(report.ambiguous_labels),
            "duplicate_mappings": len(report.duplicate_mappings),
            "unmapped_labels": len(report.unmapped_labels),
        },
        "canonical_labels": [
            _standard_label_to_dict(label) for label in report.canonical_labels
        ],
        "datasets": {
            name: {
                "dataset_name": dataset.dataset_name,
                "total_raw_labels": dataset.total_raw_labels,
                "mapped_labels": dataset.mapped_labels,
                "mappings": [_mapping_entry_to_dict(entry) for entry in dataset.mappings],
            }
            for name, dataset in sorted(report.datasets.items())
        },
        "issues": {
            "unknown_labels": [asdict(issue) for issue in report.unknown_labels],
            "ambiguous_labels": [asdict(issue) for issue in report.ambiguous_labels],
            "duplicate_mappings": [asdict(issue) for issue in report.duplicate_mappings],
            "unmapped_labels": [asdict(issue) for issue in report.unmapped_labels],
        },
    }


def save_label_mapping_json(
    report: LabelStandardizationReport,
    output_path: Path | str = DEFAULT_MAPPING_PATH,
) -> Path:
    """Save the label mapping report as JSON.

    Args:
        report: Standardization report to export.
        output_path: Destination JSON path.

    Returns:
        Path the file was written to.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(serialize_standardization_report(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved label mapping JSON to %s", path)
    return path


def _format_standard_label(label: StandardLabel) -> str:
    """Format a standard label for markdown display."""
    health = "Yes" if label.is_healthy else "No"
    return f"plant=`{label.plant}`, disease=`{label.disease}`, healthy={health}"


def generate_standardization_markdown(
    report: LabelStandardizationReport,
    output_path: Path | str = DEFAULT_REPORT_PATH,
) -> Path:
    """Write a human-readable label standardization markdown report.

    Args:
        report: Standardization report to document.
        output_path: Destination markdown path.

    Returns:
        Path the file was written to.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    total_raw = sum(dataset.total_raw_labels for dataset in report.datasets.values())
    total_mapped = sum(dataset.mapped_labels for dataset in report.datasets.values())

    lines = [
        "# Label Standardization Report",
        "",
        f"**Generated:** {report.generated_at}  ",
        f"**Label source:** `{report.source}`  ",
        f"**Datasets:** {len(report.datasets)}  ",
        f"**Raw labels:** {total_raw}  ",
        f"**Mapped labels:** {total_mapped}  ",
        f"**Canonical labels:** {len(report.canonical_labels)}  ",
        "",
        "> Read-only label mapping. No images or datasets were modified.",
        "",
        "## Universal Label Schema",
        "",
        "Every raw class label is mapped to:",
        "",
        "| Field | Description | Example |",
        "|-------|-------------|---------|",
        "| `plant` | Canonical crop name | `Tomato` |",
        "| `disease` | Canonical disease name | `Early Blight` |",
        "| `is_healthy` | Healthy sample flag | `false` |",
        "| `canonical_key` | Cross-dataset slug | `tomato|early_blight` |",
        "",
        "## Mapping Examples",
        "",
        "### PlantVillage",
        "",
        "| Raw Label | Standard Label |",
        "|-----------|----------------|",
        "| `Tomato___Early_blight` | plant=`Tomato`, disease=`Early Blight`, healthy=No |",
        "| `Apple___healthy` | plant=`Apple`, disease=`Healthy`, healthy=Yes |",
        "",
        "### PlantDoc",
        "",
        "| Raw Label | Standard Label |",
        "|-----------|----------------|",
        "| `Tomato Early blight leaf` | plant=`Tomato`, disease=`Early Blight`, healthy=No |",
        "| `Tomato leaf` | plant=`Tomato`, disease=`Healthy`, healthy=Yes |",
        "",
        "## Issue Summary",
        "",
        f"- **Unknown labels:** {len(report.unknown_labels)}",
        f"- **Ambiguous labels:** {len(report.ambiguous_labels)}",
        f"- **Duplicate mappings:** {len(report.duplicate_mappings)}",
        f"- **Unmapped labels:** {len(report.unmapped_labels)}",
        "",
    ]

    for issue_type, issues, heading in (
        ("unknown", report.unknown_labels, "Unknown Labels"),
        ("ambiguous", report.ambiguous_labels, "Ambiguous Labels"),
        ("duplicate_mapping", report.duplicate_mappings, "Duplicate Mappings"),
        ("unmapped", report.unmapped_labels, "Unmapped Labels"),
    ):
        lines.extend([f"## {heading}", ""])
        if not issues:
            lines.append("None.")
            lines.append("")
            continue

        for issue in issues:
            lines.append(
                f"- **`{issue.dataset_name}`** / `{issue.raw_label}` — {issue.message}"
            )
            if issue.candidates:
                lines.append(f"  - Candidates: {', '.join(f'`{c}`' for c in issue.candidates[:10])}")
                if len(issue.candidates) > 10:
                    lines.append(f"  - ... and {len(issue.candidates) - 10} more")
        lines.append("")

    lines.extend(["## Canonical Labels", ""])
    for label in report.canonical_labels:
        lines.append(f"- `{label.canonical_key}` → {_format_standard_label(label)}")
    lines.append("")

    for dataset_name in sorted(report.datasets.keys()):
        dataset_report = report.datasets[dataset_name]
        lines.extend(
            [
                f"## Dataset: `{dataset_name}`",
                "",
                f"- Raw labels: {dataset_report.total_raw_labels}",
                f"- Mapped: {dataset_report.mapped_labels}",
                "",
                "| Raw Label | Plant | Disease | Healthy | Status | Images |",
                "|-----------|-------|---------|---------|--------|--------|",
            ]
        )
        for entry in dataset_report.mappings:
            if entry.standard_label:
                plant = entry.standard_label.plant
                disease = entry.standard_label.disease
                healthy = "Yes" if entry.standard_label.is_healthy else "No"
            else:
                plant = disease = healthy = "—"
            count = entry.image_count if entry.image_count is not None else "—"
            lines.append(
                f"| `{entry.raw_label}` | {plant} | {disease} | {healthy} | "
                f"{entry.status.value} | {count} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved label standardization markdown to %s", path)
    return path


def run_label_standardization(
    *,
    statistics_path: Path | str = DEFAULT_STATISTICS_PATH,
    external_dir: Path | str = DEFAULT_EXTERNAL_DIR,
    mapping_path: Path | str = DEFAULT_MAPPING_PATH,
    report_path: Path | str = DEFAULT_REPORT_PATH,
) -> LabelStandardizationReport:
    """Run the full read-only label standardization pipeline.

    Discovers raw labels, maps them to the universal schema, and writes JSON
    and markdown reports under ``reports/``.

    Args:
        statistics_path: Path to audit statistics for label discovery.
        external_dir: Fallback directory for label discovery.
        mapping_path: Output JSON mapping path.
        report_path: Output markdown report path.

    Returns:
        The completed :class:`LabelStandardizationReport`.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    logger.info("Starting label standardization pipeline")
    raw_labels, source = discover_raw_labels(
        statistics_path=statistics_path,
        external_dir=external_dir,
    )

    if not raw_labels:
        logger.warning("No raw labels discovered. Reports will be empty.")

    report = build_standardization_report(raw_labels, source=source)
    save_label_mapping_json(report, mapping_path)
    generate_standardization_markdown(report, report_path)

    logger.info(
        "Label standardization complete: %d canonical labels, %d issues",
        len(report.canonical_labels),
        (
            len(report.unknown_labels)
            + len(report.ambiguous_labels)
            + len(report.unmapped_labels)
        ),
    )
    return report


if __name__ == "__main__":
    run_label_standardization()
