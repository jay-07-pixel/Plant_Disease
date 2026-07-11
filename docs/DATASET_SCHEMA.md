# PlantDiseaseAI — Universal Dataset Schema

This document defines the canonical metadata schema used across all datasets in PlantDiseaseAI. Every image record—regardless of origin—must conform to this structure before entering preprocessing, training, evaluation, or deployment pipelines.

## Why a Common Schema Matters

Plant disease research datasets are heterogeneous. PlantVillage organizes images by `Plant___Disease` folder names. PDDB uses different naming conventions and metadata fields. Future sources such as PlantDoc will introduce additional variations.

Without a unified schema:

- Preprocessing code must be rewritten for each dataset.
- Train/validation/test splits become inconsistent and leak across sources.
- Evaluation metrics cannot be compared fairly across experiments.
- The API and chatbot cannot serve predictions with reliable, structured context.
- Experiment tracking loses reproducibility when metadata formats diverge.

A single schema acts as a **contract** between data ingestion, feature engineering, model training, explainability, and production services. Each pipeline reads and writes the same `PlantDiseaseRecord` structure, enabling modular development and safe dataset expansion.

## Schema Overview

Each image in the system is represented by one `PlantDiseaseRecord` with the following fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image_id` | `str` | Yes | Globally unique identifier (e.g. UUID or `{dataset}_{hash}`). |
| `image_path` | `str` | Yes | Relative or absolute path to the image file. |
| `dataset_name` | `DatasetName` | Yes | Source dataset identifier (`plantvillage`, `pddb`, `plantdoc`, etc.). |
| `plant_name` | `str` | Yes | Canonical plant species or crop name (e.g. `tomato`, `apple`). |
| `disease_name` | `str` | Yes | Canonical disease label; use `healthy` when `is_healthy` is true. |
| `is_healthy` | `bool` | Yes | Whether the plant is disease-free. |
| `disease_category` | `DiseaseCategory` | Yes | High-level disease type (fungal, bacterial, viral, etc.). |
| `severity` | `SeverityLevel \| None` | No | Optional severity estimate when available. |
| `image_source` | `ImageSource` | Yes | How the image was captured (field, lab, web, etc.). |
| `annotation_status` | `AnnotationStatus` | Yes | Quality and completeness of labels. |
| `split` | `DatasetSplit` | Yes | Data partition (`train`, `val`, `test`). |
| `image_width` | `int` | Yes | Image width in pixels. |
| `image_height` | `int` | Yes | Image height in pixels. |
| `image_format` | `str` | Yes | File extension without dot (e.g. `jpg`, `png`). |
| `notes` | `str \| None` | No | Free-text remarks (collection conditions, caveats). |

Implementation lives in `src/config/dataset_schema.py`.

## Validation Rules

The schema enforces the following invariants at ingest time:

1. **Required fields** — All mandatory fields must be present and non-empty.
2. **Valid plant names** — `plant_name` must match a registered canonical name in `VALID_PLANT_NAMES`.
3. **Healthy consistency** — When `is_healthy` is `true`, `disease_name` must be `healthy` and `disease_category` must be `healthy`.
4. **Diseased consistency** — When `is_healthy` is `false`, `disease_name` must not be `healthy`.
5. **Valid image extensions** — `image_format` must be one of the supported formats (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`).
6. **Missing metadata reporting** — Optional fields that are absent are reported as warnings, not hard failures.

Validation results are returned as `ValidationResult` objects with `is_valid`, `errors`, and `warnings` lists.

## Merging Multiple Datasets

Future dataset integration follows a three-stage pipeline:

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────────┐
│ Raw Dataset │ ──► │ Adapter Layer │ ──► │ PlantDiseaseRecord  │
│ (any format)│     │ (per source)  │     │ (canonical schema)  │
└─────────────┘     └──────────────┘     └─────────────────────┘
                                                    │
                                                    ▼
                                          ┌─────────────────────┐
                                          │ Unified metadata    │
                                          │ (CSV / Parquet)     │
                                          └─────────────────────┘
```

### Adapter pattern

Each dataset gets a dedicated adapter that:

1. Reads the native layout (folders, JSON, CSV, etc.).
2. Maps native labels to canonical `plant_name` and `disease_name` values.
3. Populates `dataset_name`, `image_source`, and `annotation_status`.
4. Assigns or preserves `split` assignments.
5. Emits validated `PlantDiseaseRecord` instances.

Adapters are **not** implemented in this phase. The schema ensures they have a clear target structure.

### Merge considerations

| Concern | Schema support |
|---------|----------------|
| Duplicate images | Unique `image_id` per record; collisions detected at merge time. |
| Label harmonization | Canonical `plant_name` / `disease_name` registry in config. |
| Cross-dataset splits | `split` field prevents leakage when combining sources. |
| Provenance tracking | `dataset_name` and `image_source` preserve origin. |
| Partial labels | `annotation_status` flags records needing review. |

## Benefits by Pipeline Stage

### Preprocessing

- Uniform column names enable a single `DataLoader` and augmentation pipeline.
- `image_path`, `image_width`, `image_height`, and `image_format` support resize and normalization decisions.
- `split` drives deterministic train/val/test partitioning without re-scanning folders.

### Training

- `plant_name` and `disease_name` map directly to class indices via a shared label encoder.
- `is_healthy` supports binary healthy-vs-diseased auxiliary tasks.
- `disease_category` enables hierarchical or multi-task learning.
- `dataset_name` supports domain-adaptation and dataset-weighted sampling.

### Evaluation

- Per-dataset metrics via `dataset_name` grouping.
- Fair comparison when evaluating on held-out `test` split only.
- `annotation_status` filters low-confidence labels from metric computation.

### Deployment (API & Chatbot)

- Structured metadata enriches prediction responses (plant, disease, severity, source).
- `image_id` links inference logs back to training records for audit trails.
- Consistent labels prevent mismatches between model output and user-facing text.

## Supported Datasets (Planned)

| Dataset | `DatasetName` value | Status |
|---------|---------------------|--------|
| PlantVillage | `plantvillage` | Planned |
| PDDB | `pddb` | Planned |
| PlantDoc | `plantdoc` | Future |
| Other | `other` | Future |

## File Layout Convention

Raw data remains in `datasets/raw/{dataset_name}/`. Canonical metadata exports are written to `datasets/processed/metadata/` as CSV or Parquet files produced by `export_metadata()`.

## API Reference

```python
from src.config.dataset_schema import (
    PlantDiseaseRecord,
    load_dataset_metadata,
    validate_record,
    export_metadata,
)

# Placeholder — not yet implemented
records = load_dataset_metadata("datasets/processed/metadata/unified.csv")
result = validate_record(records[0])
export_metadata(records, "datasets/processed/metadata/unified.csv")
```

## Next Steps

1. Implement per-dataset adapters (PlantVillage, PDDB).
2. Build label harmonization tables mapping native names to canonical names.
3. Wire `load_dataset_metadata()` and `export_metadata()` to CSV/Parquet I/O.
4. Add unit tests under `tests/` for all validation rules.
