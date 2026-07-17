# PlantDiseaseAI

Production-quality AI system for plant disease detection, explanation, and assistance.

PlantDiseaseAI is a full-stack machine learning project designed to detect plant diseases from leaf images, explain model predictions, and assist users through an AI chatbot. The data engineering pipeline is built first — every stage is **read-only** until preprocessing is explicitly implemented.

## Vision

| Component | Status |
|-----------|--------|
| Project scaffold & configuration | Done |
| Universal dataset schema | Done |
| Dataset ingestion (ZIP extraction) | Done |
| Dataset audit & EDA | Done |
| Multi-dataset comparison | Done |
| Label standardization | Done |
| Data quality control | Done |
| Image preprocessing | Done |
| Dataset balancing strategy | Done |
| PyTorch training engine | Done |
| Transfer-learning experiments | Done |
| Grad-CAM explainability | Planned |
| FastAPI backend | Planned |
| Flutter frontend | Planned |
| AI chatbot | Planned |
| Experiment tracking (TensorBoard) | Done |

## Datasets

Two public datasets are currently ingested:

| Dataset | Location | Files | Description |
|---------|----------|-------|-------------|
| **PlantVillage** | `datasets/external/plantvillage/` | ~54,305 | Lab images, `Plant___Disease` folder labels |
| **PlantDoc** | `datasets/external/plantdoc/` | ~2,552 | Field/web images, natural-language folder labels |

Raw ZIP archives live in `datasets/raw/`. Extracted data lives in `datasets/external/`. **No images have been modified, merged, or preprocessed.**

## Data Pipeline (Completed)

The pipeline runs in order. Each stage only reads or extracts data — nothing is deleted, renamed, resized, or transformed.

```
datasets/raw/*.zip
       │
       ▼  Ingestion
datasets/external/{plantvillage, plantdoc}/
       │
       ├──▶ Audit & EDA ──────────▶ reports/
       ├──▶ Multi-dataset comparison ▶ reports/comparison/
       ├──▶ Label standardization ──▶ reports/label_mapping.json
       └──▶ Data quality scan ─────▶ reports/data_quality.json
```

### 1. Project Scaffold

Industry-standard layout with separated concerns for data, models, training, API, and deployment.

- `config.yaml` — paths, training defaults, API settings
- `requirements.txt` — Python dependencies
- `main.py` — application entry point (stub)
- Package structure under `src/` with `__init__.py` throughout

### 2. Universal Dataset Schema

**Docs:** `docs/DATASET_SCHEMA.md`  
**Code:** `src/config/dataset_schema.py`

Defines the canonical metadata contract used across all datasets:

- `PlantDiseaseRecord` dataclass — one record per image
- Enums for dataset name, split, disease category, image source, annotation status
- Validation rules (required fields, healthy/disease consistency, image formats)
- Placeholder I/O: `load_dataset_metadata()`, `validate_record()`, `export_metadata()`

Every future dataset adapter maps native labels into this schema before training or deployment.

### 3. Dataset Ingestion

**Modules:** `src/data/ingest.py`, `extract.py`, `verify.py`, `registry.py`

- Discovers all `*.zip` files in `datasets/raw/`
- Validates ZIP integrity, checks for zip-slip paths, verifies file counts
- Extracts to `datasets/external/<dataset_name>/`
- Skips re-extraction when data already exists
- Maintains `reports/dataset_registry.json` (name, version, file count, extraction date)

```bash
python -m src.data.ingest
```

### 4. Dataset Audit & EDA

**Modules:** `src/data/dataset_audit.py`, `eda.py`

Read-only analysis of datasets under `datasets/external/`:

- Image and class counts, class imbalance, resolution and format statistics
- Corrupted images, empty folders, non-image files
- Perceptual duplicate detection (reported, not removed)
- Duplicate filenames, directory depth

**Outputs in `reports/`:**

| File | Description |
|------|-------------|
| `dataset_audit.md` | Human-readable audit report |
| `dataset_statistics.json` | Machine-readable statistics |
| `class_distribution.png` | Class distribution (all, top 20, bottom 20) |
| `image_resolution.png` | Resolution histogram |
| `image_formats.png` | Format pie chart |

```bash
python -m src.data.dataset_audit
```

### 5. Multi-Dataset Comparison

**Module:** `src/data/dataset_comparison.py` (integrated with audit)

Cross-dataset comparison for PlantVillage vs PlantDoc:

- Classes, images, average resolution, formats per dataset
- Common, unique, and missing class labels (normalized leaf names)
- Side-by-side visualizations

**Outputs in `reports/comparison/`:**

| File | Description |
|------|-------------|
| `dataset_comparison.md` | Comparison report |
| `dataset_comparison.json` | Comparison data |
| `dataset_sizes.png` | Images and classes per dataset |
| `class_overlap.png` | Common / shared / unique classes |
| `resolution_comparison.png` | Average resolution by dataset |

### 6. Label Standardization

**Module:** `src/data/label_standardizer.py`

Maps dataset-specific folder labels to a universal schema:

| Field | Example |
|-------|---------|
| `plant` | `Tomato` |
| `disease` | `Early Blight` |
| `is_healthy` | `false` |
| `canonical_key` | `tomato\|early_blight` |

**Supported parsers:**

- **PlantVillage** — `Tomato___Early_blight` → Tomato / Early Blight
- **PlantDoc** — `Tomato Early blight leaf` → Tomato / Early Blight

Normalizes underscores, hyphens, spaces, capitalization, plant aliases, and healthy labels. Reports unknown, ambiguous, duplicate, and unmapped labels.

**Current results:** 65 raw labels → 43 canonical labels across both datasets (100% mapped).

**Outputs:**

- `reports/label_mapping.json`
- `reports/label_standardization.md`

```bash
python -m src.data.label_standardizer
```

### 7. Data Quality Control

**Module:** `src/data/data_quality.py`

Read-only quality assessment before preprocessing:

| Check | Method |
|-------|--------|
| Corrupted / unopenable | Pillow + OpenCV |
| Blurry images | Laplacian variance (OpenCV) |
| Duplicate images | Perceptual hashing (per dataset) |
| Extremely small images | Width/height thresholds |
| Invalid aspect ratio | Configurable ratio bounds |
| Unsupported formats | Extension validation |
| Empty folders | Directory scan |

**Statistics:** total images, good images, blurry, duplicate, corrupted, small, invalid aspect ratio.

**Outputs:**

- `reports/data_quality.json`
- `reports/data_quality.md`

```bash
python -m src.data.data_quality
```

> **Note:** A full scan of ~56,000 images takes several minutes. All checks are read-only.

## Project Structure

```
PlantDiseaseAI/
├── datasets/
│   ├── raw/              # ZIP archives (plantvillage.zip, plantdoc.zip)
│   ├── external/         # Extracted datasets (read-only source of truth)
│   ├── interim/          # Future intermediate processing
│   └── processed/        # Future canonical metadata & splits
├── docs/
│   └── DATASET_SCHEMA.md # Universal schema documentation
├── src/
│   ├── config/
│   │   └── dataset_schema.py      # Canonical metadata schema
│   ├── data/
│   │   ├── ingest.py              # Ingestion pipeline orchestrator
│   │   ├── extract.py             # ZIP extraction
│   │   ├── verify.py              # Archive & extraction verification
│   │   ├── registry.py            # Dataset registry
│   │   ├── dataset_audit.py       # Read-only dataset audit
│   │   ├── eda.py                 # Visualizations & reports
│   │   ├── dataset_comparison.py  # Cross-dataset comparison
│   │   ├── label_standardizer.py  # Universal label mapping
│   │   └── data_quality.py        # Quality control (read-only)
│   ├── preprocessing/    # Image transforms, splits, balancing
│   ├── models/           # Baseline CNN + transfer-learning classifiers
│   ├── training/         # Trainer, DataLoader, metrics, checkpoints
│   ├── evaluation/       # Metrics (planned)
│   ├── explainability/   # Grad-CAM (planned)
│   ├── api/              # FastAPI backend (planned)
│   ├── chatbot/          # AI chatbot (planned)
│   └── utils/            # Shared utilities
├── reports/              # Generated audit, mapping, and quality reports
│   └── comparison/       # Cross-dataset comparison artifacts
├── experiments/          # Per-model training experiment runners
├── logs/
├── notebooks/
├── saved_models/
├── tests/
├── config.yaml
├── main.py
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
cd PlantDiseaseAI

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate    # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### Run the Full Data Pipeline

Place ZIP files in `datasets/raw/`, then run each stage in order:

```bash
# 1. Extract datasets
python -m src.data.ingest

# 2. Audit datasets & generate comparison reports
python -m src.data.dataset_audit

# 3. Standardize labels across datasets
python -m src.data.label_standardizer

# 4. Assess image quality (read-only; may take several minutes)
python -m src.data.data_quality
```

### Configuration

Edit `config.yaml` for project paths and defaults:

```yaml
paths:
  datasets:
    raw: datasets/raw
    external: datasets/external
    processed: datasets/processed
```

## Dependencies (Active)

| Package | Used for |
|---------|----------|
| `pandas` | Statistics, EDA, blur variance summaries |
| `pillow` | Image metadata reads |
| `matplotlib` | Audit & comparison charts |
| `imagehash` | Perceptual duplicate detection |
| `opencv-python` | Laplacian blur detection |

| `torch` / `torchvision` | Model training and transfer learning |
| `torchmetrics` | Epoch accuracy, precision, recall, F1 |
| `tensorboard` | Training run logging |
| `tqdm` | Progress bars during training |
| `numpy` | Metrics and array operations |

## Training Experiments

Two-stage transfer learning (10 epochs frozen backbone + 20 epochs fine-tuning) is implemented for:

| Model | Command |
|-------|---------|
| Baseline CNN | `python -m experiments.baseline_cnn.train --train` |
| ResNet50 | `python -m experiments.resnet50.train --train` |
| ResNet101 | `python -m experiments.resnet101.train --train` |
| DenseNet121 | `python -m experiments.densenet121.train --train` |
| EfficientNet-B0 | `python -m experiments.efficientnet_b0.train --train` |
| EfficientNet-B3 | `python -m experiments.efficientnet_b3.train --train` |

Training reports are saved under `reports/`. Checkpoints go to `saved_models/` (gitignored).

## Design Principles

1. **Read-only first** — Audit, label mapping, and quality checks never modify source data.
2. **Universal schema** — One metadata contract for PlantVillage, PlantDoc, and future datasets.
3. **Modular pipelines** — Each stage is a standalone module with CLI entry point.
4. **Reports as artifacts** — JSON for machines, Markdown for humans, PNG for visualization.
5. **Production-ready structure** — Type hints, dataclasses, enums, logging, and docstrings throughout.

## Next Steps

- [ ] Final model comparison and test-set evaluation
- [ ] Grad-CAM explainability
- [ ] FastAPI inference API
- [ ] Flutter mobile app
- [ ] AI chatbot integration

## License

TBD
