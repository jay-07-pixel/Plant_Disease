# PlantDiseaseAI

Production-quality AI system for plant disease detection, explanation, and assistance.

PlantDiseaseAI is a full-stack machine learning project designed to detect plant diseases from leaf images, explain model predictions, and assist users through an AI chatbot. The data engineering and training pipelines are implemented end-to-end; deployment and explainability are next.

## Project Progress

### Completed

| Area | Details |
|------|---------|
| **Data ingestion** | ZIP extraction, verification, dataset registry |
| **Dataset audit & EDA** | Class distribution, resolution, format analysis |
| **Multi-dataset comparison** | PlantVillage vs PlantDoc overlap and stats |
| **Label standardization** | 65 raw labels → 43 canonical classes |
| **Data quality control** | Corruption, blur, duplicate, and format checks |
| **Dataset preparation** | Canonical metadata export to `datasets/processed/` |
| **Preprocessing** | Resize, normalize, train/val/test split (~56,857 images) |
| **Balancing strategy** | Class weights and training-only balancing plan |
| **PyTorch training engine** | Trainer, AMP, checkpoints, metrics, TensorBoard |
| **Model training (5/6)** | Baseline CNN, ResNet50, ResNet101, DenseNet121, EfficientNet-B0 |

### In Progress

| Area | Details |
|------|---------|
| **EfficientNet-B3 training** | Two-stage transfer learning run in progress |

### Planned

| Area | Details |
|------|---------|
| **Test-set evaluation** | Final metrics on held-out test split |
| **Model comparison** | Side-by-side analysis of all trained architectures |
| **Grad-CAM explainability** | Visual attribution for predictions |
| **FastAPI backend** | REST inference API |
| **Flutter frontend** | Mobile app for disease detection |
| **AI chatbot** | User assistance and disease guidance |

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
| Dataset preparation & splits | Done |
| Image preprocessing | Done |
| Dataset balancing strategy | Done |
| PyTorch training engine | Done |
| Baseline CNN experiment | Done |
| ResNet50 / ResNet101 experiments | Done |
| DenseNet121 experiment | Done |
| EfficientNet-B0 experiment | Done |
| EfficientNet-B3 experiment | In Progress |
| Test-set evaluation | Planned |
| Model comparison report | Planned |
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

Raw ZIP archives live in `datasets/raw/`. Extracted data lives in `datasets/external/`. Processed images and metadata live in `datasets/processed/` (gitignored — regenerate locally).

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

### 8. Dataset Preparation & Preprocessing

**Modules:** `src/data/prepare_dataset.py`, `src/preprocessing/`

- Builds canonical metadata and stratified train/val/test splits
- Resizes images, applies normalization, and exports processed copies
- Generates class weights and a training-only balancing plan

**Outputs:**

| File | Description |
|------|-------------|
| `datasets/processed/processed_metadata.csv` | Processed image records with splits |
| `reports/dataset_preparation.json` | Preparation summary |
| `reports/preprocessing_report.json` | Preprocessing statistics |
| `reports/balancing_strategy.json` | Class imbalance analysis |
| `reports/training_balancing_report.json` | Training balancing plan |

```bash
python -m src.data.prepare_dataset
python -m src.preprocessing.preprocess
python -m src.data.balancing_strategy
python -m src.preprocessing.balance_training_dataset
```

## ML Training Pipeline

Shared PyTorch training infrastructure under `src/training/` and per-model experiment runners under `experiments/`.

**Features:**

- Generic `Trainer` with AMP, gradient clipping, early stopping, and checkpointing
- `PlantDiseaseDataset` + `DataLoader` with class-weighted loss
- TorchMetrics: accuracy, precision, recall, F1, top-5
- Two-stage transfer learning: 10 epochs (frozen backbone) + 20 epochs (fine-tune last block)
- TensorBoard logging and JSON/Markdown training reports

### Model Training Results (Validation)

| Model | Status | Best Val Acc | Epochs |
|-------|--------|-------------:|-------:|
| **ResNet101** | Done | **93.53%** | 30/30 |
| **DenseNet121** | Done | 93.40% | 30/30 |
| **ResNet50** | Done | 92.93% | 30/30 |
| **EfficientNet-B0** | Done | 91.27% | 27/30 |
| **Baseline CNN** | Done | 83.95% | 30/30 |
| **EfficientNet-B3** | In Progress | — | — |

Reports: `reports/*_training_report.md`

## Project Structure

```
PlantDiseaseAI/
├── datasets/
│   ├── raw/              # ZIP archives (plantvillage.zip, plantdoc.zip)
│   ├── external/         # Extracted datasets (read-only source of truth)
│   ├── interim/          # Intermediate processing
│   └── processed/        # Processed images, metadata & splits (gitignored)
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
│   │   ├── data_quality.py        # Quality control (read-only)
│   │   ├── prepare_dataset.py     # Metadata export & splits
│   │   └── balancing_strategy.py  # Class imbalance analysis
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

## Dependencies

| Package | Used for |
|---------|----------|
| `pandas` | Statistics, EDA, metadata |
| `pillow` | Image reads and preprocessing |
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
| EfficientNet-B3 | `python -m experiments.efficientnet_b3.train --train` *(in progress)* |

Training reports are saved under `reports/`. Checkpoints go to `saved_models/` (gitignored).

Use `--resume saved_models/<model>/latest_checkpoint.pt` to continue an interrupted run.

## Design Principles

1. **Read-only first** — Audit, label mapping, and quality checks never modify source data.
2. **Universal schema** — One metadata contract for PlantVillage, PlantDoc, and future datasets.
3. **Modular pipelines** — Each stage is a standalone module with CLI entry point.
4. **Reports as artifacts** — JSON for machines, Markdown for humans, PNG for visualization.
5. **Production-ready structure** — Type hints, dataclasses, enums, logging, and docstrings throughout.

## Roadmap

### In progress
- [ ] Complete EfficientNet-B3 two-stage training

### Up next
- [ ] Test-set evaluation for all trained models
- [ ] Unified model comparison report
- [ ] Grad-CAM explainability module
- [ ] FastAPI inference API
- [ ] Flutter mobile app
- [ ] AI chatbot integration

## License

TBD
