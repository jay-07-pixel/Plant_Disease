# Dataset Preparation Report

**Generated:** 2026-07-13T07:30:57.639751+00:00  
**Label mapping source:** `reports\label_mapping.json`  
**External directory:** `C:\Projects\PlantDiseaseAI\datasets\external`  
**Metadata CSV:** `datasets\processed\dataset_metadata.csv`  
**Metadata JSON:** `datasets\processed\dataset_metadata.json`  

> Read-only metadata preparation. No images were modified, copied, or balanced.

## Summary

- **Total images:** 56857
- **Canonical classes:** 43
- **Healthy images:** 15929
- **Diseased images:** 40928
- **Unmapped images (skipped):** 0
- **Unreadable metadata:** 0

## Dataset Contribution

- **plantdoc:** 2,552 images (4.5%)
- **plantvillage:** 54,305 images (95.5%)

## Healthy vs Diseased

- **Healthy:** 15,929 (28.0%)
- **Diseased:** 40,928 (72.0%)

## Split Distribution

- **test:** 236
- **train:** 45,760
- **val:** 10,861

## Images per Plant

- **Tomato:** 18,891
- **Orange:** 5,507
- **Soybean:** 5,155
- **Corn:** 4,228
- **Grape:** 4,195
- **Apple:** 3,438
- **Peach:** 2,769
- **Pepper:** 2,607
- **Potato:** 2,374
- **Squash:** 1,964
- **Cherry:** 1,963
- **Strawberry:** 1,661
- **Blueberry:** 1,618
- **Raspberry:** 487

## Images per Class (Top 20)

- `orange|huanglongbing_(citrus_greening)`: 5,507
- `tomato|tomato_yellow_leaf_curl_virus`: 5,432
- `soybean|healthy`: 5,155
- `peach|bacterial_spot`: 2,297
- `tomato|bacterial_spot`: 2,234
- `tomato|late_blight`: 2,020
- `squash|powdery_mildew`: 1,964
- `tomato|septoria_leaf_spot`: 1,919
- `apple|healthy`: 1,736
- `tomato|spider_mites_two_spotted_spider_mite`: 1,676
- `tomato|healthy`: 1,653
- `blueberry|healthy`: 1,618
- `pepper|healthy`: 1,539
- `tomato|target_spot`: 1,404
- `grape|esca_(black_measles)`: 1,383
- `grape|black_rot`: 1,244
- `corn|common_rust`: 1,192
- `corn|healthy`: 1,162
- `potato|early_blight`: 1,117
- `strawberry|leaf_scorch`: 1,109
- ... and 23 more classes

## Class Weights (for WeightedRandomSampler)

Formula: `weight = total_samples / (num_classes × class_count)`

| Canonical Label | Count | Frequency | Weight |
|-----------------|------:|----------:|-------:|
| `orange|huanglongbing_(citrus_greening)` | 5,507 | 0.0969 | 0.240105 |
| `tomato|tomato_yellow_leaf_curl_virus` | 5,432 | 0.0955 | 0.243420 |
| `soybean|healthy` | 5,155 | 0.0907 | 0.256500 |
| `peach|bacterial_spot` | 2,297 | 0.0404 | 0.575645 |
| `tomato|bacterial_spot` | 2,234 | 0.0393 | 0.591878 |
| `tomato|late_blight` | 2,020 | 0.0355 | 0.654582 |
| `squash|powdery_mildew` | 1,964 | 0.0345 | 0.673246 |
| `tomato|septoria_leaf_spot` | 1,919 | 0.0338 | 0.689034 |
| `apple|healthy` | 1,736 | 0.0305 | 0.761668 |
| `tomato|spider_mites_two_spotted_spider_mite` | 1,676 | 0.0295 | 0.788935 |
| `tomato|healthy` | 1,653 | 0.0291 | 0.799913 |
| `blueberry|healthy` | 1,618 | 0.0285 | 0.817216 |
| `pepper|healthy` | 1,539 | 0.0271 | 0.859166 |
| `tomato|target_spot` | 1,404 | 0.0247 | 0.941778 |
| `grape|esca_(black_measles)` | 1,383 | 0.0243 | 0.956078 |
| `grape|black_rot` | 1,244 | 0.0219 | 1.062907 |
| `corn|common_rust` | 1,192 | 0.0210 | 1.109275 |
| `corn|healthy` | 1,162 | 0.0204 | 1.137914 |
| `potato|early_blight` | 1,117 | 0.0196 | 1.183756 |
| `strawberry|leaf_scorch` | 1,109 | 0.0195 | 1.192296 |
| `potato|late_blight` | 1,105 | 0.0194 | 1.196612 |
| `tomato|early_blight` | 1,083 | 0.0190 | 1.220919 |
| `grape|leaf_blight_(isariopsis_leaf_spot)` | 1,076 | 0.0189 | 1.228862 |
| `cherry|powdery_mildew` | 1,052 | 0.0185 | 1.256897 |
| `tomato|leaf_mold` | 1,043 | 0.0183 | 1.267743 |
| ... | | | 18 more classes |

## Metadata Schema

Each record in `datasets/processed/dataset_metadata.csv` contains:

| Column | Description |
|--------|-------------|
| `image_path` | Project-relative path to the image |
| `dataset_name` | Source dataset (`plantvillage`, `plantdoc`) |
| `split` | `train`, `val`, `test`, or empty |
| `plant` | Canonical plant name |
| `disease` | Canonical disease name |
| `canonical_label` | Stable cross-dataset label key |
| `is_healthy` | Healthy sample flag |
| `image_width` | Width in pixels |
| `image_height` | Height in pixels |
| `image_format` | File format extension |
| `sample_weight` | Per-sample weight for balanced sampling |
