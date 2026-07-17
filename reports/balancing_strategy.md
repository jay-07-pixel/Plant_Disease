# Dataset Balancing Strategy Report

**Generated:** 2026-07-13T07:35:19.835720+00:00  
**Metadata source:** `datasets\processed\dataset_metadata.csv`  

> Read-only strategy design. No images were modified, copied, or balanced.

## Global Imbalance Metrics

- **Total images:** 56,857
- **Total classes:** 43
- **Majority class:** `orange|huanglongbing_(citrus_greening)` (5,507 images)
- **Minority class:** `corn|gray_leaf_spot` (68 images)
- **Global imbalance ratio:** 80.99
- **Coefficient of variation:** 0.954
- **Gini coefficient:** 0.447

## Categorization Thresholds

| Tier | Minimum % of Dataset |
|------|---------------------:|
| Dominant | ≥ 7.0% |
| Common | ≥ 3.0% |
| Moderate | ≥ 1.0% |
| Rare | ≥ 0.2% |
| Extremely Rare | < 0.2% |

## Recommended Actions by Tier

| Tier | Action |
|------|--------|
| Dominant | No Action |
| Common | No Action |
| Moderate | Weighted Sampling |
| Rare | Data Augmentation |
| Extremely Rare | Weighted Sampling + Augmentation |

## Action Summary

- **Data Augmentation:** 10 classes
- **No Action:** 9 classes
- **Weighted Sampling:** 21 classes
- **Weighted Sampling + Augmentation:** 3 classes

## Category Summary

- **Common:** 6 classes
- **Dominant:** 3 classes
- **Extremely Rare:** 3 classes
- **Moderate:** 21 classes
- **Rare:** 10 classes

## Per-Class Balancing Strategy

| Canonical Label | Images | % | Imbalance Ratio | Category | Action | Norm. Weight |
|-----------------|-------:|--:|----------------:|----------|--------|-------------:|
| `orange|huanglongbing_(citrus_greening)` | 5,507 | 9.69 | 1.0 | Dominant | No Action | 0.001836 |
| `tomato|tomato_yellow_leaf_curl_virus` | 5,432 | 9.55 | 1.0 | Dominant | No Action | 0.001861 |
| `soybean|healthy` | 5,155 | 9.07 | 1.1 | Dominant | No Action | 0.001961 |
| `peach|bacterial_spot` | 2,297 | 4.04 | 2.4 | Common | No Action | 0.004401 |
| `tomato|bacterial_spot` | 2,234 | 3.93 | 2.5 | Common | No Action | 0.004525 |
| `tomato|late_blight` | 2,020 | 3.55 | 2.7 | Common | No Action | 0.005005 |
| `squash|powdery_mildew` | 1,964 | 3.45 | 2.8 | Common | No Action | 0.005147 |
| `tomato|septoria_leaf_spot` | 1,919 | 3.38 | 2.9 | Common | No Action | 0.005268 |
| `apple|healthy` | 1,736 | 3.05 | 3.2 | Common | No Action | 0.005824 |
| `tomato|spider_mites_two_spotted_spider_mite` | 1,676 | 2.95 | 3.3 | Moderate | Weighted Sampling | 0.006032 |
| `tomato|healthy` | 1,653 | 2.91 | 3.3 | Moderate | Weighted Sampling | 0.006116 |
| `blueberry|healthy` | 1,618 | 2.85 | 3.4 | Moderate | Weighted Sampling | 0.006248 |
| `pepper|healthy` | 1,539 | 2.71 | 3.6 | Moderate | Weighted Sampling | 0.006569 |
| `tomato|target_spot` | 1,404 | 2.47 | 3.9 | Moderate | Weighted Sampling | 0.007201 |
| `grape|esca_(black_measles)` | 1,383 | 2.43 | 4.0 | Moderate | Weighted Sampling | 0.007310 |
| `grape|black_rot` | 1,244 | 2.19 | 4.4 | Moderate | Weighted Sampling | 0.008127 |
| `corn|common_rust` | 1,192 | 2.10 | 4.6 | Moderate | Weighted Sampling | 0.008481 |
| `corn|healthy` | 1,162 | 2.04 | 4.7 | Moderate | Weighted Sampling | 0.008700 |
| `potato|early_blight` | 1,117 | 1.96 | 4.9 | Moderate | Weighted Sampling | 0.009051 |
| `strawberry|leaf_scorch` | 1,109 | 1.95 | 5.0 | Moderate | Weighted Sampling | 0.009116 |
| `potato|late_blight` | 1,105 | 1.94 | 5.0 | Moderate | Weighted Sampling | 0.009149 |
| `tomato|early_blight` | 1,083 | 1.90 | 5.1 | Moderate | Weighted Sampling | 0.009335 |
| `grape|leaf_blight_(isariopsis_leaf_spot)` | 1,076 | 1.89 | 5.1 | Moderate | Weighted Sampling | 0.009396 |
| `cherry|powdery_mildew` | 1,052 | 1.85 | 5.2 | Moderate | Weighted Sampling | 0.009610 |
| `tomato|leaf_mold` | 1,043 | 1.83 | 5.3 | Moderate | Weighted Sampling | 0.009693 |
| `pepper|bacterial_spot` | 997 | 1.75 | 5.5 | Moderate | Weighted Sampling | 0.010140 |
| `corn|northern_leaf_blight` | 985 | 1.73 | 5.6 | Moderate | Weighted Sampling | 0.010264 |
| `cherry|healthy` | 911 | 1.60 | 6.0 | Moderate | Weighted Sampling | 0.011097 |
| `apple|apple_scab` | 717 | 1.26 | 7.7 | Moderate | Weighted Sampling | 0.014100 |
| `apple|black_rot` | 621 | 1.09 | 8.9 | Moderate | Weighted Sampling | 0.016280 |
| `strawberry|healthy` | 552 | 0.97 | 10.0 | Rare | Data Augmentation | 0.018315 |
| `corn|cercospora_leaf_spot_gray_leaf_spot` | 513 | 0.90 | 10.7 | Rare | Data Augmentation | 0.019707 |
| `grape|healthy` | 492 | 0.87 | 11.2 | Rare | Data Augmentation | 0.020548 |
| `raspberry|healthy` | 487 | 0.86 | 11.3 | Rare | Data Augmentation | 0.020759 |
| `peach|healthy` | 472 | 0.83 | 11.7 | Rare | Data Augmentation | 0.021419 |
| `tomato|tomato_mosaic_virus` | 427 | 0.75 | 12.9 | Rare | Data Augmentation | 0.023676 |
| `apple|cedar_apple_rust` | 275 | 0.48 | 20.0 | Rare | Data Augmentation | 0.036762 |
| `corn|blight` | 192 | 0.34 | 28.7 | Rare | Data Augmentation | 0.052655 |
| `potato|healthy` | 152 | 0.27 | 36.2 | Rare | Data Augmentation | 0.066511 |
| `corn|rust` | 116 | 0.20 | 47.5 | Rare | Data Augmentation | 0.087152 |
| `apple|rust` | 89 | 0.16 | 61.9 | Extremely Rare | Weighted Sampling + Augmentation | 0.113592 |
| `pepper|leaf_spot` | 71 | 0.12 | 77.6 | Extremely Rare | Weighted Sampling + Augmentation | 0.142390 |
| `corn|gray_leaf_spot` | 68 | 0.12 | 81.0 | Extremely Rare | Weighted Sampling + Augmentation | 0.148672 |

## Weight Formulas

- **Inverse frequency weight:** `total_samples / (num_classes × class_count)`
- **Normalized class weight:** `inverse_weight / sum(all inverse_weights)`
- **Imbalance ratio:** `majority_class_count / class_count`

## Output Files

| File | Description |
|------|-------------|
| `datasets/processed/class_weights.csv` | Per-class weights and recommendations |
| `datasets/processed/class_weights.json` | Same data in JSON format |
| `reports/balancing_strategy.json` | Full strategy report |
