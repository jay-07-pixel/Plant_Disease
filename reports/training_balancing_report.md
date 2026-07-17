# Training Balancing Report

**Generated:** 2026-07-13T08:09:33.682772+00:00  
**Metadata source:** `datasets\processed\processed_metadata.csv`  
**Class weights source:** `datasets\processed\class_weights.csv`  

> Read-only balancing plan. Only the TRAIN split is analyzed. Val/test are never balanced. No images were modified or augmented.

## Summary

- **Training images:** 39,797
- **Training classes:** 43
- **Target samples per class (median):** 758
- **Total virtual augmentations required:** 7,122

## Hybrid Strategy

| Component | Applied to |
|-----------|------------|
| WeightedRandomSampler | Training split only |
| Class weights | Training split only |
| Data augmentation | Training split only (future) |
| Validation / Test | Never balanced |

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

## Per-Class Balancing Plan

| Canonical Label | Current | Target | Aug. Needed | Aug. Factor | Sampling Weight | Category | Action |
|-------------------|--------:|-------:|------------:|------------:|----------------:|----------|--------|
| `corn|gray_leaf_spot` | 48 | 758 | 710 | 15.79 | 19.2815 | Extremely Rare | Weighted Sampling + Augmentation |
| `pepper|leaf_spot` | 50 | 758 | 708 | 15.16 | 18.5102 | Extremely Rare | Weighted Sampling + Augmentation |
| `apple|rust` | 62 | 758 | 696 | 12.23 | 14.9276 | Extremely Rare | Weighted Sampling + Augmentation |
| `corn|rust` | 81 | 758 | 677 | 9.36 | 11.4261 | Rare | Data Augmentation |
| `potato|healthy` | 106 | 758 | 652 | 7.15 | 8.7312 | Rare | Data Augmentation |
| `corn|blight` | 134 | 758 | 624 | 5.66 | 6.9068 | Rare | Data Augmentation |
| `apple|cedar_apple_rust` | 192 | 758 | 566 | 3.95 | 4.8204 | Rare | Data Augmentation |
| `tomato|tomato_mosaic_virus` | 299 | 758 | 459 | 2.54 | 3.0954 | Rare | Data Augmentation |
| `peach|healthy` | 330 | 758 | 428 | 2.30 | 2.8046 | Rare | Data Augmentation |
| `raspberry|healthy` | 341 | 758 | 417 | 2.22 | 2.7141 | Rare | Data Augmentation |
| `grape|healthy` | 344 | 758 | 414 | 2.20 | 2.6904 | Rare | Data Augmentation |
| `corn|cercospora_leaf_spot_gray_leaf_spot` | 359 | 758 | 399 | 2.11 | 2.5780 | Rare | Data Augmentation |
| `strawberry|healthy` | 386 | 758 | 372 | 1.96 | 2.3977 | Rare | Data Augmentation |
| `apple|black_rot` | 435 | 435 | 0 | 1.00 | 2.1276 | Moderate | Weighted Sampling |
| `apple|apple_scab` | 502 | 502 | 0 | 1.00 | 1.8436 | Moderate | Weighted Sampling |
| `cherry|healthy` | 638 | 638 | 0 | 1.00 | 1.4506 | Moderate | Weighted Sampling |
| `corn|northern_leaf_blight` | 690 | 690 | 0 | 1.00 | 1.3413 | Moderate | Weighted Sampling |
| `pepper|bacterial_spot` | 698 | 698 | 0 | 1.00 | 1.3259 | Moderate | Weighted Sampling |
| `tomato|leaf_mold` | 730 | 730 | 0 | 1.00 | 1.2678 | Moderate | Weighted Sampling |
| `cherry|powdery_mildew` | 736 | 736 | 0 | 1.00 | 1.2575 | Moderate | Weighted Sampling |
| `grape|leaf_blight_(isariopsis_leaf_spot)` | 753 | 753 | 0 | 1.00 | 1.2291 | Moderate | Weighted Sampling |
| `tomato|early_blight` | 758 | 758 | 0 | 1.00 | 1.2210 | Moderate | Weighted Sampling |
| `potato|late_blight` | 774 | 774 | 0 | 1.00 | 1.1958 | Moderate | Weighted Sampling |
| `strawberry|leaf_scorch` | 776 | 776 | 0 | 1.00 | 1.1927 | Moderate | Weighted Sampling |
| `potato|early_blight` | 782 | 782 | 0 | 1.00 | 1.1835 | Moderate | Weighted Sampling |
| `corn|healthy` | 813 | 813 | 0 | 1.00 | 1.1384 | Moderate | Weighted Sampling |
| `corn|common_rust` | 834 | 834 | 0 | 1.00 | 1.1097 | Moderate | Weighted Sampling |
| `grape|black_rot` | 871 | 871 | 0 | 1.00 | 1.0626 | Moderate | Weighted Sampling |
| `grape|esca_(black_measles)` | 968 | 968 | 0 | 1.00 | 0.9561 | Moderate | Weighted Sampling |
| `tomato|target_spot` | 983 | 983 | 0 | 1.00 | 0.9415 | Moderate | Weighted Sampling |
| `pepper|healthy` | 1,077 | 1,077 | 0 | 1.00 | 0.8593 | Moderate | Weighted Sampling |
| `blueberry|healthy` | 1,133 | 1,133 | 0 | 1.00 | 0.8169 | Moderate | Weighted Sampling |
| `tomato|healthy` | 1,157 | 1,157 | 0 | 1.00 | 0.7999 | Moderate | Weighted Sampling |
| `tomato|spider_mites_two_spotted_spider_mite` | 1,173 | 1,173 | 0 | 1.00 | 0.7890 | Moderate | Weighted Sampling |
| `apple|healthy` | 1,215 | 1,215 | 0 | 1.00 | 0.7617 | Common | No Action |
| `tomato|septoria_leaf_spot` | 1,343 | 1,343 | 0 | 1.00 | 0.6891 | Common | No Action |
| `squash|powdery_mildew` | 1,375 | 1,375 | 0 | 1.00 | 0.6731 | Common | No Action |
| `tomato|late_blight` | 1,414 | 1,414 | 0 | 1.00 | 0.6545 | Common | No Action |
| `tomato|bacterial_spot` | 1,564 | 1,564 | 0 | 1.00 | 0.5918 | Common | No Action |
| `peach|bacterial_spot` | 1,608 | 1,608 | 0 | 1.00 | 0.5756 | Common | No Action |
| `soybean|healthy` | 3,608 | 3,608 | 0 | 1.00 | 0.2565 | Dominant | No Action |
| `tomato|tomato_yellow_leaf_curl_virus` | 3,802 | 3,802 | 0 | 1.00 | 0.2434 | Dominant | No Action |
| `orange|huanglongbing_(citrus_greening)` | 3,855 | 3,855 | 0 | 1.00 | 0.2401 | Dominant | No Action |

## Distribution Comparison (Top 15 Smallest Classes)

| Canonical Label | Before | Target |
|-------------------|-------:|-------:|
| `corn|gray_leaf_spot` | 48 | 758 |
| `pepper|leaf_spot` | 50 | 758 |
| `apple|rust` | 62 | 758 |
| `corn|rust` | 81 | 758 |
| `potato|healthy` | 106 | 758 |
| `corn|blight` | 134 | 758 |
| `apple|cedar_apple_rust` | 192 | 758 |
| `tomato|tomato_mosaic_virus` | 299 | 758 |
| `peach|healthy` | 330 | 758 |
| `raspberry|healthy` | 341 | 758 |
| `grape|healthy` | 344 | 758 |
| `corn|cercospora_leaf_spot_gray_leaf_spot` | 359 | 758 |
| `strawberry|healthy` | 386 | 758 |
| `apple|black_rot` | 435 | 435 |
| `apple|apple_scab` | 502 | 502 |

## Output Files

| File | Description |
|------|-------------|
| `datasets/processed/training_balancing_plan.csv` | Per-class balancing plan |
| `datasets/processed/training_balancing_plan.json` | Plan data (JSON) |
