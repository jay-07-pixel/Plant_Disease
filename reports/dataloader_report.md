# DataLoader Report

**Generated:** 2026-07-13T08:20:12.120618+00:00  
**Metadata source:** `datasets\processed\processed_metadata.csv`  
**Balancing plan:** `datasets\processed\training_balancing_plan.csv`  

> Data preparation only — no model training, evaluation, or augmentation on disk.

## Summary

- **Number of classes:** 43
- **Training samples:** 39,797
- **Validation samples:** 8,530
- **Test samples:** 8,530
- **Total samples:** 56,857

## Configuration

| Parameter | Value |
|-----------|------:|
| `batch_size` | 32 |
| `num_workers` | 0 |
| `image_size` | 224 |
| `pin_memory` | True |
| `persistent_workers` | False |
| `shuffle` | True |
| `random_seed` | 42 |
| `drop_last` | False |
| `use_weighted_sampler` | True |
| `use_class_weights` | True |
| `normalization_mean` | [0.485, 0.456, 0.406] |
| `normalization_std` | [0.229, 0.224, 0.225] |

## Tensor Verification

- **Single image shape:** `[3, 224, 224]`
- **Expected shape:** `[3, 224, 224]`
- **Shape matches expected:** True
- **Dtype:** `torch.float32`

## Batch Verification

| Split | Batch Shape | Batch Size | Unique Labels |
|-------|-------------|----------:|--------------:|
| train | `[32, 3, 224, 224]` | 32 | 23 |
| val | `[32, 3, 224, 224]` | 32 | 16 |
| test | `[32, 3, 224, 224]` | 32 | 23 |

## Sampler Information

- **Validation Balanced:** False
- **Test Balanced:** False
- **Weighted Random Sampler Enabled:** True
- **Num Samples Per Epoch:** 39797
- **Replacement:** True
- **Weight Column:** sampling_weight
- **Per Sample Weight Min:** 0.240080837329955
- **Per Sample Weight Max:** 19.281492248062012
- **Per Sample Weight Mean:** 1.0000000000001439
- **Classes Using Weighted Sampler:** 24
- **Classes Using Augmentation:** 13
- **Class Weights Enabled:** True
- **Class Weights Shape:** [43]
- **Class Weights Min:** 0.07899819314479828
- **Class Weights Max:** 6.344542026519775
- **Class Weights Mean:** 0.9999999403953552

## Class Distribution (Training Split)

| Class | Count |
|-------|------:|
| `orange|huanglongbing_(citrus_greening)` | 3,855 |
| `tomato|tomato_yellow_leaf_curl_virus` | 3,802 |
| `soybean|healthy` | 3,608 |
| `peach|bacterial_spot` | 1,608 |
| `tomato|bacterial_spot` | 1,564 |
| `tomato|late_blight` | 1,414 |
| `squash|powdery_mildew` | 1,375 |
| `tomato|septoria_leaf_spot` | 1,343 |
| `apple|healthy` | 1,215 |
| `tomato|spider_mites_two_spotted_spider_mite` | 1,173 |
| `tomato|healthy` | 1,157 |
| `blueberry|healthy` | 1,133 |
| `pepper|healthy` | 1,077 |
| `tomato|target_spot` | 983 |
| `grape|esca_(black_measles)` | 968 |
| `grape|black_rot` | 871 |
| `corn|common_rust` | 834 |
| `corn|healthy` | 813 |
| `potato|early_blight` | 782 |
| `strawberry|leaf_scorch` | 776 |
| `potato|late_blight` | 774 |
| `tomato|early_blight` | 758 |
| `grape|leaf_blight_(isariopsis_leaf_spot)` | 753 |
| `cherry|powdery_mildew` | 736 |
| `tomato|leaf_mold` | 730 |
| `pepper|bacterial_spot` | 698 |
| `corn|northern_leaf_blight` | 690 |
| `cherry|healthy` | 638 |
| `apple|apple_scab` | 502 |
| `apple|black_rot` | 435 |
| `strawberry|healthy` | 386 |
| `corn|cercospora_leaf_spot_gray_leaf_spot` | 359 |
| `grape|healthy` | 344 |
| `raspberry|healthy` | 341 |
| `peach|healthy` | 330 |
| `tomato|tomato_mosaic_virus` | 299 |
| `apple|cedar_apple_rust` | 192 |
| `corn|blight` | 134 |
| `potato|healthy` | 106 |
| `corn|rust` | 81 |
| `apple|rust` | 62 |
| `pepper|leaf_spot` | 50 |
| `corn|gray_leaf_spot` | 48 |

## Outputs

| File | Description |
|------|-------------|
| `src/training/dataset.py` | Custom PyTorch Dataset |
| `src/training/dataloader.py` | DataLoader factory and verification |
| `src/training/sampler.py` | WeightedRandomSampler and class weights |
| `src/training/transforms.py` | Training and evaluation transforms |
| `src/training/training_config.py` | Configurable pipeline settings |
