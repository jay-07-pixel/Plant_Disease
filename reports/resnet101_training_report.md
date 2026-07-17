# ResNet101 Transfer Learning Training Report

**Generated:** 2026-07-15T05:36:36.987184+00:00  
**Experiment:** `resnet101`  
**Device:** `cuda`  

## Summary

- **Best epoch:** 28
- **Best validation accuracy:** 0.9353
- **Final training accuracy:** 0.9951
- **Final validation accuracy:** 0.9235
- **Final validation precision:** 0.9319
- **Final validation recall:** 0.9235
- **Final validation F1 score:** 0.9219
- **Final validation top-1 accuracy:** 0.9235
- **Final validation top-5 accuracy:** 0.9777
- **Training time:** 4418.8s
- **Epochs completed:** 30

## Model

- **Trainable parameters:** 15,052,843
- **Frozen parameters:** 27,535,424

## Configuration

### Two-Stage Transfer Learning

| Stage | Epochs | Learning Rate | Trainable Layers |
|------:|-------:|--------------:|------------------|
| 1 — Feature extraction | 10 | 0.001 | Classifier head only |
| 2 — Fine-tuning | 20 | 0.0001 | layer4 + classifier |

| Parameter | Value |
|-----------|------:|
| Total epochs | 30 |
| Optimizer | Adam |
| Stage 1 learning rate | 0.001 |
| Stage 2 learning rate | 0.0001 |
| Loss | CrossEntropyLoss |
| Scheduler | ReduceLROnPlateau |
| Early stopping patience | 7 |
| Random seed | 42 |

## Per-Epoch Metrics

| Epoch | Stage | Train Loss | Val Loss | Train Acc | Val Acc | Val F1 | LR | Time (s) |
|------:|------:|-----------:|---------:|----------:|--------:|-------:|---:|---------:|
| 1 | 1 | 0.7725 | 0.5841 | 0.7275 | 0.8080 | 0.7563 | 1.00e-03 | 755.2 |
| 2 | 1 | 0.4420 | 0.6807 | 0.8349 | 0.8036 | 0.7463 | 1.00e-03 | 671.7 |
| 3 | 1 | 0.3804 | 0.4202 | 0.8545 | 0.8383 | 0.7920 | 1.00e-03 | 669.3 |
| 4 | 1 | 0.3488 | 0.5399 | 0.8670 | 0.8509 | 0.7826 | 1.00e-03 | 686.4 |
| 5 | 1 | 0.3110 | 0.4498 | 0.8761 | 0.8589 | 0.8092 | 1.00e-03 | 676.7 |
| 6 | 1 | 0.2941 | 0.4026 | 0.8796 | 0.8599 | 0.8206 | 1.00e-03 | 541.8 |
| 7 | 1 | 0.2987 | 0.4689 | 0.8836 | 0.8527 | 0.8000 | 1.00e-03 | 417.7 |
| 8 | 1 | 0.2797 | 0.4516 | 0.8890 | 0.8373 | 0.7997 | 1.00e-03 | 0.0 |
| 9 | 1 | 0.2704 | 0.5161 | 0.8928 | 0.8605 | 0.8033 | 1.00e-03 | 0.0 |
| 10 | 1 | 0.2631 | 0.4276 | 0.8925 | 0.8639 | 0.8140 | 1.00e-03 | 0.0 |
| 11 | 2 | 0.2667 | 0.2575 | 0.9118 | 0.8937 | 0.8566 | 1.00e-04 | 0.0 |
| 12 | 2 | 0.0814 | 0.1866 | 0.9594 | 0.9195 | 0.8913 | 1.00e-04 | 0.0 |
| 13 | 2 | 0.0541 | 0.2367 | 0.9704 | 0.9095 | 0.8913 | 1.00e-04 | 0.0 |
| 14 | 2 | 0.0463 | 0.1577 | 0.9757 | 0.9181 | 0.9004 | 1.00e-04 | 0.0 |
| 15 | 2 | 0.0406 | 0.2026 | 0.9793 | 0.9098 | 0.8971 | 1.00e-04 | 0.0 |
| 16 | 2 | 0.0330 | 0.1945 | 0.9821 | 0.9195 | 0.9089 | 1.00e-04 | 0.0 |
| 17 | 2 | 0.0300 | 0.1835 | 0.9833 | 0.9142 | 0.9063 | 1.00e-04 | 0.0 |
| 18 | 2 | 0.0259 | 0.1905 | 0.9850 | 0.9153 | 0.9053 | 1.00e-04 | 0.0 |
| 19 | 2 | 0.0118 | 0.1461 | 0.9913 | 0.9225 | 0.9211 | 5.00e-05 | 0.0 |
| 20 | 2 | 0.0103 | 0.1520 | 0.9932 | 0.9165 | 0.9120 | 5.00e-05 | 0.0 |
| 21 | 2 | 0.0112 | 0.1483 | 0.9927 | 0.9263 | 0.9255 | 5.00e-05 | 0.0 |
| 22 | 2 | 0.0097 | 0.1630 | 0.9936 | 0.9190 | 0.9172 | 5.00e-05 | 0.0 |
| 23 | 2 | 0.0086 | 0.1790 | 0.9941 | 0.9284 | 0.9220 | 5.00e-05 | 0.0 |
| 24 | 2 | 0.0089 | 0.1501 | 0.9946 | 0.9332 | 0.9248 | 5.00e-05 | 0.0 |
| 25 | 2 | 0.0071 | 0.1607 | 0.9948 | 0.9304 | 0.9264 | 5.00e-05 | 0.0 |
| 26 | 2 | 0.0089 | 0.1736 | 0.9944 | 0.9227 | 0.9211 | 5.00e-05 | 0.0 |
| 27 | 2 | 0.0092 | 0.1511 | 0.9951 | 0.9287 | 0.9286 | 5.00e-05 | 0.0 |
| 28 | 2 | 0.0074 | 0.1721 | 0.9949 | 0.9353 | 0.9308 | 5.00e-05 | 0.0 |
| 29 | 2 | 0.0076 | 0.1914 | 0.9956 | 0.9230 | 0.9200 | 5.00e-05 | 0.0 |
| 30 | 2 | 0.0081 | 0.1628 | 0.9951 | 0.9235 | 0.9219 | 5.00e-05 | 0.0 |

## Per-Class Validation Accuracy

| Class | Accuracy |
|-------|--------:|
| `apple|black_rot` | 1.0000 |
| `apple|cedar_apple_rust` | 1.0000 |
| `cherry|powdery_mildew` | 1.0000 |
| `corn|common_rust` | 1.0000 |
| `corn|healthy` | 1.0000 |
| `grape|esca_(black_measles)` | 1.0000 |
| `grape|leaf_blight_(isariopsis_leaf_spot)` | 1.0000 |
| `peach|bacterial_spot` | 1.0000 |
| `pepper|bacterial_spot` | 1.0000 |
| `strawberry|leaf_scorch` | 1.0000 |
| `tomato|spider_mites_two_spotted_spider_mite` | 1.0000 |
| `orange|huanglongbing_(citrus_greening)` | 0.9988 |
| `soybean|healthy` | 0.9974 |
| `squash|powdery_mildew` | 0.9932 |
| `pepper|healthy` | 0.9913 |
| `tomato|target_spot` | 0.9905 |
| `strawberry|healthy` | 0.9880 |
| `tomato|healthy` | 0.9879 |
| `corn|cercospora_leaf_spot_gray_leaf_spot` | 0.9870 |
| `grape|healthy` | 0.9865 |
| `apple|healthy` | 0.9846 |
| `peach|healthy` | 0.9718 |
| `blueberry|healthy` | 0.9712 |
| `tomato|late_blight` | 0.9670 |
| `grape|black_rot` | 0.9572 |
| `tomato|tomato_yellow_leaf_curl_virus` | 0.9571 |
| `potato|healthy` | 0.9565 |
| `apple|apple_scab` | 0.9537 |
| `potato|early_blight` | 0.9464 |
| `raspberry|healthy` | 0.9452 |
| `corn|blight` | 0.9310 |
| `potato|late_blight` | 0.9277 |
| `cherry|healthy` | 0.9270 |
| `tomato|early_blight` | 0.9259 |
| `tomato|septoria_leaf_spot` | 0.9236 |
| `corn|northern_leaf_blight` | 0.9122 |
| `tomato|leaf_mold` | 0.8846 |
| `tomato|bacterial_spot` | 0.8776 |
| `tomato|tomato_mosaic_virus` | 0.8750 |
| `apple|rust` | 0.7692 |
| `corn|gray_leaf_spot` | 0.6000 |
| `corn|rust` | 0.5882 |
| `pepper|leaf_spot` | 0.5455 |

## Confusion Matrix

Rows = true class, columns = predicted class. See JSON report for the full matrix.

- **Overall validation accuracy:** 0.9699
- **Classes:** 43
- **Samples evaluated:** 8,530

## Artifacts

| File | Description |
|------|-------------|
| `saved_models/resnet101/best_model.pth` | Best validation checkpoint |
| `saved_models/resnet101/last_model.pth` | Last epoch checkpoint |
| `saved_models/resnet101/stage1_best_model.pth` | Best Stage 1 checkpoint |
| `saved_models/resnet101/training_history.json` | Combined training history |
| `experiments/resnet101/tensorboard/` | TensorBoard event files |
