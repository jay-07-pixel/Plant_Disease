# ResNet50 Transfer Learning Training Report

**Generated:** 2026-07-14T15:18:44.245662+00:00  
**Experiment:** `resnet50`  
**Device:** `cuda`  

## Summary

- **Best epoch:** 27
- **Best validation accuracy:** 0.9293
- **Final training accuracy:** 0.9924
- **Final validation accuracy:** 0.9165
- **Final validation precision:** 0.9035
- **Final validation recall:** 0.9165
- **Final validation F1 score:** 0.9075
- **Final validation top-1 accuracy:** 0.9165
- **Final validation top-5 accuracy:** 0.9768
- **Training time:** 8739.6s
- **Epochs completed:** 30

## Model

- **Trainable parameters:** 15,052,843
- **Frozen parameters:** 8,543,296

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
| 1 | 1 | 0.7863 | 0.6039 | 0.7170 | 0.8006 | 0.7336 | 1.00e-03 | 627.6 |
| 2 | 1 | 0.4358 | 0.5525 | 0.8288 | 0.8294 | 0.7677 | 1.00e-03 | 552.2 |
| 3 | 1 | 0.3797 | 0.4777 | 0.8523 | 0.8415 | 0.7787 | 1.00e-03 | 431.5 |
| 4 | 1 | 0.3527 | 0.5083 | 0.8633 | 0.8529 | 0.7818 | 1.00e-03 | 510.1 |
| 5 | 1 | 0.3219 | 0.4794 | 0.8690 | 0.8587 | 0.7936 | 1.00e-03 | 518.2 |
| 6 | 1 | 0.3190 | 0.4387 | 0.8719 | 0.8547 | 0.8031 | 1.00e-03 | 553.5 |
| 7 | 1 | 0.3017 | 0.4775 | 0.8763 | 0.8529 | 0.7947 | 1.00e-03 | 490.5 |
| 8 | 1 | 0.2990 | 0.4803 | 0.8816 | 0.8417 | 0.7898 | 1.00e-03 | 440.6 |
| 9 | 1 | 0.2640 | 0.4450 | 0.8868 | 0.8696 | 0.8092 | 1.00e-03 | 479.9 |
| 10 | 1 | 0.2710 | 0.4855 | 0.8835 | 0.8583 | 0.8065 | 1.00e-03 | 522.0 |
| 11 | 2 | 0.2634 | 0.3140 | 0.9077 | 0.8948 | 0.8474 | 1.00e-04 | 543.7 |
| 12 | 2 | 0.0883 | 0.2448 | 0.9553 | 0.8963 | 0.8784 | 1.00e-04 | 530.0 |
| 13 | 2 | 0.0629 | 0.2307 | 0.9674 | 0.9040 | 0.8816 | 1.00e-04 | 530.1 |
| 14 | 2 | 0.0361 | 0.2003 | 0.9807 | 0.9142 | 0.9041 | 1.00e-04 | 476.2 |
| 15 | 2 | 0.0263 | 0.2311 | 0.9860 | 0.9103 | 0.8939 | 1.00e-04 | 542.0 |
| 16 | 2 | 0.0263 | 0.2596 | 0.9868 | 0.9108 | 0.8942 | 1.00e-04 | 536.2 |
| 17 | 2 | 0.0440 | 0.2092 | 0.9778 | 0.9200 | 0.8996 | 1.00e-04 | 455.3 |
| 18 | 2 | 0.0285 | 0.1801 | 0.9826 | 0.9114 | 0.8997 | 1.00e-04 | 0.0 |
| 19 | 2 | 0.0272 | 0.2073 | 0.9835 | 0.9159 | 0.9091 | 1.00e-04 | 0.0 |
| 20 | 2 | 0.0232 | 0.1922 | 0.9859 | 0.9101 | 0.9055 | 1.00e-04 | 0.0 |
| 21 | 2 | 0.0226 | 0.1736 | 0.9872 | 0.9225 | 0.9126 | 1.00e-04 | 0.0 |
| 22 | 2 | 0.0215 | 0.1693 | 0.9880 | 0.9201 | 0.9035 | 1.00e-04 | 0.0 |
| 23 | 2 | 0.0205 | 0.1880 | 0.9872 | 0.9204 | 0.9175 | 1.00e-04 | 0.0 |
| 24 | 2 | 0.0180 | 0.1725 | 0.9898 | 0.9228 | 0.9172 | 1.00e-04 | 0.0 |
| 25 | 2 | 0.0164 | 0.1584 | 0.9898 | 0.9245 | 0.9146 | 1.00e-04 | 0.0 |
| 26 | 2 | 0.0150 | 0.1518 | 0.9896 | 0.9250 | 0.9132 | 1.00e-04 | 0.0 |
| 27 | 2 | 0.0158 | 0.1746 | 0.9912 | 0.9293 | 0.9215 | 1.00e-04 | 0.0 |
| 28 | 2 | 0.0154 | 0.1675 | 0.9914 | 0.9290 | 0.9153 | 1.00e-04 | 0.0 |
| 29 | 2 | 0.0161 | 0.1857 | 0.9913 | 0.9178 | 0.9130 | 1.00e-04 | 0.0 |
| 30 | 2 | 0.0141 | 0.1802 | 0.9924 | 0.9165 | 0.9075 | 1.00e-04 | 0.0 |

## Per-Class Validation Accuracy

| Class | Accuracy |
|-------|--------:|
| `apple|black_rot` | 1.0000 |
| `apple|cedar_apple_rust` | 1.0000 |
| `cherry|powdery_mildew` | 1.0000 |
| `corn|common_rust` | 1.0000 |
| `corn|healthy` | 1.0000 |
| `grape|esca_(black_measles)` | 1.0000 |
| `grape|healthy` | 1.0000 |
| `strawberry|leaf_scorch` | 1.0000 |
| `tomato|target_spot` | 1.0000 |
| `orange|huanglongbing_(citrus_greening)` | 0.9952 |
| `peach|bacterial_spot` | 0.9942 |
| `grape|leaf_blight_(isariopsis_leaf_spot)` | 0.9938 |
| `pepper|bacterial_spot` | 0.9933 |
| `soybean|healthy` | 0.9922 |
| `tomato|spider_mites_two_spotted_spider_mite` | 0.9880 |
| `apple|healthy` | 0.9808 |
| `blueberry|healthy` | 0.9794 |
| `pepper|healthy` | 0.9740 |
| `squash|powdery_mildew` | 0.9729 |
| `peach|healthy` | 0.9718 |
| `tomato|healthy` | 0.9677 |
| `cherry|healthy` | 0.9635 |
| `corn|northern_leaf_blight` | 0.9595 |
| `raspberry|healthy` | 0.9589 |
| `grape|black_rot` | 0.9572 |
| `potato|healthy` | 0.9565 |
| `apple|apple_scab` | 0.9537 |
| `strawberry|healthy` | 0.9518 |
| `tomato|tomato_yellow_leaf_curl_virus` | 0.9497 |
| `potato|late_blight` | 0.9398 |
| `corn|cercospora_leaf_spot_gray_leaf_spot` | 0.9351 |
| `corn|blight` | 0.9310 |
| `potato|early_blight` | 0.9286 |
| `tomato|bacterial_spot` | 0.9284 |
| `tomato|septoria_leaf_spot` | 0.9167 |
| `tomato|leaf_mold` | 0.9103 |
| `tomato|late_blight` | 0.9076 |
| `tomato|tomato_mosaic_virus` | 0.8906 |
| `tomato|early_blight` | 0.8889 |
| `apple|rust` | 0.7692 |
| `corn|rust` | 0.7059 |
| `pepper|leaf_spot` | 0.4545 |
| `corn|gray_leaf_spot` | 0.4000 |

## Confusion Matrix

Rows = true class, columns = predicted class. See JSON report for the full matrix.

- **Overall validation accuracy:** 0.9662
- **Classes:** 43
- **Samples evaluated:** 8,530

## Artifacts

| File | Description |
|------|-------------|
| `saved_models/resnet50/best_model.pth` | Best validation checkpoint |
| `saved_models/resnet50/last_model.pth` | Last epoch checkpoint |
| `saved_models/resnet50/stage1_best_model.pth` | Best Stage 1 checkpoint |
| `saved_models/resnet50/training_history.json` | Combined training history |
| `experiments/resnet50/tensorboard/` | TensorBoard event files |
