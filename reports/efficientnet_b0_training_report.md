# EfficientNet-B0 Transfer Learning Training Report

**Generated:** 2026-07-15T19:21:15.368100+00:00  
**Experiment:** `efficientnet_b0`  
**Device:** `cuda`  

## Summary

- **Best epoch:** 20
- **Best validation accuracy:** 0.9127
- **Final training accuracy:** 0.9821
- **Final validation accuracy:** 0.9081
- **Final validation precision:** 0.8756
- **Final validation recall:** 0.9081
- **Final validation F1 score:** 0.8884
- **Final validation top-1 accuracy:** 0.9081
- **Final validation top-5 accuracy:** 0.9855
- **Training time:** 13619.9s
- **Epochs completed:** 27

## Model

- **Trainable parameters:** 1,184,475
- **Frozen parameters:** 2,878,156

## Configuration

### Two-Stage Transfer Learning

| Stage | Epochs | Learning Rate | Trainable Layers |
|------:|-------:|--------------:|------------------|
| 1 — Feature extraction | 10 | 0.001 | Classifier head only |
| 2 — Fine-tuning | 20 | 0.0001 | last feature stage + classifier |

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
| 1 | 1 | 0.7874 | 0.6240 | 0.7330 | 0.8018 | 0.7258 | 1.00e-03 | 619.7 |
| 2 | 1 | 0.4040 | 0.4873 | 0.8492 | 0.8367 | 0.7692 | 1.00e-03 | 621.0 |
| 3 | 1 | 0.3404 | 0.4534 | 0.8630 | 0.8481 | 0.7813 | 1.00e-03 | 642.1 |
| 4 | 1 | 0.3263 | 0.4639 | 0.8706 | 0.8519 | 0.7800 | 1.00e-03 | 641.0 |
| 5 | 1 | 0.2974 | 0.4071 | 0.8758 | 0.8636 | 0.7965 | 1.00e-03 | 630.7 |
| 6 | 1 | 0.2876 | 0.4289 | 0.8767 | 0.8553 | 0.7911 | 1.00e-03 | 636.5 |
| 7 | 1 | 0.2840 | 0.3989 | 0.8841 | 0.8586 | 0.7949 | 1.00e-03 | 626.4 |
| 8 | 1 | 0.2790 | 0.3785 | 0.8856 | 0.8606 | 0.8044 | 1.00e-03 | 626.7 |
| 9 | 1 | 0.2597 | 0.3773 | 0.8870 | 0.8721 | 0.8077 | 1.00e-03 | 414.8 |
| 10 | 1 | 0.2678 | 0.3970 | 0.8849 | 0.8612 | 0.7988 | 1.00e-03 | 432.0 |
| 11 | 2 | 0.2014 | 0.3012 | 0.9069 | 0.8859 | 0.8334 | 1.00e-04 | 631.3 |
| 12 | 2 | 0.1336 | 0.2578 | 0.9301 | 0.8892 | 0.8497 | 1.00e-04 | 628.3 |
| 13 | 2 | 0.1001 | 0.2445 | 0.9414 | 0.9032 | 0.8658 | 1.00e-04 | 614.2 |
| 14 | 2 | 0.0898 | 0.2340 | 0.9488 | 0.9055 | 0.8706 | 1.00e-04 | 461.1 |
| 15 | 2 | 0.0765 | 0.2280 | 0.9566 | 0.9044 | 0.8719 | 1.00e-04 | 411.5 |
| 16 | 2 | 0.0674 | 0.1940 | 0.9599 | 0.9087 | 0.8830 | 1.00e-04 | 406.2 |
| 17 | 2 | 0.0568 | 0.2104 | 0.9626 | 0.9058 | 0.8799 | 1.00e-04 | 405.7 |
| 18 | 2 | 0.0526 | 0.1981 | 0.9659 | 0.9068 | 0.8835 | 1.00e-04 | 404.0 |
| 19 | 2 | 0.0444 | 0.1920 | 0.9692 | 0.9047 | 0.8838 | 1.00e-04 | 401.9 |
| 20 | 2 | 0.0455 | 0.1971 | 0.9707 | 0.9127 | 0.8874 | 1.00e-04 | 404.2 |
| 21 | 2 | 0.0389 | 0.1975 | 0.9726 | 0.9076 | 0.8819 | 1.00e-04 | 397.0 |
| 22 | 2 | 0.0396 | 0.1920 | 0.9748 | 0.8995 | 0.8794 | 1.00e-04 | 562.6 |
| 23 | 2 | 0.0368 | 0.2043 | 0.9762 | 0.8993 | 0.8802 | 1.00e-04 | 446.0 |
| 24 | 2 | 0.0328 | 0.2074 | 0.9775 | 0.9004 | 0.8792 | 1.00e-04 | 402.0 |
| 25 | 2 | 0.0328 | 0.1989 | 0.9790 | 0.9005 | 0.8812 | 1.00e-04 | 433.7 |
| 26 | 2 | 0.0304 | 0.1876 | 0.9795 | 0.9059 | 0.8883 | 1.00e-04 | 365.6 |
| 27 | 2 | 0.0263 | 0.1871 | 0.9821 | 0.9081 | 0.8884 | 5.00e-05 | 353.9 |

## Per-Class Validation Accuracy

| Class | Accuracy |
|-------|--------:|
| `apple|cedar_apple_rust` | 1.0000 |
| `apple|rust` | 1.0000 |
| `cherry|powdery_mildew` | 1.0000 |
| `corn|common_rust` | 1.0000 |
| `grape|healthy` | 1.0000 |
| `strawberry|leaf_scorch` | 0.9940 |
| `grape|leaf_blight_(isariopsis_leaf_spot)` | 0.9938 |
| `pepper|bacterial_spot` | 0.9933 |
| `apple|black_rot` | 0.9892 |
| `orange|huanglongbing_(citrus_greening)` | 0.9891 |
| `tomato|target_spot` | 0.9858 |
| `pepper|healthy` | 0.9827 |
| `corn|healthy` | 0.9770 |
| `strawberry|healthy` | 0.9759 |
| `grape|esca_(black_measles)` | 0.9758 |
| `peach|healthy` | 0.9718 |
| `blueberry|healthy` | 0.9671 |
| `peach|bacterial_spot` | 0.9652 |
| `raspberry|healthy` | 0.9589 |
| `apple|healthy` | 0.9577 |
| `squash|powdery_mildew` | 0.9525 |
| `tomato|leaf_mold` | 0.9487 |
| `corn|cercospora_leaf_spot_gray_leaf_spot` | 0.9481 |
| `tomato|spider_mites_two_spotted_spider_mite` | 0.9442 |
| `grape|black_rot` | 0.9412 |
| `potato|late_blight` | 0.9398 |
| `tomato|tomato_yellow_leaf_curl_virus` | 0.9374 |
| `cherry|healthy` | 0.9343 |
| `soybean|healthy` | 0.9327 |
| `apple|apple_scab` | 0.9259 |
| `tomato|late_blight` | 0.9241 |
| `potato|healthy` | 0.9130 |
| `corn|northern_leaf_blight` | 0.8986 |
| `corn|blight` | 0.8966 |
| `tomato|bacterial_spot` | 0.8955 |
| `tomato|tomato_mosaic_virus` | 0.8906 |
| `tomato|healthy` | 0.8750 |
| `tomato|early_blight` | 0.8395 |
| `potato|early_blight` | 0.8333 |
| `tomato|septoria_leaf_spot` | 0.8125 |
| `corn|rust` | 0.5294 |
| `corn|gray_leaf_spot` | 0.5000 |
| `pepper|leaf_spot` | 0.3636 |

## Confusion Matrix

Rows = true class, columns = predicted class. See JSON report for the full matrix.

- **Overall validation accuracy:** 0.9423
- **Classes:** 43
- **Samples evaluated:** 8,530

## Artifacts

| File | Description |
|------|-------------|
| `saved_models/efficientnet_b0/best_model.pth` | Best validation checkpoint |
| `saved_models/efficientnet_b0/last_model.pth` | Last epoch checkpoint |
| `saved_models/efficientnet_b0/stage1_best_model.pth` | Best Stage 1 checkpoint |
| `saved_models/efficientnet_b0/training_history.json` | Combined training history |
| `experiments/efficientnet_b0/tensorboard/` | TensorBoard event files |
