# DenseNet121 Transfer Learning Training Report

**Generated:** 2026-07-15T15:25:48.699693+00:00  
**Experiment:** `densenet121`  
**Device:** `cuda`  

## Summary

- **Best epoch:** 24
- **Best validation accuracy:** 0.9340
- **Final training accuracy:** 0.9960
- **Final validation accuracy:** 0.9306
- **Final validation precision:** 0.9323
- **Final validation recall:** 0.9306
- **Final validation F1 score:** 0.9295
- **Final validation top-1 accuracy:** 0.9306
- **Final validation top-5 accuracy:** 0.9837
- **Training time:** 8061.6s
- **Epochs completed:** 30

## Model

- **Trainable parameters:** 2,202,155
- **Frozen parameters:** 4,795,776

## Configuration

### Two-Stage Transfer Learning

| Stage | Epochs | Learning Rate | Trainable Layers |
|------:|-------:|--------------:|------------------|
| 1 — Feature extraction | 10 | 0.001 | Classifier head only |
| 2 — Fine-tuning | 20 | 0.0001 | denseblock4 + classifier |

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
| 1 | 1 | 0.7661 | 0.5299 | 0.7383 | 0.8269 | 0.7742 | 1.00e-03 | 561.8 |
| 2 | 1 | 0.3450 | 0.4502 | 0.8734 | 0.8434 | 0.7964 | 1.00e-03 | 573.1 |
| 3 | 1 | 0.2834 | 0.3859 | 0.8906 | 0.8622 | 0.8102 | 1.00e-03 | 663.8 |
| 4 | 1 | 0.2348 | 0.4632 | 0.9056 | 0.8492 | 0.7998 | 1.00e-03 | 660.8 |
| 5 | 1 | 0.2125 | 0.3787 | 0.9130 | 0.8593 | 0.8212 | 1.00e-03 | 672.5 |
| 6 | 1 | 0.2060 | 0.3487 | 0.9168 | 0.8639 | 0.8210 | 1.00e-03 | 648.5 |
| 7 | 1 | 0.2072 | 0.3276 | 0.9108 | 0.8780 | 0.8323 | 1.00e-03 | 523.8 |
| 8 | 1 | 0.2008 | 0.3224 | 0.9153 | 0.8866 | 0.8439 | 1.00e-03 | 530.7 |
| 9 | 1 | 0.1939 | 0.3836 | 0.9160 | 0.8679 | 0.8199 | 1.00e-03 | 494.0 |
| 10 | 1 | 0.1907 | 0.3689 | 0.9192 | 0.8719 | 0.8312 | 1.00e-03 | 391.4 |
| 11 | 2 | 0.1440 | 0.2620 | 0.9393 | 0.8901 | 0.8670 | 1.00e-04 | 388.1 |
| 12 | 2 | 0.0703 | 0.2051 | 0.9614 | 0.9154 | 0.8934 | 1.00e-04 | 387.4 |
| 13 | 2 | 0.0481 | 0.1859 | 0.9710 | 0.9214 | 0.9065 | 1.00e-04 | 393.4 |
| 14 | 2 | 0.0375 | 0.1905 | 0.9775 | 0.9182 | 0.9025 | 1.00e-04 | 392.5 |
| 15 | 2 | 0.0319 | 0.1682 | 0.9810 | 0.9162 | 0.9054 | 1.00e-04 | 394.0 |
| 16 | 2 | 0.0245 | 0.1832 | 0.9843 | 0.9196 | 0.9086 | 1.00e-04 | 385.9 |
| 17 | 2 | 0.0285 | 0.1611 | 0.9853 | 0.9301 | 0.9177 | 1.00e-04 | 0.0 |
| 18 | 2 | 0.0266 | 0.1577 | 0.9869 | 0.9251 | 0.9145 | 1.00e-04 | 0.0 |
| 19 | 2 | 0.0184 | 0.1732 | 0.9889 | 0.9227 | 0.9152 | 1.00e-04 | 0.0 |
| 20 | 2 | 0.0190 | 0.1867 | 0.9881 | 0.9282 | 0.9158 | 1.00e-04 | 0.0 |
| 21 | 2 | 0.0166 | 0.1806 | 0.9906 | 0.9150 | 0.9104 | 1.00e-04 | 0.0 |
| 22 | 2 | 0.0179 | 0.1652 | 0.9908 | 0.9233 | 0.9163 | 1.00e-04 | 0.0 |
| 23 | 2 | 0.0181 | 0.1569 | 0.9916 | 0.9286 | 0.9216 | 1.00e-04 | 0.0 |
| 24 | 2 | 0.0125 | 0.1424 | 0.9942 | 0.9340 | 0.9252 | 5.00e-05 | 0.0 |
| 25 | 2 | 0.0083 | 0.1538 | 0.9948 | 0.9266 | 0.9209 | 5.00e-05 | 0.0 |
| 26 | 2 | 0.0102 | 0.1567 | 0.9942 | 0.9327 | 0.9267 | 5.00e-05 | 0.0 |
| 27 | 2 | 0.0085 | 0.1413 | 0.9949 | 0.9267 | 0.9233 | 5.00e-05 | 0.0 |
| 28 | 2 | 0.0096 | 0.1457 | 0.9959 | 0.9339 | 0.9274 | 5.00e-05 | 0.0 |
| 29 | 2 | 0.0082 | 0.1479 | 0.9956 | 0.9292 | 0.9232 | 5.00e-05 | 0.0 |
| 30 | 2 | 0.0093 | 0.1489 | 0.9960 | 0.9306 | 0.9295 | 5.00e-05 | 0.0 |

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
| `grape|leaf_blight_(isariopsis_leaf_spot)` | 1.0000 |
| `peach|bacterial_spot` | 1.0000 |
| `pepper|bacterial_spot` | 1.0000 |
| `strawberry|leaf_scorch` | 1.0000 |
| `orange|huanglongbing_(citrus_greening)` | 0.9988 |
| `soybean|healthy` | 0.9961 |
| `tomato|spider_mites_two_spotted_spider_mite` | 0.9920 |
| `corn|cercospora_leaf_spot_gray_leaf_spot` | 0.9870 |
| `squash|powdery_mildew` | 0.9797 |
| `blueberry|healthy` | 0.9794 |
| `pepper|healthy` | 0.9784 |
| `peach|healthy` | 0.9718 |
| `tomato|healthy` | 0.9718 |
| `tomato|target_spot` | 0.9716 |
| `apple|healthy` | 0.9692 |
| `strawberry|healthy` | 0.9639 |
| `cherry|healthy` | 0.9635 |
| `grape|black_rot` | 0.9572 |
| `tomato|septoria_leaf_spot` | 0.9479 |
| `tomato|tomato_yellow_leaf_curl_virus` | 0.9460 |
| `raspberry|healthy` | 0.9452 |
| `tomato|early_blight` | 0.9444 |
| `potato|early_blight` | 0.9286 |
| `tomato|late_blight` | 0.9274 |
| `apple|rust` | 0.9231 |
| `tomato|leaf_mold` | 0.9231 |
| `apple|apple_scab` | 0.9167 |
| `potato|late_blight` | 0.9157 |
| `potato|healthy` | 0.9130 |
| `corn|northern_leaf_blight` | 0.9122 |
| `tomato|bacterial_spot` | 0.8985 |
| `tomato|tomato_mosaic_virus` | 0.8906 |
| `corn|blight` | 0.8621 |
| `corn|rust` | 0.6471 |
| `pepper|leaf_spot` | 0.5455 |
| `corn|gray_leaf_spot` | 0.5000 |

## Confusion Matrix

Rows = true class, columns = predicted class. See JSON report for the full matrix.

- **Overall validation accuracy:** 0.9672
- **Classes:** 43
- **Samples evaluated:** 8,530

## Artifacts

| File | Description |
|------|-------------|
| `saved_models/densenet121/best_model.pth` | Best validation checkpoint |
| `saved_models/densenet121/last_model.pth` | Last epoch checkpoint |
| `saved_models/densenet121/stage1_best_model.pth` | Best Stage 1 checkpoint |
| `saved_models/densenet121/training_history.json` | Combined training history |
| `experiments/densenet121/tensorboard/` | TensorBoard event files |
