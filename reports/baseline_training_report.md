# Baseline CNN Training Report

**Generated:** 2026-07-13T18:05:51.369016+00:00  
**Experiment:** `baseline_cnn`  
**Device:** `cuda`  

## Summary

- **Best epoch:** 30
- **Best validation accuracy:** 0.8395
- **Final training accuracy:** 0.8606
- **Final validation accuracy:** 0.8395
- **Training time:** 13179.8s
- **Epochs completed:** 30

## Configuration

| Parameter | Value |
|-----------|------:|
| Epochs | 30 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | CrossEntropyLoss |
| Scheduler | ReduceLROnPlateau |
| Early stopping patience | 7 |
| Random seed | 42 |

## Per-Epoch Metrics

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Val F1 | LR | Time (s) |
|------:|-----------:|---------:|----------:|--------:|-------:|---:|---------:|
| 1 | 2.7388 | 3.0879 | 0.1593 | 0.3507 | 0.2075 | 1.00e-03 | 583.2 |
| 2 | 1.7816 | 2.4812 | 0.3070 | 0.4555 | 0.3144 | 1.00e-03 | 548.2 |
| 3 | 1.4878 | 2.1837 | 0.3941 | 0.5085 | 0.3566 | 1.00e-03 | 493.1 |
| 4 | 1.2935 | 1.9415 | 0.4683 | 0.5561 | 0.4205 | 1.00e-03 | 504.4 |
| 5 | 1.1351 | 1.9677 | 0.5272 | 0.5222 | 0.3960 | 1.00e-03 | 501.5 |
| 6 | 1.0148 | 1.5644 | 0.5718 | 0.6064 | 0.4820 | 1.00e-03 | 719.4 |
| 7 | 0.8968 | 1.5501 | 0.6125 | 0.6108 | 0.5003 | 1.00e-03 | 741.9 |
| 8 | 0.8195 | 1.3777 | 0.6423 | 0.6579 | 0.5622 | 1.00e-03 | 643.2 |
| 9 | 0.7216 | 1.1862 | 0.6687 | 0.6919 | 0.5939 | 1.00e-03 | 552.1 |
| 10 | 0.6736 | 1.0901 | 0.6869 | 0.7247 | 0.6242 | 1.00e-03 | 594.2 |
| 11 | 0.6286 | 1.1984 | 0.7093 | 0.7043 | 0.6094 | 1.00e-03 | 634.4 |
| 12 | 0.5671 | 1.1163 | 0.7261 | 0.6898 | 0.5988 | 1.00e-03 | 618.7 |
| 13 | 0.5511 | 1.0791 | 0.7386 | 0.7130 | 0.6283 | 1.00e-03 | 606.0 |
| 14 | 0.4869 | 0.9499 | 0.7628 | 0.7438 | 0.6678 | 1.00e-03 | 511.7 |
| 15 | 0.4924 | 0.9559 | 0.7637 | 0.7314 | 0.6561 | 1.00e-03 | 612.5 |
| 16 | 0.4515 | 0.8044 | 0.7804 | 0.7779 | 0.7038 | 1.00e-03 | 528.1 |
| 17 | 0.4287 | 0.7144 | 0.7822 | 0.7785 | 0.7038 | 1.00e-03 | 507.3 |
| 18 | 0.4174 | 0.7055 | 0.7973 | 0.7896 | 0.7269 | 1.00e-03 | 513.5 |
| 19 | 0.3933 | 0.7683 | 0.8021 | 0.7856 | 0.7198 | 1.00e-03 | 520.4 |
| 20 | 0.3659 | 0.8377 | 0.8124 | 0.7866 | 0.7106 | 1.00e-03 | 515.2 |
| 21 | 0.3574 | 0.7043 | 0.8166 | 0.8087 | 0.7392 | 1.00e-03 | 564.0 |
| 22 | 0.3420 | 0.6745 | 0.8232 | 0.8052 | 0.7477 | 1.00e-03 | 604.6 |
| 23 | 0.3351 | 0.6297 | 0.8296 | 0.8034 | 0.7404 | 1.00e-03 | 562.0 |
| 24 | 0.3141 | 0.6295 | 0.8344 | 0.8148 | 0.7685 | 1.00e-03 | 0.0 |
| 25 | 0.3017 | 0.7183 | 0.8415 | 0.8073 | 0.7593 | 1.00e-03 | 0.0 |
| 26 | 0.2853 | 0.6287 | 0.8479 | 0.8342 | 0.7713 | 1.00e-03 | 0.0 |
| 27 | 0.2825 | 0.5868 | 0.8485 | 0.8253 | 0.7691 | 1.00e-03 | 0.0 |
| 28 | 0.2601 | 0.5535 | 0.8571 | 0.8330 | 0.7795 | 1.00e-03 | 0.0 |
| 29 | 0.2773 | 0.6448 | 0.8554 | 0.7952 | 0.7387 | 1.00e-03 | 0.0 |
| 30 | 0.2577 | 0.4927 | 0.8606 | 0.8395 | 0.7880 | 1.00e-03 | 0.0 |

## Per-Class Validation Accuracy

| Class | Accuracy |
|-------|--------:|
| `apple|black_rot` | 1.0000 |
| `apple|cedar_apple_rust` | 1.0000 |
| `corn|common_rust` | 1.0000 |
| `corn|healthy` | 1.0000 |
| `potato|healthy` | 1.0000 |
| `strawberry|leaf_scorch` | 0.9940 |
| `grape|leaf_blight_(isariopsis_leaf_spot)` | 0.9938 |
| `tomato|spider_mites_two_spotted_spider_mite` | 0.9841 |
| `cherry|powdery_mildew` | 0.9810 |
| `orange|huanglongbing_(citrus_greening)` | 0.9782 |
| `corn|cercospora_leaf_spot_gray_leaf_spot` | 0.9740 |
| `tomato|healthy` | 0.9637 |
| `peach|bacterial_spot` | 0.9507 |
| `pepper|bacterial_spot` | 0.9467 |
| `grape|esca_(black_measles)` | 0.9420 |
| `pepper|healthy` | 0.9394 |
| `cherry|healthy` | 0.9270 |
| `tomato|tomato_mosaic_virus` | 0.9219 |
| `grape|healthy` | 0.9189 |
| `peach|healthy` | 0.9155 |
| `blueberry|healthy` | 0.9136 |
| `raspberry|healthy` | 0.8904 |
| `apple|healthy` | 0.8808 |
| `potato|early_blight` | 0.8571 |
| `corn|northern_leaf_blight` | 0.8514 |
| `squash|powdery_mildew` | 0.8203 |
| `apple|apple_scab` | 0.8148 |
| `potato|late_blight` | 0.8133 |
| `tomato|leaf_mold` | 0.8077 |
| `tomato|septoria_leaf_spot` | 0.8021 |
| `strawberry|healthy` | 0.7952 |
| `tomato|early_blight` | 0.7901 |
| `tomato|bacterial_spot` | 0.7672 |
| `corn|blight` | 0.7586 |
| `grape|black_rot` | 0.7219 |
| `tomato|tomato_yellow_leaf_curl_virus` | 0.7006 |
| `tomato|target_spot` | 0.6777 |
| `soybean|healthy` | 0.6727 |
| `corn|rust` | 0.6471 |
| `apple|rust` | 0.5385 |
| `tomato|late_blight` | 0.4917 |
| `corn|gray_leaf_spot` | 0.4000 |
| `pepper|leaf_spot` | 0.3636 |

## Confusion Matrix

Rows = true class, columns = predicted class. See JSON report for the full matrix.

- **Overall validation accuracy:** 0.8423
- **Classes:** 43
- **Samples evaluated:** 8,530

## Artifacts

| File | Description |
|------|-------------|
| `saved_models/baseline_cnn/best_model.pth` | Best validation checkpoint |
| `saved_models/baseline_cnn/last_model.pth` | Last epoch checkpoint |
| `saved_models/baseline_cnn/training_history.json` | Full training history |
| `experiments/baseline_cnn/tensorboard/` | TensorBoard event files |
