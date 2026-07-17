# Baseline CNN Model Report

**Generated:** 2026-07-13T08:48:30.543594+00:00  
**Model:** `BaselineCNN`  

> Architecture verification only — no training performed.

## Summary

- **Input shape:** `[1, 3, 224, 224]`
- **Output shape:** `[1, 43]`
- **Number of classes:** 43
- **Total parameters:** 399,947
- **Trainable parameters:** 399,947
- **Non-trainable parameters:** 0
- **Trainer compatible:** True
- **Verification:** Compatible with Trainer: input=[4, 3, 224, 224], output=[4, 43], loss=3.1896

## Architecture

```
Input (3×224×224)
  ↓
Conv Block 1: Conv2D → BatchNorm → ReLU → MaxPool
  ↓
Conv Block 2: Conv2D → BatchNorm → ReLU → MaxPool
  ↓
Conv Block 3: Conv2D → BatchNorm → ReLU → MaxPool
  ↓
Conv Block 4: Conv2D → BatchNorm → ReLU → MaxPool
  ↓
Global Average Pooling
  ↓
Dropout
  ↓
Fully Connected
  ↓
43-class logits (softmax at inference)
```

## Layers

| Layer | Type | Output Shape | Parameters |
|-------|------|--------------|----------:|
| `features` | Sequential | `[1, 256, 14, 14]` | 388,896 |
| `features.0` | ConvBlock | `[1, 32, 112, 112]` | 928 |
| `features.0.block` | Sequential | `[1, 32, 112, 112]` | 928 |
| `features.0.block.0` | Conv2d | `[1, 32, 224, 224]` | 864 |
| `features.0.block.1` | BatchNorm2d | `[1, 32, 224, 224]` | 64 |
| `features.0.block.2` | ReLU | `[1, 32, 224, 224]` | 0 |
| `features.0.block.3` | MaxPool2d | `[1, 32, 112, 112]` | 0 |
| `features.1` | ConvBlock | `[1, 64, 56, 56]` | 18,560 |
| `features.1.block` | Sequential | `[1, 64, 56, 56]` | 18,560 |
| `features.1.block.0` | Conv2d | `[1, 64, 112, 112]` | 18,432 |
| `features.1.block.1` | BatchNorm2d | `[1, 64, 112, 112]` | 128 |
| `features.1.block.2` | ReLU | `[1, 64, 112, 112]` | 0 |
| `features.1.block.3` | MaxPool2d | `[1, 64, 56, 56]` | 0 |
| `features.2` | ConvBlock | `[1, 128, 28, 28]` | 73,984 |
| `features.2.block` | Sequential | `[1, 128, 28, 28]` | 73,984 |
| `features.2.block.0` | Conv2d | `[1, 128, 56, 56]` | 73,728 |
| `features.2.block.1` | BatchNorm2d | `[1, 128, 56, 56]` | 256 |
| `features.2.block.2` | ReLU | `[1, 128, 56, 56]` | 0 |
| `features.2.block.3` | MaxPool2d | `[1, 128, 28, 28]` | 0 |
| `features.3` | ConvBlock | `[1, 256, 14, 14]` | 295,424 |
| `features.3.block` | Sequential | `[1, 256, 14, 14]` | 295,424 |
| `features.3.block.0` | Conv2d | `[1, 256, 28, 28]` | 294,912 |
| `features.3.block.1` | BatchNorm2d | `[1, 256, 28, 28]` | 512 |
| `features.3.block.2` | ReLU | `[1, 256, 28, 28]` | 0 |
| `features.3.block.3` | MaxPool2d | `[1, 256, 14, 14]` | 0 |
| `global_pool` | AdaptiveAvgPool2d | `[1, 256, 1, 1]` | 0 |
| `dropout` | Dropout | `[1, 256]` | 0 |
| `classifier` | Linear | `[1, 43]` | 11,051 |

## Initialization

- **Conv2d / Linear:** Kaiming normal (fan_out, ReLU)
- **BatchNorm2d:** Weight = 1, Bias = 0

## Integration

```python
from src.models.baseline_cnn import BaselineCNN
from src.training.train import create_trainer

model = BaselineCNN(num_classes=43)
trainer = create_trainer(model)
history = trainer.train()  # when ready to train
```
