# Preprocessing Report

**Generated:** 2026-07-13T07:50:31.510174+00:00  

> Source images in `datasets/external/` were not modified.

## Summary

- **Images processed:** 56,857
- **Images skipped:** 0
- **Images failed:** 0

## Configuration

- **Target size:** 224×224
- **Output format:** jpg
- **Split ratios:** train=70%, val=15%, test=15%
- **Normalization enabled:** True

## Resize Statistics

- **Mean scale factor:** 0.8500
- **Scale range:** 0.0373 – 1.9478
- **Images with padding:** 2,449
- **Mean original size:** 292 × 285 px

## Split Statistics

- **Train:** 39,797 (70.0%)
- **Validation:** 8,530 (15.0%)
- **Test:** 8,530 (15.0%)
- **Classes with small splits:** 0

## Augmentation Configuration (Training Only)

Augmentation is configured but **not applied** during preprocessing.

- **Enabled:** True
- **Target split:** train
- **Random flip:** True
- **Random rotation:** True
- **Random crop:** True
- **Color jitter:** True
- **Gaussian noise:** True

## Output Files

| File | Description |
|------|-------------|
| `datasets/processed/images/` | Processed images by split and class |
| `datasets/processed/processed_metadata.csv` | Processed metadata table |
