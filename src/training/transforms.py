"""Torchvision transform pipelines for training and evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchvision import transforms

from src.preprocessing.augmentation import AugmentationConfig, get_default_augmentation_config

if TYPE_CHECKING:
    from src.training.training_config import DataLoaderConfig


def build_train_transforms(
    config: DataLoaderConfig,
    augmentation: AugmentationConfig | None = None,
) -> transforms.Compose:
    """Build stochastic transforms for the training split.

    Pipeline: resize → random horizontal flip → random rotation → color jitter
    → random crop → tensor conversion → ImageNet normalization.

    Args:
        config: Data loader configuration (image size and normalization).
        augmentation: Optional augmentation overrides. Defaults to project preset.

    Returns:
        Composed training transform.
    """
    aug = augmentation or get_default_augmentation_config()
    size = config.image_size
    crop_padding = max(int(size * 0.125), 1)

    transform_steps: list = [
        transforms.Resize((size + crop_padding, size + crop_padding)),
    ]

    if aug.random_flip.enabled:
        transform_steps.append(
            transforms.RandomHorizontalFlip(p=aug.random_flip.horizontal_probability)
        )

    if aug.random_rotation.enabled:
        transform_steps.append(
            transforms.RandomRotation(degrees=aug.random_rotation.max_degrees)
        )

    if aug.color_jitter.enabled:
        transform_steps.append(
            transforms.ColorJitter(
                brightness=aug.color_jitter.brightness,
                contrast=aug.color_jitter.contrast,
                saturation=aug.color_jitter.saturation,
                hue=aug.color_jitter.hue,
            )
        )

    if aug.random_crop.enabled:
        transform_steps.append(transforms.RandomCrop(size))
    else:
        transform_steps.append(transforms.CenterCrop(size))

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.normalization_mean,
                std=config.normalization_std,
            ),
        ]
    )

    return transforms.Compose(transform_steps)


def build_eval_transforms(config: DataLoaderConfig) -> transforms.Compose:
    """Build deterministic transforms for validation and test splits.

    Pipeline: resize → tensor conversion → ImageNet normalization.
    No random augmentation is applied.

    Args:
        config: Data loader configuration (image size and normalization).

    Returns:
        Composed evaluation transform.
    """
    size = config.image_size
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.normalization_mean,
                std=config.normalization_std,
            ),
        ]
    )
