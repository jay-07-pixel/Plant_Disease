"""Data augmentation configuration for the training split.

Defines augmentation settings only. Augmented images are not generated during
the preprocessing pipeline — configuration is consumed later by the training
``DataLoader`` and PyTorch transforms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class AugmentationTargetSplit(str, Enum):
    """Dataset split that receives augmentation."""

    TRAIN = "train"


@dataclass(frozen=True)
class RandomFlipConfig:
    """Random horizontal/vertical flip augmentation settings.

    Attributes:
        enabled: Whether flip augmentation is enabled.
        horizontal_probability: Probability of horizontal flip.
        vertical_probability: Probability of vertical flip.
    """

    enabled: bool = True
    horizontal_probability: float = 0.5
    vertical_probability: float = 0.0


@dataclass(frozen=True)
class RandomRotationConfig:
    """Random rotation augmentation settings.

    Attributes:
        enabled: Whether rotation augmentation is enabled.
        max_degrees: Maximum absolute rotation angle in degrees.
        probability: Probability of applying rotation.
    """

    enabled: bool = True
    max_degrees: float = 20.0
    probability: float = 0.5


@dataclass(frozen=True)
class RandomCropConfig:
    """Random crop augmentation settings.

    Attributes:
        enabled: Whether random crop is enabled.
        crop_scale_min: Minimum crop area as a fraction of the image.
        crop_scale_max: Maximum crop area as a fraction of the image.
        probability: Probability of applying random crop.
    """

    enabled: bool = True
    crop_scale_min: float = 0.8
    crop_scale_max: float = 1.0
    probability: float = 0.5


@dataclass(frozen=True)
class ColorJitterConfig:
    """Color jitter augmentation settings.

    Attributes:
        enabled: Whether color jitter is enabled.
        brightness: Maximum brightness jitter factor.
        contrast: Maximum contrast jitter factor.
        saturation: Maximum saturation jitter factor.
        hue: Maximum hue jitter factor.
        probability: Probability of applying color jitter.
    """

    enabled: bool = True
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.05
    probability: float = 0.5


@dataclass(frozen=True)
class GaussianNoiseConfig:
    """Gaussian noise augmentation settings.

    Attributes:
        enabled: Whether Gaussian noise is enabled.
        mean: Noise mean.
        std: Noise standard deviation (0–1 scale on pixel values).
        probability: Probability of applying noise.
    """

    enabled: bool = True
    mean: float = 0.0
    std: float = 0.02
    probability: float = 0.3


@dataclass(frozen=True)
class AugmentationConfig:
    """Complete augmentation configuration for the training split.

    Augmentation is applied only to the training split during model training.
    No augmented images are written to disk by the preprocessing pipeline.

    Attributes:
        enabled: Master switch for training-time augmentation.
        target_split: Split that receives augmentation (always ``train``).
        random_flip: Random flip settings.
        random_rotation: Random rotation settings.
        random_crop: Random crop settings.
        color_jitter: Color jitter settings.
        gaussian_noise: Gaussian noise settings.
    """

    enabled: bool = True
    target_split: AugmentationTargetSplit = AugmentationTargetSplit.TRAIN
    random_flip: RandomFlipConfig = RandomFlipConfig()
    random_rotation: RandomRotationConfig = RandomRotationConfig()
    random_crop: RandomCropConfig = RandomCropConfig()
    color_jitter: ColorJitterConfig = ColorJitterConfig()
    gaussian_noise: GaussianNoiseConfig = GaussianNoiseConfig()


def get_default_augmentation_config() -> AugmentationConfig:
    """Return the default training augmentation configuration.

    Returns:
        Default :class:`AugmentationConfig` instance.
    """
    return AugmentationConfig()


def augmentation_config_to_dict(config: AugmentationConfig) -> dict:
    """Serialize augmentation configuration to a dictionary.

    Args:
        config: Augmentation configuration.

    Returns:
        JSON-serializable dictionary.
    """
    payload = asdict(config)
    payload["target_split"] = config.target_split.value
    return payload
