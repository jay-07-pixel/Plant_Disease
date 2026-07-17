"""Image transformation utilities for the preprocessing pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from src.preprocessing.preprocessing_config import ImageSizeConfig, NormalizationConfig

logger = logging.getLogger(__name__)

MIN_IMAGE_DIMENSION = 16
MAX_IMAGE_DIMENSION = 20_000


@dataclass
class TransformResult:
    """Result of transforming a single image.

    Attributes:
        image: Processed uint8 RGB image array with shape ``(H, W, 3)``.
        original_width: Source image width in pixels.
        original_height: Source image height in pixels.
        scale_factor: Resize scale applied before padding.
        padded: Whether letterbox padding was applied.
    """

    image: np.ndarray
    original_width: int
    original_height: int
    scale_factor: float
    padded: bool


@dataclass
class ValidationResult:
    """Outcome of validating a source image.

    Attributes:
        is_valid: Whether the image passed validation.
        width: Detected width, or ``None`` if unreadable.
        height: Detected height, or ``None`` if unreadable.
        image_format: Detected format extension.
        error: Error message when validation fails.
    """

    is_valid: bool
    width: int | None
    height: int | None
    image_format: str
    error: str | None = None


def validate_image(image_path: Path) -> ValidationResult:
    """Validate that an image can be opened and meets dimension constraints.

    Args:
        image_path: Path to the source image (read-only).

    Returns:
        A :class:`ValidationResult` describing validation outcome.
    """
    image_format = image_path.suffix.lower().lstrip(".")

    try:
        with Image.open(image_path) as img:
            width, height = img.size
            if img.format:
                image_format = img.format.lower()

            if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
                return ValidationResult(
                    is_valid=False,
                    width=width,
                    height=height,
                    image_format=image_format,
                    error=(
                        f"Image too small ({width}x{height}); "
                        f"minimum dimension is {MIN_IMAGE_DIMENSION}px."
                    ),
                )

            if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                return ValidationResult(
                    is_valid=False,
                    width=width,
                    height=height,
                    image_format=image_format,
                    error=(
                        f"Image too large ({width}x{height}); "
                        f"maximum dimension is {MAX_IMAGE_DIMENSION}px."
                    ),
                )

            return ValidationResult(
                is_valid=True,
                width=width,
                height=height,
                image_format=image_format,
            )

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return ValidationResult(
            is_valid=False,
            width=None,
            height=None,
            image_format=image_format,
            error=str(exc),
        )


def load_image_rgb(image_path: Path) -> np.ndarray:
    """Load an image as an RGB uint8 numpy array.

    Args:
        image_path: Path to the source image (read-only).

    Returns:
        RGB image array with shape ``(H, W, 3)``.

    Raises:
        ValueError: If the image cannot be loaded.
    """
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        with Image.open(image_path) as img:
            rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
            return rgb

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def convert_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an image array to RGB uint8 format.

    Args:
        image: Input image array (grayscale, RGB, or BGR).

    Returns:
        RGB uint8 array with shape ``(H, W, 3)``.
    """
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    if image.shape[2] == 3:
        return image.astype(np.uint8, copy=False)

    raise ValueError(f"Unsupported image shape: {image.shape}")


def resize_with_padding(
    image: np.ndarray,
    target_size: ImageSizeConfig,
    padding_color: tuple[int, int, int] = (0, 0, 0),
) -> tuple[np.ndarray, float, bool]:
    """Resize an image preserving aspect ratio using letterbox padding.

    Args:
        image: RGB uint8 image array.
        target_size: Target width and height.
        padding_color: RGB padding color.

    Returns:
        Tuple of padded image, scale factor, and whether padding was applied.
    """
    height, width = image.shape[:2]
    target_w, target_h = target_size.as_tuple()

    scale = min(target_w / width, target_h / height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=padding_color,
    )

    was_padded = (pad_left + pad_right + pad_top + pad_bottom) > 0
    return padded, scale, was_padded


def normalize_image(
    image: np.ndarray,
    config: NormalizationConfig,
) -> np.ndarray:
    """Normalize an RGB uint8 image to float32 using channel statistics.

    Args:
        image: RGB uint8 image array.
        config: Normalization configuration.

    Returns:
        Float32 array with shape ``(H, W, 3)`` and values roughly in [-2, 2].
    """
    image_float = image.astype(np.float32) / 255.0
    mean = np.array(config.mean, dtype=np.float32)
    std = np.array(config.std, dtype=np.float32)
    return (image_float - mean) / std


def save_processed_image(
    image: np.ndarray,
    output_path: Path,
    *,
    image_format: str = "jpg",
    jpeg_quality: int = 95,
) -> None:
    """Save a processed RGB uint8 image to disk.

    Args:
        image: RGB uint8 image array.
        output_path: Destination file path.
        image_format: Output format (``jpg`` or ``png``).
        jpeg_quality: JPEG quality when saving JPG files.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if image_format.lower() in {"jpg", "jpeg"}:
        cv2.imwrite(
            str(output_path),
            image_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
        )
        return

    if image_format.lower() == "png":
        cv2.imwrite(str(output_path), image_bgr)
        return

    raise ValueError(f"Unsupported output format: {image_format}")


def transform_image(
    image_path: Path,
    target_size: ImageSizeConfig,
    padding_color: tuple[int, int, int] = (0, 0, 0),
) -> TransformResult:
    """Load, validate, convert, and resize a single image.

    Args:
        image_path: Path to the source image (read-only).
        target_size: Target output dimensions.
        padding_color: RGB letterbox padding color.

    Returns:
        A :class:`TransformResult` with the processed image.

    Raises:
        ValueError: If validation or loading fails.
    """
    validation = validate_image(image_path)
    if not validation.is_valid:
        raise ValueError(validation.error or "Image validation failed.")

    rgb = convert_to_rgb(load_image_rgb(image_path))
    processed, scale, padded = resize_with_padding(rgb, target_size, padding_color)

    return TransformResult(
        image=processed,
        original_width=int(validation.width or 0),
        original_height=int(validation.height or 0),
        scale_factor=scale,
        padded=padded,
    )
