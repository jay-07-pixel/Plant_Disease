"""Configuration for the image preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_METADATA_CSV = Path("datasets/processed/dataset_metadata.csv")
DEFAULT_PROCESSED_IMAGES_DIR = Path("datasets/processed/images")
DEFAULT_PROCESSED_METADATA_CSV = Path("datasets/processed/processed_metadata.csv")
DEFAULT_REPORT_JSON = Path("reports/preprocessing_report.json")
DEFAULT_REPORT_MD = Path("reports/preprocessing_report.md")

# ImageNet normalization — default for pretrained vision models.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ImageSizeConfig:
    """Target image dimensions after preprocessing.

    Attributes:
        width: Target width in pixels.
        height: Target height in pixels.
    """

    width: int
    height: int

    def as_tuple(self) -> tuple[int, int]:
        """Return ``(width, height)``."""
        return self.width, self.height


IMAGE_SIZE_PRESETS: dict[int, ImageSizeConfig] = {
    224: ImageSizeConfig(224, 224),
    299: ImageSizeConfig(299, 299),
    384: ImageSizeConfig(384, 384),
}


@dataclass
class NormalizationConfig:
    """Channel-wise normalization applied during training.

    Processed images are saved as uint8 RGB on disk. Normalization parameters
    are stored for use by the PyTorch ``DataLoader`` and model pipelines.

    Attributes:
        mean: Per-channel mean for RGB (0–1 scale).
        std: Per-channel standard deviation for RGB (0–1 scale).
        enabled: Whether normalization is part of the training transform.
    """

    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    enabled: bool = True


@dataclass
class SplitConfig:
    """Stratified train/validation/test split ratios.

    Attributes:
        train_ratio: Fraction assigned to training.
        val_ratio: Fraction assigned to validation.
        test_ratio: Fraction assigned to testing.
        random_seed: Seed for reproducible stratified splitting.
    """

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")


@dataclass
class OutputConfig:
    """Output paths for processed artifacts.

    Attributes:
        images_dir: Root directory for processed images.
        metadata_csv: Path for processed metadata CSV.
        image_format: Output image format extension (``jpg`` or ``png``).
        jpeg_quality: JPEG quality when saving (1–100).
    """

    images_dir: Path = DEFAULT_PROCESSED_IMAGES_DIR
    metadata_csv: Path = DEFAULT_PROCESSED_METADATA_CSV
    image_format: str = "jpg"
    jpeg_quality: int = 95


@dataclass
class PreprocessingConfig:
    """Full preprocessing pipeline configuration.

    Attributes:
        target_size: Output image dimensions.
        normalization: Normalization parameters for training.
        split: Stratified split configuration.
        output: Output path configuration.
        source_metadata_path: Input metadata CSV path.
        project_root: Project root for resolving relative paths.
        padding_color: RGB padding color for letterbox resize.
        skip_existing: Skip processing when output file already exists.
        report_json_path: Preprocessing report JSON path.
        report_md_path: Preprocessing report markdown path.
    """

    target_size: ImageSizeConfig = field(default_factory=lambda: IMAGE_SIZE_PRESETS[224])
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    source_metadata_path: Path = DEFAULT_METADATA_CSV
    project_root: Path = field(default_factory=Path.cwd)
    padding_color: tuple[int, int, int] = (0, 0, 0)
    skip_existing: bool = True
    report_json_path: Path = DEFAULT_REPORT_JSON
    report_md_path: Path = DEFAULT_REPORT_MD


def get_preprocessing_config(
    image_size: int = 224,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    project_root: Path | str | None = None,
) -> PreprocessingConfig:
    """Build a :class:`PreprocessingConfig` from common parameters.

    Args:
        image_size: Target size preset (224, 299, or 384).
        train_ratio: Training split fraction.
        val_ratio: Validation split fraction.
        test_ratio: Test split fraction.
        random_seed: Random seed for stratified splitting.
        project_root: Project root directory.

    Returns:
        A configured :class:`PreprocessingConfig` instance.

    Raises:
        ValueError: If ``image_size`` is not a supported preset.
    """
    if image_size not in IMAGE_SIZE_PRESETS:
        supported = ", ".join(str(size) for size in sorted(IMAGE_SIZE_PRESETS))
        raise ValueError(
            f"Unsupported image size {image_size}. Supported presets: {supported}"
        )

    return PreprocessingConfig(
        target_size=IMAGE_SIZE_PRESETS[image_size],
        split=SplitConfig(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed,
        ),
        project_root=Path(project_root) if project_root else Path.cwd(),
    )
