"""Configuration for the PyTorch training data pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.preprocessing.preprocessing_config import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGE_SIZE_PRESETS,
    ImageSizeConfig,
)

DEFAULT_PROCESSED_METADATA_CSV = Path("datasets/processed/processed_metadata.csv")
DEFAULT_BALANCING_PLAN_CSV = Path("datasets/processed/training_balancing_plan.csv")
DEFAULT_DATALOADER_REPORT_JSON = Path("reports/dataloader_report.json")
DEFAULT_DATALOADER_REPORT_MD = Path("reports/dataloader_report.md")


@dataclass
class DataLoaderConfig:
    """PyTorch ``DataLoader`` and dataset configuration.

    Attributes:
        batch_size: Number of samples per batch.
        num_workers: Subprocesses for data loading (``0`` = main process only).
        image_size: Target spatial size preset (224, 299, or 384).
        pin_memory: Pin CPU tensors for faster GPU transfer.
        persistent_workers: Keep worker processes alive between epochs.
        shuffle: Shuffle training indices when no weighted sampler is used.
        random_seed: Seed for reproducible sampling and worker initialization.
        drop_last: Drop the last incomplete batch.
        use_weighted_sampler: Enable ``WeightedRandomSampler`` on training.
        use_class_weights: Expose per-class loss weights from the balancing plan.
        project_root: Project root for resolving relative image paths.
        processed_metadata_path: Path to processed metadata CSV.
        balancing_plan_path: Path to training balancing plan CSV.
        report_json_path: Output path for the dataloader verification report (JSON).
        report_md_path: Output path for the dataloader verification report (Markdown).
    """

    batch_size: int = 32
    num_workers: int = 0
    image_size: int = 224
    pin_memory: bool = True
    persistent_workers: bool = False
    shuffle: bool = True
    random_seed: int = 42
    drop_last: bool = False
    use_weighted_sampler: bool = True
    use_class_weights: bool = True
    project_root: Path = field(default_factory=Path.cwd)
    processed_metadata_path: Path = DEFAULT_PROCESSED_METADATA_CSV
    balancing_plan_path: Path = DEFAULT_BALANCING_PLAN_CSV
    report_json_path: Path = DEFAULT_DATALOADER_REPORT_JSON
    report_md_path: Path = DEFAULT_DATALOADER_REPORT_MD

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if self.image_size not in IMAGE_SIZE_PRESETS:
            supported = ", ".join(str(size) for size in sorted(IMAGE_SIZE_PRESETS))
            raise ValueError(
                f"Unsupported image_size {self.image_size}. Supported: {supported}"
            )
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError("persistent_workers requires num_workers > 0")

    @property
    def target_size(self) -> ImageSizeConfig:
        """Return the resolved image size configuration."""
        return IMAGE_SIZE_PRESETS[self.image_size]

    @property
    def normalization_mean(self) -> tuple[float, float, float]:
        """ImageNet mean used for tensor normalization."""
        return IMAGENET_MEAN

    @property
    def normalization_std(self) -> tuple[float, float, float]:
        """ImageNet standard deviation used for tensor normalization."""
        return IMAGENET_STD


def get_dataloader_config(
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int = 224,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    shuffle: bool = True,
    random_seed: int = 42,
    project_root: Path | str | None = None,
) -> DataLoaderConfig:
    """Build a :class:`DataLoaderConfig` from common parameters.

    Args:
        batch_size: Number of samples per batch.
        num_workers: Data loading worker processes.
        image_size: Spatial size preset (224, 299, or 384).
        pin_memory: Whether to pin memory for GPU transfer.
        persistent_workers: Keep workers alive between epochs.
        shuffle: Shuffle training data when no sampler is active.
        random_seed: Random seed for reproducibility.
        project_root: Project root directory.

    Returns:
        Configured :class:`DataLoaderConfig` instance.
    """
    return DataLoaderConfig(
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        shuffle=shuffle,
        random_seed=random_seed,
        project_root=Path(project_root) if project_root else Path.cwd(),
    )
