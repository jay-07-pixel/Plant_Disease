"""Custom PyTorch dataset for processed plant disease images."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from src.preprocessing.split_dataset import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL
from src.training.training_config import DataLoaderConfig

logger = logging.getLogger(__name__)


class PlantDiseaseBatchItem(NamedTuple):
    """Single sample returned by :class:`PlantDiseaseDataset`.

    Attributes:
        image: CHW image tensor after transforms.
        canonical_label: Stable cross-dataset label key.
        class_index: Integer class index for the label.
        image_path: Absolute path to the processed image file.
    """

    image: torch.Tensor
    canonical_label: str
    class_index: int
    image_path: Path


@dataclass(frozen=True)
class LabelEncoder:
    """Bidirectional mapping between canonical labels and class indices.

    Attributes:
        label_to_index: Canonical label → class index.
        index_to_label: Class index → canonical label.
    """

    label_to_index: dict[str, int]
    index_to_label: dict[int, str]

    @property
    def num_classes(self) -> int:
        """Number of distinct canonical classes."""
        return len(self.label_to_index)

    def encode(self, canonical_label: str) -> int:
        """Return the class index for a canonical label."""
        return self.label_to_index[canonical_label]

    def decode(self, class_index: int) -> str:
        """Return the canonical label for a class index."""
        return self.index_to_label[class_index]


def load_processed_metadata(metadata_path: Path | str) -> pd.DataFrame:
    """Load processed metadata CSV.

    Args:
        metadata_path: Path to ``processed_metadata.csv``.

    Returns:
        Metadata DataFrame.

    Raises:
        FileNotFoundError: If the metadata file does not exist.
    """
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Processed metadata not found: {path}")

    df = pd.read_csv(path, encoding="utf-8")
    logger.info("Loaded processed metadata from %s (%d rows)", path, len(df))
    return df


def load_balancing_plan(plan_path: Path | str) -> pd.DataFrame:
    """Load the training balancing plan CSV.

    Args:
        plan_path: Path to ``training_balancing_plan.csv``.

    Returns:
        Balancing plan DataFrame indexed by canonical label.

    Raises:
        FileNotFoundError: If the balancing plan file does not exist.
    """
    path = Path(plan_path)
    if not path.exists():
        raise FileNotFoundError(f"Training balancing plan not found: {path}")

    df = pd.read_csv(path, encoding="utf-8")
    if "canonical_label" not in df.columns:
        raise ValueError("Balancing plan must contain a 'canonical_label' column.")

    df = df.set_index("canonical_label", drop=False)
    logger.info("Loaded training balancing plan from %s (%d classes)", path, len(df))
    return df


def build_label_encoder(metadata_df: pd.DataFrame) -> LabelEncoder:
    """Build a consistent label encoder from all metadata rows.

    Labels are sorted alphabetically so indices are stable across splits and
  runs.

    Args:
        metadata_df: Full processed metadata DataFrame.

    Returns:
        :class:`LabelEncoder` instance.
    """
    labels = sorted(metadata_df["canonical_label"].astype(str).unique())
    label_to_index = {label: index for index, label in enumerate(labels)}
    index_to_label = {index: label for label, index in label_to_index.items()}
    return LabelEncoder(label_to_index=label_to_index, index_to_label=index_to_label)


def filter_split(
    metadata_df: pd.DataFrame,
    split: str,
    *,
    split_column: str = "processed_split",
) -> pd.DataFrame:
    """Filter metadata to a single dataset split.

    Args:
        metadata_df: Full processed metadata DataFrame.
        split: Split name (``train``, ``val``, or ``test``).
        split_column: Column containing split labels.

    Returns:
        Filtered DataFrame for the requested split.
    """
    filtered = metadata_df[metadata_df[split_column] == split].reset_index(drop=True)
    logger.info("Filtered %s split: %d samples", split, len(filtered))
    return filtered


def compute_split_class_distribution(
    metadata_df: pd.DataFrame,
    label_encoder: LabelEncoder,
) -> dict[str, int]:
    """Count samples per canonical label in a split DataFrame.

    Args:
        metadata_df: Metadata for a single split.
        label_encoder: Label encoder for consistent label ordering.

    Returns:
        Mapping of canonical label to sample count (all labels included).
    """
    counts = metadata_df["canonical_label"].value_counts().to_dict()
    return {
        label_encoder.index_to_label[index]: int(counts.get(label_encoder.index_to_label[index], 0))
        for index in range(label_encoder.num_classes)
    }


class PlantDiseaseDataset(Dataset):
    """PyTorch dataset over processed plant disease images.

    Each item is a :class:`PlantDiseaseBatchItem` with image tensor, canonical
    label, class index, and absolute image path.

    Args:
        metadata_df: Metadata rows for a single split.
        label_encoder: Shared label encoder across all splits.
        transform: Optional torchvision transform pipeline.
        project_root: Root used to resolve relative image paths.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        label_encoder: LabelEncoder,
        transform: Compose | None = None,
        project_root: Path | str | None = None,
    ) -> None:
        if metadata_df.empty:
            raise ValueError("Cannot create dataset from empty metadata.")

        self._records = metadata_df.reset_index(drop=True)
        self._label_encoder = label_encoder
        self._transform = transform
        self._project_root = Path(project_root) if project_root else Path.cwd()

        missing_labels = set(self._records["canonical_label"].astype(str)) - set(
            label_encoder.label_to_index
        )
        if missing_labels:
            sample = sorted(missing_labels)[:3]
            raise ValueError(f"Unknown canonical labels in metadata: {sample}")

    def __len__(self) -> int:
        return len(self._records)

    def get_canonical_label(self, index: int) -> str:
        """Return the canonical label for a dataset index without loading the image."""
        return str(self._records.iloc[index]["canonical_label"])

    def __getitem__(self, index: int) -> PlantDiseaseBatchItem:
        row = self._records.iloc[index]
        canonical_label = str(row["canonical_label"])
        class_index = self._label_encoder.encode(canonical_label)

        relative_path = Path(str(row["processed_image_path"]))
        image_path = (
            relative_path
            if relative_path.is_absolute()
            else self._project_root / relative_path
        )

        if not image_path.exists():
            raise FileNotFoundError(f"Processed image not found: {image_path}")

        with Image.open(image_path) as pil_image:
            image = pil_image.convert("RGB")

        if self._transform is not None:
            image = self._transform(image)

        if not isinstance(image, torch.Tensor):
            raise TypeError(
                "Transform pipeline must end with ToTensor(); "
                f"got {type(image).__name__}"
            )

        return PlantDiseaseBatchItem(
            image=image,
            canonical_label=canonical_label,
            class_index=class_index,
            image_path=image_path,
        )

    @property
    def metadata_records(self) -> pd.DataFrame:
        """Metadata rows backing this dataset (read-only copy)."""
        return self._records.copy()

    @property
    def label_encoder(self) -> LabelEncoder:
        """Shared label encoder for this dataset."""
        return self._label_encoder

    @property
    def canonical_labels(self) -> list[str]:
        """Canonical labels present in this split (in index order)."""
        return [
            self._label_encoder.index_to_label[index]
            for index in range(self._label_encoder.num_classes)
        ]


def create_split_datasets(
    config: DataLoaderConfig,
    *,
    train_transform: Compose | None = None,
    eval_transform: Compose | None = None,
) -> tuple[PlantDiseaseDataset, PlantDiseaseDataset, PlantDiseaseDataset, LabelEncoder]:
    """Create train, validation, and test datasets from processed metadata.

    Args:
        config: Data loader configuration with metadata paths.
        train_transform: Transform pipeline for training.
        eval_transform: Transform pipeline for validation and test.

    Returns:
        Tuple of ``(train_dataset, val_dataset, test_dataset, label_encoder)``.
    """
    metadata_df = load_processed_metadata(config.processed_metadata_path)
    label_encoder = build_label_encoder(metadata_df)

    train_df = filter_split(metadata_df, SPLIT_TRAIN)
    val_df = filter_split(metadata_df, SPLIT_VAL)
    test_df = filter_split(metadata_df, SPLIT_TEST)

    train_dataset = PlantDiseaseDataset(
        train_df,
        label_encoder,
        transform=train_transform,
        project_root=config.project_root,
    )
    val_dataset = PlantDiseaseDataset(
        val_df,
        label_encoder,
        transform=eval_transform,
        project_root=config.project_root,
    )
    test_dataset = PlantDiseaseDataset(
        test_df,
        label_encoder,
        transform=eval_transform,
        project_root=config.project_root,
    )

    return train_dataset, val_dataset, test_dataset, label_encoder
