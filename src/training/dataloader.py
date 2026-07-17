"""PyTorch DataLoader factory, verification, and reporting."""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.training.dataset import (
    PlantDiseaseBatchItem,
    PlantDiseaseDataset,
    LabelEncoder,
    compute_split_class_distribution,
    create_split_datasets,
    load_balancing_plan,
)
from src.training.sampler import (
    build_class_weights_tensor,
    build_weighted_random_sampler,
    summarize_sampler,
)
from src.training.training_config import DataLoaderConfig, get_dataloader_config
from src.training.transforms import build_eval_transforms, build_train_transforms

logger = logging.getLogger(__name__)


@dataclass
class SplitLoaderBundle:
    """DataLoader and dataset for one split.

    Attributes:
        split: Split name (``train``, ``val``, or ``test``).
        dataset: Underlying :class:`PlantDiseaseDataset`.
        dataloader: PyTorch :class:`DataLoader`.
    """

    split: str
    dataset: PlantDiseaseDataset
    dataloader: DataLoader


@dataclass
class PlantDiseaseDataLoaders:
    """Train, validation, and test data loaders with shared metadata.

    Attributes:
        train: Training split bundle (weighted sampler + class weights).
        val: Validation split bundle (no balancing).
        test: Test split bundle (no balancing).
        label_encoder: Shared label encoder across splits.
        class_weights: Per-class loss weights for training (``None`` if disabled).
        config: Data loader configuration used to build loaders.
    """

    train: SplitLoaderBundle
    val: SplitLoaderBundle
    test: SplitLoaderBundle
    label_encoder: LabelEncoder
    class_weights: torch.Tensor | None
    config: DataLoaderConfig


@dataclass
class DataLoaderReport:
    """Verification report for the PyTorch data pipeline.

    Attributes:
        generated_at: UTC ISO-8601 timestamp.
        config: Serialized data loader configuration.
        dataset_statistics: Sample counts per split.
        class_distribution: Per-class counts per split.
        tensor_statistics: Single-sample tensor shape verification.
        batch_statistics: First-batch shape verification per split.
        sampler_information: Weighted sampling summary (training only).
        num_classes: Total number of canonical classes.
        label_encoder: Mapping of class index to canonical label.
    """

    generated_at: str
    config: dict[str, Any]
    dataset_statistics: dict[str, int]
    class_distribution: dict[str, dict[str, int]]
    tensor_statistics: dict[str, Any]
    batch_statistics: dict[str, Any]
    sampler_information: dict[str, Any]
    num_classes: int
    label_encoder: dict[int, str] = field(default_factory=dict)


@dataclass
class PlantDiseaseBatch:
    """Collated batch from :class:`PlantDiseaseDataset`.

    Attributes:
        images: Batched image tensor ``(N, C, H, W)``.
        canonical_labels: Canonical label per sample.
        class_indices: Class index tensor ``(N,)``.
        image_paths: Absolute image path per sample.
    """

    images: torch.Tensor
    canonical_labels: list[str]
    class_indices: torch.Tensor
    image_paths: list[Path]


def collate_plant_disease_batch(batch: list[PlantDiseaseBatchItem]) -> PlantDiseaseBatch:
    """Collate dataset items into a batched structure for model training.

    Args:
        batch: List of :class:`PlantDiseaseBatchItem` from the dataset.

    Returns:
        :class:`PlantDiseaseBatch` with stacked tensors and label lists.
    """
    return PlantDiseaseBatch(
        images=torch.stack([item.image for item in batch]),
        canonical_labels=[item.canonical_label for item in batch],
        class_indices=torch.tensor([item.class_index for item in batch], dtype=torch.long),
        image_paths=[item.image_path for item in batch],
    )


def _worker_init_fn(worker_id: int, seed: int) -> None:
    """Seed numpy and random in DataLoader workers for reproducibility."""
    import random

    import numpy as np

    worker_seed = seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_dataloaders(config: DataLoaderConfig | None = None) -> PlantDiseaseDataLoaders:
    """Build train, validation, and test :class:`DataLoader` instances.

    Training uses ``WeightedRandomSampler`` and exposes class weights for loss
    functions. Validation and test loaders use sequential sampling without
    balancing.

    Args:
        config: Data loader configuration. Defaults to :func:`get_dataloader_config`.

    Returns:
        :class:`PlantDiseaseDataLoaders` with all three splits configured.
    """
    config = config or get_dataloader_config()
    balancing_plan = load_balancing_plan(config.balancing_plan_path)

    train_transform = build_train_transforms(config)
    eval_transform = build_eval_transforms(config)

    train_dataset, val_dataset, test_dataset, label_encoder = create_split_datasets(
        config,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )

    generator = torch.Generator()
    generator.manual_seed(config.random_seed)

    train_sampler = None
    train_shuffle = config.shuffle

    if config.use_weighted_sampler:
        train_sampler = build_weighted_random_sampler(
            train_dataset,
            balancing_plan,
            generator=generator,
        )
        train_shuffle = False

    class_weights = None
    if config.use_class_weights:
        class_weights = build_class_weights_tensor(label_encoder, balancing_plan)

    worker_init = None
    if config.num_workers > 0:
        worker_init = lambda worker_id: _worker_init_fn(worker_id, config.random_seed)  # noqa: E731

    common_loader_kwargs: dict[str, Any] = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "drop_last": config.drop_last,
        "collate_fn": collate_plant_disease_batch,
    }
    if config.num_workers > 0 and config.persistent_workers:
        common_loader_kwargs["persistent_workers"] = True
    if worker_init is not None:
        common_loader_kwargs["worker_init_fn"] = worker_init

    train_loader = DataLoader(
        train_dataset,
        shuffle=train_shuffle,
        sampler=train_sampler,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common_loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_loader_kwargs,
    )

    logger.info(
        "Built dataloaders: train=%d, val=%d, test=%d, classes=%d",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
        label_encoder.num_classes,
    )

    return PlantDiseaseDataLoaders(
        train=SplitLoaderBundle("train", train_dataset, train_loader),
        val=SplitLoaderBundle("val", val_dataset, val_loader),
        test=SplitLoaderBundle("test", test_dataset, test_loader),
        label_encoder=label_encoder,
        class_weights=class_weights,
        config=config,
    )


def _collate_batch_shapes(batch: PlantDiseaseBatch) -> dict[str, Any]:
    """Extract tensor and metadata shapes from a collated batch."""
    return {
        "image_batch_shape": list(batch.images.shape),
        "class_index_batch_shape": list(batch.class_indices.shape),
        "image_dtype": str(batch.images.dtype),
        "num_unique_labels_in_batch": len(set(batch.canonical_labels)),
    }


def verify_dataloaders(loaders: PlantDiseaseDataLoaders) -> DataLoaderReport:
    """Verify loader integrity and collect statistics for reporting.

    Args:
        loaders: Built data loaders.

    Returns:
        :class:`DataLoaderReport` with dataset, tensor, and batch statistics.
    """
    config = loaders.config
    balancing_plan = load_balancing_plan(config.balancing_plan_path)

    dataset_statistics = {
        "train": len(loaders.train.dataset),
        "val": len(loaders.val.dataset),
        "test": len(loaders.test.dataset),
        "total": (
            len(loaders.train.dataset)
            + len(loaders.val.dataset)
            + len(loaders.test.dataset)
        ),
    }

    class_distribution = {
        "train": compute_split_class_distribution(
            loaders.train.dataset.metadata_records, loaders.label_encoder
        ),
        "val": compute_split_class_distribution(
            loaders.val.dataset.metadata_records, loaders.label_encoder
        ),
        "test": compute_split_class_distribution(
            loaders.test.dataset.metadata_records, loaders.label_encoder
        ),
    }

    train_sample = loaders.train.dataset[0]
    tensor_statistics = {
        "single_image_shape": list(train_sample.image.shape),
        "single_image_dtype": str(train_sample.image.dtype),
        "channels": train_sample.image.shape[0],
        "height": train_sample.image.shape[1],
        "width": train_sample.image.shape[2],
        "expected_image_shape": [3, config.image_size, config.image_size],
        "shape_matches_expected": list(train_sample.image.shape)
        == [3, config.image_size, config.image_size],
        "sample_canonical_label": train_sample.canonical_label,
        "sample_class_index": train_sample.class_index,
        "sample_image_path": str(train_sample.image_path),
    }

    batch_statistics: dict[str, Any] = {}
    for bundle in (loaders.train, loaders.val, loaders.test):
        batch = next(iter(bundle.dataloader))
        if not isinstance(batch, PlantDiseaseBatch):
            raise TypeError(f"Unexpected batch type: {type(batch).__name__}")

        batch_statistics[bundle.split] = {
            **_collate_batch_shapes(batch),
            "batch_size": int(batch.images.shape[0]),
            "label_counter": dict(Counter(batch.canonical_labels)),
        }

    sampler_information: dict[str, Any] = {
        "validation_balanced": False,
        "test_balanced": False,
    }
    if config.use_weighted_sampler:
        sampler_information.update(
            summarize_sampler(loaders.train.dataset, balancing_plan)
        )
    else:
        sampler_information["weighted_random_sampler_enabled"] = False

    if loaders.class_weights is not None:
        sampler_information["class_weights_enabled"] = True
        sampler_information["class_weights_shape"] = list(loaders.class_weights.shape)
        sampler_information["class_weights_min"] = float(loaders.class_weights.min())
        sampler_information["class_weights_max"] = float(loaders.class_weights.max())
        sampler_information["class_weights_mean"] = float(loaders.class_weights.mean())
    else:
        sampler_information["class_weights_enabled"] = False

    config_dict = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "image_size": config.image_size,
        "pin_memory": config.pin_memory,
        "persistent_workers": config.persistent_workers,
        "shuffle": config.shuffle,
        "random_seed": config.random_seed,
        "drop_last": config.drop_last,
        "use_weighted_sampler": config.use_weighted_sampler,
        "use_class_weights": config.use_class_weights,
        "processed_metadata_path": str(config.processed_metadata_path),
        "balancing_plan_path": str(config.balancing_plan_path),
        "normalization_mean": list(config.normalization_mean),
        "normalization_std": list(config.normalization_std),
    }

    return DataLoaderReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        config=config_dict,
        dataset_statistics=dataset_statistics,
        class_distribution=class_distribution,
        tensor_statistics=tensor_statistics,
        batch_statistics=batch_statistics,
        sampler_information=sampler_information,
        num_classes=loaders.label_encoder.num_classes,
        label_encoder=dict(loaders.label_encoder.index_to_label),
    )


def save_dataloader_report_json(
    report: DataLoaderReport,
    output_path: Path | str,
) -> None:
    """Save the dataloader report as JSON.

    Args:
        report: Verification report.
        output_path: Destination file path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(report), handle, indent=2)
    logger.info("Saved dataloader report JSON to %s", path)


def save_dataloader_report_markdown(
    report: DataLoaderReport,
    output_path: Path | str,
) -> None:
    """Save the dataloader report as Markdown.

    Args:
        report: Verification report.
        output_path: Destination file path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# DataLoader Report",
        "",
        f"**Generated:** {report.generated_at}  ",
        f"**Metadata source:** `{report.config['processed_metadata_path']}`  ",
        f"**Balancing plan:** `{report.config['balancing_plan_path']}`  ",
        "",
        "> Data preparation only — no model training, evaluation, or augmentation on disk.",
        "",
        "## Summary",
        "",
        f"- **Number of classes:** {report.num_classes}",
        f"- **Training samples:** {report.dataset_statistics['train']:,}",
        f"- **Validation samples:** {report.dataset_statistics['val']:,}",
        f"- **Test samples:** {report.dataset_statistics['test']:,}",
        f"- **Total samples:** {report.dataset_statistics['total']:,}",
        "",
        "## Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|------:|",
    ]

    for key, value in report.config.items():
        if key in {"processed_metadata_path", "balancing_plan_path"}:
            continue
        lines.append(f"| `{key}` | {value} |")

    lines.extend(
        [
            "",
            "## Tensor Verification",
            "",
            f"- **Single image shape:** `{report.tensor_statistics['single_image_shape']}`",
            f"- **Expected shape:** `{report.tensor_statistics['expected_image_shape']}`",
            f"- **Shape matches expected:** {report.tensor_statistics['shape_matches_expected']}",
            f"- **Dtype:** `{report.tensor_statistics['single_image_dtype']}`",
            "",
            "## Batch Verification",
            "",
            "| Split | Batch Shape | Batch Size | Unique Labels |",
            "|-------|-------------|----------:|--------------:|",
        ]
    )

    for split in ("train", "val", "test"):
        stats = report.batch_statistics[split]
        lines.append(
            f"| {split} | `{stats['image_batch_shape']}` | {stats['batch_size']} | "
            f"{stats['num_unique_labels_in_batch']} |"
        )

    lines.extend(["", "## Sampler Information", ""])
    for key, value in report.sampler_information.items():
        lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")

    lines.extend(["", "## Class Distribution (Training Split)", "", "| Class | Count |", "|-------|------:|"])
    train_dist = report.class_distribution["train"]
    for label, count in sorted(train_dist.items(), key=lambda item: item[1], reverse=True):
        if count > 0:
            lines.append(f"| `{label}` | {count:,} |")

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "| File | Description |",
            "|------|-------------|",
            "| `src/training/dataset.py` | Custom PyTorch Dataset |",
            "| `src/training/dataloader.py` | DataLoader factory and verification |",
            "| `src/training/sampler.py` | WeightedRandomSampler and class weights |",
            "| `src/training/transforms.py` | Training and evaluation transforms |",
            "| `src/training/training_config.py` | Configurable pipeline settings |",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved dataloader report markdown to %s", path)


def run_dataloader_verification(
    config: DataLoaderConfig | None = None,
) -> tuple[PlantDiseaseDataLoaders, DataLoaderReport]:
    """Build dataloaders, verify integrity, and write reports.

    Args:
        config: Optional data loader configuration.

    Returns:
        Tuple of ``(loaders, report)``.
    """
    config = config or get_dataloader_config()
    loaders = build_dataloaders(config)
    report = verify_dataloaders(loaders)

    save_dataloader_report_json(report, config.report_json_path)
    save_dataloader_report_markdown(report, config.report_md_path)

    logger.info(
        "DataLoader verification complete: %d classes, tensor shape %s",
        report.num_classes,
        report.tensor_statistics["single_image_shape"],
    )
    return loaders, report


def main() -> None:
    """CLI entry point for dataloader verification and reporting."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_dataloader_verification()


if __name__ == "__main__":
    main()
