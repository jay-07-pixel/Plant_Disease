"""PyTorch training data pipeline for PlantDiseaseAI.

Provides dataset, transforms, weighted sampling, and DataLoader factories
for all future vision model architectures. Does not train or evaluate models.

Import from submodules directly, e.g.::

    from src.training.dataloader import build_dataloaders
    from src.training.dataset import PlantDiseaseDataset
"""

__all__ = [
    "DataLoaderConfig",
    "DataLoaderReport",
    "LabelEncoder",
    "PlantDiseaseBatch",
    "PlantDiseaseBatchItem",
    "PlantDiseaseDataLoaders",
    "PlantDiseaseDataset",
    "SplitLoaderBundle",
    "build_class_weights_tensor",
    "build_dataloaders",
    "build_eval_transforms",
    "build_label_encoder",
    "build_per_sample_weights",
    "build_train_transforms",
    "build_weighted_random_sampler",
    "collate_plant_disease_batch",
    "create_split_datasets",
    "get_dataloader_config",
    "load_balancing_plan",
    "load_processed_metadata",
    "run_dataloader_verification",
    "summarize_sampler",
    "verify_dataloaders",
    # Training engine
    "Trainer",
    "TrainingEngineConfig",
    "TrainingHistory",
    "EpochMetrics",
    "create_trainer",
    "verify_training_framework",
    "get_training_engine_config",
]


def __getattr__(name: str):
    """Lazy exports to avoid import cycles when running submodules as scripts."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name in {
        "DataLoaderReport",
        "PlantDiseaseBatch",
        "PlantDiseaseDataLoaders",
        "SplitLoaderBundle",
        "build_dataloaders",
        "collate_plant_disease_batch",
        "run_dataloader_verification",
        "verify_dataloaders",
    }:
        from src.training import dataloader as _dataloader

        return getattr(_dataloader, name)

    if name in {
        "LabelEncoder",
        "PlantDiseaseBatchItem",
        "PlantDiseaseDataset",
        "build_label_encoder",
        "create_split_datasets",
        "load_balancing_plan",
        "load_processed_metadata",
    }:
        from src.training import dataset as _dataset

        return getattr(_dataset, name)

    if name in {
        "build_class_weights_tensor",
        "build_per_sample_weights",
        "build_weighted_random_sampler",
        "summarize_sampler",
    }:
        from src.training import sampler as _sampler

        return getattr(_sampler, name)

    if name in {"DataLoaderConfig", "get_dataloader_config"}:
        from src.training import training_config as _training_config

        return getattr(_training_config, name)

    if name in {
        "Trainer",
        "TrainingEngineConfig",
        "get_training_engine_config",
    }:
        from src.training import trainer as _trainer

        return getattr(_trainer, name)

    if name in {"TrainingHistory", "EpochMetrics"}:
        from src.training import metrics as _metrics

        return getattr(_metrics, name)

    if name in {"create_trainer", "verify_training_framework"}:
        from src.training import train as _train

        return getattr(_train, name)

    if name in {"build_eval_transforms", "build_train_transforms"}:
        from src.training import transforms as _transforms

        return getattr(_transforms, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
