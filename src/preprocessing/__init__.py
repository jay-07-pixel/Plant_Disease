"""Image preprocessing pipeline for PlantDiseaseAI.

Processes images from metadata into a unified dataset under
``datasets/processed/images/`` without modifying source data in
``datasets/external/``.
"""

from src.preprocessing.preprocess import run_preprocessing
from src.preprocessing.preprocessing_config import (
    IMAGE_SIZE_PRESETS,
    PreprocessingConfig,
    get_preprocessing_config,
)

__all__ = [
    "IMAGE_SIZE_PRESETS",
    "PreprocessingConfig",
    "get_preprocessing_config",
    "run_preprocessing",
]
