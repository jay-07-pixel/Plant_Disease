"""Model architectures for plant disease classification."""

__all__ = [
    "BaselineCNN",
    "BaselineCNNConfig",
    "ConvBlock",
    "ModelSummary",
    "ParameterCounts",
    "ResNet50TransferClassifier",
    "ResNet101TransferClassifier",
    "DenseNet121TransferClassifier",
    "EfficientNetB0TransferClassifier",
    "EfficientNetB3TransferClassifier",
    "build_model_summary",
    "count_parameters",
    "print_model_summary",
    "run_baseline_verification",
    "verify_forward_pass",
    "verify_trainer_compatibility",
]


def __getattr__(name: str):
    """Lazy exports to avoid import cycles when running submodules as scripts."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name == "EfficientNetB3TransferClassifier":
        from src.models import efficientnet_b3_transfer as _efficientnet_b3_transfer

        return getattr(_efficientnet_b3_transfer, name)

    if name == "EfficientNetB0TransferClassifier":
        from src.models import efficientnet_b0_transfer as _efficientnet_b0_transfer

        return getattr(_efficientnet_b0_transfer, name)

    if name == "DenseNet121TransferClassifier":
        from src.models import densenet121_transfer as _densenet121_transfer

        return getattr(_densenet121_transfer, name)

    if name == "ResNet101TransferClassifier":
        from src.models import resnet101_transfer as _resnet101_transfer

        return getattr(_resnet101_transfer, name)

    if name in {
        "ParameterCounts",
        "ResNet50TransferClassifier",
        "verify_forward_pass",
    }:
        from src.models import resnet50_transfer as _resnet50_transfer

        return getattr(_resnet50_transfer, name)

    from src.models import baseline_cnn as _baseline_cnn

    return getattr(_baseline_cnn, name)
