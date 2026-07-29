"""Shared Model: DenseNet121 wrapper."""
import torch.nn as nn
from torchvision import models


def create_densenet(num_classes: int, variant: str = "densenet121",
                    pretrained: bool = True) -> nn.Module:
    """Create a DenseNet model with the given number of output classes.

    Args:
        num_classes: Number of output classes.
        variant: densenet121, densenet161, densenet169, densenet201.
        pretrained: Load ImageNet-pretrained weights.

    Returns:
        PyTorch model (on CPU; caller moves to device).
    """
    weights = "DEFAULT" if pretrained else None
    model = getattr(models, variant)(weights=weights)

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


def create_densenet121(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Shorthand for DenseNet121."""
    return create_densenet(num_classes, "densenet121", pretrained)
