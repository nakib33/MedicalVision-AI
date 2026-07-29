"""Shared Model: EfficientNet wrappers (B0 variant)."""
import torch
import torch.nn as nn
from torchvision import models


def create_efficientnet(num_classes: int, variant: str = "efficientnet_b0",
                        pretrained: bool = True) -> nn.Module:
    """Create an EfficientNet model with the given number of output classes.

    Args:
        num_classes: Number of output classes.
        variant: Torchvision model name (efficientnet_b0 … efficientnet_b7).
        pretrained: Load ImageNet-pretrained weights.

    Returns:
        PyTorch model (put on CPU; caller moves to device).
    """
    weights = "DEFAULT" if pretrained else None
    model = getattr(models, variant)(weights=weights)

    # Replace the classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def create_efficientnet_b0(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Shorthand for EfficientNet-B0."""
    return create_efficientnet(num_classes, "efficientnet_b0", pretrained)
