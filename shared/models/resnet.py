"""Shared Model: ResNet wrappers (ResNet18, ResNet50)."""
import torch.nn as nn
from torchvision import models


def create_resnet(num_classes: int, variant: str = "resnet50",
                  pretrained: bool = True) -> nn.Module:
    """Create a ResNet model with the given number of output classes.

    Args:
        num_classes: Number of output classes.
        variant: resnet18, resnet34, resnet50, resnet101, resnet152.
        pretrained: Load ImageNet-pretrained weights.

    Returns:
        PyTorch model (on CPU; caller moves to device).
    """
    weights = "DEFAULT" if pretrained else None
    model = getattr(models, variant)(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def create_resnet18(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Shorthand for ResNet18."""
    return create_resnet(num_classes, "resnet18", pretrained)


def create_resnet50(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Shorthand for ResNet50."""
    return create_resnet(num_classes, "resnet50", pretrained)
