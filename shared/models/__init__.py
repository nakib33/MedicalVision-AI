"""Shared Model Registry — factory function to create any supported model."""
from .efficientnet import create_efficientnet_b0
from .resnet import create_resnet18, create_resnet50
from .densenet import create_densenet121
from .custom_cnn import CustomCNN


MODEL_REGISTRY = {
    "efficientnet_b0": create_efficientnet_b0,
    "resnet18": create_resnet18,
    "resnet50": create_resnet50,
    "densenet121": create_densenet121,
    "custom_cnn": lambda num_classes, **kw: CustomCNN(num_classes, **kw),
}


def create_model(model_name: str, num_classes: int, **kwargs):
    """Factory: create a model by name.

    Args:
        model_name: Key in MODEL_REGISTRY.
        num_classes: Number of output classes.
        **kwargs: Passed to the model constructor.

    Returns:
        PyTorch nn.Module.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. "
                         f"Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](num_classes=num_classes, **kwargs)
