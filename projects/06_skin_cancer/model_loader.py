"""Skin Cancer Classification - Model loader for inference."""


import torch
from shared.config import get_trained_model_path, DEVICE
from shared.models import create_model
from .config import PROJECT_ID, MODEL_NAME, CLASSES, IMG_SIZE

_model_instance = None


def get_model() -> torch.nn.Module:
    global _model_instance
    if _model_instance is not None:
        return _model_instance
    model_path = get_trained_model_path(PROJECT_ID)
    model = create_model(MODEL_NAME, num_classes=len(CLASSES))
    if model_path.exists():
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"  Model loaded from {model_path.name}")
    else:
        print(f"  No trained model at {model_path}")
    model.to(DEVICE)
    model.eval()
    _model_instance = model
    return model


def get_model_info() -> dict:
    return {
        "project_id": PROJECT_ID,
        "model": MODEL_NAME,
        "classes": CLASSES,
        "img_size": IMG_SIZE,
        "device": str(DEVICE),
        "trained": get_trained_model_path(PROJECT_ID).exists(),
    }
