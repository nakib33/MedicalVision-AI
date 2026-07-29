"""Brain Tumor MRI — Model loader for inference."""

import torch
from pathlib import Path
from shared.config import get_trained_model_path, DEVICE
from shared.models import create_model
from .config import PROJECT_ID, MODEL_NAME, CLASSES, IMG_SIZE


_model_instance = None


def get_model() -> torch.nn.Module:
    """Load the trained model (singleton — cached after first load)."""
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    model_path = get_trained_model_path(PROJECT_ID)

    # Create model architecture
    model = create_model(MODEL_NAME, num_classes=len(CLASSES))

    # Load trained weights
    if model_path.exists():
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"  [OK] Brain Tumor model loaded from {model_path}")
    else:
        print(f"  ! No trained model found at {model_path}")
        print(f"  Please train the model first using model.ipynb")

    model.to(DEVICE)
    model.eval()
    _model_instance = model
    return model


def get_model_info() -> dict:
    """Return metadata about the model."""
    return {
        "project_id": PROJECT_ID,
        "model": MODEL_NAME,
        "classes": CLASSES,
        "img_size": IMG_SIZE,
        "device": str(DEVICE),
        "trained": get_trained_model_path(PROJECT_ID).exists(),
    }
