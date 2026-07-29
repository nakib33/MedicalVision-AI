"""MedScan-AI — Model loader with multi-modality support."""

import torch
from pathlib import Path
from shared.config import get_trained_model_path, DEVICE
from shared.models import create_model
from .config import PROJECT_ID, MODEL_NAME, CLASSES, IMG_SIZE


_model_instance = None


def get_model() -> torch.nn.Module:
    """Load the MedScan-AI model (custom CNN)."""
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    model_path = get_trained_model_path(PROJECT_ID)

    # Create custom CNN (can later be compared with EfficientNet)
    model = create_model(
        MODEL_NAME,
        num_classes=len(CLASSES),
        in_channels=3,
        base_filters=32,
        num_blocks=4,
        dropout=0.5,
    )

    if model_path.exists():
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"  [OK] MedScan-AI model loaded from {model_path.name}")
    else:
        print(f"  ! No trained model at {model_path}")
        print("  Please train the model first using model.ipynb")

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
        "modalities": [
            {
                "id": m["id"],
                "name": m["name"],
                "icon": m["icon"],
            }
            for m in __import__("importlib").import_module(
                ".config", "projects.12_medscan_ai"
            ).MODALITIES
        ],
    }
