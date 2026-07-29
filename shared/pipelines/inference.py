"""Shared Pipeline — Inference utilities for deployed models."""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from shared.pipelines.transforms import get_inference_transform


@torch.no_grad()
def predict(model, image, class_names, device=None, img_size=224):
    """Run inference on a single image.

    Args:
        model: PyTorch model (in eval mode).
        image: PIL Image, numpy array, or file path.
        class_names: List of class label strings.
        device: torch.device (auto-detected if None).
        img_size: Input size for the model.

    Returns:
        dict with:
            - predicted_class: str
            - confidence: float
            - probabilities: dict {class_name: prob, ...}
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    # Load image
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")
    elif not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Transform
    transform = get_inference_transform(img_size)
    input_tensor = transform(image).unsqueeze(0).to(device)  # (1, C, H, W)

    # Forward
    output = model(input_tensor)
    probabilities = F.softmax(output, dim=1).squeeze(0)

    probs_np = probabilities.cpu().numpy()
    pred_idx = int(probs_np.argmax())
    confidence = float(probs_np[pred_idx])

    return {
        "predicted_class": class_names[pred_idx],
        "predicted_index": pred_idx,
        "confidence": confidence,
        "probabilities": {
            name: float(probs_np[i])
            for i, name in enumerate(class_names)
        },
    }


@torch.no_grad()
def predict_batch(model, images, class_names, device=None, img_size=224):
    """Run inference on a batch of images.

    Args:
        model: PyTorch model.
        images: List of PIL Images.
        class_names: List of class labels.
        device: torch.device.
        img_size: Input size.

    Returns:
        List of prediction dicts (same as predict() per image).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    transform = get_inference_transform(img_size)
    batch = torch.stack([transform(img) for img in images]).to(device)

    outputs = model(batch)
    probs_all = F.softmax(outputs, dim=1).cpu().numpy()

    results = []
    for probs in probs_all:
        pred_idx = int(probs.argmax())
        results.append({
            "predicted_class": class_names[pred_idx],
            "predicted_index": pred_idx,
            "confidence": float(probs[pred_idx]),
            "probabilities": {
                name: float(probs[i])
                for i, name in enumerate(class_names)
            },
        })
    return results
