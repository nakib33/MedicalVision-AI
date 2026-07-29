"""XAI Factory — Run all explainers on a single image and return results as base64.

This factory orchestrates every supported explainer, making it easy to
display a full gallery of explanations on the frontend.
"""

import torch
import numpy as np
from typing import Optional

from shared.explainers.gradcam import GradCAM
from shared.explainers.gradcam_pp import GradCAMPlusPlus
from shared.explainers.scorecam import ScoreCAM
from shared.explainers.saliency import SaliencyMap
from shared.explainers.guided_backprop import GuidedBackprop
from shared.explainers.integrated_gradients import IntegratedGradients
from shared.explainers.occlusion import OcclusionSensitivity
from shared.utils.visualization import (
    overlay_heatmap, tensor_to_pil, image_to_base64
)


# ── Explainer labels (for display) ────────────────────────────
EXPLAINER_INFO = {
    "gradcam": {
        "label": "Grad-CAM",
        "description": "Gradient-weighted Class Activation Mapping — highlights regions from the final conv layer."
    },
    "gradcam_pp": {
        "label": "Grad-CAM++",
        "description": "Improved Grad-CAM with better localisation for multiple instances."
    },
    "scorecam": {
        "label": "Score-CAM",
        "description": "Score-weighted activation maps — no gradients, uses forward-pass confidence."
    },
    "saliency": {
        "label": "Saliency Map",
        "description": "Vanilla saliency — gradient of output w.r.t. input pixels."
    },
    "guided_backprop": {
        "label": "Guided Backprop",
        "description": "Modified backpropagation — sharper visualisation of features."
    },
    "integrated_gradients": {
        "label": "Integrated Gradients",
        "description": "Path-integral attribution from baseline to input (axiomatic)."
    },
    "occlusion": {
        "label": "Occlusion Sensitivity",
        "description": "Sliding-window occlusion map — measures prediction drop per patch."
    },
}


def find_last_conv(model: torch.nn.Module) -> torch.nn.Module:
    """Find the last Conv2d layer in the model."""
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No Conv2d found in model.")
    return last_conv


def run_all_explainers(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    class_names: list,
    target_class: Optional[int] = None,
    device: Optional[torch.device] = None,
    occlude_size: int = 32,
    occlude_stride: int = 16,
) -> dict:
    """Run all XAI explainers on a single image.

    Args:
        model: PyTorch model in eval mode.
        image_tensor: Input (1, C, H, W), normalised, on correct device.
        class_names: List of class labels.
        target_class: Class index to explain. Uses predicted if None.
        device: Torch device.
        occlude_size: Patch size for occlusion.
        occlude_stride: Stride for occlusion.

    Returns:
        dict with structure:
            {
                "predictions": {
                    "predicted_class": str,
                    "confidence": float,
                    "probabilities": {class: prob, ...}
                },
                "explanations": {
                    "gradcam": {
                        "label": str,
                        "description": str,
                        "heatmap_base64": str,     # heatmap-only image
                        "overlay_base64": str,     # overlay on original
                    },
                    ...
                }
            }
    """
    if device is None:
        device = image_tensor.device

    model = model.to(device)
    model.eval()

    # Get prediction
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1).squeeze(0)

    if target_class is None:
        target_class = output.argmax(dim=1).item()

    probs_np = probs.cpu().numpy()
    result = {
        "predictions": {
            "predicted_class": class_names[target_class],
            "predicted_index": target_class,
            "confidence": float(probs_np[target_class]),
            "probabilities": {
                name: float(probs_np[i])
                for i, name in enumerate(class_names)
            },
        },
        "explanations": {},
    }

    # Find target layer for CAM-based methods
    try:
        target_layer = find_last_conv(model)
    except ValueError:
        target_layer = None

    # Prepare original PIL image for overlay
    original_pil = tensor_to_pil(image_tensor.squeeze(0))

    # ── 1. Grad-CAM ────────────────────────────────────────────
    if target_layer is not None:
        try:
            cam = GradCAM(model, target_layer)
            heatmap = cam(image_tensor, target_class)
            cam_overlay = overlay_heatmap(original_pil, heatmap)
            heatmap_pil = tensor_to_pil(
                torch.tensor(heatmap).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1)
            )
            result["explanations"]["gradcam"] = {
                "label": EXPLAINER_INFO["gradcam"]["label"],
                "description": EXPLAINER_INFO["gradcam"]["description"],
                "heatmap_base64": image_to_base64(heatmap_pil),
                "overlay_base64": image_to_base64(cam_overlay),
            }
        except Exception as e:
            result["explanations"]["gradcam"] = {"error": str(e)}

    # ── 2. Grad-CAM++ ──────────────────────────────────────────
    if target_layer is not None:
        try:
            cam_pp = GradCAMPlusPlus(model, target_layer)
            heatmap = cam_pp(image_tensor, target_class)
            overlay = overlay_heatmap(original_pil, heatmap)
            heatmap_pil = tensor_to_pil(
                torch.tensor(heatmap).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1)
            )
            result["explanations"]["gradcam_pp"] = {
                "label": EXPLAINER_INFO["gradcam_pp"]["label"],
                "description": EXPLAINER_INFO["gradcam_pp"]["description"],
                "heatmap_base64": image_to_base64(heatmap_pil),
                "overlay_base64": image_to_base64(overlay),
            }
        except Exception as e:
            result["explanations"]["gradcam_pp"] = {"error": str(e)}

    # ── 3. Score-CAM ───────────────────────────────────────────
    if target_layer is not None:
        try:
            sc = ScoreCAM(model, target_layer)
            heatmap = sc(image_tensor, target_class, verbose=False)
            overlay = overlay_heatmap(original_pil, heatmap)
            heatmap_pil = tensor_to_pil(
                torch.tensor(heatmap).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1)
            )
            result["explanations"]["scorecam"] = {
                "label": EXPLAINER_INFO["scorecam"]["label"],
                "description": EXPLAINER_INFO["scorecam"]["description"],
                "heatmap_base64": image_to_base64(heatmap_pil),
                "overlay_base64": image_to_base64(overlay),
            }
        except Exception as e:
            result["explanations"]["scorecam"] = {"error": str(e)}

    # ── 4. Vanilla Saliency ────────────────────────────────────
    try:
        sal = SaliencyMap(model)
        heatmap = sal(image_tensor, target_class)
        overlay = overlay_heatmap(original_pil, heatmap)
        heatmap_pil = tensor_to_pil(
            torch.tensor(heatmap).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1)
        )
        result["explanations"]["saliency"] = {
            "label": EXPLAINER_INFO["saliency"]["label"],
            "description": EXPLAINER_INFO["saliency"]["description"],
            "heatmap_base64": image_to_base64(heatmap_pil),
            "overlay_base64": image_to_base64(overlay),
        }
    except Exception as e:
        result["explanations"]["saliency"] = {"error": str(e)}

    # ── 5. Guided Backpropagation ──────────────────────────────
    try:
        gbp = GuidedBackprop(model)
        heatmap = gbp(image_tensor, target_class)
        overlay = overlay_heatmap(original_pil, heatmap)
        heatmap_pil = tensor_to_pil(
            torch.tensor(heatmap).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1)
        )
        result["explanations"]["guided_backprop"] = {
            "label": EXPLAINER_INFO["guided_backprop"]["label"],
            "description": EXPLAINER_INFO["guided_backprop"]["description"],
            "heatmap_base64": image_to_base64(heatmap_pil),
            "overlay_base64": image_to_base64(overlay),
        }
    except Exception as e:
        result["explanations"]["guided_backprop"] = {"error": str(e)}

    # ── 6. Integrated Gradients ────────────────────────────────
    try:
        ig = IntegratedGradients(model)
        heatmap = ig(image_tensor, target_class, steps=50)
        overlay = overlay_heatmap(original_pil, heatmap)
        heatmap_pil = tensor_to_pil(
            torch.tensor(heatmap).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1)
        )
        result["explanations"]["integrated_gradients"] = {
            "label": EXPLAINER_INFO["integrated_gradients"]["label"],
            "description": EXPLAINER_INFO["integrated_gradients"]["description"],
            "heatmap_base64": image_to_base64(heatmap_pil),
            "overlay_base64": image_to_base64(overlay),
        }
    except Exception as e:
        result["explanations"]["integrated_gradients"] = {"error": str(e)}

    # ── 7. Occlusion Sensitivity ───────────────────────────────
    try:
        occ = OcclusionSensitivity(model)
        heatmap = occ(image_tensor, target_class,
                      patch_size=occlude_size, stride=occlude_stride,
                      verbose=False)
        overlay = overlay_heatmap(original_pil, heatmap)
        heatmap_pil = tensor_to_pil(
            torch.tensor(heatmap).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1)
        )
        result["explanations"]["occlusion"] = {
            "label": EXPLAINER_INFO["occlusion"]["label"],
            "description": EXPLAINER_INFO["occlusion"]["description"],
            "heatmap_base64": image_to_base64(heatmap_pil),
            "overlay_base64": image_to_base64(overlay),
        }
    except Exception as e:
        result["explanations"]["occlusion"] = {"error": str(e)}

    return result
