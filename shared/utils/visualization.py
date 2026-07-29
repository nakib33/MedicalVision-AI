"""Shared Visualization — Heatmap overlays, confusion matrices, and plot helpers."""

import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
import torch


# ── Colourmaps ─────────────────────────────────────────────────
HEATMAP_CMAP = "jet"
OVERLAY_ALPHA = 0.5


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a normalized image tensor (C, H, W) to a PIL Image."""
    img = tensor.detach().cpu()
    # Denormalise roughly
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray,
                    alpha: float = OVERLAY_ALPHA,
                    cmap: str = HEATMAP_CMAP) -> Image.Image:
    """Overlay a heatmap (H, W) onto a PIL image and return the composite.

    Args:
        image: Original PIL image (RGB).
        heatmap: 2D numpy array of saliency values (H, W).
        alpha: Transparency of heatmap overlay.
        cmap: Matplotlib colourmap name.

    Returns:
        PIL Image with heatmap overlay.
    """
    # Resize heatmap to match image size
    heatmap_resized = np.array(Image.fromarray(heatmap).resize(
        image.size, Image.BILINEAR
    ))

    # Normalise heatmap to [0, 1]
    h_min, h_max = heatmap_resized.min(), heatmap_resized.max()
    if h_max - h_min > 1e-8:
        heatmap_norm = (heatmap_resized - h_min) / (h_max - h_min)
    else:
        heatmap_norm = np.zeros_like(heatmap_resized)

    # Apply colourmap
    cmap_obj = plt.get_cmap(cmap)
    heatmap_colored = cmap_obj(heatmap_norm)[:, :, :3]  # Drop alpha
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Blend
    img_np = np.array(image)
    blended = (1 - alpha) * img_np + alpha * heatmap_colored
    blended = blended.astype(np.uint8)
    return Image.fromarray(blended)


def overlay_heatmap_on_tensor(image_tensor: torch.Tensor,
                               heatmap: np.ndarray,
                               alpha: float = OVERLAY_ALPHA) -> Image.Image:
    """Convenience: overlay heatmap on a tensor (C, H, W)."""
    pil_img = tensor_to_pil(image_tensor)
    return overlay_heatmap(pil_img, heatmap, alpha)


def fig_to_base64(fig) -> str:
    """Convert a matplotlib Figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                pad_inches=0.1)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return img_b64


def image_to_base64(image: Image.Image) -> str:
    """Convert a PIL Image to a base64-encoded PNG string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def plot_confusion_matrix(cm: np.ndarray, class_names: list) -> str:
    """Plot a confusion matrix and return it as a base64 PNG string.

    Args:
        cm: Confusion matrix array (K x K).
        class_names: List of class label strings.

    Returns:
        Base64-encoded PNG string.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=11)

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_probability_bars(probabilities: dict) -> str:
    """Plot class probabilities as horizontal bar chart.

    Args:
        probabilities: {class_name: prob, ...}

    Returns:
        Base64-encoded PNG string.
    """
    names = list(probabilities.keys())
    values = list(probabilities.values())
    colors = plt.cm.Spectral(np.linspace(0.2, 0.8, len(names)))

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names, values, color=colors, edgecolor="white")
    ax.set_xlim(0, 1)
    ax.set_title("Class Probabilities", fontsize=14, fontweight="bold")
    ax.set_xlabel("Probability")

    # Annotate bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=10)

    plt.tight_layout()
    return fig_to_base64(fig)
