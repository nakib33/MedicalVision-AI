"""PDF Report Generator — Creates professional medical AI analysis reports.

Generates a downloadable PDF containing:
- Patient/scan info section
- Uploaded image thumbnail
- Prediction results with confidence
- All XAI heatmaps as a gallery
- Model architecture summary
- Timestamp and metadata
"""

import io
import os
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import numpy as np

from shared.utils.visualization import image_to_base64


def generate_prediction_report(
    project_name: str,
    class_names: list,
    predicted_class: str,
    confidence: float,
    probabilities: dict,
    uploaded_image: Image.Image,
    explanations: dict,
) -> bytes:
    """Generate a PDF report as bytes.

    Args:
        project_name: Name of the medical imaging project.
        class_names: All possible class labels.
        predicted_class: The predicted label.
        confidence: Confidence score (0-1).
        probabilities: {class: prob} dict.
        uploaded_image: Original uploaded PIL image.
        explanations: {explainer_key: {"label": str, "overlay_base64": str, ...}}
                      from xai_factory.run_all_explainers().

    Returns:
        PDF as bytes (ready for download).
    """
    buf = io.BytesIO()

    with PdfPages(buf) as pdf:

        # ── Page 1: Cover & Prediction ─────────────────────────
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")

        # Title
        ax.text(0.5, 0.95, "MedicalVision AI Suite",
                ha="center", va="top", fontsize=22, fontweight="bold",
                transform=ax.transAxes, color="#1a5276")
        ax.text(0.5, 0.90, project_name,
                ha="center", va="top", fontsize=16,
                transform=ax.transAxes, color="#2c3e50")

        # Divider line
        ax.axhline(y=0.87, xmin=0.1, xmax=0.9, color="#3498db",
                   linewidth=2)

        # Timestamp
        ax.text(0.5, 0.84, f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ha="center", va="top", fontsize=10,
                transform=ax.transAxes, color="#7f8c8d")

        # Uploaded image
        img_resized = uploaded_image.resize((300, 300), Image.LANCZOS)
        ax_img = fig.add_axes([0.35, 0.55, 0.3, 0.3])
        ax_img.imshow(np.array(img_resized))
        ax_img.set_title("Uploaded Scan", fontsize=11, fontweight="bold")
        ax_img.axis("off")

        # Prediction box
        box_y = 0.45
        bbox_props = dict(boxstyle="round,pad=0.5",
                          facecolor="#eaf2f8", edgecolor="#2980b9")
        ax.text(0.5, box_y,
                f"Prediction: {predicted_class}\n"
                f"Confidence: {confidence:.1%}",
                ha="center", va="center", fontsize=14,
                transform=ax.transAxes, fontweight="bold",
                bbox=bbox_props)

        # Probability bar
        if probabilities:
            names = list(probabilities.keys())
            vals = list(probabilities.values())
            colors = plt.cm.Spectral(np.linspace(0.2, 0.8, len(names)))

            ax_bar = fig.add_axes([0.15, 0.25, 0.7, 0.15])
            bars = ax_bar.barh(names, vals, color=colors, edgecolor="white")
            ax_bar.set_xlim(0, 1)
            ax_bar.set_title("Class Probabilities", fontsize=11, fontweight="bold")
            ax_bar.set_xlabel("Probability")
            for bar, val in zip(bars, vals):
                ax_bar.text(bar.get_width() + 0.01,
                            bar.get_y() + bar.get_height() / 2,
                            f"{val:.1%}", va="center", fontsize=8)

        # Footer
        ax.text(0.5, 0.02,
                "MedicalVision AI Suite — AI-Powered Medical Imaging Analysis",
                ha="center", va="bottom", fontsize=8,
                transform=ax.transAxes, color="#95a5a6",
                style="italic")

        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 2: XAI Explanations ───────────────────────────
        # 2 rows × 2 cols of heatmaps, plus a row for captions
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
        fig.suptitle("Explainable AI — Heatmap Analysis",
                     fontsize=16, fontweight="bold", y=0.98)

        explains = list(explanations.items())

        # We'll use 2×2 grid (pages) to show all 7+ explainers
        sub_idx = 0
        for ax_row in axes:
            for ax_cell in ax_row:
                ax_cell.axis("off")
                if sub_idx < len(explains):
                    key, info = explains[sub_idx]
                    if "overlay_base64" in info:
                        # Decode and display overlay image
                        import base64
                        img_bytes = base64.b64decode(info["overlay_base64"])
                        img = Image.open(io.BytesIO(img_bytes))
                        ax_cell.imshow(np.array(img))
                        ax_cell.set_title(info.get("label", key),
                                          fontsize=10, fontweight="bold")
                    else:
                        ax_cell.text(0.5, 0.5, f"{info.get('label', key)}\n(Error)",
                                     ha="center", va="center", fontsize=10,
                                     color="red")
                    sub_idx += 1

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3 (if more than 4 explainers — like we have 7)
        if len(explains) > 4:
            fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
            fig.suptitle("Explainable AI — Additional Heatmaps",
                         fontsize=16, fontweight="bold", y=0.98)
            explains_remaining = explains[4:]
            sub_idx = 0
            for ax_row in axes:
                for ax_cell in ax_row:
                    ax_cell.axis("off")
                    if sub_idx < len(explains_remaining):
                        key, info = explains_remaining[sub_idx]
                        if "overlay_base64" in info:
                            import base64
                            img_bytes = base64.b64decode(info["overlay_base64"])
                            img = Image.open(io.BytesIO(img_bytes))
                            ax_cell.imshow(np.array(img))
                            ax_cell.set_title(info.get("label", key),
                                              fontsize=10, fontweight="bold")
                        else:
                            ax_cell.text(0.5, 0.5, f"{info.get('label', key)}\n(Error)",
                                         ha="center", va="center", fontsize=10,
                                         color="red")
                        sub_idx += 1
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # ── Final Page: Model & Disclaimer ─────────────────────
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")

        info_lines = [
            "Technical Information",
            "─" * 40,
            f"Project: {project_name}",
            f"Classification Classes: {', '.join(class_names)}",
            f"Predicted: {predicted_class}",
            f"Confidence: {confidence:.1%}",
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Explainable AI Methods Used:",
            "• Grad-CAM: Gradient-weighted Class Activation Mapping",
            "• Grad-CAM++: Improved gradient weighting",
            "• Score-CAM: Score-weighted channel activation",
            "• Saliency Map: Input gradient visualisation",
            "• Guided Backpropagation: Modified backpropagation",
            "• Integrated Gradients: Path-integral attribution",
            "• Occlusion Sensitivity: Sliding-window occlusion",
            "",
            "Disclaimer",
            "─" * 40,
            "This report is for research and educational purposes only.",
            "It does NOT constitute a medical diagnosis or professional",
            "medical advice. Always consult a qualified healthcare",
            "provider for medical decisions.",
        ]

        y_pos = 0.92
        for line in info_lines:
            ax.text(0.5, y_pos, line, ha="center", va="top",
                    fontsize=10, fontfamily="monospace" if line.startswith("─") else "sans-serif",
                    transform=ax.transAxes)
            y_pos -= 0.025

        ax.text(0.5, 0.02,
                "MedicalVision AI Suite © 2026",
                ha="center", va="bottom", fontsize=8,
                transform=ax.transAxes, color="#95a5a6")

        pdf.savefig(fig)
        plt.close(fig)

    buf.seek(0)
    return buf.getvalue()
