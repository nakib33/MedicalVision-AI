"""Integrated Gradients — Axiomatic attribution method.

Reference: Sundararajan et al., "Axiomatic Attribution for Deep Networks",
ICML 2017.

Integrated Gradients computes the integral of gradients along a straight-line
path from a baseline (e.g., black image) to the input. It satisfies two
axioms: sensitivity and implementation invariance.
"""

import torch
import numpy as np
from typing import Optional


class IntegratedGradients:
    """Integrated Gradients attribution generator.

    Usage:
        ig = IntegratedGradients(model)
        heatmap = ig(image_tensor, target_class=pred_idx, steps=50)
    """

    def __init__(self, model: torch.nn.Module):
        """
        Args:
            model: PyTorch model in eval mode.
        """
        self.model = model
        self.model.eval()

    def __call__(self, image_tensor: torch.Tensor,
                 target_class: Optional[int] = None,
                 steps: int = 50,
                 baseline: Optional[torch.Tensor] = None) -> np.ndarray:
        """Generate Integrated Gradients attribution.

        Args:
            image_tensor: Input (1, C, H, W), normalised.
            target_class: Target class. Uses predicted if None.
            steps: Number of interpolation steps (recommended: 50-200).
            baseline: Baseline input (e.g., black image). If None, uses zeros.

        Returns:
            2D attribution map (H, W) in [0, 1].
        """
        device = image_tensor.device

        # Default baseline: black image (after normalisation)
        if baseline is None:
            baseline = torch.zeros_like(image_tensor, device=device)

        if target_class is None:
            with torch.no_grad():
                output = self.model(image_tensor)
            target_class = output.argmax(dim=1).item()

        # Interpolate between baseline and input
        # Scale from 0 to 1 over `steps` points
        # Using trapezoidal rule: steps interpolations, scaled_sum = sum(grads) * (1/steps)
        scaled_gradients = torch.zeros_like(image_tensor, device=device)

        for alpha in np.linspace(0, 1, steps):
            # Interpolated input
            interpolated = baseline + alpha * (image_tensor - baseline)
            interpolated = interpolated.detach().requires_grad_(True)

            # Forward
            output = self.model(interpolated)

            # Backward
            self.model.zero_grad()
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1.0
            output.backward(gradient=one_hot, retain_graph=True)

            if interpolated.grad is None:
                continue

            scaled_gradients += interpolated.grad.data / steps

        # (Input - Baseline) * Integrated_gradients
        attribution = (image_tensor - baseline) * scaled_gradients

        # Max across channels → 2D map
        attr_map, _ = attribution.abs().max(dim=1)  # (1, H, W)
        attr_map = attr_map.squeeze().cpu().numpy()

        # Normalise
        a_min, a_max = attr_map.min(), attr_map.max()
        if a_max - a_min > 1e-8:
            attr_map = (attr_map - a_min) / (a_max - a_min)
        else:
            attr_map = np.zeros_like(attr_map)

        return attr_map
