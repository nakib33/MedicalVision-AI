"""Vanilla Saliency — Visualise gradients of output w.r.t. input pixels.

Reference: Simonyan et al., "Deep Inside Convolutional Networks:
Visualising Image Classification Models and Saliency Maps", ICLR 2014.

Produces a 2D map showing which input pixels most influence the prediction.
The absolute gradient value is taken across colour channels.
"""

import torch
import numpy as np
from typing import Optional


class SaliencyMap:
    """Vanilla Saliency Map generator.

    Usage:
        sal = SaliencyMap(model)
        heatmap = sal(image_tensor, target_class=pred_idx)
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
                 abs_values: bool = True) -> np.ndarray:
        """Generate a saliency map.

        Args:
            image_tensor: Input (1, C, H, W), requires_grad=True.
            target_class: Target class index. Uses predicted if None.
            abs_values: If True, take absolute value of gradients (default).
                        If False, use only positive gradients.

        Returns:
            Saliency map as 2D numpy array (H, W) in [0, 1].
        """
        self.model.zero_grad()

        # Ensure input requires gradients
        if not image_tensor.requires_grad:
            image_tensor.requires_grad_(True)

        # Forward
        output = self.model(image_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        # Get gradients w.r.t. input
        gradients = image_tensor.grad.data  # (1, C, H, W)
        if abs_values:
            saliency = gradients.abs()
        else:
            saliency = torch.clamp(gradients, min=0)

        # Max over colour channels
        saliency, _ = saliency.max(dim=1)  # (1, H, W)
        saliency = saliency.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        s_min, s_max = saliency.min(), saliency.max()
        if s_max - s_min > 1e-8:
            saliency = (saliency - s_min) / (s_max - s_min)
        else:
            saliency = np.zeros_like(saliency)

        return saliency
