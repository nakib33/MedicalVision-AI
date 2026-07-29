"""Occlusion Sensitivity — Systematic occlusion-based explanations.

Reference: Zeiler & Fergus, "Visualizing and Understanding Convolutional
Networks", ECCV 2014.

Systematically occludes patches of the input image and measures how the
prediction changes. Regions that cause a large drop in confidence when
occluded are considered important for the prediction.
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class OcclusionSensitivity:
    """Occlusion sensitivity heatmap generator.

    Usage:
        occ = OcclusionSensitivity(model)
        heatmap = occ(image_tensor, target_class=pred_idx, patch_size=16, stride=8)
    """

    def __init__(self, model: torch.nn.Module):
        """
        Args:
            model: PyTorch model in eval mode.
        """
        self.model = model
        self.model.eval()

    def __call__(self, image_tensor: torch.Tensor,
                 target_class: int = None,
                 patch_size: int = 32,
                 stride: int = 16,
                 baseline_value: float = 0.0,
                 verbose: bool = False) -> np.ndarray:
        """Generate an occlusion sensitivity map.

        Args:
            image_tensor: Input (1, C, H, W), normalised.
            target_class: Target class. Uses predicted if None.
            patch_size: Size of the occlusion patch (square).
            stride: Sliding window stride (smaller = finer but slower).
            baseline_value: Value to fill occluded patch with
                           (0=black, 0.5=gray after normalisation).
            verbose: Show progress bar.

        Returns:
            2D sensitivity map (H, W) in [0, 1].
            Higher values = region is more important for prediction.
        """
        _, _, h, w = image_tensor.shape
        device = image_tensor.device

        # Get baseline (unoccluded) prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            probs = F.softmax(output, dim=1)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        baseline_prob = probs[0, target_class].item()

        # Create occlusion grid
        heatmap = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)

        y_positions = list(range(0, h - patch_size + 1, stride))
        x_positions = list(range(0, w - patch_size + 1, stride))
        total_patches = len(y_positions) * len(x_positions)

        patch_iter = range(total_patches)
        if verbose:
            patch_iter = tqdm(patch_iter, desc="Occlusion")

        for idx in patch_iter:
            yi = y_positions[idx // len(x_positions)]
            xi = x_positions[idx % len(x_positions)]

            # Create occluded image
            occluded = image_tensor.clone()
            occluded[:, :, yi:yi + patch_size, xi:xi + patch_size] = baseline_value

            # Measure drop in confidence
            with torch.no_grad():
                occ_output = self.model(occluded)
                occ_prob = F.softmax(occ_output, dim=1)[0, target_class].item()

            # Importance = drop in probability
            importance = max(0, baseline_prob - occ_prob)

            heatmap[yi:yi + patch_size, xi:xi + patch_size] += importance
            count_map[yi:yi + patch_size, xi:xi + patch_size] += 1

        # Average overlapping patches
        count_map[count_map == 0] = 1
        heatmap = heatmap / count_map

        # Normalise to [0, 1]
        h_min, h_max = heatmap.min(), heatmap.max()
        if h_max - h_min > 1e-8:
            heatmap = (heatmap - h_min) / (h_max - h_min)
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap
