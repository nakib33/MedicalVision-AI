"""Score-CAM: Score-weighted Class Activation Mapping.

Reference: Wang et al., "Score-CAM: Score-Weighted Visual Explanations for
Convolutional Neural Networks", CVPRW 2020.

Unlike Grad-CAM which uses gradients, Score-CAM uses the increase in
confidence score for each activation channel as its weight — removing
the reliance on sometimes-noisy gradients.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from tqdm import tqdm


class ScoreCAM:
    """Score-CAM heatmap generator.

    Usage:
        cam = ScoreCAM(model, target_layer)
        heatmap = cam(image_tensor, target_class=pred_idx)
    """

    def __init__(self, model: torch.nn.Module,
                 target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.model.eval()
        self.hook = None
        self.activations = None

        if target_layer is None:
            target_layer = self._find_last_conv()
        self.target_layer = target_layer
        self._register_hook()

    def _find_last_conv(self) -> torch.nn.Module:
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError("No Conv2d found. Specify target_layer.")
        return last_conv

    def _register_hook(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        self.hook = self.target_layer.register_forward_hook(forward_hook)

    def _remove_hook(self):
        if self.hook is not None:
            self.hook.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._remove_hook()

    def __call__(self, image_tensor: torch.Tensor,
                 target_class: Optional[int] = None,
                 verbose: bool = False) -> np.ndarray:
        """Generate Score-CAM heatmap.

        Args:
            image_tensor: Input (1, C, H, W).
            target_class: Target class. Uses predicted if None.
            verbose: Show progress bar.

        Returns:
            Heatmap as 2D numpy array (H, W) in [0, 1].
        """
        self.model.zero_grad()

        # Get baseline prediction
        output = self.model(image_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        baseline_score = F.softmax(output, dim=1)[0, target_class].item()

        # Get activations
        _ = self.model(image_tensor)  # Forward to fill activations
        activations = self.activations.squeeze()  # (C, H', W')
        num_channels = activations.shape[0]
        h, w = activations.shape[1:]

        # Upsample activations to input size
        input_h, input_w = image_tensor.shape[2:]
        activations_upsampled = F.interpolate(
            self.activations, size=(input_h, input_w),
            mode="bilinear", align_corners=False
        ).squeeze()  # (C, H, W)

        # For each channel, compute the score increase
        weights = torch.zeros(num_channels, device=image_tensor.device)
        channel_iter = range(num_channels)
        if verbose:
            channel_iter = tqdm(channel_iter, desc="Score-CAM channels")

        for i in channel_iter:
            # Normalise channel activation to [0, 1]
            channel_map = activations_upsampled[i]
            cm_min, cm_max = channel_map.min(), channel_map.max()
            if cm_max - cm_min > 1e-8:
                channel_map = (channel_map - cm_min) / (cm_max - cm_min)
            else:
                channel_map = torch.zeros_like(channel_map)

            # Mask input with this channel's activation
            masked_input = image_tensor * channel_map.unsqueeze(0)
            with torch.no_grad():
                score = F.softmax(self.model(masked_input), dim=1)
            weights[i] = score[0, target_class].item() - baseline_score

        # Weighted combination of activations
        weights = weights.view(-1, 1, 1)  # (C, 1, 1)
        cam = (weights * activations).sum(dim=0)  # (H', W')
        cam = F.relu(cam)

        # Resize to input size
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(input_h, input_w),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise
        cmin, cmax = cam.min(), cam.max()
        if cmax - cmin > 1e-8:
            cam = (cam - cmin) / (cmax - cmin)
        else:
            cam = np.zeros_like(cam)

        return cam
