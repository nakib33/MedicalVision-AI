"""Grad-CAM: Gradient-weighted Class Activation Mapping.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-Based Localization", ICCV 2017.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


class GradCAM:
    """Grad-CAM heatmap generator.

    Produces a coarse localisation map highlighting regions the model
    focuses on for a given prediction.

    Usage:
        cam = GradCAM(model, target_layer=model.features[-1])
        heatmap = cam(image_tensor, target_class=pred_idx)
        # Visualise with overlay_heatmap()
    """

    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        """
        Args:
            model: PyTorch model (in eval mode).
            target_layer: The convolutional layer to hook.
                          If None, attempts to automatically find the last Conv2d.
        """
        self.model = model
        self.model.eval()
        self.hooks = []

        self.activations = None
        self.gradients = None

        if target_layer is None:
            target_layer = self._find_last_conv()

        self.target_layer = target_layer
        self._register_hooks()

    def _find_last_conv(self) -> torch.nn.Module:
        """Walk the model and return the last Conv2d module."""
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError("No Conv2d layer found in model. "
                             "Specify target_layer explicitly.")
        return last_conv

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._remove_hooks()

    def __call__(self, image_tensor: torch.Tensor,
                 target_class: Optional[int] = None) -> np.ndarray:
        """Generate a Grad-CAM heatmap.

        Args:
            image_tensor: Input tensor (1, C, H, W) — normalised, on correct device.
            target_class: Class index to explain. If None, uses the predicted class.

        Returns:
            Heatmap as 2D numpy array (H, W) in range [0, 1].
        """
        self.model.zero_grad()

        # Forward
        output = self.model(image_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks did not fire. Check target_layer.")

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # Only positive influence

        # Resize to input size
        cam = F.interpolate(cam, size=image_tensor.shape[2:],
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cmin, cmax = cam.min(), cam.max()
        if cmax - cmin > 1e-8:
            cam = (cam - cmin) / (cmax - cmin)
        else:
            cam = np.zeros_like(cam)

        return cam
