"""Grad-CAM++: Improved visual explanations.

Reference: Chattopadhay et al., "Grad-CAM++: Generalized Gradient-Based Visual
Explanations for Deep Convolutional Networks", WACV 2018.

The key difference from Grad-CAM is the weighting scheme: Grad-CAM++ uses a
weighted average of positive gradients that accounts for the importance of
each pixel location, giving better localisation — especially when an image
contains multiple instances of the same class.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


class GradCAMPlusPlus:
    """Grad-CAM++ heatmap generator.

    Usage:
        cam = GradCAMPlusPlus(model, target_layer)
        heatmap = cam(image_tensor, target_class=pred_idx)
    """

    def __init__(self, model: torch.nn.Module,
                 target_layer: Optional[torch.nn.Module] = None):
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
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError("No Conv2d layer found. Specify target_layer.")
        return last_conv

    def _register_hooks(self):
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
        """Generate a Grad-CAM++ heatmap.

        Args:
            image_tensor: Input (1, C, H, W), normalised.
            target_class: Target class index. Uses predicted if None.

        Returns:
            Heatmap as 2D numpy array (H, W) in [0, 1].
        """
        self.model.zero_grad()

        output = self.model(image_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks did not fire.")

        activations = self.activations  # (1, C, H, W)
        gradients = self.gradients       # (1, C, H, W)

        # Grad-CAM++ weighting
        # α_c_k = (∂y^c / ∂A_k) / (2 * ∂y^c / ∂A_k + Σ_a Σ_b A_k_ab * ∂y^c / ∂A_k)
        # where A_k are activations and ∂y^c/∂A_k are gradients

        # First-order gradients
        first = gradients

        # Second-order gradients (approximate via element-wise square)
        second = gradients.pow(2)

        # Third-order gradients
        third = gradients.pow(3)

        # Compute alpha weights
        # alpha = second / (2 * second + (activations * third).sum([2,3], keepdim=True) + eps)
        alpha_numer = second
        alpha_denom = 2 * second + (activations * third).sum(dim=(2, 3), keepdim=True) + 1e-8
        alpha = alpha_numer / alpha_denom

        # Weighted activation sum
        # w_c_k = Σ_i Σ_j alpha_ij * ReLU(grad_ij)
        positive_grads = F.relu(gradients)
        weights = (alpha * positive_grads).sum(dim=(2, 3), keepdim=True)

        # Weighted combination of activations
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Resize
        cam = F.interpolate(cam, size=image_tensor.shape[2:],
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise
        cmin, cmax = cam.min(), cam.max()
        if cmax - cmin > 1e-8:
            cam = (cam - cmin) / (cmax - cmin)
        else:
            cam = np.zeros_like(cam)

        return cam
