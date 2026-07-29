"""Guided Backpropagation — Visualisation through modified backpropagation.

Reference: Springenberg et al., "Striving for Simplicity: The All Convolutional
Net", ICLRW 2015.

Guided Backpropagation modifies the ReLU backward pass: it only allows gradients
to flow back where both the input AND the gradient are positive. This produces
sharper, more interpretable visualisations than vanilla saliency.
"""

import torch
import numpy as np
from typing import Optional


class GuidedBackprop:
    """Guided Backpropagation heatmap generator.

    Usage:
        gbp = GuidedBackprop(model)
        heatmap = gbp(image_tensor, target_class=pred_idx)
    """

    def __init__(self, model: torch.nn.Module):
        """
        Args:
            model: PyTorch model in eval mode.
        """
        self.model = model
        self.model.eval()
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register hooks on all ReLU modules to modify backward pass."""
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                # We need to replace the forward hook AND backward hook
                self.hooks.append(module.register_forward_hook(
                    self._make_forward_hook(module)
                ))
                self.hooks.append(module.register_full_backward_hook(
                    self._make_backward_hook()
                ))

    @staticmethod
    def _make_forward_hook(module):
        """Store the forward input for use in backward."""
        def hook(m, input, output):
            m.saved_input = input[0].detach()
        return hook

    @staticmethod
    def _make_backward_hook():
        """Modify backward: only propagate positive gradient * positive input."""
        def hook(m, grad_in, grad_out):
            if hasattr(m, "saved_input") and m.saved_input is not None:
                # Guided backprop: grad * (input > 0) * (grad_out[0] > 0)
                # grad_in[0] is the incoming gradient, grad_out[0] is gradient of output
                input_mask = (m.saved_input > 0).float()
                grad_mask = (grad_out[0] > 0).float()
                grad = grad_in[0] * input_mask * grad_mask
                return (grad,)
            return grad_in
        return hook

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._remove_hooks()

    def __call__(self, image_tensor: torch.Tensor,
                 target_class: Optional[int] = None) -> np.ndarray:
        """Generate a Guided Backpropagation visualisation.

        Args:
            image_tensor: Input (1, C, H, W), normalised.
            target_class: Target class. Uses predicted if None.

        Returns:
            2D heatmap (H, W) in [0, 1].
        """
        self.model.zero_grad()

        if not image_tensor.requires_grad:
            image_tensor.requires_grad_(True)

        output = self.model(image_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        if image_tensor.grad is None:
            raise RuntimeError("No gradients computed. Check hooks.")

        # Gradient magnitude across channels
        grad = image_tensor.grad.data  # (1, C, H, W)
        gbp, _ = grad.abs().max(dim=1)  # (1, H, W)
        gbp = gbp.squeeze().cpu().numpy()

        # Normalise
        g_min, g_max = gbp.min(), gbp.max()
        if g_max - g_min > 1e-8:
            gbp = (gbp - g_min) / (g_max - g_min)
        else:
            gbp = np.zeros_like(gbp)

        return gbp
