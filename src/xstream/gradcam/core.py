"""GradCAM implementation with phased execution for benchmark timing."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .hooks import HookHandle, register_hooks


class GradCAMStepper:
    """
    Splits GradCAM into discrete phases so the caller can insert
    timing/energy measurements between forward, backward, and heatmap steps.

    Usage:
        stepper = GradCAMStepper(model, input_tensor, "layer4.2")
        logits = stepper.forward()
        stepper.backward(target_class=None)  # None = argmax
        heatmap = stepper.heatmap()
        stepper.cleanup()
    """

    def __init__(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target_layer_name: str,
    ) -> None:
        self.model = model
        self.input_tensor = input_tensor
        self.target_layer_name = target_layer_name

        self._handle: HookHandle | None = None
        self._hooks: list = []
        self._logits: torch.Tensor | None = None

    def forward(self) -> torch.Tensor:
        """Run forward pass, register hooks, return logits."""
        self._handle, self._hooks = register_hooks(
            self.model, self.target_layer_name
        )
        self._logits = self.model(self.input_tensor)
        return self._logits

    def backward(self, target_class: int | None = None) -> None:
        """Backward from class score. Gradients captured via hook."""
        assert self._logits is not None, "Call forward() first"
        assert self._handle is not None

        if target_class is None:
            target_class = self._logits.argmax(dim=1).item()

        self.model.zero_grad()
        class_score = self._logits[0, target_class]
        class_score.backward(retain_graph=False)

    def heatmap(self) -> torch.Tensor:
        """Compute GradCAM heatmap: ReLU(weighted_sum(activations, gradients))."""
        assert self._handle is not None
        assert self._handle.activations is not None
        assert self._handle.gradients is not None

        # Channel-wise weights via global average pooling of gradients
        weights = self._handle.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of feature maps
        cam = (weights * self._handle.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)

        # Resize to input spatial dimensions
        cam = F.interpolate(
            cam,
            size=self.input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        return cam.squeeze()  # (H, W)

    def cleanup(self) -> None:
        """Remove hooks and free captured tensors."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        if self._handle is not None:
            self._handle.clear()
            self._handle = None
        self._logits = None
