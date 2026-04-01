"""Forward/backward hook management for capturing activations and gradients."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class HookHandle:
    """Stores captured activations and gradients for a target layer."""

    activations: torch.Tensor | None = field(default=None, repr=False)
    gradients: torch.Tensor | None = field(default=None, repr=False)

    def clear(self) -> None:
        self.activations = None
        self.gradients = None


def _resolve_layer(model: nn.Module, layer_name: str) -> nn.Module:
    """Resolve a dotted layer name (e.g. 'layer4.2' or 'features.28') to a module."""
    return operator.attrgetter(layer_name)(model)


def register_hooks(
    model: nn.Module, target_layer_name: str
) -> tuple[HookHandle, list[torch.utils.hooks.RemovableHook]]:
    """
    Register forward and backward hooks on the named layer.

    Returns:
        (handle, removable_hooks) — caller must call .remove() on each hook after use.
    """
    layer = _resolve_layer(model, target_layer_name)
    handle = HookHandle()

    def forward_hook(module: nn.Module, input: tuple, output: torch.Tensor) -> torch.Tensor:
        # Clone to avoid issues with in-place operations (e.g. VGG's inplace ReLU)
        out = output.clone()
        handle.activations = out.detach()
        return out

    def backward_hook(
        module: nn.Module, grad_input: tuple, grad_output: tuple
    ) -> None:
        handle.gradients = grad_output[0].detach()

    fh = layer.register_forward_hook(forward_hook)
    bh = layer.register_full_backward_hook(backward_hook)

    return handle, [fh, bh]
