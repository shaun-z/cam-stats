"""Tests for GradCAM core functionality."""

import torch
import torch.nn as nn

from xstream.gradcam.core import GradCAMStepper
from xstream.gradcam.hooks import register_hooks


class SimpleConvNet(nn.Module):
    """Minimal CNN for testing."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def _make_model_and_input():
    model = SimpleConvNet()
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    return model, x


def test_hooks_capture_activations_and_gradients():
    model, x = _make_model_and_input()
    handle, hooks = register_hooks(model, "conv1")

    out = model(x)
    out[0, 0].backward()

    assert handle.activations is not None
    assert handle.gradients is not None
    assert handle.activations.shape[1] == 8  # 8 channels
    assert handle.gradients.shape[1] == 8

    for h in hooks:
        h.remove()


def test_hooks_cleanup():
    model, x = _make_model_and_input()
    handle, hooks = register_hooks(model, "conv1")

    for h in hooks:
        h.remove()
    handle.clear()

    assert handle.activations is None
    assert handle.gradients is None


def test_gradcam_stepper_heatmap_shape():
    model, x = _make_model_and_input()
    stepper = GradCAMStepper(model, x, "conv1")

    stepper.forward()
    stepper.backward()
    heatmap = stepper.heatmap()
    stepper.cleanup()

    assert heatmap.shape == (32, 32)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_gradcam_stepper_with_target_class():
    model, x = _make_model_and_input()
    stepper = GradCAMStepper(model, x, "conv1")

    stepper.forward()
    stepper.backward(target_class=5)
    heatmap = stepper.heatmap()
    stepper.cleanup()

    assert heatmap.shape == (32, 32)


def test_gradcam_stepper_cleanup_removes_hooks():
    model, x = _make_model_and_input()
    stepper = GradCAMStepper(model, x, "conv1")

    stepper.forward()
    stepper.backward()
    stepper.heatmap()
    stepper.cleanup()

    assert stepper._handle is None
    assert len(stepper._hooks) == 0
