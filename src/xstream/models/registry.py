"""Model registry for benchmark targets."""

from __future__ import annotations

from torch import nn
from torchvision import models


MODEL_REGISTRY: dict[str, tuple] = {
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "vgg16": (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1),
}


def _disable_inplace_relu(model: nn.Module) -> None:
    """Disable in-place ReLU to allow backward hooks on preceding layers."""
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False


def load_model(name: str, device: str = "cuda:0") -> nn.Module:
    """Load a pretrained model by name, move to device, set eval mode."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")

    factory, weights = MODEL_REGISTRY[name]
    model = factory(weights=weights)
    _disable_inplace_relu(model)
    model = model.to(device)
    model.eval()
    return model
