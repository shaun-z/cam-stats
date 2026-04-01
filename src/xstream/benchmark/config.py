"""Benchmark configuration and Figure 12 presets."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelLayerConfig:
    """A model and its target GradCAM layers."""

    model_name: str
    layer_names: list[str]


@dataclass
class BenchmarkConfig:
    """Full benchmark configuration."""

    models: list[ModelLayerConfig]
    num_images: int = 100
    warmup_runs: int = 5
    device: str = "cuda:0"
    platform: str = "auto"  # "auto" | "desktop" | "jetson" | "cpu" | "tpu"
    imagenet_root: str = "/data/imagenet/val"
    energy_sample_interval_s: float = 0.005
    output_dir: str = "results"


# Exact models and layers from Figure 12 of the X-Stream paper
FIGURE_12_MODELS = [
    ModelLayerConfig(
        model_name="resnet34",
        layer_names=["layer4.2", "layer3.5", "layer2.3", "layer1.2", "conv1"],
    ),
    ModelLayerConfig(
        model_name="resnet50",
        layer_names=["layer4.2", "layer3.5", "layer2.3", "layer1.2", "conv1"],
    ),
    ModelLayerConfig(
        model_name="vgg16",
        layer_names=[
            "features.28",  # conv5_3
            "features.21",  # conv4_3
            "features.14",  # conv3_3
            "features.7",   # conv2_2
            "features.0",   # conv1_1
        ],
    ),
]

# VGG layer name mapping for human-readable output
VGG_LAYER_ALIASES: dict[str, str] = {
    "features.28": "conv5_3",
    "features.21": "conv4_3",
    "features.14": "conv3_3",
    "features.7": "conv2_2",
    "features.0": "conv1_1",
}

FIGURE_12_CONFIG = BenchmarkConfig(models=FIGURE_12_MODELS)
