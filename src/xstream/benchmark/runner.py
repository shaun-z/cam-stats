"""Main benchmark orchestrator and CLI entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..data.loader import get_imagenet_val_loader
from ..gradcam.core import GradCAMStepper
from ..models.registry import load_model
from .config import FIGURE_12_CONFIG, BenchmarkConfig, VGG_LAYER_ALIASES
from .energy import create_energy_monitor
from .results import LayerResult, print_summary_table, save_results_csv, save_results_json
from .timer import DeviceTimer, TimingBreakdown


def _display_layer_name(model_name: str, layer_name: str) -> str:
    """Return a human-readable layer name for display."""
    if model_name == "vgg16" and layer_name in VGG_LAYER_ALIASES:
        return VGG_LAYER_ALIASES[layer_name]
    return layer_name


def benchmark_gradcam_layer(
    model: torch.nn.Module,
    model_name: str,
    dataloader: torch.utils.data.DataLoader,
    layer_name: str,
    config: BenchmarkConfig,
) -> LayerResult:
    """Benchmark GradCAM on a single model-layer pair across all images."""
    device = config.device
    timer = DeviceTimer(device)
    display_name = _display_layer_name(model_name, layer_name)

    raw_times: list[TimingBreakdown] = []
    raw_energies: list[float] = []

    # Warmup runs
    warmup_loader = iter(dataloader)
    for i in range(min(config.warmup_runs, len(dataloader))):
        try:
            images, _ = next(warmup_loader)
        except StopIteration:
            break
        images = images.to(device)
        stepper = GradCAMStepper(model, images, layer_name)
        stepper.forward()
        stepper.backward()
        stepper.heatmap()
        stepper.cleanup()

    # Benchmark runs
    for batch_idx, (images, _) in enumerate(dataloader):
        images = images.to(device)
        energy_monitor = create_energy_monitor(
            device,
            platform=config.platform,
            sample_interval_s=config.energy_sample_interval_s,
        )

        stepper = GradCAMStepper(model, images, layer_name)

        energy_monitor.start()

        t0 = timer.sync_and_time()
        stepper.forward()
        t1 = timer.sync_and_time()
        stepper.backward()
        t2 = timer.sync_and_time()
        heatmap = stepper.heatmap()
        t3 = timer.sync_and_time()

        energy_j = energy_monitor.stop()

        stepper.cleanup()

        raw_times.append(TimingBreakdown(
            forward_s=t1 - t0,
            backward_s=t2 - t1,
            heatmap_s=t3 - t2,
        ))
        raw_energies.append(energy_j)

    return LayerResult(
        model_name=model_name,
        layer_name=display_name,
        num_images=len(raw_times),
        raw_times=raw_times,
        raw_energies=raw_energies,
    )


def run_benchmark(config: BenchmarkConfig) -> list[LayerResult]:
    """Run GradCAM benchmarks for all model-layer pairs in the config."""
    results: list[LayerResult] = []

    dataloader = get_imagenet_val_loader(
        root=config.imagenet_root,
        num_images=config.num_images,
    )

    for model_cfg in config.models:
        print(f"\n{'='*60}")
        print(f"Model: {model_cfg.model_name}")
        print(f"{'='*60}")

        model = load_model(model_cfg.model_name, device=config.device)

        for layer_name in model_cfg.layer_names:
            display_name = _display_layer_name(model_cfg.model_name, layer_name)
            print(f"  Benchmarking layer: {display_name} ...", end=" ", flush=True)

            result = benchmark_gradcam_layer(
                model=model,
                model_name=model_cfg.model_name,
                dataloader=dataloader,
                layer_name=layer_name,
                config=config,
            )
            results.append(result)
            print(f"done (avg {result.mean_total_s * 1000:.3f} ms, {result.mean_energy_j:.4f} J)")

        # Free device memory between models
        del model
        if config.device.startswith("cuda"):
            torch.cuda.empty_cache()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="GradCAM Benchmark (X-Stream Figure 12)")
    parser.add_argument("--num-images", type=int, default=100, help="Number of images to benchmark")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup runs")
    parser.add_argument("--device", default="cuda:0", help="Device to benchmark on")
    parser.add_argument(
        "--platform", default="auto",
        choices=["auto", "desktop", "jetson", "cpu", "tpu"],
        help="Platform for energy monitoring (auto-detected by default)",
    )
    parser.add_argument("--imagenet-root", default="/data/imagenet/val", help="ImageNet val path")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--energy-interval", type=float, default=0.05, help="Energy sampling interval (seconds)")
    args = parser.parse_args()

    config = BenchmarkConfig(
        models=FIGURE_12_CONFIG.models,
        num_images=args.num_images,
        warmup_runs=args.warmup,
        device=args.device,
        platform=args.platform,
        imagenet_root=args.imagenet_root,
        output_dir=args.output_dir,
        energy_sample_interval_s=args.energy_interval,
    )

    print(f"GradCAM Benchmark — X-Stream Figure 12")
    print(f"Device: {config.device}  Platform: {config.platform}")
    print(f"Images: {config.num_images}")
    print(f"Models: {[m.model_name for m in config.models]}")

    results = run_benchmark(config)

    # Save results
    output_dir = Path(config.output_dir)
    save_results_json(results, output_dir / "gradcam_benchmark.json")
    save_results_csv(results, output_dir / "gradcam_benchmark.csv")

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print_summary_table(results)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
