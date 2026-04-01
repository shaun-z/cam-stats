# CAM-Stats

GradCAM benchmark framework for measuring per-layer inference time and energy consumption across multiple hardware platforms. Built for reproducing X-Stream Figure 12.

## Supported Models & Layers

| Model | Target Layers |
|-------|---------------|
| ResNet-34 | layer4.2, layer3.5, layer2.3, layer1.2, conv1 |
| ResNet-50 | layer4.2, layer3.5, layer2.3, layer1.2, conv1 |
| VGG-16 | conv5_3, conv4_3, conv3_3, conv2_2, conv1_1 |

## Supported Platforms

| Platform | Device | Energy Backend | Extra Dependency |
|----------|--------|----------------|------------------|
| Desktop GPU | `cuda:0` | pynvml (NVML API) | `pynvml` |
| Jetson Orin NX | `cuda:0` | sysfs INA3221 | none |
| CPU | `cpu` | RAPL (sysfs / MSR) | none |
| TPU | `xla:0` | N/A | `torch-xla` |

## Install

```bash
# Desktop CUDA
uv sync --extra cuda

# Jetson Orin NX / CPU (no extra dependencies)
uv sync

# TPU
uv sync --extra tpu
```

## Usage

```bash
# Desktop GPU (auto-detected)
uv run xstream-bench --num-images 100 --device cuda:0

# Jetson Orin NX
uv run xstream-bench --num-images 100 --device cuda:0 --platform jetson

# CPU
uv run xstream-bench --num-images 100 --device cpu

# TPU
uv run xstream-bench --num-images 100 --device xla:0 --platform tpu
```

### Options

```
--num-images N       Number of images to benchmark (default: 100)
--warmup N           Number of warmup runs (default: 5)
--device DEVICE      PyTorch device (default: cuda:0)
--platform PLATFORM  auto | desktop | jetson | cpu | tpu (default: auto)
--imagenet-root PATH Path to ImageNet val set (default: /data/imagenet/val)
--output-dir DIR     Output directory (default: results/)
--energy-interval S  Energy sampling interval in seconds (default: 0.05)
```

### Output

Results are saved as `gradcam_benchmark.json` and `gradcam_benchmark.csv` in the output directory. Each entry contains per-image breakdown of forward, backward, and heatmap time (ms) plus energy (J).

### CPU Energy on AMD EPYC (no RAPL kernel module)

If `/sys/class/powercap/` has no RAPL entries, the tool falls back to direct MSR reads. This requires `CAP_SYS_RAWIO`:

```bash
sudo setcap cap_sys_rawio+ep $(readlink -f $(which python3))
```

## Test

```bash
uv run python -m pytest tests/ -v
```

## Project Structure

```
src/xstream/
  gradcam/
    hooks.py      # Forward/backward hook management
    core.py       # GradCAMStepper (phased forward/backward/heatmap)
  benchmark/
    config.py     # BenchmarkConfig + Figure 12 presets
    timer.py      # DeviceTimer (CUDA / XLA / CPU sync)
    energy.py     # EnergyMonitor (Nvidia / Jetson / RAPL / NoOp)
    results.py    # JSON/CSV export + summary table
    runner.py     # Orchestrator + CLI entry point
  data/
    loader.py     # ImageNet loader with synthetic fallback
  models/
    registry.py   # Model registry with in-place ReLU fix
```
