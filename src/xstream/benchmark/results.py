"""Result storage and export."""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .timer import TimingBreakdown


@dataclass
class LayerResult:
    """Aggregated benchmark results for one model-layer pair."""

    model_name: str
    layer_name: str
    num_images: int
    raw_times: list[TimingBreakdown] = field(repr=False)
    raw_energies: list[float] = field(repr=False)

    @property
    def mean_forward_s(self) -> float:
        return statistics.mean(t.forward_s for t in self.raw_times)

    @property
    def mean_backward_s(self) -> float:
        return statistics.mean(t.backward_s for t in self.raw_times)

    @property
    def mean_heatmap_s(self) -> float:
        return statistics.mean(t.heatmap_s for t in self.raw_times)

    @property
    def mean_total_s(self) -> float:
        return statistics.mean(t.total_s for t in self.raw_times)

    @property
    def std_total_s(self) -> float:
        if len(self.raw_times) < 2:
            return 0.0
        return statistics.stdev(t.total_s for t in self.raw_times)

    @property
    def mean_energy_j(self) -> float:
        return statistics.mean(self.raw_energies)

    @property
    def std_energy_j(self) -> float:
        if len(self.raw_energies) < 2:
            return 0.0
        return statistics.stdev(self.raw_energies)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "layer_name": self.layer_name,
            "num_images": self.num_images,
            "mean_forward_s": self.mean_forward_s,
            "mean_backward_s": self.mean_backward_s,
            "mean_heatmap_s": self.mean_heatmap_s,
            "mean_total_s": self.mean_total_s,
            "std_total_s": self.std_total_s,
            "mean_energy_j": self.mean_energy_j,
            "std_energy_j": self.std_energy_j,
            "raw_times": [
                {"forward_s": t.forward_s, "backward_s": t.backward_s, "heatmap_s": t.heatmap_s}
                for t in self.raw_times
            ],
            "raw_energies": self.raw_energies,
        }


def save_results_json(results: list[LayerResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [r.to_dict() for r in results]
    path.write_text(json.dumps(data, indent=2))


def save_results_csv(results: list[LayerResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_name", "layer_name", "num_images",
        "mean_forward_s", "mean_backward_s", "mean_heatmap_s",
        "mean_total_s", "std_total_s", "mean_energy_j", "std_energy_j",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.to_dict()[k] for k in fieldnames})


def print_summary_table(results: list[LayerResult]) -> None:
    header = f"{'Model':<12} {'Layer':<14} {'Forward(ms)':>12} {'Backward(ms)':>13} {'Heatmap(ms)':>12} {'Total(ms)':>10} {'Energy(J)':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.model_name:<12} {r.layer_name:<14} "
            f"{r.mean_forward_s * 1000:>12.3f} {r.mean_backward_s * 1000:>13.3f} "
            f"{r.mean_heatmap_s * 1000:>12.3f} {r.mean_total_s * 1000:>10.3f} "
            f"{r.mean_energy_j:>10.4f}"
        )
