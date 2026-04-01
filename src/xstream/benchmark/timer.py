"""Device-aware timing utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch


@dataclass
class TimingBreakdown:
    """Per-image timing breakdown in seconds."""

    forward_s: float
    backward_s: float
    heatmap_s: float

    @property
    def total_s(self) -> float:
        return self.forward_s + self.backward_s + self.heatmap_s


class DeviceTimer:
    """
    Timing abstraction that handles device synchronization.

    Supports CUDA, TPU (XLA), and CPU devices.
    """

    def __init__(self, device: str) -> None:
        self._device = torch.device(device)
        self._is_cuda = device.startswith("cuda")
        self._is_xla = device.startswith("xla")

        if self._is_xla:
            import torch_xla.core.xla_model as xm
            self._xm = xm

    def sync_and_time(self) -> float:
        """Synchronize device and return current time."""
        if self._is_cuda:
            torch.cuda.synchronize(self._device)
        elif self._is_xla:
            self._xm.mark_step()
            self._xm.wait_device_ops()
        return time.perf_counter()
