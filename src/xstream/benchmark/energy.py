"""Pluggable energy measurement abstraction."""

from __future__ import annotations

import glob
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path


class EnergyMonitor(ABC):
    """Abstract energy monitor. Subclass for each hardware platform."""

    @abstractmethod
    def start(self) -> None:
        """Start energy sampling."""

    @abstractmethod
    def stop(self) -> float:
        """Stop sampling, return total energy in Joules."""


# ---------------------------------------------------------------------------
# Shared helper: trapezoidal integration over (timestamp, power_watts) samples
# ---------------------------------------------------------------------------

def _trapezoidal_energy(samples: list[tuple[float, float]]) -> float:
    if len(samples) < 2:
        return 0.0
    energy_j = 0.0
    for i in range(len(samples) - 1):
        t0, p0 = samples[i]
        t1, p1 = samples[i + 1]
        energy_j += (t1 - t0) * (p0 + p1) / 2.0
    return energy_j


# ---------------------------------------------------------------------------
# Desktop NVIDIA GPU  (pynvml)
# ---------------------------------------------------------------------------

class NvidiaEnergyMonitor(EnergyMonitor):
    """Samples GPU power via pynvml in a background thread."""

    def __init__(self, device_index: int = 0, sample_interval_s: float = 0.005) -> None:
        import pynvml

        self._pynvml = pynvml
        self._sample_interval_s = sample_interval_s
        self._stop_event = threading.Event()
        self._samples: list[tuple[float, float]] = []
        self._thread: threading.Thread | None = None

        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

    def _read_power_w(self) -> float:
        return self._pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            self._samples.append((time.perf_counter(), self._read_power_w()))
            self._stop_event.wait(self._sample_interval_s)

    def start(self) -> None:
        self._samples.clear()
        self._stop_event.clear()
        self._samples.append((time.perf_counter(), self._read_power_w()))
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._samples.append((time.perf_counter(), self._read_power_w()))
        return _trapezoidal_energy(self._samples)


# ---------------------------------------------------------------------------
# Jetson (Orin NX / AGX Orin / etc.)  — sysfs INA3221 power rails
# ---------------------------------------------------------------------------

# Rail name priority: prefer GPU-specific → module total → any rail.
_JETSON_RAIL_PRIORITY = [
    "VDD_GPU_SOC",
    "VDD_CPU_GPU_CV",
    "GPU",
    "VDD_IN",
    "VDD_MODULE",
]


def _is_jetson() -> bool:
    """Detect Jetson platform via /etc/nv_tegra_release or device-tree model."""
    if Path("/etc/nv_tegra_release").exists():
        return True
    dt_model = Path("/proc/device-tree/model")
    if dt_model.exists():
        try:
            return "jetson" in dt_model.read_text().lower()
        except OSError:
            pass
    return False


def _find_jetson_power_rail() -> Path | None:
    """
    Scan /sys/class/hwmon/ for INA3221 power rail sysfs entries.

    Returns the path to the best power*_input file (microwatts), or None.

    Jetson Orin exposes power via hwmon devices backed by ina3221x.
    Each channel has:
      - power<N>_input   (µW, instantaneous)
      - power<N>_label   (rail name, e.g. "VDD_GPU_SOC")
    """
    hwmon_dirs = sorted(glob.glob("/sys/class/hwmon/hwmon*"))

    # Collect all (priority, power_input_path) pairs
    candidates: list[tuple[int, Path]] = []

    for hwmon in hwmon_dirs:
        hwmon_path = Path(hwmon)
        # Check each power channel
        for label_file in sorted(hwmon_path.glob("power*_label")):
            try:
                rail_name = label_file.read_text().strip()
            except OSError:
                continue
            # Derive the corresponding power*_input file
            input_file = label_file.with_name(
                label_file.name.replace("_label", "_input")
            )
            if not input_file.exists():
                continue

            # Assign priority (lower = better)
            try:
                prio = _JETSON_RAIL_PRIORITY.index(rail_name)
            except ValueError:
                prio = len(_JETSON_RAIL_PRIORITY)  # unknown rail, lowest prio
            candidates.append((prio, input_file))

    if not candidates:
        # Fallback: any power*_input file (no label)
        for hwmon in hwmon_dirs:
            for p in sorted(Path(hwmon).glob("power*_input")):
                return p
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


class JetsonEnergyMonitor(EnergyMonitor):
    """
    Samples Jetson module/GPU power via sysfs INA3221 sensors.

    Reads power*_input (microwatts) in a background thread and integrates
    via the trapezoidal rule — same methodology as NvidiaEnergyMonitor.
    """

    def __init__(
        self,
        power_rail_path: Path | None = None,
        sample_interval_s: float = 0.005,
    ) -> None:
        self._rail_path = power_rail_path or _find_jetson_power_rail()
        if self._rail_path is None:
            raise RuntimeError(
                "No Jetson power rail found in /sys/class/hwmon/. "
                "Ensure the INA3221 driver is loaded."
            )
        self._sample_interval_s = sample_interval_s
        self._stop_event = threading.Event()
        self._samples: list[tuple[float, float]] = []
        self._thread: threading.Thread | None = None

        # Sanity-check: try one read
        self._read_power_w()

    @property
    def rail_path(self) -> Path:
        assert self._rail_path is not None
        return self._rail_path

    def _read_power_w(self) -> float:
        """Read instantaneous power in watts from sysfs (value is in µW)."""
        assert self._rail_path is not None
        raw = self._rail_path.read_text().strip()
        return int(raw) / 1_000_000.0

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            self._samples.append((time.perf_counter(), self._read_power_w()))
            self._stop_event.wait(self._sample_interval_s)

    def start(self) -> None:
        self._samples.clear()
        self._stop_event.clear()
        self._samples.append((time.perf_counter(), self._read_power_w()))
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._samples.append((time.perf_counter(), self._read_power_w()))
        return _trapezoidal_energy(self._samples)


# ---------------------------------------------------------------------------
# CPU — Linux RAPL (Running Average Power Limit) energy counters
# ---------------------------------------------------------------------------
#
# Two backends, tried in order:
#   1. sysfs  /sys/class/powercap/{intel,amd}-rapl:0/energy_uj
#      Requires intel_rapl_msr or amd_energy kernel module.
#   2. MSR    /dev/cpu/0/msr  (direct register read)
#      Works without kernel modules; needs read permission on the MSR device
#      (root, or `sudo chmod a+r /dev/cpu/*/msr`).

import os
import struct

_RAPL_BASE = Path("/sys/class/powercap")

# MSR addresses (AMD and Intel share the same interface)
_MSR_RAPL_POWER_UNIT_INTEL = 0x606
_MSR_PKG_ENERGY_STATUS_INTEL = 0x611
_MSR_RAPL_POWER_UNIT_AMD = 0xC0010299
_MSR_PKG_ENERGY_STATUS_AMD = 0xC001029B


def _find_rapl_energy_file() -> Path | None:
    """
    Find the RAPL package-0 energy counter via sysfs.

    RAPL exposes cumulative energy (µJ) via:
      /sys/class/powercap/intel-rapl:0/energy_uj   (Intel)
      /sys/class/powercap/amd-rapl:0/energy_uj     (AMD, kernel ≥6.x)

    Returns the path to the best energy_uj file, or None.
    """
    for pattern in ("intel-rapl:0", "amd-rapl:0", "intel-rapl:1"):
        candidate = _RAPL_BASE / pattern / "energy_uj"
        if candidate.exists():
            return candidate

    for d in sorted(_RAPL_BASE.glob("*-rapl:0")):
        f = d / "energy_uj"
        if f.exists():
            return f
    return None


def _detect_cpu_vendor() -> str:
    """Return 'amd' or 'intel' based on /proc/cpuinfo vendor string."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("vendor_id"):
                    if "AMD" in line:
                        return "amd"
                    return "intel"
    except OSError:
        pass
    return "intel"  # default


def _read_msr(msr_addr: int, cpu: int = 0) -> int:
    """Read a 64-bit MSR register from /dev/cpu/<cpu>/msr."""
    fd = os.open(f"/dev/cpu/{cpu}/msr", os.O_RDONLY)
    try:
        os.lseek(fd, msr_addr, os.SEEK_SET)
        raw = os.read(fd, 8)
        return struct.unpack("<Q", raw)[0]
    finally:
        os.close(fd)


class RaplEnergyMonitor(EnergyMonitor):
    """
    Reads CPU package energy via Linux RAPL counters.

    Tries sysfs first (requires kernel module), then falls back to
    direct MSR reads (requires /dev/cpu/0/msr read permission).

    No background thread needed — just two reads of a cumulative counter.
    """

    def __init__(self, energy_file: Path | None = None) -> None:
        self._backend: str  # "sysfs" or "msr"
        self._energy_file: Path | None = None
        self._energy_unit_j: float = 0.0   # joules per MSR tick
        self._msr_energy_addr: int = 0
        self._max_uj: int = 2**63
        self._start_val: int = 0

        # Try explicit file or sysfs discovery
        sysfs_file = energy_file or _find_rapl_energy_file()
        if sysfs_file is not None:
            self._backend = "sysfs"
            self._energy_file = sysfs_file
            max_file = sysfs_file.parent / "max_energy_range_uj"
            if max_file.exists():
                self._max_uj = int(max_file.read_text().strip())
            self._read_counter()  # sanity-check
            return

        # Fallback: direct MSR read
        vendor = _detect_cpu_vendor()
        if vendor == "amd":
            unit_addr = _MSR_RAPL_POWER_UNIT_AMD
            self._msr_energy_addr = _MSR_PKG_ENERGY_STATUS_AMD
        else:
            unit_addr = _MSR_RAPL_POWER_UNIT_INTEL
            self._msr_energy_addr = _MSR_PKG_ENERGY_STATUS_INTEL

        unit_raw = _read_msr(unit_addr)
        energy_unit_bits = (unit_raw >> 8) & 0x1F
        self._energy_unit_j = 1.0 / (2**energy_unit_bits)
        # MSR counter is 32-bit
        self._max_uj = int((2**32) * self._energy_unit_j * 1_000_000)
        self._backend = "msr"
        self._read_counter()  # sanity-check

    @property
    def energy_file(self) -> Path | None:
        return self._energy_file

    @property
    def backend(self) -> str:
        return self._backend

    def _read_counter(self) -> int:
        """Read current energy counter value (units depend on backend)."""
        if self._backend == "sysfs":
            assert self._energy_file is not None
            return int(self._energy_file.read_text().strip())
        else:
            raw = _read_msr(self._msr_energy_addr)
            return raw & 0xFFFFFFFF  # 32-bit counter

    def start(self) -> None:
        self._start_val = self._read_counter()

    def stop(self) -> float:
        end_val = self._read_counter()
        delta = end_val - self._start_val

        if self._backend == "sysfs":
            if delta < 0:
                delta += self._max_uj
            return delta / 1_000_000.0  # µJ → J
        else:
            # MSR: delta is in RAPL ticks
            if delta < 0:
                delta += 2**32
            return delta * self._energy_unit_j


# ---------------------------------------------------------------------------
# NoOp fallback
# ---------------------------------------------------------------------------

class NoOpEnergyMonitor(EnergyMonitor):
    """Fallback that returns 0. Used on platforms without energy monitoring."""

    def start(self) -> None:
        pass

    def stop(self) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_energy_monitor(
    device: str,
    platform: str = "auto",
    **kwargs,
) -> EnergyMonitor:
    """
    Create an energy monitor for the given device and platform.

    Args:
        device: PyTorch device string (e.g. "cuda:0", "xla:0", "cpu")
        platform: "auto" | "desktop" | "jetson" | "cpu" | "tpu"
            auto    — detect Jetson → pynvml → RAPL → NoOp.
            desktop — force NvidiaEnergyMonitor (pynvml).
            jetson  — force JetsonEnergyMonitor (sysfs INA3221).
            cpu     — force RaplEnergyMonitor (Linux RAPL).
            tpu     — NoOp (no power API).
    """
    if platform == "tpu":
        return NoOpEnergyMonitor()

    if platform == "jetson" or (platform == "auto" and _is_jetson()):
        try:
            return JetsonEnergyMonitor(**kwargs)
        except Exception:
            return NoOpEnergyMonitor()

    if platform == "cpu" or (platform == "auto" and device == "cpu"):
        try:
            return RaplEnergyMonitor()
        except Exception:
            return NoOpEnergyMonitor()

    if device.startswith("cuda") and platform in ("auto", "desktop"):
        try:
            device_index = 0
            if ":" in device:
                device_index = int(device.split(":")[1])
            return NvidiaEnergyMonitor(device_index=device_index, **kwargs)
        except Exception:
            return NoOpEnergyMonitor()

    return NoOpEnergyMonitor()
