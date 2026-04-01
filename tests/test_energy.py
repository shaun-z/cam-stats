"""Tests for energy monitoring backends."""

import time
from pathlib import Path
from unittest.mock import patch

from xstream.benchmark.energy import (
    JetsonEnergyMonitor,
    NoOpEnergyMonitor,
    RaplEnergyMonitor,
    _JetsonRail,
    _find_jetson_power_rail,
    _find_rapl_energy_file,
    _is_jetson,
    _trapezoidal_energy,
    create_energy_monitor,
)

import pytest


def test_trapezoidal_energy_constant_power():
    """Constant 10W for 1 second → 10 J."""
    samples = [(0.0, 10.0), (0.5, 10.0), (1.0, 10.0)]
    assert abs(_trapezoidal_energy(samples) - 10.0) < 1e-9


def test_trapezoidal_energy_linear_ramp():
    """Power ramps 0→10W over 1s → area = 5 J."""
    samples = [(0.0, 0.0), (1.0, 10.0)]
    assert abs(_trapezoidal_energy(samples) - 5.0) < 1e-9


def test_trapezoidal_energy_too_few_samples():
    assert _trapezoidal_energy([]) == 0.0
    assert _trapezoidal_energy([(0.0, 5.0)]) == 0.0


def test_noop_monitor():
    mon = NoOpEnergyMonitor()
    mon.start()
    assert mon.stop() == 0.0


def test_jetson_monitor_with_fake_sysfs(tmp_path: Path):
    """Simulate a Jetson sysfs power rail file."""
    # Write 15W = 15_000_000 µW
    rail_file = tmp_path / "power1_input"
    rail_file.write_text("15000000\n")

    mon = JetsonEnergyMonitor(power_rail_path=rail_file, sample_interval_s=0.002)
    mon.start()
    time.sleep(0.05)  # let it collect some samples
    energy = mon.stop()

    # ~15W × 0.05s ≈ 0.75 J, but timing isn't exact — just check > 0
    assert energy > 0.0
    assert mon.rail_path == rail_file


def test_jetson_monitor_with_fake_vi_sysfs(tmp_path: Path):
    """Simulate a Jetson sysfs voltage × current rail (no power*_input)."""
    # 5V × 3A = 15W → voltage 5000 mV, current 3000 mA
    voltage_file = tmp_path / "in1_input"
    current_file = tmp_path / "curr1_input"
    voltage_file.write_text("5000\n")
    current_file.write_text("3000\n")

    rail = _JetsonRail(voltage_path=voltage_file, current_path=current_file)
    mon = JetsonEnergyMonitor(power_rail_path=rail, sample_interval_s=0.002)
    mon.start()
    time.sleep(0.05)
    energy = mon.stop()

    assert energy > 0.0


def test_find_jetson_power_rail_none_on_non_jetson():
    """On a non-Jetson machine, should return None; on Jetson, a valid rail."""
    rail = _find_jetson_power_rail()
    # Can't assert None (might run on Jetson CI), but it shouldn't crash.
    if rail is not None:
        # Verify it can actually read a power value
        assert rail.read_power_w() >= 0.0


def test_create_energy_monitor_jetson_fallback():
    """If Jetson sysfs is absent, fall back to NoOp; on Jetson, use Jetson monitor."""
    mon = create_energy_monitor("cuda:0", platform="jetson")
    if _is_jetson():
        assert isinstance(mon, JetsonEnergyMonitor)
    else:
        assert isinstance(mon, NoOpEnergyMonitor)


def test_create_energy_monitor_tpu():
    mon = create_energy_monitor("xla:0", platform="tpu")
    assert isinstance(mon, NoOpEnergyMonitor)


# ---------------------------------------------------------------------------
# RAPL (CPU) tests
# ---------------------------------------------------------------------------


def test_rapl_monitor_with_fake_sysfs(tmp_path: Path):
    """Simulate a RAPL energy counter file."""
    energy_file = tmp_path / "energy_uj"
    max_file = tmp_path / "max_energy_range_uj"

    # Start at 1,000,000 µJ (1 J)
    energy_file.write_text("1000000\n")
    max_file.write_text("262143328850\n")

    mon = RaplEnergyMonitor(energy_file=energy_file)
    assert mon.backend == "sysfs"
    mon.start()

    # Simulate 5 J consumed → counter advances to 6,000,000 µJ
    energy_file.write_text("6000000\n")
    energy = mon.stop()

    assert abs(energy - 5.0) < 1e-9
    assert mon.energy_file == energy_file


def test_rapl_monitor_wrap_around(tmp_path: Path):
    """Counter wrap-around should be handled correctly."""
    energy_file = tmp_path / "energy_uj"
    max_file = tmp_path / "max_energy_range_uj"

    max_file.write_text("100000000\n")  # 100 J max
    energy_file.write_text("99000000\n")  # start near max

    mon = RaplEnergyMonitor(energy_file=energy_file)
    mon.start()

    # Counter wraps: 99 J → 0 → 2 J  (total delta = 3 J)
    energy_file.write_text("2000000\n")
    energy = mon.stop()

    assert abs(energy - 3.0) < 1e-9


def test_create_energy_monitor_cpu_auto():
    """--device cpu with auto platform: Jetson detected first, else RAPL/NoOp."""
    mon = create_energy_monitor("cpu", platform="auto")
    if _is_jetson():
        assert isinstance(mon, JetsonEnergyMonitor)
    else:
        assert isinstance(mon, (RaplEnergyMonitor, NoOpEnergyMonitor))


def test_create_energy_monitor_cpu_explicit_fallback():
    """--platform cpu on a machine without RAPL should fall back to NoOp."""
    # On the dev server RAPL may or may not exist; either way no crash.
    mon = create_energy_monitor("cpu", platform="cpu")
    assert isinstance(mon, (RaplEnergyMonitor, NoOpEnergyMonitor))
