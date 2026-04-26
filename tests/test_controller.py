"""Skeleton tests for core.precision_controller.

The full controller tests — effective-setpoint monotonicity, the
hysteresis ladder behaviour, and decision emission — follow the
implementation. For now we pin the interface and the defaults from
ARCHITECTURE.md § 4.2.
"""

from __future__ import annotations

import inspect

import pytest

from core.precision_controller import (
    DEFAULT_HYSTERESIS_BAND,
    DEFAULT_SETPOINT,
    DEFAULT_TICK_INTERVAL,
    PIDPrecisionController,
)


def test_defaults():
    assert DEFAULT_SETPOINT == pytest.approx(0.01)
    assert DEFAULT_TICK_INTERVAL == 100
    assert DEFAULT_HYSTERESIS_BAND == pytest.approx(0.20)


def test_interface_shape():
    ctrl = PIDPrecisionController(num_layers=32)
    for name in ("reset", "effective_setpoint", "tick", "level"):
        assert hasattr(ctrl, name)
    sig = inspect.signature(ctrl.tick)
    assert list(sig.parameters) == [
        "delta_hat",
        "kv_utilization",
        "precision_budget",
    ]
