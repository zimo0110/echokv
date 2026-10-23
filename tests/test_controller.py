"""Tests for ``core.precision_controller.PIDPrecisionController``.

Coverage:
  * Defaults pinned to ARCHITECTURE.md § 4.2 values.
  * ``effective_setpoint`` monotonicity in u and b.
  * Hysteresis ladder: upgrade above τ_eff, downgrade below τ_eff·(1−H),
    hold inside the deadband.
  * Stall counter prevents pathological flapping.
  * Three-tier ladder respects (INT2, INT4, INT8) bounds.
  * Decisions emitted match the level vector after the tick.
"""

from __future__ import annotations

import inspect

import pytest

from core.block_allocator import Precision
from core.precision_controller import (
    DEFAULT_HYSTERESIS_BAND,
    DEFAULT_SETPOINT,
    DEFAULT_TICK_INTERVAL,
    ControllerDirection,
    PIDPrecisionController,
)


# ---------------------------------------------------------------------------
# Constants and interface contract
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# effective_setpoint monotonicity
# ---------------------------------------------------------------------------


def test_effective_setpoint_baseline_is_setpoint():
    ctrl = PIDPrecisionController(num_layers=4, setpoint=0.01)
    # u=0, b=1 → g=1 → τ_eff = τ
    assert ctrl.effective_setpoint(0.0, 1.0) == pytest.approx(0.01)


def test_effective_setpoint_monotone_in_utilization():
    ctrl = PIDPrecisionController(num_layers=4, setpoint=0.01)
    a = ctrl.effective_setpoint(0.0, 1.0)
    b = ctrl.effective_setpoint(0.5, 1.0)
    c = ctrl.effective_setpoint(1.0, 1.0)
    assert a < b < c, "τ_eff must rise with utilization"


def test_effective_setpoint_monotone_decreasing_in_budget():
    ctrl = PIDPrecisionController(num_layers=4, setpoint=0.01)
    a = ctrl.effective_setpoint(0.5, 1.0)
    b = ctrl.effective_setpoint(0.5, 0.5)
    c = ctrl.effective_setpoint(0.5, 0.0)
    assert a < b < c, "tight budget (low b) must yield higher τ_eff"


# ---------------------------------------------------------------------------
# Hysteresis ladder behaviour
# ---------------------------------------------------------------------------


def test_initial_level_default_is_int4():
    ctrl = PIDPrecisionController(num_layers=4)
    for i in range(4):
        assert ctrl.level(i) == Precision.INT4


def test_upgrade_when_delta_above_setpoint():
    """δ̂ > τ_eff → upgrade INT4 → INT8 (over enough ticks)."""
    ctrl = PIDPrecisionController(
        num_layers=1, setpoint=0.01, hysteresis_band=0.2,
        min_dwell_ticks=0,
    )
    # δ̂ well above τ_eff (=0.01 at u=0, b=1)
    for _ in range(3):
        decisions = ctrl.tick([0.05], kv_utilization=0.0, precision_budget=1.0)
    assert ctrl.level(0) == Precision.INT8
    assert decisions[0].direction == ControllerDirection.HOLD  # already at top


def test_downgrade_when_delta_below_lower_band():
    """δ̂ < τ_eff · (1 − H) → downgrade INT4 → INT2 (over ticks)."""
    ctrl = PIDPrecisionController(
        num_layers=1, setpoint=0.01, hysteresis_band=0.2,
        min_dwell_ticks=0,
    )
    # δ̂ well below τ_eff · 0.8 = 0.008
    for _ in range(3):
        decisions = ctrl.tick([0.0001], kv_utilization=0.0, precision_budget=1.0)
    assert ctrl.level(0) == Precision.INT2


def test_hold_inside_deadband():
    """δ̂ ∈ (τ_eff · (1−H), τ_eff] → hold."""
    ctrl = PIDPrecisionController(
        num_layers=1, setpoint=0.01, hysteresis_band=0.2,
        min_dwell_ticks=0,
    )
    # τ_eff = 0.01; deadband = (0.008, 0.01]
    for _ in range(5):
        ctrl.tick([0.009], kv_utilization=0.0, precision_budget=1.0)
    assert ctrl.level(0) == Precision.INT4


def test_no_upgrade_past_int8():
    ctrl = PIDPrecisionController(num_layers=1, min_dwell_ticks=0)
    # Already INT8 → upgrade is a no-op
    ctrl.reset(initial_level=Precision.INT8)
    decisions = ctrl.tick([1.0], 0.0, 1.0)
    assert ctrl.level(0) == Precision.INT8
    assert decisions[0].direction == ControllerDirection.HOLD


def test_no_downgrade_past_int2():
    ctrl = PIDPrecisionController(num_layers=1, min_dwell_ticks=0)
    ctrl.reset(initial_level=Precision.INT2)
    decisions = ctrl.tick([0.0], 0.0, 1.0)
    assert ctrl.level(0) == Precision.INT2
    assert decisions[0].direction == ControllerDirection.HOLD


# ---------------------------------------------------------------------------
# Stall counter prevents flapping
# ---------------------------------------------------------------------------


def test_stall_counter_prevents_flap():
    """A layer that just transitioned holds for ``min_dwell_ticks`` ticks
    even if signal flips."""
    ctrl = PIDPrecisionController(
        num_layers=1, setpoint=0.01, hysteresis_band=0.2,
        min_dwell_ticks=3,
    )
    # Tick 1: δ̂ high → upgrade INT4 → INT8
    ctrl.tick([0.1], 0.0, 1.0)
    assert ctrl.level(0) == Precision.INT8
    # Ticks 2-3: δ̂ flips low — but stall counter blocks downgrade
    for _ in range(2):
        ctrl.tick([0.0001], 0.0, 1.0)
    assert ctrl.level(0) == Precision.INT8, "stall counter must block flap"
    # Tick 4: dwell exhausted, downgrade allowed (3-tick min after change at tick 1)
    ctrl.tick([0.0001], 0.0, 1.0)
    assert ctrl.level(0) == Precision.INT4


def test_per_layer_independence():
    """Each layer is controlled independently — strong signal on one
    layer doesn't perturb a quiet neighbour. Deadband at τ=0.01,
    band=0.2 is (0.008, 0.01]."""
    ctrl = PIDPrecisionController(
        num_layers=4, setpoint=0.01, hysteresis_band=0.2, min_dwell_ticks=0,
    )
    delta = [0.05, 0.0001, 0.009, 0.05]  # noisy / quiet / in-band / noisy
    for _ in range(4):
        ctrl.tick(delta, 0.0, 1.0)
    assert ctrl.level(0) == Precision.INT8  # noisy → upgraded twice
    assert ctrl.level(1) == Precision.INT2  # quiet → downgraded twice
    assert ctrl.level(2) == Precision.INT4  # in deadband → hold
    assert ctrl.level(3) == Precision.INT8


# ---------------------------------------------------------------------------
# Decisions reflect post-tick state
# ---------------------------------------------------------------------------


def test_tick_returns_one_decision_per_layer():
    ctrl = PIDPrecisionController(num_layers=8, min_dwell_ticks=0)
    decisions = ctrl.tick([0.005] * 8, 0.0, 1.0)
    assert len(decisions) == 8
    assert all(d.layer_id == i for i, d in enumerate(decisions))


def test_decision_direction_matches_outcome():
    ctrl = PIDPrecisionController(
        num_layers=3, setpoint=0.01, hysteresis_band=0.2, min_dwell_ticks=0,
    )
    # 0.05 > 0.01 (upgrade); 0.009 ∈ (0.008, 0.01] deadband (hold);
    # 0.0001 < 0.008 (downgrade).
    decisions = ctrl.tick([0.05, 0.009, 0.0001], 0.0, 1.0)
    assert decisions[0].direction == ControllerDirection.UPGRADE
    assert decisions[0].target == Precision.INT8
    assert decisions[1].direction == ControllerDirection.HOLD
    assert decisions[1].target == Precision.INT4
    assert decisions[2].direction == ControllerDirection.DOWNGRADE
    assert decisions[2].target == Precision.INT2


def test_tick_validates_delta_hat_length():
    ctrl = PIDPrecisionController(num_layers=4)
    with pytest.raises(ValueError, match="entries"):
        ctrl.tick([0.01, 0.01], 0.0, 1.0)


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


def test_reset_returns_to_initial_level():
    ctrl = PIDPrecisionController(num_layers=2, min_dwell_ticks=0)
    ctrl.tick([0.1, 0.0001], 0.0, 1.0)
    assert ctrl.level(0) == Precision.INT8
    assert ctrl.level(1) == Precision.INT2
    ctrl.reset()
    assert ctrl.level(0) == Precision.INT4
    assert ctrl.level(1) == Precision.INT4


def test_reset_with_explicit_level():
    ctrl = PIDPrecisionController(num_layers=2)
    ctrl.reset(initial_level=Precision.INT8)
    assert ctrl.level(0) == Precision.INT8
