"""Tests for ``core.slo_budget.SLOBudget``.

Coverage:
  * Rolling-window TPOT samples → p99 → budget.
  * Pre-computed p99 path equivalent.
  * Budget monotonicity: tighter SLO pressure → lower budget.
  * Clamps at min_budget / max_budget.
  * ``broadcast_precision`` is synchronous and idempotent on each call;
    every call increments ``num_broadcasts`` and invokes the callback
    with the exact levels passed in.
  * No callback registered → broadcast still counts (controller-only
    unit tests of the budget don't need a real scheduler).
"""

from __future__ import annotations

import pytest

from core.block_allocator import Precision
from core.slo_budget import (
    DEFAULT_MAX_BUDGET,
    DEFAULT_MIN_BUDGET,
    DEFAULT_TARGET_P99_TPOT_MS,
    DEFAULT_WINDOW_SIZE,
    SLOBudget,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_defaults():
    assert DEFAULT_TARGET_P99_TPOT_MS == pytest.approx(50.0)
    assert DEFAULT_MIN_BUDGET == pytest.approx(0.1)
    assert DEFAULT_MAX_BUDGET == pytest.approx(1.0)
    assert DEFAULT_WINDOW_SIZE == 200


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_initial_budget_is_max():
    budget = SLOBudget()
    assert budget.budget() == pytest.approx(DEFAULT_MAX_BUDGET)


def test_construct_validates_inputs():
    with pytest.raises(ValueError):
        SLOBudget(target_p99_tpot_ms=0)
    with pytest.raises(ValueError):
        SLOBudget(min_budget=0.5, max_budget=0.4)
    with pytest.raises(ValueError):
        SLOBudget(window_size=0)


# ---------------------------------------------------------------------------
# Rolling-window path
# ---------------------------------------------------------------------------


def test_rolling_window_caps_at_window_size():
    budget = SLOBudget(window_size=10)
    for i in range(50):
        budget.update_tpot_sample(float(i))
    assert budget.num_samples == 10  # cap


def test_under_slo_yields_max_budget():
    """All samples well under target → ratio < 1 → max_budget."""
    budget = SLOBudget(target_p99_tpot_ms=50.0, window_size=10)
    for _ in range(10):
        budget.update_tpot_sample(20.0)  # well under 50ms
    assert budget.budget() == pytest.approx(DEFAULT_MAX_BUDGET)


def test_severe_overrun_yields_min_budget():
    """All samples ≥ 2× target → ratio ≥ 2 → min_budget."""
    budget = SLOBudget(target_p99_tpot_ms=50.0, window_size=10)
    for _ in range(10):
        budget.update_tpot_sample(150.0)  # 3× target
    assert budget.budget() == pytest.approx(DEFAULT_MIN_BUDGET)


def test_budget_monotone_decreasing_in_p99():
    """Increasing latency must yield non-increasing budget."""
    budget_lo = SLOBudget(target_p99_tpot_ms=50.0)
    budget_hi = SLOBudget(target_p99_tpot_ms=50.0)
    budget_lo.update_p99(60.0)   # 1.2× target
    budget_hi.update_p99(80.0)   # 1.6× target
    assert budget_lo.budget() > budget_hi.budget()


def test_pre_computed_p99_path_equivalent_to_window():
    """Updating via ``update_p99`` directly must yield the same budget
    as a window whose p99 equals that value."""
    target = 50.0
    p99 = 75.0  # 1.5× target
    a = SLOBudget(target_p99_tpot_ms=target)
    a.update_p99(p99)
    b = SLOBudget(target_p99_tpot_ms=target, window_size=20)
    for _ in range(20):
        b.update_tpot_sample(p99)
    assert a.budget() == pytest.approx(b.budget(), abs=1e-9)


def test_update_validates_inputs():
    budget = SLOBudget()
    with pytest.raises(ValueError):
        budget.update_tpot_sample(-1.0)
    with pytest.raises(ValueError):
        budget.update_p99(-1.0)


# ---------------------------------------------------------------------------
# Broadcast (INV-3 ingredient)
# ---------------------------------------------------------------------------


def test_broadcast_with_no_callback_just_counts():
    """Useful for unit-testing the controller without a real scheduler."""
    budget = SLOBudget()
    budget.broadcast_precision([Precision.INT4, Precision.INT8], tp_world_size=2)
    assert budget.num_broadcasts == 1
    budget.broadcast_precision([Precision.INT2], tp_world_size=1)
    assert budget.num_broadcasts == 2


def test_broadcast_invokes_callback_synchronously():
    received: list[tuple[list[Precision], int]] = []

    def cb(levels, tp):
        received.append((list(levels), tp))

    budget = SLOBudget()
    budget.register_broadcast_callback(cb)
    levels = [Precision.INT8, Precision.INT4, Precision.INT4, Precision.INT2]
    budget.broadcast_precision(levels, tp_world_size=4)

    assert received == [(levels, 4)]
    assert budget.num_broadcasts == 1


def test_broadcast_validates_tp_world_size():
    budget = SLOBudget()
    with pytest.raises(ValueError):
        budget.broadcast_precision([Precision.INT4], tp_world_size=0)


def test_register_then_clear_callback():
    received: list[int] = []
    budget = SLOBudget()
    budget.register_broadcast_callback(lambda l, t: received.append(t))
    budget.broadcast_precision([Precision.INT4], 1)
    budget.register_broadcast_callback(None)
    budget.broadcast_precision([Precision.INT4], 1)  # no-op via callback
    assert received == [1]
    assert budget.num_broadcasts == 2  # both still counted
