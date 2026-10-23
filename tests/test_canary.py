"""Behavioural tests for core.canary_estimator.

Coverage:
  - Interface shape and defaults (retained from session-1 skeleton)
  - EMA dynamics: cold start, convergence, multi-layer independence
  - INV-4: observe(..., is_canary=False) is a strict no-op
  - Scratchpad ring semantics: cap at scratchpad_depth, FIFO eviction
  - Thread safety: concurrent writes do not corrupt state
  - Validation: layer_id bounds, negative KL, NaN
  - reset() zeroes all state
"""

from __future__ import annotations

import random
import threading
import time

import pytest

from core.canary_estimator import (
    DEFAULT_CANARY_RATE,
    DEFAULT_EMA_ALPHA,
    DEFAULT_SCRATCHPAD_DEPTH,
    CanarySample,
    CanaryEstimator,
)


# ---------------------------------------------------------------------------
# Interface / defaults (retained from session-1 skeleton)
# ---------------------------------------------------------------------------


def test_default_canary_rate_is_one_in_sixtyfour():
    assert DEFAULT_CANARY_RATE == pytest.approx(1.0 / 64.0)


def test_default_ema_alpha():
    assert DEFAULT_EMA_ALPHA == pytest.approx(0.1)


def test_interface_has_required_methods():
    est = CanaryEstimator(num_layers=4)
    for name in ("observe", "ema", "snapshot", "reset", "is_canary_step"):
        assert hasattr(est, name), f"missing method: {name}"


def test_constructor_accepts_custom_params():
    est = CanaryEstimator(
        num_layers=32,
        canary_rate=1.0 / 32.0,
        ema_alpha=0.2,
        scratchpad_depth=128,
    )
    assert est.num_layers == 32
    assert est.canary_rate == pytest.approx(1.0 / 32.0)
    assert est.ema_alpha == pytest.approx(0.2)
    assert est.scratchpad_depth == 128


# ---------------------------------------------------------------------------
# INV-4 — observe(is_canary=False) must be a strict no-op
# ---------------------------------------------------------------------------


def test_inv4_non_canary_observe_is_noop():
    """EMA stays at 0, scratchpad stays empty when is_canary=False (INV-4)."""
    est = CanaryEstimator(num_layers=4)
    for step in range(200):
        est.observe(layer_id=0, step=step, kl=None, is_canary=False)
    assert est.ema(0) == 0.0
    assert est.num_samples(0) == 0
    assert est.snapshot() == [0.0, 0.0, 0.0, 0.0]


def test_inv4_non_canary_observe_accepts_none_kl():
    """kl=None is only valid when is_canary=False; must not raise."""
    est = CanaryEstimator(num_layers=2)
    est.observe(layer_id=0, step=0, kl=None, is_canary=False)  # must not raise


# ---------------------------------------------------------------------------
# EMA dynamics
# ---------------------------------------------------------------------------


def test_ema_cold_start_uses_first_sample_directly():
    """First canary sample seeds the EMA at its value, not blended from zero."""
    est = CanaryEstimator(num_layers=1, ema_alpha=0.1)
    est.observe(0, step=0, kl=0.5, is_canary=True)
    assert est.ema(0) == pytest.approx(0.5)


def test_ema_update_law():
    """EMA follows δ̂ = (1-α)·δ̂_prev + α·δ_new after the first sample."""
    alpha = 0.1
    est = CanaryEstimator(num_layers=1, ema_alpha=alpha)
    est.observe(0, step=0, kl=1.0, is_canary=True)   # cold start → 1.0
    est.observe(0, step=1, kl=0.0, is_canary=True)   # blend toward 0
    expected = (1.0 - alpha) * 1.0 + alpha * 0.0
    assert est.ema(0) == pytest.approx(expected)


def test_ema_converges_toward_constant_signal():
    """After many identical samples the EMA converges to that value."""
    target = 0.05
    est = CanaryEstimator(num_layers=1, ema_alpha=0.1)
    for step in range(200):
        est.observe(0, step=step, kl=target, is_canary=True)
    assert est.ema(0) == pytest.approx(target, abs=1e-4)


def test_ema_is_independent_across_layers():
    """Observing on layer 0 must not change layer 1's EMA."""
    est = CanaryEstimator(num_layers=4, ema_alpha=0.5)
    est.observe(0, step=0, kl=1.0, is_canary=True)
    assert est.ema(1) == 0.0
    assert est.ema(2) == 0.0
    assert est.ema(3) == 0.0


def test_snapshot_returns_all_layers():
    est = CanaryEstimator(num_layers=4)
    est.observe(0, step=0, kl=0.1, is_canary=True)
    est.observe(2, step=0, kl=0.3, is_canary=True)
    snap = est.snapshot()
    assert len(snap) == 4
    assert snap[0] == pytest.approx(0.1)
    assert snap[1] == pytest.approx(0.0)
    assert snap[2] == pytest.approx(0.3)
    assert snap[3] == pytest.approx(0.0)


def test_snapshot_is_a_copy():
    """Mutating the returned snapshot must not affect the internal state."""
    est = CanaryEstimator(num_layers=2)
    est.observe(0, step=0, kl=0.2, is_canary=True)
    snap = est.snapshot()
    snap[0] = 999.0
    assert est.ema(0) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Scratchpad ring semantics
# ---------------------------------------------------------------------------


def test_scratchpad_records_raw_samples():
    est = CanaryEstimator(num_layers=2, scratchpad_depth=10)
    for i in range(5):
        est.observe(0, step=i, kl=float(i) * 0.01, is_canary=True)
    samples = est.scratchpad_samples(0)
    assert len(samples) == 5
    assert all(isinstance(s, CanarySample) for s in samples)
    assert samples[0].kl == pytest.approx(0.0)
    assert samples[4].kl == pytest.approx(0.04)


def test_scratchpad_ring_evicts_oldest():
    """Once scratchpad_depth entries are present, older entries are evicted."""
    depth = 5
    est = CanaryEstimator(num_layers=1, scratchpad_depth=depth)
    for i in range(10):
        est.observe(0, step=i, kl=float(i), is_canary=True)
    samples = est.scratchpad_samples(0)
    assert len(samples) == depth
    # The retained samples should be the last `depth` steps (5 through 9).
    kls = [s.kl for s in samples]
    assert kls == pytest.approx([5.0, 6.0, 7.0, 8.0, 9.0])


def test_scratchpad_samples_are_isolated_per_layer():
    est = CanaryEstimator(num_layers=2, scratchpad_depth=10)
    est.observe(0, step=0, kl=0.1, is_canary=True)
    est.observe(1, step=0, kl=0.9, is_canary=True)
    assert len(est.scratchpad_samples(0)) == 1
    assert len(est.scratchpad_samples(1)) == 1
    assert est.scratchpad_samples(0)[0].kl == pytest.approx(0.1)
    assert est.scratchpad_samples(1)[0].kl == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


def test_reset_clears_ema_and_scratchpad():
    est = CanaryEstimator(num_layers=3)
    for i in range(5):
        est.observe(0, step=i, kl=0.5, is_canary=True)
    est.reset()
    assert est.ema(0) == 0.0
    assert est.num_samples(0) == 0
    assert est.snapshot() == [0.0, 0.0, 0.0]


def test_cold_start_resumes_after_reset():
    """After reset(), the first new sample should again seed the EMA directly."""
    est = CanaryEstimator(num_layers=1, ema_alpha=0.1)
    est.observe(0, step=0, kl=1.0, is_canary=True)
    est.reset()
    est.observe(0, step=1, kl=0.3, is_canary=True)
    assert est.ema(0) == pytest.approx(0.3)  # cold-start, not blended from 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_layer_id_raises():
    est = CanaryEstimator(num_layers=4)
    with pytest.raises(ValueError, match="out of range"):
        est.observe(-1, step=0, kl=0.1, is_canary=True)
    with pytest.raises(ValueError, match="out of range"):
        est.observe(4, step=0, kl=0.1, is_canary=True)


def test_negative_kl_raises():
    est = CanaryEstimator(num_layers=2)
    with pytest.raises(ValueError, match="non-negative"):
        est.observe(0, step=0, kl=-0.001, is_canary=True)


def test_nan_kl_raises():
    est = CanaryEstimator(num_layers=2)
    with pytest.raises(ValueError):
        est.observe(0, step=0, kl=float("nan"), is_canary=True)


def test_missing_kl_when_canary_raises():
    est = CanaryEstimator(num_layers=2)
    with pytest.raises(ValueError, match="kl must be provided"):
        est.observe(0, step=0, kl=None, is_canary=True)


def test_constructor_rejects_invalid_params():
    with pytest.raises(ValueError):
        CanaryEstimator(num_layers=0)
    with pytest.raises(ValueError):
        CanaryEstimator(num_layers=4, canary_rate=0.0)
    with pytest.raises(ValueError):
        CanaryEstimator(num_layers=4, ema_alpha=0.0)
    with pytest.raises(ValueError):
        CanaryEstimator(num_layers=4, scratchpad_depth=0)


# ---------------------------------------------------------------------------
# is_canary_step — Bernoulli sampling
# ---------------------------------------------------------------------------


def test_is_canary_step_deterministic_with_seeded_rng():
    rng = random.Random(42)
    est = CanaryEstimator(num_layers=4, canary_rate=0.5)
    results = [est.is_canary_step(rng=rng) for _ in range(100)]
    # Should have roughly half True — check both extremes are present.
    assert any(results)
    assert not all(results)


def test_is_canary_step_rate_one_always_true():
    est = CanaryEstimator(num_layers=1, canary_rate=1.0)
    for _ in range(20):
        assert est.is_canary_step() is True


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_concurrent_writes_do_not_corrupt_ema():
    """Multiple threads writing to different layers concurrently must not
    produce corrupted EMA values (wrong layer, out-of-range, etc.)."""
    num_layers = 8
    est = CanaryEstimator(num_layers=num_layers, ema_alpha=0.1)
    errors: list[str] = []

    def worker(layer_id: int) -> None:
        for step in range(50):
            try:
                est.observe(layer_id, step=step, kl=0.01 * (step + 1), is_canary=True)
            except Exception as exc:
                errors.append(f"layer {layer_id} step {step}: {exc}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_layers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"thread errors: {errors}"
    snap = est.snapshot()
    assert len(snap) == num_layers
    for layer_id, val in enumerate(snap):
        assert val >= 0.0, f"layer {layer_id} EMA is negative: {val}"
