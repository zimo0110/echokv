"""Tests for the TP-aware canary install/read helpers.

The helpers in ``evaluation.serving.canary_tp`` need to:
- Install ``CanaryHarness`` against a model handed in directly
  (instead of unpacked from a vLLM ``LLM`` object).
- Be idempotent: installing twice without remove should not leave
  duplicate hooks behind.
- Snapshot per-rank state in a JSON-serializable shape.
- Remove cleanly.

These tests use a CPU-only mock that mimics vLLM's Attention layout
(modules whose ``type(...).__name__ == 'Attention'`` with the right
attributes and a ``layers.N`` parent name). No GPU or vLLM required.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from evaluation.serving.canary_tp import (
    install_canary_on_worker,
    remove_canary_on_worker,
    seed_tiers_on_worker,
    snapshot_canary_on_worker,
)


# A class literally named ``Attention`` so ``type(mod).__name__ ==
# "Attention"`` matches CanaryHarness's collector filter. Each instance
# carries the head-count attributes the harness reads.
class Attention(nn.Module):
    def __init__(self, num_heads: int = 4, num_kv_heads: int = 2,
                 head_size: int = 8) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size

    def forward(self, query, key, value, *args, **kwargs):
        # vLLM's real Attention does much more; we only need the shape
        # contract for the pre-hook to fire correctly.
        return torch.zeros_like(query)


class _LayerBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = Attention()


class _MockModel(nn.Module):
    """Mock with module names like ``layers.0.attn`` so the
    ``layers.(\\d+)`` regex finds layer ids."""

    def __init__(self, n_layers: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_LayerBlock() for _ in range(n_layers)])


def _make_kwargs(**overrides):
    """Default canary kwargs that won't fire by chance during tests.

    ``canary_rate=1e-9`` rather than zero — the constructor rejects 0
    (must be in (0, 1]). At 1e-9 the chance of any fire across all
    test prompts is vanishingly small.
    """
    base = {
        "canary_rate": 1e-9,
        "tau": 0.01,
        "tick_interval": 100,
        "tier_application": False,
        "seed": 0,
    }
    base.update(overrides)
    return base


def test_install_returns_layer_count():
    model = _MockModel(n_layers=6)
    n = install_canary_on_worker(model, harness_kwargs=_make_kwargs())
    assert n == 6
    remove_canary_on_worker(model)


def test_install_via_functools_partial():
    """Reproduce the driver-side TPCanaryHandle pattern: ``apply_model``
    only takes a single-arg callable, so kwargs are bound via
    ``functools.partial``. Verify that path works."""
    from functools import partial
    model = _MockModel(n_layers=4)
    fn = partial(install_canary_on_worker,
                 harness_kwargs=_make_kwargs())
    n = fn(model)
    assert n == 4
    remove_canary_on_worker(model)


def test_install_attaches_harness_to_model():
    model = _MockModel(n_layers=4)
    install_canary_on_worker(model, harness_kwargs=_make_kwargs())
    # The helper stores the harness on the model itself so we can
    # find it again from a different RPC call.
    assert hasattr(model, "_echokv_canary_harness")
    remove_canary_on_worker(model)


def test_install_registers_hooks_on_every_attention():
    model = _MockModel(n_layers=5)
    install_canary_on_worker(model, harness_kwargs=_make_kwargs())
    n_with_hooks = 0
    for mod in model.modules():
        if type(mod).__name__ == "Attention":
            # Each Attention should have at least one forward-pre-hook
            # registered (PyTorch stores them on the module's
            # ``_forward_pre_hooks`` OrderedDict).
            if len(mod._forward_pre_hooks) > 0:
                n_with_hooks += 1
    assert n_with_hooks == 5
    remove_canary_on_worker(model)


def test_install_is_idempotent():
    model = _MockModel(n_layers=3)
    install_canary_on_worker(model, harness_kwargs=_make_kwargs())
    # A second install without an explicit remove should swap out the
    # old harness and leave exactly one hook per Attention — not two.
    install_canary_on_worker(model, harness_kwargs=_make_kwargs())
    for mod in model.modules():
        if type(mod).__name__ == "Attention":
            assert len(mod._forward_pre_hooks) == 1
    remove_canary_on_worker(model)


def test_remove_clears_hooks():
    model = _MockModel(n_layers=3)
    install_canary_on_worker(model, harness_kwargs=_make_kwargs())
    removed = remove_canary_on_worker(model)
    assert removed is True
    for mod in model.modules():
        if type(mod).__name__ == "Attention":
            assert len(mod._forward_pre_hooks) == 0
    # Helper now reports False since no harness is attached.
    assert remove_canary_on_worker(model) is False


def test_snapshot_shape_with_no_fires():
    model = _MockModel(n_layers=4)
    install_canary_on_worker(model, harness_kwargs=_make_kwargs())
    snap = snapshot_canary_on_worker(model)
    assert snap["num_layers"] == 4
    assert len(snap["tier_vector"]) == 4
    assert all(t in {2, 4, 8} for t in snap["tier_vector"])
    assert len(snap["per_layer_delta"]) == 4
    assert all(isinstance(v, float) for v in snap["per_layer_delta"])
    assert "stats" in snap
    assert snap["stats"]["canary_fires"] == 0
    remove_canary_on_worker(model)


def test_snapshot_empty_when_not_installed():
    model = _MockModel(n_layers=2)
    snap = snapshot_canary_on_worker(model)
    assert snap == {}


def test_seed_tiers_overrides_controller():
    model = _MockModel(n_layers=4)
    install_canary_on_worker(model, harness_kwargs=_make_kwargs())
    n = seed_tiers_on_worker(model, levels=[2, 4, 8, 4])
    assert n == 4
    snap = snapshot_canary_on_worker(model)
    assert snap["tier_vector"] == [2, 4, 8, 4]
    remove_canary_on_worker(model)


def test_seed_tiers_rejects_wrong_length():
    model = _MockModel(n_layers=4)
    install_canary_on_worker(model, harness_kwargs=_make_kwargs())
    try:
        seed_tiers_on_worker(model, levels=[2, 4, 8])  # one short
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on length mismatch")
    finally:
        remove_canary_on_worker(model)


def test_multi_precision_kwargs_emits_per_tier_delta():
    model = _MockModel(n_layers=3)
    install_canary_on_worker(
        model,
        harness_kwargs=_make_kwargs(multi_precision=True, tau_budget=0.05),
    )
    snap = snapshot_canary_on_worker(model)
    assert "per_tier_delta" in snap
    # Should have entries for each of the three tiers (Precision.INT2/INT4/INT8).
    # The exact key shape depends on MultiPrecisionEstimator.snapshot_per_tier;
    # we just check it's a dict with three keys.
    assert isinstance(snap["per_tier_delta"], dict)
    assert len(snap["per_tier_delta"]) == 3
    remove_canary_on_worker(model)
