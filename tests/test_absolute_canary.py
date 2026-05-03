"""Tests for the absolute-canary mode (sliding-window FP16 K_ref).

Architecture: ``ARCHITECTURE.md`` § 9.9 (sensor redesign after the
Phase-4 max_num_seqs=1 INT2 cascade).

The legacy canary measures ``KL(softmax(Q · quant(K)^T) ‖
softmax(Q · K^T))`` where K is the current step's pre-tier-application
K projection. This is a per-batch attention KL where T_key equals the
batch's decode-token count, so:

- T_key = 1 (max_num_seqs=1 in pure decode) → softmax over 1 element
  is trivially [1.0], KL = 0, controller sees no signal, picks INT2
  (lowest tier passing budget) → uniform-INT2 cascade.
- T_key small but >1 → KL is non-zero but the attention pattern
  doesn't match any real attention the model performs.
- Cache-state coupling → multiple equilibria (INT2 cascade in one
  batch regime, INT8 cascade in another).

The absolute-canary fix: maintain a sliding-window buffer of the last
N FP16 K rows seen on each layer. Use that buffer as the canary's key
context, computing KL between FP16 attention and quant-attention
both against the same FP16 reference. Decouples measurement from
batch size T and from cache state.

These tests verify:
- Buffer fills correctly under prefill-style large appends and
  decode-style small appends.
- Cross-tier monotonicity holds: δ_abs(2) ≥ δ_abs(4) ≥ δ_abs(8).
- Signal is non-trivial at T=1 (the failure mode the redesign fixes).
- Disabling the flag preserves the legacy behaviour.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from evaluation.serving.canary_vllm import CanaryHarness


# ---------------------------------------------------------------------------
# CPU-only mock model (matches tests/test_canary_tp.py shape)
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    def __init__(self, num_heads: int = 4, num_kv_heads: int = 2,
                 head_size: int = 32) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size

    def forward(self, query, key, value, *args, **kwargs):
        return torch.zeros_like(query)


class _LayerBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = Attention()


class _MockModel(nn.Module):
    def __init__(self, n_layers: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_LayerBlock() for _ in range(n_layers)])


def _make_harness(**overrides):
    base = dict(
        canary_rate=0.999,             # near-1 so canary fires on every call
        tau=0.01,
        tick_interval=10000,           # don't tick during the unit test
        tier_application=False,
        multi_precision=True,
        tau_budget=0.05,
        absolute_canary=True,
        kref_window_size=16,
        seed=0,
    )
    base.update(overrides)
    model = _MockModel(n_layers=4)
    return CanaryHarness(model=model, **base), model


# ---------------------------------------------------------------------------
# Buffer mechanics
# ---------------------------------------------------------------------------


def test_buffer_unallocated_until_first_call():
    h, _ = _make_harness()
    assert 0 not in h._kref_buf
    assert h._kref_view(0) is None


def test_buffer_appends_decode_step_one_row():
    h, _ = _make_harness(kref_window_size=8)
    # One decode step: T=1, layout is (1, num_kv_heads * head_size)
    key = torch.randn(1, 2 * 32)  # num_kv_heads=2, head_size=32
    h._kref_update(layer_id=0, key=key, num_kv_heads=2)
    assert h._kref_fill[0] == 1
    assert h._kref_view(0) is None  # need >=2 rows
    h._kref_update(layer_id=0, key=key, num_kv_heads=2)
    assert h._kref_fill[0] == 2
    assert h._kref_view(0).shape == (2, 2, 32)


def test_buffer_fifo_eviction_after_full():
    h, _ = _make_harness(kref_window_size=4)
    # Fill with 4 distinct rows tagged by their first element.
    for tag in range(4):
        key = torch.full((1, 2 * 32), float(tag))
        h._kref_update(layer_id=0, key=key, num_kv_heads=2)
    buf = h._kref_view(0)
    assert buf.shape == (4, 2, 32)
    # Now append two more — circular buffer overwrites positions 0, 1.
    for tag in range(4, 6):
        key = torch.full((1, 2 * 32), float(tag))
        h._kref_update(layer_id=0, key=key, num_kv_heads=2)
    # Order is undefined (circular), but the multiset of first-elements
    # should now be {2, 3, 4, 5}.
    seen_tags = sorted({float(buf[i, 0, 0]) for i in range(4)})
    assert seen_tags == [2.0, 3.0, 4.0, 5.0]


def test_buffer_prefill_replaces_when_T_exceeds_window():
    h, _ = _make_harness(kref_window_size=4)
    # Prefill chunk T=10 > window=4. Buffer should hold the LAST 4
    # rows of the chunk.
    rows = torch.stack(
        [torch.full((2 * 32,), float(i)) for i in range(10)],
        dim=0,
    )
    h._kref_update(layer_id=0, key=rows, num_kv_heads=2)
    buf = h._kref_view(0)
    assert buf.shape == (4, 2, 32)
    seen_tags = sorted({float(buf[i, 0, 0]) for i in range(4)})
    assert seen_tags == [6.0, 7.0, 8.0, 9.0]


def test_buffer_disabled_when_absolute_canary_off():
    h, _ = _make_harness(absolute_canary=False)
    key = torch.randn(1, 2 * 32)
    h._kref_update(layer_id=0, key=key, num_kv_heads=2)
    # Should be a no-op when the flag is off.
    assert 0 not in h._kref_buf


# ---------------------------------------------------------------------------
# Cross-tier monotonicity on real tensors
# ---------------------------------------------------------------------------


def test_kl_at_bits_monotonic_on_random_kref():
    """δ_abs(2) ≥ δ_abs(4) ≥ δ_abs(8) on realistic K_ref / Q tensors.

    Uses the harness's actual ``_kl_at_bits`` against a manually
    constructed K_ref tensor. This is the structural invariant from
    Session 13's MultiPrecisionEstimator; we check it again here on
    the new path's call shape (T_query may be small, T_key = window).
    """
    torch.manual_seed(0)
    h, _ = _make_harness(kref_window_size=32)
    # Random Q (1 query token, 4 heads, head_size 8) and K_ref (32, 4, 8)
    # already GQA-replicated for the harness's expected shape.
    q = torch.randn(1, 4, 1, 32).float()
    k = torch.randn(1, 4, 32, 32).float()

    d2 = h._kl_at_bits(q, k, bits=2)
    d4 = h._kl_at_bits(q, k, bits=4)
    d8 = h._kl_at_bits(q, k, bits=8)
    assert d2 >= d4, f"INT2 should be >= INT4 KL: got {d2:.6f} vs {d4:.6f}"
    assert d4 >= d8, f"INT4 should be >= INT8 KL: got {d4:.6f} vs {d8:.6f}"
    assert d8 >= 0.0, f"KL must be non-negative: got {d8:.6f}"


def test_kl_at_bits_nonzero_at_T_query_one():
    """The canary's KL is non-trivial when T_query=1 if T_key > 1.

    This is the property the absolute-canary redesign was built for.
    The legacy canary used T_key = T_query = batch size, so T=1 made
    KL trivially 0. Here T_key is the K_ref window, decoupled from
    the query batch, so KL can be non-zero even at T_query = 1.
    """
    torch.manual_seed(1)
    h, _ = _make_harness(kref_window_size=16)
    q = torch.randn(1, 4, 1, 32).float()           # T_query = 1
    k = torch.randn(1, 4, 16, 32).float() * 2.0    # T_key = 16, scaled up
    d4 = h._kl_at_bits(q, k, bits=4)
    assert d4 > 0.0, f"Expected non-trivial INT4 KL at T_query=1; got {d4}"


def test_legacy_path_unchanged_when_flag_off():
    """``absolute_canary=False`` preserves the legacy code path. We
    can't easily exercise the full pre-hook in a CPU test, but we can
    verify that constructing the harness with the flag off doesn't
    allocate any K_ref state and that the helper view returns None."""
    h, _ = _make_harness(absolute_canary=False, multi_precision=True)
    assert h._kref_buf == {}
    assert h._kref_view(0) is None
    # Buffer-update is a no-op when the flag is off (already covered)
    assert h.absolute_canary is False


# ---------------------------------------------------------------------------
# Constructor parameter validation
# ---------------------------------------------------------------------------


def test_window_size_must_be_at_least_two():
    import pytest
    with pytest.raises(ValueError):
        _make_harness(kref_window_size=1)
