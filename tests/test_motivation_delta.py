"""Tests for evaluation.motivation.delta — KL + attention pipeline."""

from __future__ import annotations

import math

import pytest
import torch

from evaluation.motivation.delta import (
    AttentionTrace,
    _attention_weights,
    _kl_divergence,
    compute_layer_delta,
    compute_request_delta,
)


def _trace(batch=1, heads=4, q_len=8, k_len=8, head_dim=32, seed=0):
    torch.manual_seed(seed)
    return AttentionTrace(
        q=torch.randn(batch, heads, q_len, head_dim),
        k=torch.randn(batch, heads, k_len, head_dim),
        v=torch.randn(batch, heads, k_len, head_dim),
    )


def test_kl_divergence_zero_for_equal_distributions():
    p = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    q = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    assert torch.allclose(_kl_divergence(p, q), torch.zeros(1), atol=1e-6)


def test_kl_divergence_positive_for_different_distributions():
    p = torch.tensor([[0.9, 0.05, 0.05]])
    q = torch.tensor([[0.2, 0.4, 0.4]])
    assert _kl_divergence(p, q).item() > 0.0


def test_attention_weights_sum_to_one_per_query():
    q = torch.randn(1, 2, 3, 8)
    k = torch.randn(1, 2, 5, 8)
    w = _attention_weights(q, k, causal=False)
    assert w.shape == (1, 2, 3, 5)
    assert torch.allclose(w.sum(-1), torch.ones(w.shape[:-1]), atol=1e-6)


def test_attention_weights_causal_mask_applies_when_square():
    # q_len == k_len triggers the causal mask.
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    w = _attention_weights(q, k, causal=True)
    # upper triangle (strictly) must be zero
    upper = torch.triu(w[0, 0], diagonal=1)
    assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6)


def test_layer_delta_is_nonnegative_and_zero_under_no_quantization_change():
    """When the quantized and reference tensors are identical (e.g.
    with extreme bits that recover the input), KL must be zero.
    We approximate that by using 16-bit quantization on a tensor that
    lives well within the INT16 range — the round-trip is then
    essentially identity, and the KL should be numerically tiny.
    """
    trace = _trace()
    delta = compute_layer_delta(trace, bits=16, group_size=32).item()
    # KL is mathematically non-negative; allow ~1e-6 of float32 noise
    # from the clamp-then-log path.
    assert delta >= -1e-6
    assert delta < 1e-3  # numerically tiny


def test_layer_delta_grows_as_bits_shrink():
    trace = _trace(head_dim=64)
    d16 = compute_layer_delta(trace, bits=16, group_size=32).item()
    d8 = compute_layer_delta(trace, bits=8, group_size=32).item()
    d4 = compute_layer_delta(trace, bits=4, group_size=32).item()
    d2 = compute_layer_delta(trace, bits=2, group_size=32).item()
    # monotone non-decreasing — tighter quantization = more error.
    assert d16 <= d8 + 1e-6
    assert d8 <= d4 + 1e-6
    assert d4 <= d2 + 1e-6
    assert d2 > d16  # clearly separable


def test_request_delta_returns_one_scalar_per_layer():
    traces = [_trace(seed=i) for i in range(12)]
    d = compute_request_delta(traces, bits=4, group_size=32)
    assert d.shape == (12,)
    assert torch.all(torch.isfinite(d))
    assert torch.all(d >= 0)


def test_layer_delta_reduction_none_returns_per_position():
    trace = _trace(batch=2, heads=3, q_len=5, head_dim=32)
    out = compute_layer_delta(
        trace, bits=4, group_size=32, reduction="none"
    )
    assert out.shape == (2, 3, 5)
