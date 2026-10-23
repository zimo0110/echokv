"""Tests for evaluation.motivation.kivi — the quantization math."""

from __future__ import annotations

import pytest
import torch

from evaluation.motivation.kivi import (
    DEFAULT_BITS,
    DEFAULT_GROUP_SIZE,
    QuantizedTensor,
    dequantize,
    quantize,
    simulate_quantization,
)


def _linear(shape, device="cpu"):
    n = 1
    for s in shape:
        n *= s
    return torch.linspace(-1.0, 1.0, n, device=device).reshape(shape)


def test_group_size_divisibility_is_enforced():
    x = torch.randn(1, 4, 16, 65)  # 65 not divisible by 32
    with pytest.raises(ValueError):
        quantize(x, bits=4, group_size=32, group_axis=-1)


def test_quantize_returns_integer_valued_q_int():
    x = torch.randn(2, 64)
    qt = quantize(x, bits=4, group_size=32, group_axis=-1)
    # integer-valued floats, within [0, 15]
    assert torch.all(qt.q_int == qt.q_int.round())
    assert qt.q_int.min() >= 0
    assert qt.q_int.max() <= 15
    # scale and zero_point shape: one per group along the grouped axis.
    assert qt.scale.shape == (2, 2)
    assert qt.zero_point.shape == (2, 2)


def test_dequantize_round_trip_is_bounded_by_group_span():
    torch.manual_seed(0)
    x = torch.randn(4, 8, 128)
    qt = quantize(x, bits=4, group_size=32, group_axis=-1)
    x_hat = dequantize(qt)
    # per-group, the reconstruction must lie within
    # [group_min - eps, group_max + scale]. Use the measured scale.
    scale = qt.scale.unsqueeze(-1)
    # rearrange to per-group view
    x_grouped = x.reshape(4, 8, 4, 32)
    x_hat_grouped = x_hat.reshape(4, 8, 4, 32)
    mn = x_grouped.amin(-1, keepdim=True)
    mx = x_grouped.amax(-1, keepdim=True)
    assert torch.all(x_hat_grouped >= mn - scale - 1e-6)
    assert torch.all(x_hat_grouped <= mx + scale + 1e-6)


def test_simulate_preserves_dtype_and_shape():
    x = torch.randn(3, 64, dtype=torch.float32)
    y = simulate_quantization(x, bits=4, group_size=32)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_int4_group32_has_at_most_16_levels_per_group():
    """The information-theoretic axiom (ARCHITECTURE.md § 2) rests on
    INT4 producing at most 2**4 = 16 distinct values per group. Make
    that hold empirically on random input."""
    torch.manual_seed(1)
    x = torch.randn(1, 8, 32)  # one group per (batch, head)
    y = simulate_quantization(x, bits=4, group_size=32, group_axis=-1)
    for b in range(x.shape[0]):
        for h in range(x.shape[1]):
            vals = torch.unique(y[b, h])
            assert vals.numel() <= 16, (
                f"group ({b},{h}) produced {vals.numel()} distinct values"
            )


def test_constant_group_quantizes_losslessly():
    # A group of identical values has span 0; our scale floor avoids
    # divide-by-zero and produces the same value back.
    x = torch.full((1, 32), 0.42)
    y = simulate_quantization(x, bits=4, group_size=32, group_axis=-1)
    assert torch.allclose(y, x, atol=1e-6)


def test_quantize_on_different_axis():
    x = torch.randn(2, 128, 8)  # group along the middle axis (128)
    qt = quantize(x, bits=4, group_size=32, group_axis=1)
    # scale: one value per (batch, group, tail_dim)
    assert qt.scale.shape == (2, 4, 8)
    assert qt.zero_point.shape == (2, 4, 8)
    x_hat = dequantize(qt)
    assert x_hat.shape == x.shape


def test_defaults_match_architecture():
    assert DEFAULT_BITS == 4
    assert DEFAULT_GROUP_SIZE == 32


def test_qt_carries_no_aux_shadow():
    """INV-1 local form for the quantization module: a QuantizedTensor
    stores only (q_int, scale, zero_point, bits, group_size,
    group_axis, original_shape). No FP8 shadow, no alternate
    representation.
    """
    allowed = {
        "q_int",
        "scale",
        "zero_point",
        "bits",
        "group_size",
        "group_axis",
        "original_shape",
    }
    qt = quantize(
        torch.randn(1, 32), bits=4, group_size=32, group_axis=-1
    )
    present = set(vars(qt).keys())
    assert present == allowed, f"unexpected fields: {present - allowed}"
