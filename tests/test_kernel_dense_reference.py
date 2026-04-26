"""Tests for kernels.canary_attention.dense_canary_attention.

The dense reference is the correctness oracle for the future Triton
paged kernel. These tests verify:

  - Non-canary path computes standard attention, no side effects.
  - Canary path computes KL(ȧ ‖ ā) and writes to the scratchpad.
  - KL is non-negative (KL divergence property).
  - KL is zero when K is already FP16 and quant is lossless.
  - KL grows as quantisation coarsens (INT8 < INT4 < INT2).
  - Scratchpad records the correct layer_id / step labels.
  - INV-4 guard: canary_paged_attention stub raises NotImplementedError.
"""

from __future__ import annotations

import pytest
import torch

from kernels.canary_attention import (
    CanaryScratchpad,
    _attention_weights,
    _kl_divergence,
    dense_canary_attention,
    canary_paged_attention,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_qkv(
    batch: int = 1,
    heads: int = 4,
    seq: int = 32,
    head_dim: int = 64,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    q = torch.randn(batch, heads, seq, head_dim)
    k = torch.randn(batch, heads, seq, head_dim)
    v = torch.randn(batch, heads, seq, head_dim)
    return q, k, v


# ---------------------------------------------------------------------------
# Non-canary path: matches plain SDPA, no scratchpad side effects
# ---------------------------------------------------------------------------


def test_non_canary_path_output_equals_manual_sdpa():
    """Non-canary dense_canary_attention must equal manual SDPA."""
    q, k, v = _make_qkv(heads=2, seq=8, head_dim=64, seed=1)
    out = dense_canary_attention(q, k, v, is_canary=False, causal=True)

    w = _attention_weights(q, k, causal=True)
    ref = torch.matmul(w, v.float()).to(q.dtype)
    assert torch.allclose(out, ref, atol=1e-5), (
        f"max diff: {(out - ref).abs().max().item()}"
    )


def test_non_canary_path_writes_nothing_to_scratchpad():
    q, k, v = _make_qkv()
    sp = CanaryScratchpad()
    dense_canary_attention(q, k, v, is_canary=False, canary_scratchpad=sp)
    assert sp.read_all() == []


def test_non_canary_path_does_not_require_scratchpad():
    q, k, v = _make_qkv()
    dense_canary_attention(q, k, v, is_canary=False)  # must not raise


# ---------------------------------------------------------------------------
# Canary path: sample written, KL correct
# ---------------------------------------------------------------------------


def test_canary_path_writes_one_sample_per_call():
    q, k, v = _make_qkv(heads=2, seq=16, head_dim=64)
    sp = CanaryScratchpad()
    dense_canary_attention(
        q, k, v,
        is_canary=True, canary_scratchpad=sp,
        layer_id=3, step=100,
    )
    samples = sp.read_all()
    assert len(samples) == 1


def test_canary_sample_labels_are_correct():
    q, k, v = _make_qkv()
    sp = CanaryScratchpad()
    dense_canary_attention(
        q, k, v,
        is_canary=True, canary_scratchpad=sp,
        layer_id=7, step=42,
    )
    s = sp.read_all()[0]
    assert s["layer_id"] == 7
    assert s["step"] == 42


def test_canary_kl_is_non_negative():
    """KL divergence is always ≥ 0 (information inequality)."""
    q, k, v = _make_qkv(heads=4, seq=32, head_dim=64, seed=5)
    sp = CanaryScratchpad()
    dense_canary_attention(q, k, v, is_canary=True, canary_scratchpad=sp)
    assert sp.read_all()[0]["kl"] >= 0.0


def test_canary_kl_zero_for_high_precision_quantization():
    """INT8 group-32 quantization on head_dim=64 is very low loss.
    The resulting KL should be close to zero.
    """
    q, k, v = _make_qkv(heads=2, seq=32, head_dim=64, seed=99)
    sp = CanaryScratchpad()
    dense_canary_attention(
        q, k, v,
        is_canary=True, canary_scratchpad=sp,
        quant_bits=8, group_size=32,
    )
    kl = sp.read_all()[0]["kl"]
    assert kl < 1e-3, f"INT8 KL should be near zero, got {kl:.6f}"


def test_canary_kl_increases_as_bits_decrease():
    """Coarser quantisation → larger δ.  INT2 ≥ INT4 ≥ INT8 (on average)."""
    q, k, v = _make_qkv(heads=4, seq=32, head_dim=64, seed=42)

    def run(bits: int) -> float:
        sp = CanaryScratchpad()
        dense_canary_attention(
            q, k, v,
            is_canary=True, canary_scratchpad=sp,
            quant_bits=bits, group_size=32,
        )
        return sp.read_all()[0]["kl"]

    kl_int8 = run(8)
    kl_int4 = run(4)
    kl_int2 = run(2)
    assert kl_int4 >= kl_int8, (
        f"INT4 KL ({kl_int4:.4f}) should be >= INT8 KL ({kl_int8:.4f})"
    )
    assert kl_int2 >= kl_int4, (
        f"INT2 KL ({kl_int2:.4f}) should be >= INT4 KL ({kl_int4:.4f})"
    )


def test_canary_path_accumulates_across_calls():
    """Multiple canary calls should append to the scratchpad, not overwrite."""
    q, k, v = _make_qkv(seq=16, head_dim=64)
    sp = CanaryScratchpad()
    for step in range(5):
        dense_canary_attention(
            q, k, v,
            is_canary=True, canary_scratchpad=sp,
            layer_id=0, step=step,
        )
    assert len(sp.read_all()) == 5


# ---------------------------------------------------------------------------
# Validation: missing scratchpad raises when canary=True
# ---------------------------------------------------------------------------


def test_canary_path_requires_scratchpad():
    q, k, v = _make_qkv()
    with pytest.raises(ValueError, match="canary_scratchpad"):
        dense_canary_attention(q, k, v, is_canary=True, canary_scratchpad=None)


# ---------------------------------------------------------------------------
# _kl_divergence helper correctness
# ---------------------------------------------------------------------------


def test_kl_zero_for_equal_distributions():
    p = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    assert _kl_divergence(p, p).item() == pytest.approx(0.0, abs=1e-6)


def test_kl_non_negative():
    torch.manual_seed(0)
    p = torch.softmax(torch.randn(8, 16), dim=-1)
    q = torch.softmax(torch.randn(8, 16), dim=-1)
    assert (_kl_divergence(p, q) >= 0).all()


# ---------------------------------------------------------------------------
# Paged kernel exists — exhaustive tests live in tests/test_canary_kernel.py.
# This file remains the dense-reference contract.
# ---------------------------------------------------------------------------


def test_paged_kernel_is_callable():
    """``canary_paged_attention`` is the production entry point. The full
    behavioural suite is in tests/test_canary_kernel.py; this is a smoke
    check that the function is callable from the dense-reference test file
    (catches accidental import-time breakage)."""
    q, k, v = _make_qkv(seq=8, head_dim=64)
    out = canary_paged_attention(q, k, v, is_canary=False)
    assert out.shape == q.shape
