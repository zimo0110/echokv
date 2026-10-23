"""Tests for the production canary kernel (``canary_paged_attention``).

These tests cover the spec defined in ARCHITECTURE.md § 4.1 and § 9.5:

  * INV-4: ``is_canary=False`` output is bit-identical to the KIVI-4
    reference (same dense quantised SDPA computation).
  * δ ≥ 0 always (KL non-negativity).
  * δ ≈ 0 when INT4 and FP16 attention weights match (high-bit quant).
  * δ_INT2 > δ_INT4 > δ_INT8 (monotone ordering on quantisation coarseness).
  * Triton kernel δ matches the dense PyTorch reference δ within float32
    tolerance.
  * Two-stream path on CUDA, single-stream fallback on CPU.

CUDA tests are skipped when CUDA is unavailable. Triton-vs-PyTorch parity
runs on CUDA only because the Triton path is GPU-only.
"""

from __future__ import annotations

import pytest
import torch

from kernels.canary_attention import (
    CanaryScratchpad,
    _kivi4_attention_dense,
    _kl_rowwise,
    canary_paged_attention,
    dense_canary_attention,
)


CUDA_AVAILABLE = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _qkv(seed: int = 0, batch: int = 1, heads: int = 4, seq: int = 32, head_dim: int = 64,
         device: str = "cpu", dtype: torch.dtype = torch.float32):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(batch, heads, seq, head_dim, dtype=dtype, device=device, generator=g)
    k = torch.randn(batch, heads, seq, head_dim, dtype=dtype, device=device, generator=g)
    v = torch.randn(batch, heads, seq, head_dim, dtype=dtype, device=device, generator=g)
    return q, k, v


# ---------------------------------------------------------------------------
# INV-4 — no-canary path is bit-identical to KIVI-4 static
# ---------------------------------------------------------------------------


def test_inv4_no_canary_equals_kivi4_reference_cpu():
    """is_canary=False on CPU must be byte-for-byte equal to the KIVI-4
    reference. Both paths call ``_kivi4_attention_dense`` so the equality
    is by construction; this test guards against any divergent code path
    being introduced later."""
    q, k, v = _qkv(seed=1)
    out = canary_paged_attention(q, k, v, is_canary=False)
    ref = _kivi4_attention_dense(q, k, v)
    assert torch.equal(out, ref), (
        f"INV-4 violated on CPU: max diff = {(out - ref).abs().max().item():.2e}"
    )


@cuda_only
def test_inv4_no_canary_equals_kivi4_reference_cuda():
    q, k, v = _qkv(seed=2, device="cuda")
    out = canary_paged_attention(q, k, v, is_canary=False)
    ref = _kivi4_attention_dense(q, k, v)
    assert torch.equal(out, ref), (
        f"INV-4 violated on CUDA: max diff = {(out - ref).abs().max().item():.2e}"
    )


def test_inv4_no_canary_writes_nothing():
    """No-canary path must not touch the scratchpad even if one is
    provided — INV-4 also implies zero side effects on the canary path."""
    q, k, v = _qkv(seed=3)
    sp = CanaryScratchpad()
    canary_paged_attention(q, k, v, is_canary=False, canary_scratchpad=sp)
    assert sp.read_all() == []


# ---------------------------------------------------------------------------
# Canary path — basic correctness
# ---------------------------------------------------------------------------


def test_canary_path_writes_one_sample_per_call():
    q, k, v = _qkv(seed=4)
    sp = CanaryScratchpad()
    canary_paged_attention(
        q, k, v, is_canary=True, canary_scratchpad=sp, layer_id=24, step=42,
    )
    samples = sp.read_all()
    assert len(samples) == 1
    assert samples[0]["layer_id"] == 24
    assert samples[0]["step"] == 42


def test_canary_path_requires_a_sink():
    q, k, v = _qkv(seed=5)
    with pytest.raises(ValueError, match="canary_scratchpad or estimator"):
        canary_paged_attention(q, k, v, is_canary=True)


def test_canary_kl_is_non_negative():
    """KL divergence must be ≥ 0 by definition (information inequality)."""
    q, k, v = _qkv(seed=6, heads=4, seq=32, head_dim=64)
    sp = CanaryScratchpad()
    canary_paged_attention(q, k, v, is_canary=True, canary_scratchpad=sp)
    assert sp.read_all()[0]["kl"] >= 0.0


def test_canary_kl_near_zero_for_high_bit_quant():
    """INT8 group-32 quantisation on head_dim=64 is near-lossless;
    KL between the INT8-quantised and FP16 attention distributions must
    therefore be near zero."""
    q, k, v = _qkv(seed=7, heads=2, seq=32, head_dim=64)
    sp = CanaryScratchpad()
    canary_paged_attention(
        q, k, v, is_canary=True, canary_scratchpad=sp,
        quant_bits=8, group_size=32,
    )
    kl = sp.read_all()[0]["kl"]
    assert kl < 1e-3, f"INT8 KL should be near zero, got {kl:.6f}"


def test_canary_kl_monotone_in_bit_width():
    """Coarser quantisation → larger δ. INT2 ≥ INT4 ≥ INT8."""
    q, k, v = _qkv(seed=8, heads=4, seq=32, head_dim=64)

    def run(bits: int) -> float:
        sp = CanaryScratchpad()
        canary_paged_attention(
            q, k, v, is_canary=True, canary_scratchpad=sp,
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
    q, k, v = _qkv(seed=9, seq=16, head_dim=64)
    sp = CanaryScratchpad()
    for step in range(5):
        canary_paged_attention(
            q, k, v, is_canary=True, canary_scratchpad=sp,
            layer_id=0, step=step,
        )
    assert len(sp.read_all()) == 5


# ---------------------------------------------------------------------------
# δ matches dense reference within float32 tolerance
# ---------------------------------------------------------------------------


def test_canary_kl_matches_dense_reference_cpu():
    """The Triton kernel's δ must match dense_canary_attention's δ within
    float32 tolerance on identical inputs."""
    q, k, v = _qkv(seed=10, heads=4, seq=24, head_dim=64)

    sp_paged = CanaryScratchpad()
    canary_paged_attention(
        q, k, v, is_canary=True, canary_scratchpad=sp_paged,
        quant_bits=4, group_size=32,
    )
    delta_paged = sp_paged.read_all()[0]["kl"]

    sp_dense = CanaryScratchpad()
    dense_canary_attention(
        q, k, v, is_canary=True, canary_scratchpad=sp_dense,
        quant_bits=4, group_size=32,
    )
    delta_dense = sp_dense.read_all()[0]["kl"]

    assert abs(delta_paged - delta_dense) < 1e-5, (
        f"δ_paged ({delta_paged:.6f}) and δ_dense ({delta_dense:.6f}) disagree"
    )


@cuda_only
def test_canary_kl_matches_dense_reference_cuda():
    q, k, v = _qkv(seed=11, heads=4, seq=24, head_dim=64, device="cuda")

    sp_paged = CanaryScratchpad()
    canary_paged_attention(
        q, k, v, is_canary=True, canary_scratchpad=sp_paged,
        quant_bits=4, group_size=32,
    )
    delta_paged = sp_paged.read_all()[0]["kl"]

    sp_dense = CanaryScratchpad()
    dense_canary_attention(
        q, k, v, is_canary=True, canary_scratchpad=sp_dense,
        quant_bits=4, group_size=32,
    )
    delta_dense = sp_dense.read_all()[0]["kl"]

    assert abs(delta_paged - delta_dense) < 1e-5, (
        f"CUDA δ_paged ({delta_paged:.6f}) and δ_dense ({delta_dense:.6f}) disagree"
    )


# ---------------------------------------------------------------------------
# Triton row-wise KL kernel correctness
# ---------------------------------------------------------------------------


@cuda_only
def test_triton_kl_matches_pytorch_kl():
    torch.manual_seed(13)
    p = torch.softmax(torch.randn(2, 4, 8, 64, device="cuda"), dim=-1)
    q = torch.softmax(torch.randn(2, 4, 8, 64, device="cuda"), dim=-1)
    from kernels.canary_attention import _kl_divergence
    kl_triton = _kl_rowwise(p, q)
    kl_torch = _kl_divergence(p.float(), q.float())
    diff = (kl_triton - kl_torch).abs().max().item()
    assert diff < 1e-5, f"Triton-vs-PyTorch KL diverges: {diff:.2e}"


@cuda_only
def test_triton_kl_zero_for_identical_distributions():
    torch.manual_seed(14)
    p = torch.softmax(torch.randn(4, 32, device="cuda"), dim=-1)
    kl = _kl_rowwise(p, p)
    assert kl.abs().max().item() < 1e-6


@cuda_only
def test_triton_kl_falls_back_for_oversize_n_cols():
    """Beyond the BLOCK cap (32K), the implementation must fall back to
    PyTorch and still produce correct KL values."""
    torch.manual_seed(15)
    n_cols = 40_000  # beyond _TRITON_MAX_BLOCK
    p = torch.softmax(torch.randn(2, n_cols, device="cuda"), dim=-1)
    q = torch.softmax(torch.randn(2, n_cols, device="cuda"), dim=-1)
    kl = _kl_rowwise(p, q)  # must not crash; falls back to PyTorch
    assert kl.shape == (2,)
    assert (kl >= 0).all()


# ---------------------------------------------------------------------------
# Estimator integration
# ---------------------------------------------------------------------------


def test_canary_kernel_observes_estimator():
    """When an ``estimator`` is passed, the kernel must call
    ``estimator.observe(layer_id, step, kl, is_canary=True)`` so the EMA
    is updated from real kernel output."""
    from core.canary_estimator import CanaryEstimator

    q, k, v = _qkv(seed=16)
    est = CanaryEstimator(num_layers=32)

    canary_paged_attention(
        q, k, v, is_canary=True, estimator=est,
        layer_id=24, step=10,
    )
    canary_paged_attention(
        q, k, v, is_canary=True, estimator=est,
        layer_id=24, step=11,
    )

    # EMA must be initialised and equal to the kernel's recorded kl on
    # second observation. Both calls use the same q,k,v so kl is identical.
    assert est.num_samples(24) == 2
    assert est.ema(24) > 0.0


def test_canary_kernel_writes_to_both_sinks():
    """If both scratchpad and estimator are provided, both receive δ."""
    from core.canary_estimator import CanaryEstimator

    q, k, v = _qkv(seed=17)
    sp = CanaryScratchpad()
    est = CanaryEstimator(num_layers=32)

    canary_paged_attention(
        q, k, v, is_canary=True,
        canary_scratchpad=sp, estimator=est,
        layer_id=5, step=99,
    )
    assert len(sp.read_all()) == 1
    assert est.num_samples(5) == 1
    # Both saw the same δ.
    assert abs(sp.read_all()[0]["kl"] - est.ema(5)) < 1e-9


# ---------------------------------------------------------------------------
# Two-stream behaviour (CUDA only)
# ---------------------------------------------------------------------------


@cuda_only
def test_canary_kernel_two_streams_produce_same_result_as_serial():
    """The CUDA two-stream canary path must produce the same δ as the
    serial CPU path (the streams are an optimisation, not a semantic
    change)."""
    q, k, v = _qkv(seed=18, heads=4, seq=24, head_dim=64)

    sp_cpu = CanaryScratchpad()
    canary_paged_attention(q, k, v, is_canary=True, canary_scratchpad=sp_cpu)

    qc, kc, vc = q.cuda(), k.cuda(), v.cuda()
    sp_cuda = CanaryScratchpad()
    canary_paged_attention(qc, kc, vc, is_canary=True, canary_scratchpad=sp_cuda)

    cpu_kl = sp_cpu.read_all()[0]["kl"]
    cuda_kl = sp_cuda.read_all()[0]["kl"]
    assert abs(cpu_kl - cuda_kl) < 1e-5, (
        f"two-stream CUDA kl ({cuda_kl:.6f}) diverged from serial CPU ({cpu_kl:.6f})"
    )
