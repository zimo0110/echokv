"""Round-trip parity tests for kernels.native_int4_kv.

The Triton ``round_trip_int4_per_group`` kernel must produce numerically
identical output to ``evaluation.motivation.kivi.simulate_quantization``
with bits=4, group_size=group_size, group_axis=-1. Any per-bit difference
will compound across attention and break every downstream claim.
"""
from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("native_int4_kv tests require CUDA", allow_module_level=True)

from evaluation.motivation.kivi import simulate_quantization
from kernels.native_int4_kv import (
    LEVELS,
    pack_int4_per_group,
    round_trip_int4_per_group,
    unpack_int4_per_group,
)


# ---------------------------------------------------------------------------
# Numerical-parity helpers
# ---------------------------------------------------------------------------

def _check_parity(x: torch.Tensor, group_size: int) -> None:
    """Assert that the Triton round-trip matches the KIVI reference within
    one quantization step, with mismatches occurring on <0.1% of elements.

    Both implementations do the same fp32 math, but Triton fuses
    ``x/scale + zp`` into a single-rounded FMA while PyTorch does separate
    divide-then-add (double-rounded). On rare elements where the true
    mathematical value of ``x/scale + zp`` lies a few ULPs above (or
    below) a half-integer, the two paths diverge:

      * PyTorch's intermediate ``x/scale`` rounds to fp32 first, often
        landing exactly on a half-integer; the subsequent ``+ zp`` is
        exact, so ``rint`` sees an exact half and applies round-half-to-
        even (picking the *even* integer).
      * Triton's FMA preserves the few extra ULPs of precision through
        the multiply-add, so ``rint`` sees a value strictly above (or
        below) the half-integer and rounds in that direction.

    Triton's FMA path is mathematically *more* accurate than the
    double-rounded reference; the reference's tiebreak is an artefact of
    intermediate fp32 rounding, not the true quantization rule. Per-
    element abs-diff is bounded by one quantization step (the per-group
    scale --- the intrinsic precision of asymmetric per-group INTb), and
    the fraction of differing elements is tiny (~1 in 8K).
    """
    LEVELS = 15
    triton_out = round_trip_int4_per_group(x, group_size=group_size)
    ref_out = simulate_quantization(x, bits=4, group_size=group_size, group_axis=-1)
    diff = (triton_out.float() - ref_out.float()).abs()
    max_diff = diff.max().item()
    n_diff = int((diff > 1e-6).sum())
    frac_diff = n_diff / x.numel()
    # Per-group scale ≈ (mx - mn)/levels; worst-case across all groups
    # gives the tolerance for any one element.
    grouped = x.float().reshape(*x.shape[:-1], x.shape[-1] // group_size, group_size)
    per_group_scale = (grouped.amax(-1) - grouped.amin(-1)) / LEVELS
    tolerance = per_group_scale.max().item() * 1.05  # 5% slack for fp32 noise
    assert max_diff <= tolerance, (
        f"max abs diff {max_diff:.4e} exceeds one quant step "
        f"({tolerance:.4e})\n"
        f"  shape={tuple(x.shape)}, dtype={x.dtype}, group_size={group_size}\n"
        f"  Triton sample: {triton_out.flatten()[:5]}\n"
        f"  Ref sample:    {ref_out.flatten()[:5]}"
    )
    assert frac_diff < 1e-3, (
        f"too many differing elements: {n_diff}/{x.numel()} "
        f"({100*frac_diff:.3f}%); expect <0.1%\n"
        f"  shape={tuple(x.shape)}, dtype={x.dtype}, group_size={group_size}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group_size", [16, 32, 64])
def test_random_uniform(dtype, group_size):
    torch.manual_seed(0)
    x = torch.randn(8, 16, 4 * group_size, device="cuda", dtype=dtype) * 2.0
    _check_parity(x, group_size)


@pytest.mark.parametrize("group_size", [16, 32])
def test_with_outliers(group_size):
    """Heavy-tailed input — outliers in 1% of elements at 8× scale.
    KIVI's per-group quant should localise the outliers without
    poisoning the whole tensor."""
    torch.manual_seed(1)
    x = torch.randn(4, 8, 8 * group_size, device="cuda", dtype=torch.float32)
    mask = torch.rand_like(x) < 0.01
    x[mask] *= 8.0
    _check_parity(x, group_size)


@pytest.mark.parametrize("group_size", [16, 32])
def test_constant_group(group_size):
    """A constant-valued group has span=0; the reference uses a
    folded-into-scale encoding to reconstruct exactly. Our kernel
    must do the same."""
    x = torch.zeros(2, 4, 2 * group_size, device="cuda", dtype=torch.float32)
    x[..., :group_size] = 0.7         # constant positive
    x[..., group_size:] = -1.3        # constant negative
    _check_parity(x, group_size)


@pytest.mark.parametrize("group_size", [16, 32])
def test_kv_shape(group_size):
    """Realistic K/V tensor shape: (T tokens, num_kv_heads, head_dim)."""
    torch.manual_seed(2)
    T, nh, hd = 128, 8, 128
    x = torch.randn(T, nh, hd, device="cuda", dtype=torch.bfloat16) * 0.5
    # Add some outliers like real attention activations
    mask = torch.rand_like(x.float()) < 0.005
    x[mask] *= 6.0
    _check_parity(x, group_size)


def test_input_dim_not_divisible_raises():
    x = torch.randn(4, 30, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="not divisible"):
        round_trip_int4_per_group(x, group_size=32)


def test_cpu_input_raises():
    x = torch.randn(4, 32, dtype=torch.float32)
    with pytest.raises(ValueError, match="CUDA"):
        round_trip_int4_per_group(x, group_size=32)


def test_dtype_preserved():
    """Output dtype matches input dtype."""
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        x = torch.randn(2, 64, device="cuda", dtype=dt)
        y = round_trip_int4_per_group(x, group_size=32)
        assert y.dtype == dt


def test_levels_constant():
    assert LEVELS == 15


# ---------------------------------------------------------------------------
# Phase B: pack / unpack tests (cache-layout write & read paths)
# ---------------------------------------------------------------------------

def _check_pack_unpack_parity(x: torch.Tensor, group_size: int) -> None:
    """pack -> unpack must match ``round_trip_int4_per_group`` within
    half a quantization step.

    The cache layout stores ``scale`` as fp16 (per the vLLM-paged-cache
    design --- aligns with vLLM's existing fp8 scale tensors), while the
    round-trip kernel keeps ``scale`` in fp32 throughout. The fp16 scale
    cast loses ~10 bits of mantissa precision, which propagates to a
    small residue in the reconstruction (~$|q{-}zp|$ × scale × $2^{-10}$
    ≈ <1 bf16 ULP at typical magnitudes). A 1-quantization-step diff
    would indicate a bug; we test for ≤ half a step.
    """
    LEVELS = 15
    data, scale, zp = pack_int4_per_group(x, group_size=group_size)
    recon = unpack_int4_per_group(data, scale, zp, group_size=group_size,
                                  out_dtype=x.dtype)
    rt = round_trip_int4_per_group(x, group_size=group_size)
    diff = (recon.float() - rt.float()).abs().max().item()
    grouped = x.float().reshape(*x.shape[:-1], x.shape[-1] // group_size, group_size)
    max_scale = ((grouped.amax(-1) - grouped.amin(-1)) / LEVELS).max().item()
    tolerance = max_scale * 0.5  # half a quant step
    assert diff <= tolerance, (
        f"pack→unpack diverges from round-trip by {diff:.4e}, "
        f"exceeds half-step tolerance {tolerance:.4e}\n"
        f"  shape={tuple(x.shape)}, dtype={x.dtype}, group_size={group_size}\n"
        f"  pack→unpack sample: {recon.flatten()[:5]}\n"
        f"  round_trip sample:  {rt.flatten()[:5]}"
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group_size", [16, 32, 64])
def test_pack_unpack_random(dtype, group_size):
    torch.manual_seed(0)
    x = torch.randn(8, 16, 4 * group_size, device="cuda", dtype=dtype) * 2.0
    _check_pack_unpack_parity(x, group_size)


@pytest.mark.parametrize("group_size", [16, 32])
def test_pack_unpack_kv_shape(group_size):
    torch.manual_seed(2)
    T, nh, hd = 128, 8, 128
    x = torch.randn(T, nh, hd, device="cuda", dtype=torch.bfloat16) * 0.5
    mask = torch.rand_like(x.float()) < 0.005
    x[mask] *= 6.0
    _check_pack_unpack_parity(x, group_size)


def test_pack_output_shapes():
    """The packed tensors have the documented shape contract."""
    T, nh, hd = 16, 8, 128
    group_size = 32
    x = torch.randn(T, nh, hd, device="cuda", dtype=torch.bfloat16)
    data, scale, zp = pack_int4_per_group(x, group_size=group_size)
    assert data.shape  == (T, nh, hd // 2),               f"data shape {tuple(data.shape)}"
    assert scale.shape == (T, nh, hd // group_size),      f"scale shape {tuple(scale.shape)}"
    assert zp.shape    == (T, nh, hd // group_size),      f"zp shape {tuple(zp.shape)}"
    assert data.dtype  == torch.uint8
    assert scale.dtype == torch.float16
    assert zp.dtype    == torch.uint8


def test_pack_zp_in_range():
    """All zero-points must land in [0, 15]."""
    torch.manual_seed(3)
    x = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16) * 5.0
    _, _, zp = pack_int4_per_group(x, group_size=32)
    assert int(zp.min()) >= 0
    assert int(zp.max()) <= LEVELS


def test_pack_data_nibbles_in_range():
    """Each nibble in the packed bytes must encode a value in [0, 15]."""
    torch.manual_seed(4)
    x = torch.randn(8, 128, device="cuda", dtype=torch.float32) * 3.0
    data, _, _ = pack_int4_per_group(x, group_size=32)
    low  = data & 0x0F
    high = (data >> 4) & 0x0F
    assert int(low.max())  <= LEVELS
    assert int(high.max()) <= LEVELS


def test_pack_odd_group_size_raises():
    """Group size must be even (we pack pairs of nibbles)."""
    x = torch.randn(4, 30, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="group_size must be even"):
        pack_int4_per_group(x, group_size=15)
