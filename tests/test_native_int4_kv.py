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
from kernels.native_int4_kv import round_trip_int4_per_group, LEVELS


# ---------------------------------------------------------------------------
# Numerical-parity helpers
# ---------------------------------------------------------------------------

def _check_parity(x: torch.Tensor, group_size: int) -> None:
    """Assert that the Triton round-trip matches the PyTorch reference
    bit-for-bit (or to within FP32 ULP).

    The reference does its math in fp32 internally; we do the same. The
    only difference between the two should be reduction order, which for
    min/max over a single contiguous group is deterministic. So the
    expected error is exactly zero.
    """
    triton_out = round_trip_int4_per_group(x, group_size=group_size)
    ref_out = simulate_quantization(x, bits=4, group_size=group_size, group_axis=-1)
    diff = (triton_out.float() - ref_out.float()).abs()
    max_diff = diff.max().item()
    rel = (diff / (ref_out.abs().float() + 1e-9)).max().item()
    assert max_diff < 1e-5, (
        f"max abs diff {max_diff:.4e} exceeds 1e-5 (rel max {rel:.4e})\n"
        f"  shape={tuple(x.shape)}, dtype={x.dtype}, group_size={group_size}\n"
        f"  Triton sample: {triton_out.flatten()[:5]}\n"
        f"  Ref sample:    {ref_out.flatten()[:5]}"
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
