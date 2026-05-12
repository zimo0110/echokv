"""Phase D (proof-of-concept): inline-dequant attention parity tests.

Validates that ``sdpa_int4_packed`` produces output equivalent to
running standard SDPA on the cache layout's reference reconstruction
(``unpack_int4_per_group``). The reference establishes "what the cache
layout actually represents" --- the inline-dequant kernel must not
introduce any additional precision loss beyond what the cache layout
already commits to.
"""
from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("Phase D attn tests require CUDA", allow_module_level=True)

from kernels.native_int4_attn import sdpa_int4_packed
from kernels.native_int4_kv import (
    pack_int4_per_group,
    unpack_int4_per_group,
)


def _ref_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              softmax_scale: float) -> torch.Tensor:
    """Reference SDPA in fp32: same kernel structure as the Triton path
    so the only difference between this and ``sdpa_int4_packed`` is
    where the dequant happens (eager vs fused)."""
    T, H, D = q.shape
    S = k.shape[0]
    qf = q.float()
    kf = k.float()
    vf = v.float()
    out = torch.empty_like(qf)
    for h in range(H):
        scores = (qf[:, h, :] @ kf[:, h, :].T) * softmax_scale  # (T, S)
        attn = torch.softmax(scores, dim=-1)
        out[:, h, :] = attn @ vf[:, h, :]
    return out.to(q.dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group_size", [16, 32])
def test_inline_dequant_attn_matches_unpack_then_attn(dtype, group_size):
    """sdpa_int4_packed(q, packed) ≈ ref_sdpa(q, unpack(packed)).

    Tolerance is roughly fp16-scale precision propagated through one
    softmax, ~few percent of the output magnitude. Larger than fp32
    precision because the cache stores scale as fp16; smaller than one
    quantization step because no q-code rounding is involved (the codes
    were committed to disk during pack)."""
    torch.manual_seed(0)
    T, H, D = 4, 4, 64
    S = 32
    softmax_scale = 1.0 / (D ** 0.5)

    q = torch.randn(T, H, D, device="cuda", dtype=dtype) * 0.5
    # K/V drawn from a similar distribution to attention activations.
    k = torch.randn(S, H, D, device="cuda", dtype=dtype) * 0.5
    v = torch.randn(S, H, D, device="cuda", dtype=dtype) * 0.5

    # Pack K and V via Phase B kernel.
    k_data, k_scale, k_zp = pack_int4_per_group(k, group_size=group_size)
    v_data, v_scale, v_zp = pack_int4_per_group(v, group_size=group_size)

    # Triton inline-dequant attention.
    out_triton = sdpa_int4_packed(
        q, k_data, k_scale, k_zp, v_data, v_scale, v_zp,
        group_size=group_size, softmax_scale=softmax_scale,
    )

    # Reference: dequantize K and V via Phase B unpack, then standard SDPA.
    k_recon = unpack_int4_per_group(k_data, k_scale, k_zp,
                                     group_size=group_size, out_dtype=dtype)
    v_recon = unpack_int4_per_group(v_data, v_scale, v_zp,
                                     group_size=group_size, out_dtype=dtype)
    out_ref = _ref_sdpa(q, k_recon, v_recon, softmax_scale)

    # The two paths should agree to within accumulated fp16/fp32
    # rounding noise. Output magnitudes are ~O(1) for these inputs.
    diff = (out_triton.float() - out_ref.float()).abs()
    max_diff = diff.max().item()
    out_mag = out_ref.float().abs().max().item()
    rel = max_diff / max(out_mag, 1e-9)
    # Empirically, fp16 scale precision + softmax accumulation give ~1%
    # relative error on these inputs; we use 5% as a safe upper bound.
    assert rel < 5e-2, (
        f"inline-dequant attention diverges from unpack-then-attention\n"
        f"  max abs diff: {max_diff:.4e} (relative: {rel:.4e})\n"
        f"  out_triton sample: {out_triton.flatten()[:5]}\n"
        f"  out_ref    sample: {out_ref.flatten()[:5]}\n"
        f"  shape={tuple(q.shape)}, dtype={dtype}, group_size={group_size}"
    )


def test_attn_shape_and_dtype():
    """Output preserves Q's shape and dtype."""
    T, H, D = 2, 8, 32
    S = 16
    q = torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(S, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(S, H, D, device="cuda", dtype=torch.bfloat16)
    k_data, k_scale, k_zp = pack_int4_per_group(k, group_size=16)
    v_data, v_scale, v_zp = pack_int4_per_group(v, group_size=16)
    out = sdpa_int4_packed(
        q, k_data, k_scale, k_zp, v_data, v_scale, v_zp, group_size=16,
    )
    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert out.is_cuda


def test_attn_uniform_v_returns_v():
    """Sanity: if all V rows are identical, attention output should equal
    that row regardless of Q (softmax weighted average of equal values)."""
    T, H, D = 2, 4, 32
    S = 16
    torch.manual_seed(7)
    q = torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    # All V rows equal
    v_row = torch.randn(1, H, D, device="cuda", dtype=torch.bfloat16) * 0.5
    v = v_row.expand(S, -1, -1).contiguous()
    k_data, k_scale, k_zp = pack_int4_per_group(k, group_size=16)
    v_data, v_scale, v_zp = pack_int4_per_group(v, group_size=16)
    out = sdpa_int4_packed(
        q, k_data, k_scale, k_zp, v_data, v_scale, v_zp, group_size=16,
    )
    # Every output row should approximately equal the (dequantised) v_row.
    v_recon = unpack_int4_per_group(v_data, v_scale, v_zp, group_size=16,
                                    out_dtype=torch.bfloat16)
    expected = v_recon[0]  # (H, D), the constant row after pack→unpack
    for t in range(T):
        diff = (out[t].float() - expected.float()).abs().max().item()
        out_mag = expected.float().abs().max().item() + 1e-6
        assert diff / out_mag < 0.05, (
            f"uniform-V failure at t={t}: rel diff {diff/out_mag:.3e}"
        )
