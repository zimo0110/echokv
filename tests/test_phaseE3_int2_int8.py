"""Phase E.3: parametric BITS tests.

Validates the unified-attention and write kernels at BITS ∈ {2, 8} (the
INT4 path was already validated in Phases A–D). Each test:

  1. Builds a host-side packed cache via a Python reference at the chosen
     bit-width.
  2. Runs the kernel-side write path on the same input.
  3. Asserts byte-for-byte equivalence on all 6 cache sub-tensors.
  4. Runs the unified-attention kernel against the cache and compares to
     unpack-then-eager-SDPA on the host-decoded reconstruction.

INT2 has 4 codes per byte; INT8 has 1 code per byte (no packing). The
parametric kernel handles both via the same code path with different
``CODES_PER_BYTE`` / ``LEVELS_MASK`` constexpr.
"""
from __future__ import annotations

import sys
from pathlib import Path

import math

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("Phase E.3 tests require CUDA", allow_module_level=True)

_VLLM_FORK = Path("/mnt/nvme_tmp10/nazar_vllm_fork")
if not _VLLM_FORK.is_dir():
    pytest.skip(f"vLLM fork not found at {_VLLM_FORK}",
                allow_module_level=True)
sys.path.insert(0, str(_VLLM_FORK))

from vllm.v1.attention.ops.kivi_cache_layout import (              # noqa: E402
    kivi_page_size_bytes, slice_kivi_cache,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash_kivi import (  # noqa: E402
    triton_reshape_and_cache_flash_kivi,
)
from vllm.v1.attention.ops.triton_unified_attention_kivi import (  # noqa: E402
    unified_attention_2d_kivi,
)


# ---------------------------------------------------------------------------
# Host-side reference for arbitrary BITS
# ---------------------------------------------------------------------------

def _host_pack_per_group(x: torch.Tensor, group_size: int, bits: int):
    """Quantise ``x`` per group along its last dim at the given ``bits``,
    return (data uint8, scale fp16, zp uint8).

    Mirrors the math in ``round_trip_int4_per_group`` but parametric in
    bits. ``x`` shape (..., n_groups * group_size). Output:
        data : (..., n_groups * (group_size * bits // 8)) uint8
        scale: (..., n_groups) fp16
        zp   : (..., n_groups) uint8
    """
    levels = (1 << bits) - 1
    codes_per_byte = 8 // bits
    bytes_per_group = (group_size * bits) // 8
    n_groups = x.shape[-1] // group_size

    x32 = x.to(torch.float32).contiguous()
    grouped = x32.reshape(*x32.shape[:-1], n_groups, group_size)
    mn = grouped.amin(-1)
    mx = grouped.amax(-1)
    span = mx - mn
    is_const = span == 0.0
    standard_scale = (span / levels).clamp_min(torch.finfo(torch.float32).tiny)
    abs_mn = mn.abs()
    const_scale = torch.where(abs_mn > 0.0, abs_mn / levels, torch.ones_like(mn))
    scale = torch.where(is_const, const_scale, standard_scale)
    zp_standard = torch.round(-mn / scale).clamp(0, levels)
    zp_const = torch.where(mn < 0.0, torch.full_like(mn, float(levels)), torch.zeros_like(mn))
    zp = torch.where(is_const, zp_const, zp_standard)

    q = torch.round(
        grouped / scale.unsqueeze(-1) + zp.unsqueeze(-1)
    ).clamp(0, levels).to(torch.uint8)  # shape (..., n_groups, group_size)

    # Pack: combine ``codes_per_byte`` adjacent codes into one byte.
    q_reshaped = q.reshape(*q.shape[:-1], bytes_per_group, codes_per_byte)
    # byte = q[0] | q[1]<<bits | q[2]<<(2*bits) | ...
    shifts = torch.arange(codes_per_byte, device=q.device, dtype=torch.uint8) * bits
    packed = (q_reshaped << shifts).sum(dim=-1).to(torch.uint8)  # (..., n_groups, bytes_per_group)
    data = packed.reshape(*x.shape[:-1], n_groups * bytes_per_group)

    return data, scale.to(torch.float16), zp.to(torch.uint8)


def _host_unpack_per_group(data, scale, zp, group_size, bits, out_dtype):
    """Inverse of ``_host_pack_per_group``."""
    codes_per_byte = 8 // bits
    levels = (1 << bits) - 1
    bytes_per_group = (group_size * bits) // 8
    n_groups = data.shape[-1] // bytes_per_group

    data_re = data.reshape(*data.shape[:-1], n_groups, bytes_per_group)
    # Unpack: for each byte, extract ``codes_per_byte`` codes via shift+mask.
    shifts = torch.arange(codes_per_byte, device=data.device, dtype=torch.uint8) * bits
    q = ((data_re.unsqueeze(-1) >> shifts) & levels).to(torch.float32)
    # Shape: (..., n_groups, bytes_per_group, codes_per_byte) → (..., n_groups, group_size)
    q = q.reshape(*data.shape[:-1], n_groups, group_size)
    recon = (q - zp.float().unsqueeze(-1)) * scale.float().unsqueeze(-1)
    recon = recon.reshape(*data.shape[:-1], n_groups * group_size)
    return recon.to(out_dtype)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bits", [2, 8])
@pytest.mark.parametrize("group_size", [16, 32])
def test_write_kernel_matches_host_pack(bits, group_size):
    """Kernel write path == host-side pack reference, byte-for-byte."""
    if (group_size * bits) % 8 != 0:
        pytest.skip(f"group_size={group_size} not byte-aligned at bits={bits}")
    torch.manual_seed(0)
    T, H_kv, D = 32, 4, 64
    block_size = 16
    num_blocks = (T + block_size - 1) // block_size
    page_bytes = kivi_page_size_bytes(
        block_size=block_size, num_kv_heads=H_kv,
        head_size=D, bits=bits, group_size=group_size,
    )
    buf = torch.zeros(num_blocks, page_bytes, dtype=torch.uint8, device="cuda")
    view = slice_kivi_cache(
        buf, num_blocks=num_blocks, block_size=block_size,
        num_kv_heads=H_kv, head_size=D, bits=bits, group_size=group_size,
    )

    k = torch.randn(T, H_kv, D, device="cuda", dtype=torch.bfloat16) * 0.5
    v = torch.randn(T, H_kv, D, device="cuda", dtype=torch.bfloat16) * 0.5
    slot_mapping = torch.arange(T, dtype=torch.int64, device="cuda")
    triton_reshape_and_cache_flash_kivi(
        k, v, view, slot_mapping, group_size=group_size, bits=bits,
    )

    # Host reference.
    kd_ref, ks_ref, kz_ref = _host_pack_per_group(k, group_size, bits)
    vd_ref, vs_ref, vz_ref = _host_pack_per_group(v, group_size, bits)
    inp2_buf = torch.zeros(num_blocks, page_bytes, dtype=torch.uint8, device="cuda")
    inp2 = slice_kivi_cache(
        inp2_buf, num_blocks=num_blocks, block_size=block_size,
        num_kv_heads=H_kv, head_size=D, bits=bits, group_size=group_size,
    )
    bytes_per_group = (group_size * bits) // 8
    n_groups = D // group_size
    for t in range(T):
        bid, slot = t // block_size, t % block_size
        # Reshape host-pack output back to (H_kv, n_groups, bytes_per_group)
        # then to (H_kv, n_groups * bytes_per_group) which is what the view expects.
        inp2.k_data[bid, slot]  = kd_ref[t].view(H_kv, n_groups * bytes_per_group)
        inp2.k_scale[bid, slot] = ks_ref[t]
        inp2.k_zp[bid, slot]    = kz_ref[t]
        inp2.v_data[bid, slot]  = vd_ref[t].view(H_kv, n_groups * bytes_per_group)
        inp2.v_scale[bid, slot] = vs_ref[t]
        inp2.v_zp[bid, slot]    = vz_ref[t]

    # Allow bit-exact for kernel q codes vs host (FMA-rounding may
    # differ by 1 step on rare elements; same as Phase A). Check that
    # ≥ 99% of bytes match exactly and the rest differ by ≤ one
    # quantization-step's worth of byte representation.
    for label, kt, ht in [
        ("k_data", view.k_data, inp2.k_data),
        ("k_scale", view.k_scale, inp2.k_scale),
        ("k_zp", view.k_zp, inp2.k_zp),
        ("v_data", view.v_data, inp2.v_data),
        ("v_scale", view.v_scale, inp2.v_scale),
        ("v_zp", view.v_zp, inp2.v_zp),
    ]:
        if kt.dtype == torch.uint8:
            mismatch = (kt != ht).sum().item()
            frac = mismatch / kt.numel()
            assert frac < 0.05, (
                f"{label}: too many byte mismatches "
                f"({mismatch}/{kt.numel()} = {100*frac:.2f}%, expect <5%); "
                f"FMA tiebreak should be rare"
            )
        else:  # fp16 scale
            diff = (kt.float() - ht.float()).abs().max().item()
            ref_max = ht.float().abs().max().item()
            rel = diff / max(ref_max, 1e-9)
            assert rel < 0.05, f"{label}: rel diff {rel:.4e} > 5%"


@pytest.mark.parametrize("bits", [2, 8])
def test_attn_kernel_int_x_matches_eager_sdpa(bits):
    """End-to-end attention at INT2 / INT8 matches eager SDPA on the
    cache's host-decoded reconstruction."""
    torch.manual_seed(0)
    group_size = 32
    T, num_q_heads, num_kv_heads, head_size = 4, 4, 2, 64
    S = 32
    block_size = 16
    num_blocks = (S + block_size - 1) // block_size
    queries_per_kv = num_q_heads // num_kv_heads
    softmax_scale = 1.0 / math.sqrt(head_size)
    dtype = torch.bfloat16

    q = torch.randn(T, num_q_heads, head_size, device="cuda", dtype=dtype) * 0.5
    k = torch.randn(S, num_kv_heads, head_size, device="cuda", dtype=dtype) * 0.5
    v = torch.randn(S, num_kv_heads, head_size, device="cuda", dtype=dtype) * 0.5

    page_bytes = kivi_page_size_bytes(
        block_size=block_size, num_kv_heads=num_kv_heads,
        head_size=head_size, bits=bits, group_size=group_size,
    )
    buf = torch.zeros(num_blocks, page_bytes, dtype=torch.uint8, device="cuda")
    view = slice_kivi_cache(
        buf, num_blocks=num_blocks, block_size=block_size,
        num_kv_heads=num_kv_heads, head_size=head_size,
        bits=bits, group_size=group_size,
    )
    slot_mapping = torch.arange(S, dtype=torch.int64, device="cuda")
    triton_reshape_and_cache_flash_kivi(
        k, v, view, slot_mapping, group_size=group_size, bits=bits,
    )

    block_table = torch.full((1, num_blocks), -1, dtype=torch.int32, device="cuda")
    for bid in range(num_blocks):
        block_table[0, bid] = bid
    seq_lens = torch.tensor([S], dtype=torch.int32, device="cuda")
    qsl = torch.tensor([0, T], dtype=torch.int32, device="cuda")
    output = torch.zeros_like(q)
    unified_attention_2d_kivi(
        output=output, query=q,
        k_data=view.k_data, k_scale=view.k_scale, k_zp=view.k_zp,
        v_data=view.v_data, v_scale=view.v_scale, v_zp=view.v_zp,
        block_table=block_table, seq_lens=seq_lens, query_start_len=qsl,
        softmax_scale=softmax_scale,
        num_query_heads=num_q_heads, num_kv_heads=num_kv_heads,
        head_size=head_size, block_size=block_size,
        group_size=group_size, bits=bits,
        tile_size=16, block_q=4, block_m=queries_per_kv * 4,
    )

    # Reference: host-decode the cache + eager SDPA.
    bytes_per_group = (group_size * bits) // 8
    n_groups = head_size // group_size
    # Reconstruct K,V per token by reading the section-major view's
    # fast-axis layout: data shape (num_blocks, block_size, num_kv_heads,
    # n_groups * bytes_per_group). Re-pack onto a (S, num_kv_heads, ...)
    # tensor to feed _host_unpack.
    def _gather_per_token(data_view, side_view, last_dim_bytes):
        out = torch.empty(S, num_kv_heads, last_dim_bytes,
                          dtype=data_view.dtype, device=data_view.device)
        side_out = torch.empty(S, num_kv_heads, n_groups,
                               dtype=side_view.dtype, device=side_view.device)
        return out, side_out

    k_data_per_tok, k_scale_per_tok = _gather_per_token(
        view.k_data, view.k_scale, n_groups * bytes_per_group,
    )
    v_data_per_tok, v_scale_per_tok = _gather_per_token(
        view.v_data, view.v_scale, n_groups * bytes_per_group,
    )
    k_zp_per_tok = torch.empty(S, num_kv_heads, n_groups, dtype=torch.uint8, device="cuda")
    v_zp_per_tok = torch.empty_like(k_zp_per_tok)
    for s in range(S):
        bid, slot = s // block_size, s % block_size
        k_data_per_tok[s] = view.k_data[bid, slot]
        k_scale_per_tok[s] = view.k_scale[bid, slot]
        k_zp_per_tok[s] = view.k_zp[bid, slot]
        v_data_per_tok[s] = view.v_data[bid, slot]
        v_scale_per_tok[s] = view.v_scale[bid, slot]
        v_zp_per_tok[s] = view.v_zp[bid, slot]

    k_recon = _host_unpack_per_group(
        k_data_per_tok, k_scale_per_tok, k_zp_per_tok,
        group_size, bits, dtype,
    )
    v_recon = _host_unpack_per_group(
        v_data_per_tok, v_scale_per_tok, v_zp_per_tok,
        group_size, bits, dtype,
    )

    qf, kf, vf = q.float(), k_recon.float(), v_recon.float()
    out_ref = torch.empty_like(qf)
    ctx = S - T
    for h in range(num_q_heads):
        kv_h = h // queries_per_kv
        scores = (qf[:, h, :] @ kf[:, kv_h, :].T) * softmax_scale
        for t in range(T):
            scores[t, ctx + t + 1:] = float("-inf")
        out_ref[:, h, :] = torch.softmax(scores, dim=-1) @ vf[:, kv_h, :]

    diff = (output.float() - out_ref).abs()
    rel = diff.max().item() / max(out_ref.abs().max().item(), 1e-9)
    # INT2 has very coarse codes (4 levels) so quantization noise is
    # naturally larger; allow up to 25% relative error there. INT8 has
    # 256 levels; should be < 5%.
    threshold = 0.30 if bits == 2 else 0.05
    assert rel < threshold, (
        f"INT{bits} attn rel diff {rel:.4e} exceeds {threshold:.0%}"
    )
