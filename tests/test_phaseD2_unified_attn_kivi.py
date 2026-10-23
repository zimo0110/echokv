"""Phase D.2: end-to-end parity test for the new
``kernel_unified_attention_2d_kivi`` against an unpack-then-attention
reference.

Constructs a vLLM-shape paged cache populated from random K/V via the
Phase B ``pack_int4_per_group`` kernel, runs the new attention kernel
on it, and compares against eager SDPA on the dequantised cache.

This is the smallest end-to-end test that exercises the full new kernel
within vLLM's tiling structure (BLOCK_M, TILE_SIZE, HEAD_SIZE_PADDED)
plus the paged block-table indexing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("Phase D.2 tests require CUDA", allow_module_level=True)

# Load the vLLM fork (Phase C+D) before any vllm import.
_VLLM_FORK = Path("/mnt/nvme_tmp10/nazar_vllm_fork")
if not _VLLM_FORK.is_dir():
    pytest.skip(f"vLLM fork not found at {_VLLM_FORK}",
                allow_module_level=True)
sys.path.insert(0, str(_VLLM_FORK))

# Import after path mutation.
from vllm.v1.attention.ops.triton_unified_attention_kivi import (   # noqa: E402
    unified_attention_2d_kivi,
)
from kernels.native_int4_kv import (                                 # noqa: E402
    pack_int4_per_group,
    unpack_int4_per_group,
)


def _build_paged_cache(
    k: torch.Tensor,         # (S, num_kv_heads, head_size), bf16
    v: torch.Tensor,
    *,
    num_blocks: int,
    block_size: int,
    group_size: int,
):
    """Pack K and V into a vLLM-shape paged cache.

    K/V are (S, H_kv, D); we lay them out into (num_blocks, block_size, H_kv, …).
    For this test we use a single sequence; tokens fill blocks contiguously.
    """
    S, H_kv, D = k.shape
    half = D // 2
    n_groups = D // group_size

    # Pre-allocate the paged tensors.
    k_data  = torch.zeros(num_blocks, block_size, H_kv, half,     dtype=torch.uint8,   device=k.device)
    v_data  = torch.zeros_like(k_data)
    k_scale = torch.zeros(num_blocks, block_size, H_kv, n_groups, dtype=torch.float16, device=k.device)
    v_scale = torch.zeros_like(k_scale)
    k_zp    = torch.zeros(num_blocks, block_size, H_kv, n_groups, dtype=torch.uint8,   device=k.device)
    v_zp    = torch.zeros_like(k_zp)

    # Pack once with the Phase B kernel; result has shape (S, H_kv, …).
    kd, ks, kz = pack_int4_per_group(k, group_size=group_size)
    vd, vs, vz = pack_int4_per_group(v, group_size=group_size)

    # Lay tokens into the paged buffers, one slot at a time.
    block_table = torch.full((1, (S + block_size - 1) // block_size),
                              -1, dtype=torch.int32, device=k.device)
    for s in range(S):
        block_id = s // block_size
        slot = s % block_size
        block_table[0, block_id] = block_id          # contiguous mapping
        k_data[block_id, slot]  = kd[s]
        k_scale[block_id, slot] = ks[s]
        k_zp[block_id, slot]    = kz[s]
        v_data[block_id, slot]  = vd[s]
        v_scale[block_id, slot] = vs[s]
        v_zp[block_id, slot]    = vz[s]
    return k_data, k_scale, k_zp, v_data, v_scale, v_zp, block_table


def _ref_sdpa(q, k, v, scale):
    """Eager fp32 SDPA reference: q (T,Hq,D), k/v (S,H_kv,D)."""
    T, Hq, D = q.shape
    S, H_kv, _ = k.shape
    queries_per_kv = Hq // H_kv
    out = torch.empty_like(q, dtype=torch.float32)
    qf, kf, vf = q.float(), k.float(), v.float()
    for h in range(Hq):
        kv_h = h // queries_per_kv
        scores = (qf[:, h, :] @ kf[:, kv_h, :].T) * scale  # (T, S)
        # Causal mask: with cur_batch_query_len = T and context_len = S - T,
        # query t can attend to keys in [0, context_len + t].
        ctx = S - T
        for t in range(T):
            scores[t, ctx + t + 1:] = float("-inf")
        attn = torch.softmax(scores, dim=-1)
        out[:, h, :] = attn @ vf[:, kv_h, :]
    return out.to(q.dtype)


@pytest.mark.parametrize("group_size", [16, 32])
def test_kernel_2d_kivi_matches_unpack_then_attn(group_size):
    """End-to-end: KIVI kernel ≈ eager SDPA on unpacked K/V.

    Tolerance: fp16 scale precision propagated through softmax (~5%).
    """
    torch.manual_seed(0)
    T, num_q_heads, num_kv_heads, head_size = 4, 4, 2, 64
    S = 32
    block_size = 16
    num_blocks = (S + block_size - 1) // block_size
    queries_per_kv = num_q_heads // num_kv_heads
    softmax_scale = 1.0 / (head_size ** 0.5)
    dtype = torch.bfloat16

    q = torch.randn(T, num_q_heads, head_size, device="cuda", dtype=dtype) * 0.5
    k = torch.randn(S, num_kv_heads, head_size, device="cuda", dtype=dtype) * 0.5
    v = torch.randn(S, num_kv_heads, head_size, device="cuda", dtype=dtype) * 0.5

    k_data, k_scale, k_zp, v_data, v_scale, v_zp, block_table = _build_paged_cache(
        k, v,
        num_blocks=num_blocks,
        block_size=block_size,
        group_size=group_size,
    )

    seq_lens = torch.tensor([S], dtype=torch.int32, device="cuda")
    query_start_len = torch.tensor([0, T], dtype=torch.int32, device="cuda")

    output = torch.zeros_like(q)
    unified_attention_2d_kivi(
        output=output,
        query=q,
        k_data=k_data, k_scale=k_scale, k_zp=k_zp,
        v_data=v_data, v_scale=v_scale, v_zp=v_zp,
        block_table=block_table,
        seq_lens=seq_lens,
        query_start_len=query_start_len,
        softmax_scale=softmax_scale,
        num_query_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        group_size=group_size,
        tile_size=16,
        block_q=4,
        block_m=queries_per_kv * 4,  # BLOCK_M = num_queries_per_kv * BLOCK_Q
    )

    # Reference: dequant via Phase B unpack, then eager SDPA.
    k_recon = unpack_int4_per_group(
        pack_int4_per_group(k, group_size=group_size)[0],
        pack_int4_per_group(k, group_size=group_size)[1],
        pack_int4_per_group(k, group_size=group_size)[2],
        group_size=group_size, out_dtype=dtype,
    )
    # Cleaner: just unpack what we packed during _build_paged_cache.
    # But since pack is deterministic, this is equivalent.
    kd, ks, kz = pack_int4_per_group(k, group_size=group_size)
    vd, vs, vz = pack_int4_per_group(v, group_size=group_size)
    k_ref = unpack_int4_per_group(kd, ks, kz, group_size=group_size, out_dtype=dtype)
    v_ref = unpack_int4_per_group(vd, vs, vz, group_size=group_size, out_dtype=dtype)
    out_ref = _ref_sdpa(q, k_ref, v_ref, softmax_scale)

    diff = (output.float() - out_ref.float()).abs()
    max_diff = diff.max().item()
    out_mag = out_ref.float().abs().max().item()
    rel = max_diff / max(out_mag, 1e-9)
    assert rel < 5e-2, (
        f"KIVI unified-attention diverges from reference by rel {rel:.4e} "
        f"(abs {max_diff:.4e}, out_mag {out_mag:.4e})\n"
        f"  group_size={group_size}, T={T}, S={S}, "
        f"num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}, head_size={head_size}\n"
        f"  out_triton sample: {output.flatten()[:5]}\n"
        f"  out_ref    sample: {out_ref.flatten()[:5]}"
    )
