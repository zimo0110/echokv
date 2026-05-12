"""Phase D scaffold: Triton attention kernel with inline INT4 dequant.

Single-sequence, single-query-group. Reads K and V from the EchoKV-native
cache layout (uint8 data + fp16 scale + uint8 zp, matching
``kernels/native_int4_kv.py``) and produces attention output equivalent
to running standard SDPA on ``unpack_int4_per_group(data, scale, zp)``,
i.e., dequant fused into the K/V load instead of materialising a full
FP16 K/V tensor first.

This proves the inline-dequant pattern that Phase D.2's full integration
into ``vllm/v1/attention/ops/triton_unified_attention.py`` will use.
Production features (paging, softcap, sliding window, GQA broadcast,
multi-stage reduction, autotuning) are deliberately omitted.

Public API:
    sdpa_int4_packed(q, k_data, k_scale, k_zp, v_data, v_scale, v_zp,
                     group_size=32, scale=None) -> output

Shapes:
    q      : (T, H, D)            fp16/bf16
    k_data : (S, H, D // 2)        uint8
    k_scale: (S, H, D // group_size) fp16
    k_zp   : (S, H, D // group_size) uint8
    v_data : (S, H, D // 2)        uint8
    v_scale: (S, H, D // group_size) fp16
    v_zp   : (S, H, D // group_size) uint8
    output : (T, H, D)            same dtype as q
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sdpa_int4_packed_kernel(
    q_ptr,                # (T, H, D), fp16/bf16
    k_data_ptr,           # (S, H, D//2), uint8
    k_scale_ptr,          # (S, H, n_groups), fp16
    k_zp_ptr,             # (S, H, n_groups), uint8
    v_data_ptr,           # (S, H, D//2), uint8
    v_scale_ptr,          # (S, H, n_groups), fp16
    v_zp_ptr,             # (S, H, n_groups), uint8
    out_ptr,              # (T, H, D), same dtype as q
    softmax_scale,        # 1 / sqrt(D) typically
    T,                    # num query tokens
    S,                    # num key tokens
    H: tl.constexpr,
    D: tl.constexpr,
    HALF: tl.constexpr,        # D // 2
    N_GROUPS: tl.constexpr,    # D // group_size
    GROUP_SIZE: tl.constexpr,
):
    """One program per (query-token, head). Iterates keys one row at a
    time, dequantising K[s] and V[s] from the packed layout into full-D
    fp32 vectors via index-tensor gathers, and accumulates attention
    output via online softmax.
    """
    pid_t = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    if pid_t >= T:
        return

    # ---- load Q for this (token, head) into registers ----
    arange_d = tl.arange(0, D)
    q_off = pid_t * (H * D) + pid_h * D + arange_d
    q = tl.load(q_ptr + q_off).to(tl.float32)

    # ---- per-D index helpers (constant across s loop) ----
    # For each output-dim d:
    #   byte index j = d // 2  (since two nibbles per byte)
    #   parity bit p = d % 2   (low nibble if 0, high nibble if 1)
    #   group index g = d // GROUP_SIZE
    byte_idx = arange_d // 2          # (D,)
    parity   = arange_d % 2           # (D,)
    group_idx = arange_d // GROUP_SIZE # (D,)

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((D,), dtype=tl.float32)

    # Iterate keys row-by-row. BLOCK_S=1 effectively; this is the
    # simplest correct form. Phase D.2 will tile this.
    for s in range(0, S):
        base_data = s * (H * HALF) + pid_h * HALF
        base_side = s * (H * N_GROUPS) + pid_h * N_GROUPS

        # ---- dequantise K[s, pid_h, :] inline into full-D fp32 ----
        k_bytes = tl.load(k_data_ptr + base_data + byte_idx)        # (D,) uint8
        k_q     = tl.where(parity == 0, k_bytes & 0x0F, (k_bytes >> 4) & 0x0F).to(tl.float32)
        k_sc    = tl.load(k_scale_ptr + base_side + group_idx).to(tl.float32)
        k_zp_v  = tl.load(k_zp_ptr    + base_side + group_idx).to(tl.float32)
        k       = (k_q - k_zp_v) * k_sc                              # (D,) fp32

        # qk_s = dot(q, k) * softmax_scale
        qk = tl.sum(q * k, axis=0) * softmax_scale

        # online softmax update
        m_new = tl.maximum(m_i, qk)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)
        acc = acc * alpha

        # ---- dequantise V[s, pid_h, :] inline into full-D fp32 ----
        v_bytes = tl.load(v_data_ptr + base_data + byte_idx)
        v_q     = tl.where(parity == 0, v_bytes & 0x0F, (v_bytes >> 4) & 0x0F).to(tl.float32)
        v_sc    = tl.load(v_scale_ptr + base_side + group_idx).to(tl.float32)
        v_zp_v  = tl.load(v_zp_ptr    + base_side + group_idx).to(tl.float32)
        v       = (v_q - v_zp_v) * v_sc

        acc = acc + p * v
        l_i = l_i * alpha + p
        m_i = m_new

    out = acc / l_i
    out_off = pid_t * (H * D) + pid_h * D + arange_d
    tl.store(out_ptr + out_off, out)


def sdpa_int4_packed(
    q: torch.Tensor,
    k_data: torch.Tensor, k_scale: torch.Tensor, k_zp: torch.Tensor,
    v_data: torch.Tensor, v_scale: torch.Tensor, v_zp: torch.Tensor,
    *,
    group_size: int = 32,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention with K/V dequantised inline from the
    EchoKV cache layout (see module docstring for shape contracts)."""
    if not q.is_cuda:
        raise ValueError("sdpa_int4_packed requires CUDA tensors")
    T, H, D = q.shape
    S = k_data.shape[0]
    n_groups = D // group_size
    half = D // 2
    assert k_data.shape  == (S, H, half),    f"k_data shape {tuple(k_data.shape)}"
    assert v_data.shape  == (S, H, half),    f"v_data shape {tuple(v_data.shape)}"
    assert k_scale.shape == (S, H, n_groups)
    assert k_zp.shape    == (S, H, n_groups)
    assert v_scale.shape == (S, H, n_groups)
    assert v_zp.shape    == (S, H, n_groups)
    assert D % group_size == 0
    assert group_size % 2 == 0

    if softmax_scale is None:
        softmax_scale = 1.0 / (D ** 0.5)

    out = torch.empty_like(q)
    grid = (T, H)
    _sdpa_int4_packed_kernel[grid](
        q, k_data, k_scale, k_zp, v_data, v_scale, v_zp, out,
        softmax_scale,
        T, S,
        H=H, D=D, HALF=half, N_GROUPS=n_groups, GROUP_SIZE=group_size,
    )
    return out
