"""Phase D.3: parity tests for the optional features layered into the
KIVI unified-attention kernel (softcap, sliding window, ALiBi slopes).

For each feature the test runs the KIVI kernel with the feature enabled,
runs an eager-mode SDPA reference with the same feature applied to the
unpacked K/V, and asserts the two match within fp16-scale-precision-
through-softmax tolerance.

This locks down the contract that ``kernel_unified_attention_2d_kivi``
is a drop-in replacement for ``kernel_unified_attention_2d`` modulo the
inline INT4 dequant.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("Phase D.3 tests require CUDA", allow_module_level=True)

_VLLM_FORK = Path("/mnt/nvme_tmp10/nazar_vllm_fork")
if not _VLLM_FORK.is_dir():
    pytest.skip(f"vLLM fork not found at {_VLLM_FORK}",
                allow_module_level=True)
sys.path.insert(0, str(_VLLM_FORK))

from vllm.v1.attention.ops.triton_unified_attention_kivi import (   # noqa: E402
    unified_attention_2d_kivi,
)
from kernels.native_int4_kv import (                                 # noqa: E402
    pack_int4_per_group,
    unpack_int4_per_group,
)


def _build_paged_cache(k, v, *, num_blocks, block_size, group_size):
    S, H_kv, D = k.shape
    half = D // 2
    n_groups = D // group_size
    k_data  = torch.zeros(num_blocks, block_size, H_kv, half, dtype=torch.uint8, device=k.device)
    v_data  = torch.zeros_like(k_data)
    k_scale = torch.zeros(num_blocks, block_size, H_kv, n_groups, dtype=torch.float16, device=k.device)
    v_scale = torch.zeros_like(k_scale)
    k_zp    = torch.zeros(num_blocks, block_size, H_kv, n_groups, dtype=torch.uint8, device=k.device)
    v_zp    = torch.zeros_like(k_zp)
    kd, ks, kz = pack_int4_per_group(k, group_size=group_size)
    vd, vs, vz = pack_int4_per_group(v, group_size=group_size)
    block_table = torch.full((1, (S + block_size - 1) // block_size),
                              -1, dtype=torch.int32, device=k.device)
    for s in range(S):
        bid, slot = s // block_size, s % block_size
        block_table[0, bid] = bid
        k_data[bid, slot] = kd[s];  k_scale[bid, slot] = ks[s];  k_zp[bid, slot] = kz[s]
        v_data[bid, slot] = vd[s];  v_scale[bid, slot] = vs[s];  v_zp[bid, slot] = vz[s]
    return k_data, k_scale, k_zp, v_data, v_scale, v_zp, block_table


def _ref_sdpa_with_features(
    q, k, v, scale,
    *,
    softcap: float = 0.0,
    sliding_window: int = 0,
    alibi_slopes: torch.Tensor | None = None,
):
    """Eager fp32 SDPA with the same feature surface as the kernel."""
    T, Hq, D = q.shape
    S, H_kv, _ = k.shape
    queries_per_kv = Hq // H_kv
    out = torch.empty_like(q, dtype=torch.float32)
    qf, kf, vf = q.float(), k.float(), v.float()
    ctx = S - T
    for h in range(Hq):
        kv_h = h // queries_per_kv
        scores = (qf[:, h, :] @ kf[:, kv_h, :].T) * scale  # (T, S)
        # Causal: t can attend to keys [0, ctx + t]
        for t in range(T):
            scores[t, ctx + t + 1:] = float("-inf")
        # Sliding window: t can only attend to keys >= (ctx + t - sw + 1)
        if sliding_window > 0:
            for t in range(T):
                cutoff = ctx + t - sliding_window + 1
                if cutoff > 0:
                    scores[t, :cutoff] = float("-inf")
        # Softcap (applied AFTER scale, BEFORE mask of -inf — matches kernel order)
        # But we need to apply softcap to the scaled-but-unmasked scores. To
        # mirror the kernel exactly, recompute:
        if softcap > 0.0:
            raw = (qf[:, h, :] @ kf[:, kv_h, :].T) * scale
            cap = softcap * torch.tanh(raw / softcap)
            # Now re-apply the masks from above.
            for t in range(T):
                cap[t, ctx + t + 1:] = float("-inf")
            if sliding_window > 0:
                for t in range(T):
                    cutoff = ctx + t - sliding_window + 1
                    if cutoff > 0:
                        cap[t, :cutoff] = float("-inf")
            scores = cap
        if alibi_slopes is not None:
            slope = alibi_slopes[h].item()
            # offset = key_pos - context_len  → (S,)
            seq_offset = torch.arange(S, device=q.device, dtype=torch.float32) - ctx
            scores = scores + slope * seq_offset[None, :]
        attn = torch.softmax(scores, dim=-1)
        out[:, h, :] = attn @ vf[:, kv_h, :]
    return out.to(q.dtype)


# ---------------------------------------------------------------------------
# Common test setup
# ---------------------------------------------------------------------------

def _setup_inputs(group_size=32):
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
        k, v, num_blocks=num_blocks, block_size=block_size, group_size=group_size,
    )
    seq_lens = torch.tensor([S], dtype=torch.int32, device="cuda")
    query_start_len = torch.tensor([0, T], dtype=torch.int32, device="cuda")
    return dict(
        q=q, k=k, v=v,
        k_data=k_data, k_scale=k_scale, k_zp=k_zp,
        v_data=v_data, v_scale=v_scale, v_zp=v_zp,
        block_table=block_table, seq_lens=seq_lens, query_start_len=query_start_len,
        softmax_scale=softmax_scale,
        T=T, S=S, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
        head_size=head_size, block_size=block_size, group_size=group_size,
        queries_per_kv=queries_per_kv, dtype=dtype,
    )


def _run_kivi(inp, **kwargs):
    output = torch.zeros_like(inp["q"])
    unified_attention_2d_kivi(
        output=output,
        query=inp["q"],
        k_data=inp["k_data"], k_scale=inp["k_scale"], k_zp=inp["k_zp"],
        v_data=inp["v_data"], v_scale=inp["v_scale"], v_zp=inp["v_zp"],
        block_table=inp["block_table"],
        seq_lens=inp["seq_lens"],
        query_start_len=inp["query_start_len"],
        softmax_scale=inp["softmax_scale"],
        num_query_heads=inp["num_q_heads"],
        num_kv_heads=inp["num_kv_heads"],
        head_size=inp["head_size"],
        block_size=inp["block_size"],
        group_size=inp["group_size"],
        tile_size=16,
        block_q=4,
        block_m=inp["queries_per_kv"] * 4,
        **kwargs,
    )
    return output


def _ref_with_features(inp, **kwargs):
    kd, ks, kz = pack_int4_per_group(inp["k"], group_size=inp["group_size"])
    vd, vs, vz = pack_int4_per_group(inp["v"], group_size=inp["group_size"])
    k_ref = unpack_int4_per_group(kd, ks, kz, group_size=inp["group_size"], out_dtype=inp["dtype"])
    v_ref = unpack_int4_per_group(vd, vs, vz, group_size=inp["group_size"], out_dtype=inp["dtype"])
    return _ref_sdpa_with_features(inp["q"], k_ref, v_ref, inp["softmax_scale"], **kwargs)


def _assert_close(out, ref, label):
    diff = (out.float() - ref.float()).abs()
    out_mag = ref.float().abs().max().item()
    rel = diff.max().item() / max(out_mag, 1e-9)
    assert rel < 5e-2, (
        f"{label}: rel diff {rel:.4e} too large\n"
        f"  abs max {diff.max().item():.4e} (out_mag {out_mag:.4e})"
    )


# ---------------------------------------------------------------------------
# Feature tests
# ---------------------------------------------------------------------------

def test_softcap():
    """USE_SOFTCAP applies softcap(scale * Q·K^T) before the causal mask
    and softmax. Used by Gemma-style models."""
    inp = _setup_inputs()
    out = _run_kivi(inp, softcap=30.0)
    ref = _ref_with_features(inp, softcap=30.0)
    _assert_close(out, ref, "softcap=30")


@pytest.mark.parametrize("sw", [4, 8, 16])
def test_sliding_window(sw):
    """SLIDING_WINDOW restricts each query to keys within sw positions."""
    inp = _setup_inputs()
    out = _run_kivi(inp, sliding_window=sw)
    ref = _ref_with_features(inp, sliding_window=sw)
    _assert_close(out, ref, f"sliding_window={sw}")


def test_alibi_slopes():
    """USE_ALIBI_SLOPES adds slope * (key_pos - context_len) to scores.
    Default linear ALiBi (USE_ALIBI_SQRT=False)."""
    inp = _setup_inputs()
    # ALiBi slopes typically negative powers of 2 per head.
    slopes = torch.tensor([-2.0, -1.0, -0.5, -0.25],
                          device="cuda", dtype=torch.float32)
    assert slopes.numel() == inp["num_q_heads"]
    out = _run_kivi(inp, alibi_slopes=slopes)
    ref = _ref_with_features(inp, alibi_slopes=slopes)
    _assert_close(out, ref, "alibi_slopes")


def test_softcap_and_sliding_window():
    """Both features simultaneously — covers the constexpr-branch
    composition path."""
    inp = _setup_inputs()
    out = _run_kivi(inp, softcap=20.0, sliding_window=8)
    ref = _ref_with_features(inp, softcap=20.0, sliding_window=8)
    _assert_close(out, ref, "softcap+sw")
