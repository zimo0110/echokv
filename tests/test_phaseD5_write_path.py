"""Phase D.5: write-path (reshape_and_cache_kivi) parity tests.

Validates:
  1. ``triton_reshape_and_cache_flash_kivi`` is bit-equivalent to running
     ``pack_int4_per_group`` per token then writing into the cache view
     slot-by-slot. This is the contract that lets the production write
     path replace the host-side packing we used as the reference in D.4.
  2. End-to-end: write via the new kernel → read via the inline-dequant
     attention kernel → matches the eager-SDPA-on-unpacked reference.
  3. Slot-mapping semantics: padding tokens (slot < 0) are skipped.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("Phase D.5 tests require CUDA", allow_module_level=True)

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

from kernels.native_int4_kv import (                                # noqa: E402
    pack_int4_per_group, unpack_int4_per_group,
)


def _setup(num_tokens=32, num_kv_heads=2, head_size=64,
           block_size=16, group_size=32, dtype=torch.bfloat16):
    bits = 4
    n_groups = head_size // group_size
    num_blocks = (num_tokens + block_size - 1) // block_size
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
    return dict(
        buf=buf, view=view,
        num_tokens=num_tokens, num_kv_heads=num_kv_heads, head_size=head_size,
        block_size=block_size, group_size=group_size, num_blocks=num_blocks,
        n_groups=n_groups, dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("group_size", [16, 32])
def test_write_kernel_matches_pack_then_slot_write(dtype, group_size):
    """Write kernel == per-token pack_int4_per_group then slot-write."""
    inp = _setup(group_size=group_size, dtype=dtype)
    torch.manual_seed(0)
    T = inp["num_tokens"]
    H_kv = inp["num_kv_heads"]
    D = inp["head_size"]
    block_size = inp["block_size"]
    view = inp["view"]
    bs = block_size

    k = torch.randn(T, H_kv, D, device="cuda", dtype=dtype) * 0.5
    v = torch.randn(T, H_kv, D, device="cuda", dtype=dtype) * 0.5

    # Slot mapping: simple contiguous (slot t -> token t).
    slot_mapping = torch.arange(T, dtype=torch.int64, device="cuda")

    # Run the write kernel.
    triton_reshape_and_cache_flash_kivi(
        k, v, view, slot_mapping, group_size=group_size,
    )

    # Reference: pack each token via Phase B then write into a fresh view.
    inp2 = _setup(group_size=group_size, dtype=dtype)
    ref = inp2["view"]
    kd, ks, kz = pack_int4_per_group(k, group_size=group_size)
    vd, vs, vz = pack_int4_per_group(v, group_size=group_size)
    for t in range(T):
        bid, slot = t // bs, t % bs
        ref.k_data[bid, slot]  = kd[t]; ref.k_scale[bid, slot] = ks[t]; ref.k_zp[bid, slot] = kz[t]
        ref.v_data[bid, slot]  = vd[t]; ref.v_scale[bid, slot] = vs[t]; ref.v_zp[bid, slot] = vz[t]

    assert torch.equal(view.k_data,  ref.k_data),  "k_data mismatch"
    assert torch.equal(view.k_scale, ref.k_scale), "k_scale mismatch"
    assert torch.equal(view.k_zp,    ref.k_zp),    "k_zp mismatch"
    assert torch.equal(view.v_data,  ref.v_data),  "v_data mismatch"
    assert torch.equal(view.v_scale, ref.v_scale), "v_scale mismatch"
    assert torch.equal(view.v_zp,    ref.v_zp),    "v_zp mismatch"


def test_write_skip_padding_tokens():
    """slot_mapping entries == -1 are padding tokens; the kernel must
    leave those slots unchanged."""
    inp = _setup(num_tokens=8)
    view = inp["view"]
    block_size = inp["block_size"]
    H_kv, D = inp["num_kv_heads"], inp["head_size"]

    # Pre-populate slot 3 with a known pattern.
    view.k_data.fill_(0)
    view.k_data[0, 3].fill_(0xCC)
    view.v_data[0, 3].fill_(0xDD)

    # Tokens 0,1,2 → slots 0,1,2. Token 3 → slot -1 (padding). Token 4 → slot 3 not touched.
    T = 4
    k = torch.randn(T, H_kv, D, device="cuda", dtype=torch.bfloat16) * 0.5
    v = torch.randn_like(k)
    slot_mapping = torch.tensor([0, 1, 2, -1], dtype=torch.int64, device="cuda")
    triton_reshape_and_cache_flash_kivi(
        k, v, view, slot_mapping, group_size=inp["group_size"],
    )
    # Slot 3 should still be 0xCC / 0xDD because the corresponding token had slot=-1.
    assert (view.k_data[0, 3] == 0xCC).all(), "padding token wrote into slot 3"
    assert (view.v_data[0, 3] == 0xDD).all(), "padding token wrote into slot 3 (V)"


def test_write_then_read_matches_unpack_then_attn():
    """End-to-end: write via D.5 kernel, read via D.3 attention kernel,
    compare against unpack-then-eager-SDPA on the same tokens."""
    inp = _setup(num_tokens=32)
    T = inp["num_tokens"]
    H_kv = inp["num_kv_heads"]
    D = inp["head_size"]
    block_size = inp["block_size"]
    group_size = inp["group_size"]
    view = inp["view"]

    torch.manual_seed(7)
    num_q_heads = H_kv * 2
    queries_per_kv = num_q_heads // H_kv
    Tq = 4
    softmax_scale = 1.0 / (D ** 0.5)

    # Source K, V for all 32 tokens; query for the last 4 only.
    k_src = torch.randn(T, H_kv, D, device="cuda", dtype=torch.bfloat16) * 0.5
    v_src = torch.randn(T, H_kv, D, device="cuda", dtype=torch.bfloat16) * 0.5
    q = torch.randn(Tq, num_q_heads, D, device="cuda", dtype=torch.bfloat16) * 0.5

    # Write all 32 tokens into the cache via the new kernel.
    slot_mapping = torch.arange(T, dtype=torch.int64, device="cuda")
    triton_reshape_and_cache_flash_kivi(
        k_src, v_src, view, slot_mapping, group_size=group_size,
    )

    # Run attention.
    block_table = torch.full(
        (1, inp["num_blocks"]), -1, dtype=torch.int32, device="cuda"
    )
    for bid in range(inp["num_blocks"]):
        block_table[0, bid] = bid
    seq_lens = torch.tensor([T], dtype=torch.int32, device="cuda")
    qsl = torch.tensor([0, Tq], dtype=torch.int32, device="cuda")
    output = torch.zeros_like(q)
    unified_attention_2d_kivi(
        output=output, query=q,
        k_data=view.k_data, k_scale=view.k_scale, k_zp=view.k_zp,
        v_data=view.v_data, v_scale=view.v_scale, v_zp=view.v_zp,
        block_table=block_table, seq_lens=seq_lens, query_start_len=qsl,
        softmax_scale=softmax_scale,
        num_query_heads=num_q_heads, num_kv_heads=H_kv,
        head_size=D, block_size=block_size, group_size=group_size,
        tile_size=16, block_q=4, block_m=queries_per_kv * 4,
    )

    # Reference path: pack on host → unpack → eager SDPA.
    kd, ks, kz = pack_int4_per_group(k_src, group_size=group_size)
    vd, vs, vz = pack_int4_per_group(v_src, group_size=group_size)
    k_ref = unpack_int4_per_group(kd, ks, kz, group_size=group_size, out_dtype=torch.bfloat16)
    v_ref = unpack_int4_per_group(vd, vs, vz, group_size=group_size, out_dtype=torch.bfloat16)

    qf, kf, vf = q.float(), k_ref.float(), v_ref.float()
    out_ref = torch.empty_like(qf)
    ctx = T - Tq
    for h in range(num_q_heads):
        kv_h = h // queries_per_kv
        scores = (qf[:, h, :] @ kf[:, kv_h, :].T) * softmax_scale
        for t in range(Tq):
            scores[t, ctx + t + 1:] = float("-inf")
        out_ref[:, h, :] = torch.softmax(scores, dim=-1) @ vf[:, kv_h, :]

    diff = (output.float() - out_ref).abs()
    rel = diff.max().item() / max(out_ref.abs().max().item(), 1e-9)
    assert rel < 5e-2, (
        f"write→read rel diff {rel:.4e} (abs {diff.max().item():.4e}) "
        f"exceeds 5%; kernel likely diverging from pack reference"
    )
