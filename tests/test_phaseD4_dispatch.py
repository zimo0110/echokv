"""Phase D.4: dispatcher + cache-layout helper tests.

Validates:
  1. ``kivi_page_size_bytes`` matches ``GroupQuantizedAttentionSpec``'s
     ``real_page_size_bytes`` for the same config (single source of truth).
  2. ``slice_kivi_cache`` produces views with the correct shapes, dtypes,
     and write-through semantics.
  3. End-to-end: pack K/V into a sliced flat buffer, run
     ``unified_attention_2d_kivi`` through the slicer, compare against
     unpack-then-eager-SDPA reference.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("Phase D.4 tests require CUDA", allow_module_level=True)

_VLLM_FORK = Path("/mnt/nvme_tmp10/nazar_vllm_fork")
if not _VLLM_FORK.is_dir():
    pytest.skip(f"vLLM fork not found at {_VLLM_FORK}",
                allow_module_level=True)
sys.path.insert(0, str(_VLLM_FORK))

from vllm.v1.attention.ops.kivi_cache_layout import (              # noqa: E402
    is_kivi_dtype,
    kivi_bits,
    kivi_page_size_bytes,
    slice_kivi_cache,
)
from vllm.v1.attention.ops.triton_unified_attention_kivi import (  # noqa: E402
    unified_attention_2d_kivi,
)
from vllm.v1.kv_cache_interface import GroupQuantizedAttentionSpec  # noqa: E402

from kernels.native_int4_kv import (                                # noqa: E402
    pack_int4_per_group,
    unpack_int4_per_group,
)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype,expected_bits", [
    ("int2_kivi", 2), ("int4_kivi", 4), ("int8_kivi", 8), ("nvfp4_e2m1", 4),
])
def test_is_kivi_and_bits(dtype, expected_bits):
    assert is_kivi_dtype(dtype)
    assert kivi_bits(dtype) == expected_bits


@pytest.mark.parametrize("dtype", ["fp8", "fp8_e4m3", "auto", "bfloat16"])
def test_non_kivi_rejected(dtype):
    assert not is_kivi_dtype(dtype)


def test_page_size_matches_spec():
    """The paged page-size helper must agree with
    ``GroupQuantizedAttentionSpec.real_page_size_bytes`` — they're the
    single source of truth for the *paged* cache layout.

    Phase F.2: ``real_page_size_bytes`` returns data-only bytes, so we
    compare against ``kivi_data_page_size_bytes`` (the helper for that
    same data-only page). The combined ``kivi_page_size_bytes`` is now
    a legacy single-buffer helper used only by kernel unit tests."""
    from vllm.v1.attention.ops.kivi_cache_layout import (
        kivi_data_page_size_bytes,
    )
    common = dict(block_size=16, num_kv_heads=8, head_size=128)
    for bits in (2, 4, 8):
        for group_size in (16, 32, 64):
            spec = GroupQuantizedAttentionSpec(
                **common, dtype=torch.bfloat16,
                bits=bits, group_size=group_size,
                quantize_v=True,
            )
            helper_bytes = kivi_data_page_size_bytes(
                bits=bits, group_size=group_size,
                quantize_v=True, **common,
            )
            assert helper_bytes == spec.real_page_size_bytes, (
                f"mismatch at bits={bits} group_size={group_size}: "
                f"helper={helper_bytes}, spec={spec.real_page_size_bytes}"
            )


def test_slice_shapes():
    """slice_kivi_cache produces views with the documented contract."""
    num_blocks, block_size, num_kv_heads, head_size = 4, 16, 8, 128
    bits, group_size = 4, 32
    page_bytes = kivi_page_size_bytes(
        block_size=block_size, num_kv_heads=num_kv_heads,
        head_size=head_size, bits=bits, group_size=group_size,
    )
    buf = torch.zeros(num_blocks, page_bytes, dtype=torch.uint8, device="cuda")
    view = slice_kivi_cache(
        buf,
        num_blocks=num_blocks, block_size=block_size,
        num_kv_heads=num_kv_heads, head_size=head_size,
        bits=bits, group_size=group_size,
    )
    half = head_size // 2
    n_groups = head_size // group_size
    assert view.k_data.shape  == (num_blocks, block_size, num_kv_heads, half)
    assert view.v_data.shape  == (num_blocks, block_size, num_kv_heads, half)
    assert view.k_scale.shape == (num_blocks, block_size, num_kv_heads, n_groups)
    assert view.v_scale.shape == (num_blocks, block_size, num_kv_heads, n_groups)
    assert view.k_zp.shape    == (num_blocks, block_size, num_kv_heads, n_groups)
    assert view.v_zp.shape    == (num_blocks, block_size, num_kv_heads, n_groups)
    assert view.k_data.dtype  == torch.uint8
    assert view.k_scale.dtype == torch.float16
    assert view.k_zp.dtype    == torch.uint8


def test_slice_write_through():
    """Writes to view tensors must land in the underlying buffer (zero-copy).

    Layout is section-major: K_data section spans the first
    ``num_blocks * block_size * num_kv_heads * (head_size//2)`` bytes,
    then K_scale section, then K_zp, then V_data, V_scale, V_zp.
    """
    num_blocks, block_size, num_kv_heads, head_size = 2, 16, 4, 64
    bits, group_size = 4, 32
    page_bytes = kivi_page_size_bytes(
        block_size=block_size, num_kv_heads=num_kv_heads,
        head_size=head_size, bits=bits, group_size=group_size,
    )
    buf = torch.zeros(num_blocks, page_bytes, dtype=torch.uint8, device="cuda")
    flat = buf.view(-1)
    view = slice_kivi_cache(
        buf,
        num_blocks=num_blocks, block_size=block_size,
        num_kv_heads=num_kv_heads, head_size=head_size,
        bits=bits, group_size=group_size,
    )
    # Write a known byte through k_data and verify it lands at the
    # start of the K_data section (offset 0 in flat).
    view.k_data[0, 0, 0, 0] = 0xAB
    assert int(flat[0]) == 0xAB

    # Section sizes (across all blocks):
    n_groups = head_size // group_size
    SEC_DATA  = num_blocks * block_size * num_kv_heads * (head_size // 2)
    SEC_SCALE = num_blocks * block_size * num_kv_heads * n_groups * 2

    # Write a known fp16 scale at K_scale[0,0,0,0] → bytes [0x00, 0x38] at SEC_DATA.
    view.k_scale[0, 0, 0, 0] = 0.5
    assert int(flat[SEC_DATA])     == 0x00
    assert int(flat[SEC_DATA + 1]) == 0x38

    # All views must be C-contiguous (the whole point of section-major).
    for t in (view.k_data, view.k_scale, view.k_zp,
              view.v_data, view.v_scale, view.v_zp):
        assert t.is_contiguous(), f"{t.shape} should be contiguous"


def test_slice_layout_disjoint():
    """The six views together cover the full buffer with no overlap."""
    num_blocks, block_size, num_kv_heads, head_size = 1, 16, 4, 64
    bits, group_size = 4, 32
    page_bytes = kivi_page_size_bytes(
        block_size=block_size, num_kv_heads=num_kv_heads,
        head_size=head_size, bits=bits, group_size=group_size,
    )
    buf = torch.zeros(num_blocks, page_bytes, dtype=torch.uint8, device="cuda")
    view = slice_kivi_cache(
        buf,
        num_blocks=num_blocks, block_size=block_size,
        num_kv_heads=num_kv_heads, head_size=head_size,
        bits=bits, group_size=group_size,
    )
    # Fill each view with a distinctive byte and verify the union is the whole buffer.
    view.k_data.fill_(0x01)
    view.k_scale.view(torch.uint8).fill_(0x02)
    view.k_zp.fill_(0x03)
    view.v_data.fill_(0x04)
    view.v_scale.view(torch.uint8).fill_(0x05)
    view.v_zp.fill_(0x06)
    # No 0x00 left → all bytes covered.
    assert int(buf.min()) >= 0x01
    assert int(buf.max()) <= 0x06
    # Each marker should be present.
    for marker in (1, 2, 3, 4, 5, 6):
        assert (buf == marker).any(), f"marker 0x0{marker} missing"


# ---------------------------------------------------------------------------
# End-to-end: slicing -> kernel
# ---------------------------------------------------------------------------

def _ref_sdpa(q, k, v, scale):
    T, Hq, D = q.shape
    S, H_kv, _ = k.shape
    queries_per_kv = Hq // H_kv
    out = torch.empty_like(q, dtype=torch.float32)
    qf, kf, vf = q.float(), k.float(), v.float()
    ctx = S - T
    for h in range(Hq):
        kv_h = h // queries_per_kv
        scores = (qf[:, h, :] @ kf[:, kv_h, :].T) * scale
        for t in range(T):
            scores[t, ctx + t + 1:] = float("-inf")
        attn = torch.softmax(scores, dim=-1)
        out[:, h, :] = attn @ vf[:, kv_h, :]
    return out.to(q.dtype)


def test_end_to_end_slice_then_kernel():
    """Allocate a flat byte buffer, slice it, populate via Phase B pack,
    run the new kernel, compare against eager SDPA on the unpacked
    cache. Same parity bound as Phase D.2 (~5% relative)."""
    torch.manual_seed(0)
    T, num_q_heads, num_kv_heads, head_size = 4, 4, 2, 64
    S = 32
    block_size = 16
    bits, group_size = 4, 32
    queries_per_kv = num_q_heads // num_kv_heads
    softmax_scale = 1.0 / (head_size ** 0.5)
    dtype = torch.bfloat16
    num_blocks = (S + block_size - 1) // block_size

    q = torch.randn(T, num_q_heads, head_size, device="cuda", dtype=dtype) * 0.5
    k = torch.randn(S, num_kv_heads, head_size, device="cuda", dtype=dtype) * 0.5
    v = torch.randn(S, num_kv_heads, head_size, device="cuda", dtype=dtype) * 0.5

    # Allocate flat byte buffer (this is what the vLLM allocator hands us
    # for kv_cache_dtype="int4_kivi" after Phase D.4's get_kv_cache_shape).
    page_bytes = kivi_page_size_bytes(
        block_size=block_size, num_kv_heads=num_kv_heads,
        head_size=head_size, bits=bits, group_size=group_size,
    )
    buf = torch.zeros(num_blocks, page_bytes, dtype=torch.uint8, device="cuda")
    view = slice_kivi_cache(
        buf,
        num_blocks=num_blocks, block_size=block_size,
        num_kv_heads=num_kv_heads, head_size=head_size,
        bits=bits, group_size=group_size,
    )

    # Populate the cache via Phase B pack, slot-by-slot.
    kd, ks, kz = pack_int4_per_group(k, group_size=group_size)
    vd, vs, vz = pack_int4_per_group(v, group_size=group_size)
    block_table = torch.full(
        (1, num_blocks), -1, dtype=torch.int32, device="cuda"
    )
    for s in range(S):
        bid, slot = s // block_size, s % block_size
        block_table[0, bid] = bid
        view.k_data[bid, slot]  = kd[s]
        view.k_scale[bid, slot] = ks[s]
        view.k_zp[bid, slot]    = kz[s]
        view.v_data[bid, slot]  = vd[s]
        view.v_scale[bid, slot] = vs[s]
        view.v_zp[bid, slot]    = vz[s]

    seq_lens = torch.tensor([S], dtype=torch.int32, device="cuda")
    query_start_len = torch.tensor([0, T], dtype=torch.int32, device="cuda")

    output = torch.zeros_like(q)
    unified_attention_2d_kivi(
        output=output, query=q,
        k_data=view.k_data, k_scale=view.k_scale, k_zp=view.k_zp,
        v_data=view.v_data, v_scale=view.v_scale, v_zp=view.v_zp,
        block_table=block_table, seq_lens=seq_lens,
        query_start_len=query_start_len,
        softmax_scale=softmax_scale,
        num_query_heads=num_q_heads, num_kv_heads=num_kv_heads,
        head_size=head_size, block_size=block_size, group_size=group_size,
        tile_size=16, block_q=4, block_m=queries_per_kv * 4,
    )

    # Reference path: unpack-then-eager-SDPA.
    k_ref = unpack_int4_per_group(kd, ks, kz, group_size=group_size, out_dtype=dtype)
    v_ref = unpack_int4_per_group(vd, vs, vz, group_size=group_size, out_dtype=dtype)
    ref = _ref_sdpa(q, k_ref, v_ref, softmax_scale)

    diff = (output.float() - ref.float()).abs()
    rel = diff.max().item() / max(ref.float().abs().max().item(), 1e-9)
    assert rel < 5e-2, f"slice→kernel rel {rel:.4e} exceeds 5%"
