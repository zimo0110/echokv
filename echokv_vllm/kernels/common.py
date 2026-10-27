"""Triton helpers shared by the read kernels (decode, fused prefill, dequantize).

Reading a tile of TILE tokens from one block of the tiered store:

* ``block_info`` reads the block's tier and residual mapping once per block.
* ``load_kv_tile`` dequantizes K and V for TILE tokens of that block to fp32.

``TILE`` must divide ``KG``, so a tile lies inside one K group. K's
per-channel scale/min are then one ``[D]`` row per tile (not a TILE x D
gather), and whether the tile is the block's open group (read from the
full-precision residual) is a per-tile scalar, so only one of codes or
residual is loaded. V's per-token stats are loaded as ``[TILE, D / VG]``
and broadcast over each group of VG elements.
"""

from __future__ import annotations

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # macOS, CPU-only installs
    HAS_TRITON = False

if HAS_TRITON:

    @triton.jit
    def block_info(blk, h, tag_ptr, tag_s0, block_res_ptr, res_group_ptr,
                   H: tl.constexpr, D: tl.constexpr, BS: tl.constexpr):
        bits = tl.load(tag_ptr + blk * tag_s0).to(tl.int32)
        # Never-written block (tag 0; vLLM's warmup/profiling batches read
        # these): address it as INT2 so every access stays inside its page.
        bits = tl.where(bits == 0, 2, bits)
        offs_d = tl.arange(0, D)
        rslot = tl.load(block_res_ptr + blk)
        rgrp = tl.load(res_group_ptr + tl.maximum(rslot, 0))
        return (
            bits,
            (2 * BS) // bits,  # tokens per chunk
            (H * D * bits) // 8,  # bytes of K (or V) codes per token row
            h * ((D * bits) // 8),  # this head's offset inside a side
            (offs_d * bits) // 8,  # byte of each channel
            (offs_d * bits) % 8,  # bit shift of each channel
            (1 << bits) - 1,  # code mask
            rslot,
            rgrp,
        )

    @triton.jit
    def load_kv_tile(
        blk, t0, n_valid, h,
        tpc, side_bytes, head_off, byte_d, shift_d, mask_val, rslot, rgrp,
        table_ptr, table_s0,
        data_ptr, data_s0, data_s1,
        ks_ptr, ks_s0, ks_s1, ks_s2, km_ptr,
        vs_ptr, vs_s0, vs_s1, vs_s2, vm_ptr,
        res_ptr, res_s0, res_s1, res_s2,
        D: tl.constexpr, KG: tl.constexpr, VG: tl.constexpr, KC: tl.constexpr, TILE: tl.constexpr,
    ):
        """K, V (fp32 ``[TILE, D]``) for tokens ``t0 .. t0 + TILE`` of block ``blk``.

        Tokens at offset >= ``n_valid`` are masked (returned as 0). Also
        returns the in-block offsets and the validity mask.
        """
        offs_t = t0 + tl.arange(0, TILE)
        offs_d = tl.arange(0, D)
        valid = offs_t < n_valid

        # Where each token's row lives: chunk j of the block.
        j = offs_t // tpc
        ext = (j > 0) & valid
        ids = tl.load(table_ptr + blk * table_s0 + (j - 1), mask=ext, other=0).to(tl.int64)
        page = tl.where(ext, ids // KC, blk)
        slot = tl.where(ext, ids % KC, 0)
        row = page * data_s0 + slot * data_s1 + ((offs_t % tpc) * (2 * side_bytes)).to(tl.int64)

        # K: one group per tile. Either its codes (+ per-channel stats) or,
        # for the block's open group, the full-precision residual.
        grp = t0 // KG
        in_res = (rslot >= 0) & (grp == rgrp)
        k_raw = tl.load(
            data_ptr + row[:, None] + head_off + byte_d[None, :],
            mask=valid[:, None] & (in_res == 0), other=0,
        ).to(tl.int32)
        k_code = (k_raw >> shift_d[None, :]) & mask_val
        st = blk * ks_s0 + grp * ks_s1 + h * ks_s2 + offs_d
        k_scale = tl.load(ks_ptr + st).to(tl.float32)
        k_min = tl.load(km_ptr + st).to(tl.float32)
        k = k_code.to(tl.float32) * k_scale[None, :] + k_min[None, :]
        k_res = tl.load(
            res_ptr + tl.maximum(rslot, 0) * res_s0 + (offs_t % KG)[:, None] * res_s1 + h * res_s2 + offs_d[None, :],
            mask=valid[:, None] & in_res, other=0.0,
        ).to(tl.float32)
        k = tl.where(in_res, k_res, k)

        # V: per-token stats, one per VG channels, broadcast over the group.
        v_raw = tl.load(
            data_ptr + row[:, None] + side_bytes + head_off + byte_d[None, :], mask=valid[:, None], other=0
        ).to(tl.int32)
        v_code = ((v_raw >> shift_d[None, :]) & mask_val).to(tl.float32)
        G: tl.constexpr = D // VG
        vs_off = blk * vs_s0 + offs_t[:, None] * vs_s1 + h * vs_s2 + tl.arange(0, G)[None, :]
        v_scale = tl.load(vs_ptr + vs_off, mask=valid[:, None], other=0.0).to(tl.float32)
        v_min = tl.load(vm_ptr + vs_off, mask=valid[:, None], other=0.0).to(tl.float32)
        v = tl.reshape(
            tl.reshape(v_code, [TILE, G, VG]) * v_scale[:, :, None] + v_min[:, :, None], [TILE, D]
        )
        return k, v, offs_t, valid
