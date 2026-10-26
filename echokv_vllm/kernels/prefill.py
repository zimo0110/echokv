"""Fused prefill (chunked-prefill) attention over a TieredKVCache.

A prefill chunk's queries attend to the request's whole cached context,
which the write path has already stored, this chunk included. The Triton
kernel reads K/V straight from the tiered pages and dequantizes tile by
tile in registers, so no full-precision copy of the context is
materialized (the PyTorch path gathers the context and calls SDPA).

Work is split into query tiles: up to ``BLOCK_M`` consecutive query tokens
of one request, times the GQA group of query heads that share a KV head,
padded to 64 rows. One program per (tile, KV head). Each row has its own
causal limit (key position <= query position), so a chunk anywhere inside
a long prompt is handled. ``build_tiles`` turns the step's prefill list
into the tile table, once per step.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right

from echokv_vllm.kernels.decode import HAS_TRITON, _use_triton
from echokv_vllm.tiered_cache import TieredKVCache

if HAS_TRITON:
    import triton
    import triton.language as tl

ROWS = 64  # query rows per program (tokens x padded GQA group)


def gqa_pad(gqa: int) -> int:
    return 1 << (gqa - 1).bit_length()


def block_m(gqa: int) -> int:
    return max(1, ROWS // gqa_pad(gqa))


def build_tiles(prefill: list[tuple[int, int, int, int]], req_of: list[int], gqa: int, device) -> torch.Tensor:
    """Tile table ``[n, 5]`` int32: (q_row0, n_tokens, q_pos0, ctx, request).

    ``prefill`` holds (q_start, q_end, ctx, _) per prefill request, ``req_of``
    the matching request index (row of the block table).
    """
    bm = block_m(gqa)
    rows = []
    for (qs, qe, ctx, _), r in zip(prefill, req_of):
        nq = qe - qs
        pos0 = ctx - nq
        for t0 in range(0, nq, bm):
            rows.append((qs + t0, min(bm, nq - t0), pos0 + t0, ctx, r))
    return torch.tensor(rows, dtype=torch.int32).reshape(-1, 5).to(device)


def prefill_attention(
    store: TieredKVCache,
    query: torch.Tensor,
    output: torch.Tensor,
    prefill: list[tuple[int, int, int, int]],
    req_of: list[int],
    block_table: torch.Tensor,
    softmax_scale: float,
    tiles: torch.Tensor | None = None,
    gather: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """Write attention for every prefill chunk into ``output [T, Hq, D]``.

    ``gather`` is the (blk, off) index over all prefill contexts, used only
    by the PyTorch path.
    """
    if not prefill:
        return
    d = query.shape[-1]
    if _use_triton(query.device) and d & (d - 1) == 0 and store.layout.block_size % 32 == 0:
        gqa = query.shape[1] // store.layout.num_kv_heads
        if tiles is None:
            tiles = build_tiles(prefill, req_of, gqa, query.device)
        _prefill_triton(store, query, output, tiles, block_table, softmax_scale)
        return
    prefill_attention_torch(store, query, output, prefill, block_table, softmax_scale, gather, req_of)


def prefill_attention_torch(store, query, output, prefill, block_table, softmax_scale, gather=None, req_of=None):
    """Reference: gather each context, dequantize, SDPA with a bottom-right causal mask."""
    bs = store.layout.block_size
    if gather is None:
        # Build the index here; contexts are then laid out back to back.
        blks, offs, starts, g = [], [], [], 0
        for (_, _, ctx, _), r in zip(prefill, req_of):
            pos = torch.arange(ctx, device=query.device)
            blks.append(block_table[r, pos // bs].long())
            offs.append(pos % bs)
            starts.append(g)
            g += ctx
        gather = (torch.cat(blks), torch.cat(offs))
        prefill = [(qs, qe, ctx, s) for (qs, qe, ctx, _), s in zip(prefill, starts)]
    k_all, v_all = store.gather(*gather)
    k_all, v_all = k_all.to(query.dtype), v_all.to(query.dtype)
    for qs, qe, ctx, g in prefill:
        nq = qe - qs
        out = F.scaled_dot_product_attention(
            query[qs:qe].transpose(0, 1).unsqueeze(0),
            k_all[g : g + ctx].transpose(0, 1).unsqueeze(0),
            v_all[g : g + ctx].transpose(0, 1).unsqueeze(0),
            attn_mask=causal_lower_right(nq, ctx),
            scale=softmax_scale,
            enable_gqa=True,
        )
        output[qs:qe] = out.squeeze(0).transpose(0, 1).to(output.dtype)


if HAS_TRITON:

    @triton.jit
    def _prefill_kernel(
        q_ptr, q_s0, q_s1,
        o_ptr, o_s0, o_s1,
        tiles_ptr,
        bt_ptr, bt_s0,
        tag_ptr, tag_s0,
        table_ptr, table_s0,
        data_ptr, data_s0, data_s1,
        ks_ptr, ks_s0, ks_s1, ks_s2,
        km_ptr,
        vs_ptr, vs_s0, vs_s1, vs_s2,
        vm_ptr,
        res_ptr, res_s0, res_s1, res_s2,
        block_res_ptr, res_group_ptr,
        softmax_scale,
        H: tl.constexpr,
        GQA: tl.constexpr,
        GQA_PAD: tl.constexpr,
        BLOCK_M: tl.constexpr,
        D: tl.constexpr,
        BS: tl.constexpr,
        KG: tl.constexpr,
        VG: tl.constexpr,
        KC: tl.constexpr,
        TILE: tl.constexpr,
    ):
        tile = tl.program_id(0)
        h = tl.program_id(1)
        q_row0 = tl.load(tiles_ptr + tile * 5 + 0)
        n_tok = tl.load(tiles_ptr + tile * 5 + 1)
        q_pos0 = tl.load(tiles_ptr + tile * 5 + 2)
        ctx = tl.load(tiles_ptr + tile * 5 + 3)
        req = tl.load(tiles_ptr + tile * 5 + 4)

        M: tl.constexpr = BLOCK_M * GQA_PAD
        offs_m = tl.arange(0, M)
        tok = offs_m // GQA_PAD
        grp_head = offs_m % GQA_PAD
        row_ok = (tok < n_tok) & (grp_head < GQA)
        head = h * GQA + grp_head
        qpos = q_pos0 + tok
        offs_d = tl.arange(0, D)
        q = tl.load(
            q_ptr + (q_row0 + tok)[:, None].to(tl.int64) * q_s0 + head[:, None] * q_s1 + offs_d[None, :],
            mask=row_ok[:, None],
            other=0.0,
        ).to(tl.float16)

        m_i = tl.full([M], float("-inf"), tl.float32)
        l_i = tl.zeros([M], tl.float32)
        acc = tl.zeros([M, D], tl.float32)

        kv_end = tl.minimum(ctx, q_pos0 + n_tok)  # keys beyond the last query are masked anyway
        for bi in range(0, tl.cdiv(kv_end, BS)):
            blk = tl.load(bt_ptr + req.to(tl.int64) * bt_s0 + bi).to(tl.int64)
            bits = tl.load(tag_ptr + blk * tag_s0).to(tl.int32)
            bits = tl.where(bits == 0, 2, bits)  # never-written block: stay in bounds
            tpc = (2 * BS) // bits
            side_bytes = (H * D * bits) // 8
            head_off = h * ((D * bits) // 8)
            byte_d = (offs_d * bits) // 8
            shift_d = (offs_d * bits) % 8
            mask_val = (1 << bits) - 1
            rslot = tl.load(block_res_ptr + blk)
            rgrp = tl.load(res_group_ptr + tl.maximum(rslot, 0))

            for t0 in tl.static_range(0, BS, TILE):
                offs_t = t0 + tl.arange(0, TILE)
                kpos = bi * BS + offs_t
                valid = kpos < kv_end

                j = offs_t // tpc
                ext = (j > 0) & valid
                ids = tl.load(table_ptr + blk * table_s0 + (j - 1), mask=ext, other=0).to(tl.int64)
                page = tl.where(ext, ids // KC, blk)
                slot = tl.where(ext, ids % KC, 0)
                row = page * data_s0 + slot * data_s1 + ((offs_t % tpc) * (2 * side_bytes)).to(tl.int64)

                k_raw = tl.load(
                    data_ptr + row[:, None] + head_off + byte_d[None, :], mask=valid[:, None], other=0
                ).to(tl.int32)
                k_code = (k_raw >> shift_d[None, :]) & mask_val
                grp = offs_t // KG
                ks_off = blk * ks_s0 + grp[:, None] * ks_s1 + h * ks_s2 + offs_d[None, :]
                k_scale = tl.load(ks_ptr + ks_off, mask=valid[:, None], other=0.0).to(tl.float32)
                k_min = tl.load(km_ptr + ks_off, mask=valid[:, None], other=0.0).to(tl.float32)
                k = k_code.to(tl.float32) * k_scale + k_min
                in_res = valid & (rslot >= 0) & (grp == rgrp)
                k_res = tl.load(
                    res_ptr + tl.maximum(rslot, 0) * res_s0 + (offs_t % KG)[:, None] * res_s1
                    + h * res_s2 + offs_d[None, :],
                    mask=in_res[:, None],
                    other=0.0,
                ).to(tl.float32)
                k = tl.where(in_res[:, None], k_res, k)

                v_raw = tl.load(
                    data_ptr + row[:, None] + side_bytes + head_off + byte_d[None, :],
                    mask=valid[:, None],
                    other=0,
                ).to(tl.int32)
                v_code = (v_raw >> shift_d[None, :]) & mask_val
                vs_off = blk * vs_s0 + offs_t[:, None] * vs_s1 + h * vs_s2 + (offs_d // VG)[None, :]
                v_scale = tl.load(vs_ptr + vs_off, mask=valid[:, None], other=0.0).to(tl.float32)
                v_min = tl.load(vm_ptr + vs_off, mask=valid[:, None], other=0.0).to(tl.float32)
                v = v_code.to(tl.float32) * v_scale + v_min

                s = tl.dot(q, tl.trans(k.to(tl.float16))) * softmax_scale
                causal = valid[None, :] & (kpos[None, :] <= qpos[:, None])
                s = tl.where(causal, s, float("-inf"))
                m_new = tl.maximum(m_i, tl.max(s, axis=1))
                m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
                alpha = tl.exp(m_i - m_safe)
                p = tl.exp(s - m_safe[:, None])
                l_i = l_i * alpha + tl.sum(p, axis=1)
                acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v.to(tl.float16))
                m_i = m_new

        out = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
        tl.store(
            o_ptr + (q_row0 + tok)[:, None].to(tl.int64) * o_s0 + head[:, None] * o_s1 + offs_d[None, :],
            out.to(o_ptr.dtype.element_ty),
            mask=row_ok[:, None],
        )


def _prefill_triton(store, query, output, tiles, block_table, softmax_scale):
    L = store.layout
    h = L.num_kv_heads
    gqa = query.shape[1] // h
    res = store.res_k
    if tiles.numel() == 0:
        return
    _prefill_kernel[(tiles.shape[0], h)](
        query, query.stride(0), query.stride(1),
        output, output.stride(0), output.stride(1),
        tiles,
        block_table, block_table.stride(0),
        store.tag, store.tag.stride(0),
        store.table, store.table.stride(0),
        store.data, store.data.stride(0), store.data.stride(1),
        store.k_scale, store.k_scale.stride(0), store.k_scale.stride(1), store.k_scale.stride(2),
        store.k_min,
        store.v_scale, store.v_scale.stride(0), store.v_scale.stride(1), store.v_scale.stride(2),
        store.v_min,
        res, res.stride(0), res.stride(1), res.stride(2),
        store.block_res, store.res_group,
        softmax_scale,
        H=h,
        GQA=gqa,
        GQA_PAD=gqa_pad(gqa),
        BLOCK_M=block_m(gqa),
        D=L.head_size,
        BS=L.block_size,
        KG=L.k_group,
        VG=L.v_group,
        KC=L.chunks_per_page,
        TILE=min(32, L.block_size),
        num_stages=1,
    )
