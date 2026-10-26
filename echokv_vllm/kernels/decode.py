"""Paged decode attention over a TieredKVCache (one query token per sequence).

``decode_attention`` is the entry point. It dispatches to the Triton kernel
when Triton is usable for the query's device (CUDA, or anything under
``TRITON_INTERPRET=1``) and to a batched PyTorch implementation otherwise.
The PyTorch path is also the oracle the kernel is tested against.

Kernel layout: one program per (sequence, KV head, split). It owns the KV
head's GQA group of query heads and walks its share of the sequence's
blocks with an online softmax (flash-decoding). Partial results are merged
across splits in PyTorch. Each block's tier is read from its page header at
runtime and drives all addressing (chunk, row, byte, bit shift), so INT2,
INT4 and INT8 blocks share one code path. K for a block's open group is
taken from the full-precision residual buffer.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from echokv_vllm.tiered_cache import TieredKVCache

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # macOS, CPU-only installs
    HAS_TRITON = False


def _use_triton(device: torch.device) -> bool:
    if not HAS_TRITON or os.environ.get("ECHOKV_DISABLE_TRITON") == "1":
        return False
    return device.type == "cuda" or os.environ.get("TRITON_INTERPRET") == "1"


BLOCKS_PER_SPLIT = 16


def decode_attention(
    store: TieredKVCache,
    query: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    softmax_scale: float,
    blocks_per_split: int = BLOCKS_PER_SPLIT,
) -> torch.Tensor:
    """Attention of ``query [S, Hq, D]`` (one token per sequence) over each sequence's cache.

    ``block_table [S, >=ceil(L/bs)]`` and ``seq_lens [S]`` (cache length,
    including the query token's own K/V) follow vLLM's conventions.
    Returns float32 ``[S, Hq, D]``.

    Batch-invariant: a sequence's result does not depend on which other
    sequences share the batch. The kernel splits each sequence at fixed
    ``blocks_per_split`` boundaries (empty trailing splits add exact zeros);
    the PyTorch path reduces each sequence on its own.
    """
    if _use_triton(query.device):
        return _decode_triton(store, query, block_table, seq_lens, softmax_scale, blocks_per_split)
    return decode_attention_torch(store, query, block_table, seq_lens, softmax_scale)


def decode_attention_torch(
    store: TieredKVCache,
    query: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Reference: one gather for every sequence, then attention per sequence.

    The gather is batched; the reductions are per sequence (unpadded), so
    results do not depend on the rest of the batch.
    """
    s, hq, d = query.shape
    bs = store.layout.block_size
    h = store.layout.num_kv_heads
    lens = seq_lens.to(query.device).long()
    lens_l = lens.tolist()
    max_len = max(lens_l)
    pos = torch.arange(max_len, device=query.device)
    valid = pos.unsqueeze(0) < lens.unsqueeze(1)  # [S, maxL]
    blocks = block_table[:, : -(-max_len // bs)].long()
    blk = torch.gather(blocks, 1, (pos // bs).unsqueeze(0).expand(s, -1))
    off = (pos % bs).unsqueeze(0).expand(s, -1)
    kv, vv = store.gather(blk[valid], off[valid])
    rep = hq // h
    out = torch.empty(s, hq, d, device=query.device)
    start = 0
    for i, n in enumerate(lens_l):
        k = kv[start : start + n].repeat_interleave(rep, dim=1).transpose(0, 1)  # [Hq, n, D]
        v = vv[start : start + n].repeat_interleave(rep, dim=1).transpose(0, 1)
        out[i] = F.scaled_dot_product_attention(
            query[i].float().unsqueeze(1), k, v, scale=softmax_scale
        ).squeeze(1)
        start += n
    return out


if HAS_TRITON:

    @triton.jit
    def _decode_kernel(
        q_ptr, q_s0, q_s1,
        bt_ptr, bt_s0,
        lens_ptr,
        tag_ptr, tag_s0,
        table_ptr, table_s0,
        data_ptr, data_s0, data_s1,
        ks_ptr, ks_s0, ks_s1, ks_s2,
        km_ptr,
        vs_ptr, vs_s0, vs_s1, vs_s2,
        vm_ptr,
        res_ptr, res_s0, res_s1, res_s2,
        block_res_ptr, res_group_ptr,
        acc_ptr, ml_ptr,
        softmax_scale,
        num_splits,
        blocks_per_split,
        H: tl.constexpr,
        GQA: tl.constexpr,
        QBLOCK: tl.constexpr,
        D: tl.constexpr,
        BS: tl.constexpr,
        KG: tl.constexpr,
        VG: tl.constexpr,
        KC: tl.constexpr,
    ):
        seq = tl.program_id(0)
        h = tl.program_id(1)
        split = tl.program_id(2)

        seq_len = tl.load(lens_ptr + seq)
        n_blocks = tl.cdiv(seq_len, BS)
        b_start = split * blocks_per_split
        b_end = tl.minimum(b_start + blocks_per_split, n_blocks)

        offs_q = tl.arange(0, QBLOCK)
        offs_d = tl.arange(0, D)
        offs_t = tl.arange(0, BS)
        q_mask = offs_q < GQA
        q = tl.load(
            q_ptr + seq * q_s0 + (h * GQA + offs_q)[:, None] * q_s1 + offs_d[None, :],
            mask=q_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        m_i = tl.full([QBLOCK], float("-inf"), tl.float32)
        l_i = tl.zeros([QBLOCK], tl.float32)
        acc = tl.zeros([QBLOCK, D], tl.float32)

        for bi in range(b_start, b_end):
            blk = tl.load(bt_ptr + seq * bt_s0 + bi).to(tl.int64)
            bits = tl.load(tag_ptr + blk * tag_s0).to(tl.int32)
            # Never-written block (tag 0; vLLM's warmup/profiling batches read
            # these): address it as INT2 so every access stays inside its page.
            bits = tl.where(bits == 0, 2, bits)
            tok = bi * BS + offs_t
            valid = tok < seq_len

            # Where each token's row lives: chunk j of the block.
            tpc = (2 * BS) // bits
            j = offs_t // tpc
            ext = (j > 0) & valid
            ids = tl.load(table_ptr + blk * table_s0 + (j - 1), mask=ext, other=0).to(tl.int64)
            page = tl.where(ext, ids // KC, blk)
            slot = tl.where(ext, ids % KC, 0)
            side_bytes = (H * D * bits) // 8
            row = page * data_s0 + slot * data_s1 + ((offs_t % tpc) * (2 * side_bytes)).to(tl.int64)
            head_off = h * ((D * bits) // 8)
            byte_d = (offs_d * bits) // 8
            shift_d = (offs_d * bits) % 8
            mask_val = (1 << bits) - 1

            # K: per-channel stats of the token's group; open group from the residual.
            k_raw = tl.load(
                data_ptr + row[:, None] + head_off + byte_d[None, :], mask=valid[:, None], other=0
            ).to(tl.int32)
            k_code = (k_raw >> shift_d[None, :]) & mask_val
            grp = offs_t // KG
            ks_off = blk * ks_s0 + grp[:, None] * ks_s1 + h * ks_s2 + offs_d[None, :]
            k_scale = tl.load(ks_ptr + ks_off, mask=valid[:, None], other=0.0).to(tl.float32)
            k_min = tl.load(km_ptr + ks_off, mask=valid[:, None], other=0.0).to(tl.float32)
            k = k_code.to(tl.float32) * k_scale + k_min
            rslot = tl.load(block_res_ptr + blk)
            rgrp = tl.load(res_group_ptr + tl.maximum(rslot, 0))
            in_res = valid & (rslot >= 0) & (grp == rgrp)
            k_res = tl.load(
                res_ptr + tl.maximum(rslot, 0) * res_s0 + (offs_t % KG)[:, None] * res_s1
                + h * res_s2 + offs_d[None, :],
                mask=in_res[:, None],
                other=0.0,
            ).to(tl.float32)
            k = tl.where(in_res[:, None], k_res, k)

            # V: per-token stats along head_dim groups.
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

            s = tl.dot(q, tl.trans(k), input_precision="ieee") * softmax_scale
            s = tl.where(valid[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
            m_i = m_new

        out_row = (seq * (H * GQA) + h * GQA + offs_q) * num_splits + split
        tl.store(acc_ptr + out_row[:, None] * D + offs_d[None, :], acc, mask=q_mask[:, None])
        tl.store(ml_ptr + out_row * 2, m_i, mask=q_mask)
        tl.store(ml_ptr + out_row * 2 + 1, l_i, mask=q_mask)


def _decode_triton(store, query, block_table, seq_lens, softmax_scale, blocks_per_split):
    L = store.layout
    s, hq, d = query.shape
    h = L.num_kv_heads
    gqa = hq // h
    if d & (d - 1) or L.block_size & (L.block_size - 1):
        return decode_attention_torch(store, query, block_table, seq_lens, softmax_scale)
    # Fixed split boundaries per sequence: the grid covers the longest
    # possible sequence, and shorter sequences' extra splits stay empty.
    num_splits = max(1, -(-block_table.shape[1] // blocks_per_split))
    query = query.contiguous()
    lens = seq_lens.to(torch.int32).contiguous()
    acc = torch.empty(s, hq, num_splits, d, device=query.device, dtype=torch.float32)
    ml = torch.empty(s, hq, num_splits, 2, device=query.device, dtype=torch.float32)
    res = store.res_k
    _decode_kernel[(s, h, num_splits)](
        query, query.stride(0), query.stride(1),
        block_table, block_table.stride(0),
        lens,
        store.tag, store.tag.stride(0),
        store.table, store.table.stride(0),
        store.data, store.data.stride(0), store.data.stride(1),
        store.k_scale, store.k_scale.stride(0), store.k_scale.stride(1), store.k_scale.stride(2),
        store.k_min,
        store.v_scale, store.v_scale.stride(0), store.v_scale.stride(1), store.v_scale.stride(2),
        store.v_min,
        res, res.stride(0), res.stride(1), res.stride(2),
        store.block_res, store.res_group,
        acc, ml,
        softmax_scale,
        num_splits,
        blocks_per_split,
        H=h,
        GQA=gqa,
        QBLOCK=max(16, triton.next_power_of_2(gqa)),
        D=d,
        BS=L.block_size,
        KG=L.k_group,
        VG=L.v_group,
        KC=L.chunks_per_page,
    )
    m, l = ml[..., 0], ml[..., 1]
    m_max = m.max(dim=2, keepdim=True).values
    w = torch.exp(m - m_max)  # empty splits have m = -inf -> weight 0
    return (acc * w.unsqueeze(-1)).sum(2) / (l * w).sum(2, keepdim=True)
