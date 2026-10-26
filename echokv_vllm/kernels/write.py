"""Write kernels for TieredKVCache: per-token V quantization and K group finalize.

Both take each block's tier from its page header at runtime, so one launch
covers INT2, INT4 and INT8 blocks. Numerics mirror ``echokv_vllm.quant``
bit for bit: fp16-rounded stats, IEEE division, round-half-to-even.
``write_v`` / ``finalize_k`` dispatch to Triton when usable, else PyTorch.
"""

from __future__ import annotations

import torch

from echokv_vllm.kernels.decode import HAS_TRITON, _use_triton
from echokv_vllm.quant import SUPPORTED_BITS, quantize, quantize_with, stats_along

if HAS_TRITON:
    import triton
    import triton.language as tl

_FP16_TINY = float(torch.finfo(torch.float16).tiny)


def write_v(store, value: torch.Tensor, rows: torch.Tensor, slots: torch.Tensor) -> None:
    """Quantize ``value[rows]`` per token and store it at ``slots``."""
    if slots.numel() == 0:
        return
    if _use_triton(value.device) and _triton_shapes_ok(store):
        _write_v_triton(store, value, rows, slots)
    else:
        _write_v_torch(store, value, rows, slots)


def finalize_k(store, key: torch.Tensor, block, group, res_slot, fin_rows) -> None:
    """Quantize each completed K group per channel and store its codes and stats."""
    if block.numel() == 0:
        return
    if _use_triton(key.device) and _triton_shapes_ok(store):
        _finalize_k_triton(store, key, block, group, res_slot, fin_rows)
    else:
        _finalize_k_torch(store, key, block, group, res_slot, fin_rows)


def _triton_shapes_ok(store) -> bool:
    d = store.layout.head_size
    return d & (d - 1) == 0


# -- PyTorch -----------------------------------------------------------------


def _write_v_torch(store, value, rows, slots):
    L = store.layout
    blk, off = slots // L.block_size, slots % L.block_size
    tags = store.tag[blk]
    for bits in SUPPORTED_BITS:
        m = tags == bits
        if not bool(m.any()):
            continue
        codes, scale, mn = quantize(value[rows[m]], bits, L.v_group)
        store.v_scale[blk[m], off[m]] = scale
        store.v_min[blk[m], off[m]] = mn
        store._write_side(blk[m], off[m], bits, codes, side=1)


def _finalize_k_torch(store, key, block, group, res_slot, fin_rows):
    L = store.layout
    g = L.k_group
    f = block.numel()
    from_key = fin_rows >= 0
    full = torch.zeros(f, g, L.num_kv_heads, L.head_size, device=key.device)
    if bool((~from_key).any()):
        rs = res_slot.clamp_min(0).unsqueeze(1).expand(f, g)
        pos = torch.arange(g, device=key.device).unsqueeze(0).expand(f, g)
        full[~from_key] = store.res_k[rs[~from_key], pos[~from_key]].float()
    full[from_key] = key[fin_rows[from_key]].float()
    bits_all = store.tag[block]
    for bits in SUPPORTED_BITS:
        m = bits_all == bits
        if not bool(m.any()):
            continue
        kg = full[m]
        scale, mn = stats_along(kg, bits, dim=1)
        store.k_scale[block[m], group[m]] = scale
        store.k_min[block[m], group[m]] = mn
        codes = quantize_with(kg, scale.unsqueeze(1), mn.unsqueeze(1), bits)
        nb = int(m.sum())
        tb = block[m].repeat_interleave(g)
        to = (group[m].unsqueeze(1) * g + torch.arange(g, device=key.device)).reshape(-1)
        store._write_side(tb, to, bits, codes.reshape(nb * g, L.num_kv_heads, L.head_size), side=0)


# -- Triton ------------------------------------------------------------------
#
# The kernels take the layer's raw ``pages`` tensor once and derive typed
# pointers from byte offsets. Passing several overlapping views of the same
# storage would also work on a GPU, but this keeps one aliasing argument per
# launch (and keeps the Triton CPU interpreter, which copies each argument
# separately, faithful).

if HAS_TRITON:

    @triton.jit
    def _round_half_even(x):
        t = x + 0.5
        r = tl.floor(t)
        odd = (r - 2.0 * tl.floor(r * 0.5)) != 0.0
        return tl.where((t == r) & odd, r - 1.0, r)

    @triton.jit
    def _row_base(pages_ptr, PS, blk, off, bits,
                  H: tl.constexpr, D: tl.constexpr, BS: tl.constexpr, KC: tl.constexpr,
                  O_DATA: tl.constexpr, CB: tl.constexpr):
        """Byte address offset (from ``pages_ptr``) of token ``off``'s K+V row in ``blk``."""
        tpc = (2 * BS) // bits
        j = off // tpc
        ext = j > 0
        table = (pages_ptr + blk * PS + 4).to(tl.pointer_type(tl.int32))
        ids = tl.load(table + (j - 1), mask=ext, other=0).to(tl.int64)
        page = tl.where(ext, ids // KC, blk)
        slot = tl.where(ext, ids % KC, 0)
        return page * PS + O_DATA + slot * CB + ((off % tpc) * ((2 * H * D * bits) // 8)).to(tl.int64)

    @triton.jit
    def _write_v_kernel(
        v_ptr, v_s0, v_s1,
        rows_ptr, slots_ptr,
        pages_ptr, PS,
        tiny,
        H: tl.constexpr, D: tl.constexpr, BS: tl.constexpr, VG: tl.constexpr, KC: tl.constexpr,
        O_VS: tl.constexpr, O_VM: tl.constexpr, O_DATA: tl.constexpr, CB: tl.constexpr,
    ):
        i = tl.program_id(0)
        h = tl.program_id(1)
        row = tl.load(rows_ptr + i)
        slot_id = tl.load(slots_ptr + i)
        blk = slot_id // BS
        off = slot_id % BS
        bits = tl.load(pages_ptr + blk * PS).to(tl.int32)
        bits = tl.where(bits == 0, 2, bits)  # never-opened block: stay in bounds
        levels = ((1 << bits) - 1).to(tl.float32)

        x = tl.load(v_ptr + row * v_s0 + h * v_s1 + tl.arange(0, D)).to(tl.float32)
        xg = tl.reshape(x, [D // VG, VG])
        mn16 = tl.min(xg, axis=1).to(tl.float16)
        mn = mn16.to(tl.float32)
        scale16 = tl.maximum(tl.math.div_rn(tl.max(xg, axis=1) - mn, levels), tiny).to(tl.float16)
        scale = scale16.to(tl.float32)
        q = _round_half_even(tl.math.div_rn(xg - mn[:, None], scale[:, None]))
        q = tl.minimum(tl.maximum(q, 0.0), levels).to(tl.int32)
        q = tl.reshape(q, [D])

        elem = (off * H + h) * (D // VG) + tl.arange(0, D // VG)
        tl.store((pages_ptr + blk * PS + O_VS).to(tl.pointer_type(tl.float16)) + elem, scale16)
        tl.store((pages_ptr + blk * PS + O_VM).to(tl.pointer_type(tl.float16)) + elem, mn16)
        base = _row_base(pages_ptr, PS, blk, off, bits, H, D, BS, KC, O_DATA, CB)
        base += (H * D * bits) // 8 + h * ((D * bits) // 8)  # V half, then this head
        if bits == 8:
            tl.store(pages_ptr + base + tl.arange(0, D), q.to(tl.uint8))
        elif bits == 4:
            # Distinct names per branch: the compiler requires a name assigned in
            # several branches to have one type, and these shapes differ.
            c4 = tl.reshape(q, [D // 2, 2])
            sh4 = tl.arange(0, 2) * 4
            tl.store(pages_ptr + base + tl.arange(0, D // 2), tl.sum(c4 << sh4[None, :], axis=1).to(tl.uint8))
        else:
            c2 = tl.reshape(q, [D // 4, 4])
            sh2 = tl.arange(0, 4) * 2
            tl.store(pages_ptr + base + tl.arange(0, D // 4), tl.sum(c2 << sh2[None, :], axis=1).to(tl.uint8))

    @triton.jit
    def _finalize_k_kernel(
        k_ptr, k_s0, k_s1,
        blk_ptr, grp_ptr, rslot_ptr, frows_ptr,
        res_ptr, res_s0, res_s1, res_s2,
        pages_ptr, PS,
        tiny,
        H: tl.constexpr, D: tl.constexpr, BS: tl.constexpr, KG: tl.constexpr, KC: tl.constexpr,
        O_KS: tl.constexpr, O_KM: tl.constexpr, O_DATA: tl.constexpr, CB: tl.constexpr,
    ):
        f = tl.program_id(0)
        h = tl.program_id(1)
        blk = tl.load(blk_ptr + f)
        grp = tl.load(grp_ptr + f)
        rslot = tl.load(rslot_ptr + f)
        bits = tl.load(pages_ptr + blk * PS).to(tl.int32)
        bits = tl.where(bits == 0, 2, bits)  # never-opened block: stay in bounds
        levels = ((1 << bits) - 1).to(tl.float32)

        offs_p = tl.arange(0, KG)
        offs_d = tl.arange(0, D)
        rows = tl.load(frows_ptr + f * KG + offs_p)
        from_key = rows >= 0
        xk = tl.load(k_ptr + tl.maximum(rows, 0)[:, None] * k_s0 + h * k_s1 + offs_d[None, :],
                     mask=from_key[:, None], other=0.0).to(tl.float32)
        xr = tl.load(res_ptr + tl.maximum(rslot, 0) * res_s0 + offs_p[:, None] * res_s1 + h * res_s2
                     + offs_d[None, :], mask=(~from_key)[:, None], other=0.0).to(tl.float32)
        x = tl.where(from_key[:, None], xk, xr)  # [KG, D]

        mn16 = tl.min(x, axis=0).to(tl.float16)
        mn = mn16.to(tl.float32)
        scale16 = tl.maximum(tl.math.div_rn(tl.max(x, axis=0) - mn, levels), tiny).to(tl.float16)
        scale = scale16.to(tl.float32)
        q = _round_half_even(tl.math.div_rn(x - mn[None, :], scale[None, :]))
        q = tl.minimum(tl.maximum(q, 0.0), levels).to(tl.int32)

        elem = (grp * H + h) * D + offs_d
        tl.store((pages_ptr + blk * PS + O_KS).to(tl.pointer_type(tl.float16)) + elem, scale16)
        tl.store((pages_ptr + blk * PS + O_KM).to(tl.pointer_type(tl.float16)) + elem, mn16)
        off = grp * KG + offs_p
        base = _row_base(pages_ptr, PS, blk, off, bits, H, D, BS, KC, O_DATA, CB)
        base += h * ((D * bits) // 8)  # K half, this head
        if bits == 8:
            tl.store(pages_ptr + base[:, None] + offs_d[None, :], q.to(tl.uint8))
        elif bits == 4:
            c4 = tl.reshape(q, [KG, D // 2, 2])
            sh4 = tl.arange(0, 2) * 4
            packed4 = tl.sum(c4 << sh4[None, None, :], axis=2).to(tl.uint8)
            tl.store(pages_ptr + base[:, None] + tl.arange(0, D // 2)[None, :], packed4)
        else:
            c2 = tl.reshape(q, [KG, D // 4, 4])
            sh2 = tl.arange(0, 4) * 2
            packed2 = tl.sum(c2 << sh2[None, None, :], axis=2).to(tl.uint8)
            tl.store(pages_ptr + base[:, None] + tl.arange(0, D // 4)[None, :], packed2)


def _common(store):
    L = store.layout
    o = L.offsets
    return dict(
        H=L.num_kv_heads, D=L.head_size, BS=L.block_size, KC=L.chunks_per_page,
        O_DATA=o["data"], CB=L.chunk_bytes,
    )


def _write_v_triton(store, value, rows, slots):
    o = store.layout.offsets
    _write_v_kernel[(slots.numel(), store.layout.num_kv_heads)](
        value, value.stride(0), value.stride(1),
        rows.contiguous(), slots.contiguous(),
        store.pages, store.pages.stride(0),
        _FP16_TINY,
        VG=store.layout.v_group, O_VS=o["v_scale"], O_VM=o["v_min"],
        **_common(store),
    )


def _finalize_k_triton(store, key, block, group, res_slot, fin_rows):
    o = store.layout.offsets
    res = store.res_k
    _finalize_k_kernel[(block.numel(), store.layout.num_kv_heads)](
        key, key.stride(0), key.stride(1),
        block.contiguous(), group.contiguous(), res_slot.contiguous(), fin_rows.contiguous(),
        res, res.stride(0), res.stride(1), res.stride(2),
        store.pages, store.pages.stride(0),
        _FP16_TINY,
        KG=store.layout.k_group, O_KS=o["k_scale"], O_KM=o["k_min"],
        **_common(store),
    )
