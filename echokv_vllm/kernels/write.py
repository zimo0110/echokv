"""Write kernels for TieredKVCache: one launch per token batch, one per completed K group.

``write_tokens`` handles every token row of a step in one launch: V is
quantized per token and stored; K goes to the residual (open groups), is
quantized against stored stats (groups sealed by eviction), and is copied
to the layer's K_ref ring. ``finalize_k`` quantizes each completed K group
per channel. Both take each block's tier from its page header at runtime,
so one launch covers INT2, INT4 and INT8 blocks, and both read everything
else from device metadata (``WriteMeta``), so a CUDA graph can capture
them. Numerics mirror ``echokv_vllm.quant`` bit for bit: fp16-rounded
stats, IEEE division, round-half-to-even. Each dispatches to Triton when
usable, else PyTorch.
"""

from __future__ import annotations

import torch

from echokv_vllm.kernels.decode import HAS_TRITON, _use_triton
from echokv_vllm.quant import SUPPORTED_BITS, quantize, quantize_with, stats_along

if HAS_TRITON:
    import triton
    import triton.language as tl

_FP16_TINY = float(torch.finfo(torch.float16).tiny)


def write_tokens(store, key: torch.Tensor, value: torch.Tensor, meta, kref: torch.Tensor | None = None) -> None:
    """Store the ``meta.n_rows`` token rows of ``key``/``value`` as ``meta.tok`` says."""
    if meta.n_rows == 0:
        return
    if _use_triton(value.device) and _triton_shapes_ok(store):
        _write_tokens_triton(store, key, value, meta, kref)
    else:
        _write_tokens_torch(store, key, value, meta, kref)


def finalize_k(store, key: torch.Tensor, meta, full_grid: bool = False) -> None:
    """Quantize each completed K group per channel and store its codes and stats.

    ``full_grid`` launches one program per token row, masked by the device
    count, instead of one per group (the count is then not needed on the host).
    """
    if _use_triton(key.device) and _triton_shapes_ok(store):
        n = meta.n_rows if full_grid else meta.n_fin
        if n:
            _finalize_k_triton(store, key, meta, n)
    elif meta.n_fin:
        _finalize_k_torch(store, key, meta.fin[: meta.n_fin])


def _triton_shapes_ok(store) -> bool:
    d = store.layout.head_size
    return d & (d - 1) == 0


# -- PyTorch -----------------------------------------------------------------


def _write_tokens_torch(store, key, value, meta, kref):
    L = store.layout
    tok = meta.tok[:, : meta.n_rows]
    rows = torch.nonzero(tok[0] >= 0).squeeze(1)
    if rows.numel() == 0:
        return
    slot, res, sealed, kidx = (tok[i, rows] for i in range(4))
    blk, off = slot // L.block_size, slot % L.block_size
    tags = store.tag[blk]
    for bits in SUPPORTED_BITS:
        m = tags == bits
        if not bool(m.any()):
            continue
        codes, scale, mn = quantize(value[rows[m]], bits, L.v_group)
        store.v_scale[blk[m], off[m]] = scale
        store.v_min[blk[m], off[m]] = mn
        store._write_side(blk[m], off[m], bits, codes, side=1)
    m = res >= 0
    if bool(m.any()):
        store.res_k[res[m], off[m] % L.k_group] = key[rows[m]].to(store.res_k.dtype)
    m = sealed != 0
    if bool(m.any()):
        b_, o_, r_ = blk[m], off[m], rows[m]
        for bits in SUPPORTED_BITS:
            mb = store.tag[b_] == bits
            if bool(mb.any()):
                sc = store.k_scale[b_[mb], o_[mb] // L.k_group]
                mn = store.k_min[b_[mb], o_[mb] // L.k_group]
                store._write_side(b_[mb], o_[mb], bits, quantize_with(key[r_[mb]], sc, mn, bits), 0)
    if kref is not None:
        m = kidx >= 0
        if bool(m.any()):
            kref[kidx[m]] = key[rows[m]].to(kref.dtype)


def _finalize_k_torch(store, key, fin):
    L = store.layout
    g = L.k_group
    f = fin.shape[0]
    block, group, res_slot, fin_rows = fin[:, 0], fin[:, 1], fin[:, 2], fin[:, 3:]
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
    def _store_codes(pages_ptr, base, q, bits, D: tl.constexpr):
        """Pack one head's ``[D]`` int32 codes at ``bits`` and store them at byte ``base``."""
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
    def _write_tokens_kernel(
        k_ptr, k_s0, k_s1,
        v_ptr, v_s0, v_s1,
        tok_ptr, tok_s0,
        res_ptr, res_s0, res_s1, res_s2,
        kref_ptr, kref_s0, kref_s1,
        pages_ptr, PS,
        tiny,
        H: tl.constexpr, D: tl.constexpr, BS: tl.constexpr, VG: tl.constexpr, KG: tl.constexpr,
        KC: tl.constexpr, O_VS: tl.constexpr, O_VM: tl.constexpr, O_KS: tl.constexpr,
        O_KM: tl.constexpr, O_DATA: tl.constexpr, CB: tl.constexpr, HAS_KREF: tl.constexpr,
    ):
        t = tl.program_id(0)
        h = tl.program_id(1)
        slot_id = tl.load(tok_ptr + t)
        if slot_id < 0:  # padding, or a token vLLM does not store
            return
        blk = slot_id // BS
        off = slot_id % BS
        bits = tl.load(pages_ptr + blk * PS).to(tl.int32)
        bits = tl.where(bits == 0, 2, bits)  # never-opened block: stay in bounds
        levels = ((1 << bits) - 1).to(tl.float32)
        offs_d = tl.arange(0, D)
        row_base = _row_base(pages_ptr, PS, blk, off, bits, H, D, BS, KC, O_DATA, CB)

        # V: per token, groups of VG along head_dim.
        x = tl.load(v_ptr + t * v_s0 + h * v_s1 + offs_d).to(tl.float32)
        xg = tl.reshape(x, [D // VG, VG])
        mn16 = tl.min(xg, axis=1).to(tl.float16)
        mn = mn16.to(tl.float32)
        scale16 = tl.maximum(tl.math.div_rn(tl.max(xg, axis=1) - mn, levels), tiny).to(tl.float16)
        scale = scale16.to(tl.float32)
        q = _round_half_even(tl.math.div_rn(xg - mn[:, None], scale[:, None]))
        q = tl.minimum(tl.maximum(q, 0.0), levels).to(tl.int32)
        elem = (off * H + h) * (D // VG) + tl.arange(0, D // VG)
        tl.store((pages_ptr + blk * PS + O_VS).to(tl.pointer_type(tl.float16)) + elem, scale16)
        tl.store((pages_ptr + blk * PS + O_VM).to(tl.pointer_type(tl.float16)) + elem, mn16)
        _store_codes(pages_ptr, row_base + (H * D * bits) // 8 + h * ((D * bits) // 8),
                     tl.reshape(q, [D]), bits, D)

        # K: residual (open group), codes under stored stats (sealed group), K_ref.
        kx = tl.load(k_ptr + t * k_s0 + h * k_s1 + offs_d)
        rslot = tl.load(tok_ptr + tok_s0 + t)
        if rslot >= 0:
            tl.store(res_ptr + rslot * res_s0 + (off % KG) * res_s1 + h * res_s2 + offs_d,
                     kx.to(res_ptr.dtype.element_ty))
        sealed = tl.load(tok_ptr + 2 * tok_s0 + t)
        if sealed != 0:
            kelem = ((off // KG) * H + h) * D + offs_d
            ks = tl.load((pages_ptr + blk * PS + O_KS).to(tl.pointer_type(tl.float16)) + kelem).to(tl.float32)
            km = tl.load((pages_ptr + blk * PS + O_KM).to(tl.pointer_type(tl.float16)) + kelem).to(tl.float32)
            kq = _round_half_even(tl.math.div_rn(kx.to(tl.float32) - km, ks))
            kq = tl.minimum(tl.maximum(kq, 0.0), levels).to(tl.int32)
            _store_codes(pages_ptr, row_base + h * ((D * bits) // 8), kq, bits, D)
        if HAS_KREF:
            kidx = tl.load(tok_ptr + 3 * tok_s0 + t)
            if kidx >= 0:
                tl.store(kref_ptr + kidx * kref_s0 + h * kref_s1 + offs_d, kx.to(kref_ptr.dtype.element_ty))

    @triton.jit
    def _finalize_k_kernel(
        k_ptr, k_s0, k_s1,
        fin_ptr, count_ptr,
        res_ptr, res_s0, res_s1, res_s2,
        pages_ptr, PS,
        tiny,
        H: tl.constexpr, D: tl.constexpr, BS: tl.constexpr, KG: tl.constexpr, KC: tl.constexpr,
        O_KS: tl.constexpr, O_KM: tl.constexpr, O_DATA: tl.constexpr, CB: tl.constexpr,
    ):
        f = tl.program_id(0)
        h = tl.program_id(1)
        if f >= tl.load(count_ptr):
            return
        row = fin_ptr + f * (3 + KG)
        blk = tl.load(row)
        grp = tl.load(row + 1)
        rslot = tl.load(row + 2)
        bits = tl.load(pages_ptr + blk * PS).to(tl.int32)
        bits = tl.where(bits == 0, 2, bits)  # never-opened block: stay in bounds
        levels = ((1 << bits) - 1).to(tl.float32)

        offs_p = tl.arange(0, KG)
        offs_d = tl.arange(0, D)
        rows = tl.load(row + 3 + offs_p)
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


def _consts(store) -> dict:
    """The layout's compile-time constants, computed once per store."""
    c = getattr(store, "_kernel_consts", None)
    if c is None:
        L = store.layout
        o = L.offsets
        c = store._kernel_consts = dict(
            H=L.num_kv_heads, D=L.head_size, BS=L.block_size, KG=L.k_group, KC=L.chunks_per_page,
            O_KS=o["k_scale"], O_KM=o["k_min"], O_DATA=o["data"], CB=L.chunk_bytes,
        )
    return c


def _write_tokens_triton(store, key, value, meta, kref):
    L = store.layout
    o = L.offsets
    res = store.res_k
    tok = meta.tok
    has_kref = kref is not None
    kr = kref if has_kref else res  # any valid pointer; unused
    _write_tokens_kernel[(meta.n_rows, L.num_kv_heads)](
        key, key.stride(0), key.stride(1),
        value, value.stride(0), value.stride(1),
        tok, tok.stride(0),
        res, res.stride(0), res.stride(1), res.stride(2),
        kr, kr.stride(0), kr.stride(1),
        store.pages, store.pages.stride(0),
        _FP16_TINY,
        VG=L.v_group, O_VS=o["v_scale"], O_VM=o["v_min"], HAS_KREF=has_kref,
        **_consts(store),
    )


def _finalize_k_triton(store, key, meta, n):
    res = store.res_k
    _finalize_k_kernel[(n, store.layout.num_kv_heads)](
        key, key.stride(0), key.stride(1),
        meta.fin, meta.fin_count,
        res, res.stride(0), res.stride(1), res.stride(2),
        store.pages, store.pages.stride(0),
        _FP16_TINY,
        **_consts(store),
    )
