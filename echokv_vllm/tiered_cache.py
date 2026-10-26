"""Per-block mixed-precision paged KV store inside a vLLM page budget.

Quantization (chosen by measurement, see docs/ECHOKV_VLLM_PLUGIN.md):

* V is quantized per token, in groups of ``v_group`` along head_dim.
* K is quantized per channel, over consecutive groups of ``k_group``
  tokens (KIVI). A K group can only be quantized once all its tokens exist,
  so each block's open group lives in a full-precision *residual* buffer
  until it fills.

Both store an fp16 scale and an fp16 min per group (``code * scale + min``).

Each layer receives its slice of vLLM's paged KV buffer as raw bytes. Every
page (one vLLM block of ``bs`` tokens) is laid out as:

    header     bs bytes. [0] block tier (0 = never opened, else 2/4/8),
               [1] bitmask of K groups sealed early, [4:16] int32 ids of up
               to 3 extra pool chunks.
    V stats    fp16 scale, fp16 min, each [bs, H, D / v_group]
    K stats    fp16 scale, fp16 min, each [bs / k_group, H, D]
    data       k_c = mem_bits // 2 chunks of bs*H*D/2 bytes. One chunk holds
               a whole block's K+V codes at INT2.

Token rows are ``[K codes | V codes]`` packed along head_dim. Chunk ``j`` of a
block holds tokens ``[j * tpc, (j + 1) * tpc)``, ``tpc = 2 * bs / bits``.

A block's tier is fixed when its first token (offset 0) is written, using
the layer's tier at that moment. That is the forward-only rule: changing a
layer's tier never touches blocks already written. An INT2 block uses chunk
0 of its own page; INT4 needs 1 extra chunk and INT8 needs 3, popped from a
per-layer pool made of chunk slots 1..k_c-1 of every page. ``mem_bits`` is a
hard memory budget: when the pool is empty, a block opens at the best tier
that fits (INT2 always fits). When vLLM reassigns a block (offset 0 is
written again), its old extra chunks and residual slot are released first.

Residual slots are a fixed side buffer per layer (its last row is a trash
row for masked writes). A request has at most one open K group per layer,
so ``max_num_seqs`` slots plus slack are enough. Slots of finished requests
are reclaimed when their block is reused, or by LRU eviction under
pressure. Eviction *seals* the group: the tokens present are quantized now,
and any later tokens in that group reuse the sealed stats (clamped). That
is slightly lossy but never corrupts data.

Writes are split into a host-side ``WritePlan`` (see write_plan.py: which
blocks open, which groups complete, residual slot assignment) and a
device-side ``apply_plan`` that issues no host synchronization on the
common path: the chunk allocator is vectorized on the device, V and K
codes are written by the kernels in kernels/write.py (Triton on CUDA,
PyTorch elsewhere). ``write`` plans and applies in one call.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from echokv_vllm.quant import (
    SUPPORTED_BITS,
    dequantize,
    dequantize_with,
    pack,
    quantize_with,
    stats_along,
    unpack,
)
from echokv_vllm.write_plan import Eviction, WritePlan, WritePlanner

# Extra pool chunks a block of each tier needs beyond its in-page chunk 0.
NEXTRA = {2: 0, 4: 1, 8: 3}


@dataclass(frozen=True)
class TieredLayout:
    num_kv_heads: int
    head_size: int
    block_size: int
    mem_bits: int
    k_group: int = 32
    v_group: int = 32

    def __post_init__(self) -> None:
        if self.mem_bits not in (2, 4, 6, 8):
            raise ValueError(f"mem_bits must be one of 2/4/6/8, got {self.mem_bits}")
        if self.block_size % 16 != 0:
            raise ValueError(f"block_size must be a multiple of 16, got {self.block_size}")
        if self.block_size % self.k_group != 0 or self.block_size // self.k_group > 8:
            raise ValueError(
                f"block_size {self.block_size} must be 1-8 multiples of k_group {self.k_group}"
            )
        if self.head_size % self.v_group != 0 or self.head_size % 4 != 0:
            raise ValueError(
                f"head_size {self.head_size} must be divisible by v_group {self.v_group} and by 4"
            )

    @staticmethod
    def per_token_bytes(
        num_kv_heads: int, head_size: int, mem_bits: int, k_group: int = 32, v_group: int = 32
    ) -> int:
        """Bytes per token vLLM must budget: header + stats + data at mem_bits."""
        h, d = num_kv_heads, head_size
        raw = 1 + 4 * h * (d // v_group) + 4 * h * d // k_group + (mem_bits // 2) * h * d // 2
        return -(-raw // 4) * 4  # whole number of fp32 elements

    @property
    def k_groups_per_block(self) -> int:
        return self.block_size // self.k_group

    @property
    def v_groups(self) -> int:
        return self.head_size // self.v_group

    @property
    def chunks_per_page(self) -> int:
        return self.mem_bits // 2

    @property
    def offsets(self) -> dict[str, int]:
        bs, h, d = self.block_size, self.num_kv_heads, self.head_size
        o = {"v_scale": bs}
        o["v_min"] = o["v_scale"] + bs * h * self.v_groups * 2
        o["k_scale"] = o["v_min"] + bs * h * self.v_groups * 2
        o["k_min"] = o["k_scale"] + self.k_groups_per_block * h * d * 2
        o["data"] = o["k_min"] + self.k_groups_per_block * h * d * 2
        o["end"] = o["data"] + self.chunks_per_page * self.chunk_bytes
        return o

    @property
    def chunk_bytes(self) -> int:
        return self.block_size * self.num_kv_heads * self.head_size // 2

    @property
    def page_bytes(self) -> int:
        return self.offsets["end"]

    def side_bytes(self, bits: int) -> int:
        """Bytes of K (or V) codes per token at ``bits``."""
        return self.num_kv_heads * self.head_size * bits // 8

    def row_bytes(self, bits: int) -> int:
        return 2 * self.side_bytes(bits)

    def tokens_per_chunk(self, bits: int) -> int:
        return 2 * self.block_size // bits


class TieredKVCache:
    """One layer's tiered store over ``pages``: a uint8 ``[num_blocks, >=page_bytes]`` view."""

    def __init__(
        self,
        layout: TieredLayout,
        pages: torch.Tensor,
        residual_slots: int = 256,
        residual_dtype: torch.dtype = torch.float32,
        residual: torch.Tensor | None = None,
    ) -> None:
        if pages.dtype != torch.uint8 or pages.dim() != 2:
            raise ValueError("pages must be a 2-D uint8 tensor")
        if pages.shape[1] < layout.page_bytes:
            raise ValueError(f"page has {pages.shape[1]} bytes, layout needs {layout.page_bytes}")
        if pages.stride(1) != 1 or pages.stride(0) % 16 != 0:
            raise ValueError("pages must be byte-contiguous with a 16-byte aligned row stride")
        L = layout
        n = pages.shape[0]
        h, d, bs = L.num_kv_heads, L.head_size, L.block_size
        o = L.offsets
        dev = pages.device
        self.layout = L
        self.num_blocks = n
        self.device = dev
        self.pages = pages
        self.tag = pages[:, 0]
        self.sealed = pages[:, 1]
        self.table = pages[:, 4:16].view(torch.int32)

        def f16(name, nxt, *shape):
            return pages[:, o[name] : o[nxt]].view(torch.float16).view(n, *shape)

        self.v_scale = f16("v_scale", "v_min", bs, h, L.v_groups)
        self.v_min = f16("v_min", "k_scale", bs, h, L.v_groups)
        self.k_scale = f16("k_scale", "k_min", L.k_groups_per_block, h, d)
        self.k_min = f16("k_min", "data", L.k_groups_per_block, h, d)
        self.data = pages[:, o["data"] : o["end"]].view(n, L.chunks_per_page, L.chunk_bytes)
        pages[:, :bs].zero_()

        # Chunk pool, managed on the host: every input to an allocation decision
        # (blocks opened, target tier, old tier, fill, recency) is host-known, so
        # the device only receives small tag/table updates and nothing syncs.
        k = L.chunks_per_page
        self._free: list[int] = (
            np.arange(n * k).reshape(n, k)[:, 1:].reshape(-1)[::-1].tolist()
        )
        self.pool_size = len(self._free)
        self._tag_h = np.zeros(n, dtype=np.int64)
        self._table_h = np.zeros((n, 3), dtype=np.int64)
        self._fill_h = np.zeros(n, dtype=np.int64)
        self._last_h = np.zeros(n, dtype=np.int64)

        if residual is None:
            residual = torch.zeros(residual_slots + 1, L.k_group, h, d, dtype=residual_dtype, device=dev)
        if residual.shape[1:] != (L.k_group, h, d) or residual.shape[0] < 2:
            raise ValueError(f"residual buffer shape {tuple(residual.shape)} does not match layout")
        r = residual.shape[0] - 1  # last row is the trash row
        self.res_k = residual
        self.res_group = torch.zeros(r + 1, dtype=torch.long, device=dev)
        self.block_res = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.planner = WritePlanner(bs, L.k_group, r)

        self._tokens = torch.zeros(9, dtype=torch.long, device=dev)
        self.reset_counters()

    # -- counters (read lazily; each read syncs) ---------------------------

    @property
    def tokens_written(self) -> dict[int, int]:
        return {b: int(self._tokens[b]) for b in SUPPORTED_BITS}

    def reset_counters(self) -> None:
        # Callable from outside vLLM's forward (llm.apply_model), where the
        # counter, created under inference mode, may only be updated inside it.
        with torch.inference_mode():
            self._tokens.zero_()
        self.blocks_opened = {b: 0 for b in SUPPORTED_BITS}
        self.downgraded_blocks = 0
        self.demoted_blocks = 0
        self.sealed_groups = 0

    @property
    def pool_free(self) -> int:
        return len(self._free)

    @property
    def residual_free(self) -> int:
        return len(self.planner.free)

    # -- chunk pool (host-side decisions) ---------------------------------

    def open_blocks(self, blocks, target_bits: int, protect=None) -> None:
        """(Re)open ``blocks`` (unique ids) at ``target_bits`` or the best tier that fits.

        Old extra chunks go back to the pool first. If the pool cannot give a
        block ``target_bits``, the least recently written full blocks are
        demoted to INT2 in place to free chunks (never ``protect`` blocks or
        blocks opened here). Only then does a new block open lower, down to
        INT2, which needs no extra chunks.
        """
        if target_bits not in SUPPORTED_BITS:
            raise ValueError(f"unsupported target bits {target_bits}")
        blocks = np.asarray(blocks.cpu() if torch.is_tensor(blocks) else blocks, dtype=np.int64)
        for b in blocks.tolist():
            ne = NEXTRA.get(int(self._tag_h[b]), 0)
            self._free.extend(self._table_h[b, :ne].tolist())
            self._tag_h[b] = 0
            self._fill_h[b] = 0
        need = NEXTRA[target_bits] * blocks.size
        if need > len(self._free):
            keep = set(blocks.tolist()) | (set() if protect is None else set(protect))
            self._demote_lru(need - len(self._free), keep)
        # Greedy in block order: as many as fit at target, then the next tier down.
        tiers = np.full(blocks.size, 2, dtype=np.int64)
        start = 0
        for bits in (x for x in (8, 4) if x <= target_bits):
            ne = NEXTRA[bits]
            take = min(blocks.size - start, len(self._free) // ne)
            if take:
                sel = blocks[start : start + take]
                ids = self._free[len(self._free) - take * ne :][::-1]
                del self._free[len(self._free) - take * ne :]
                self._table_h[sel, :ne] = np.asarray(ids, dtype=np.int64).reshape(take, ne)
                tiers[start : start + take] = bits
                start += take
        for bits in SUPPORTED_BITS:
            self.blocks_opened[bits] += int((tiers == bits).sum())
        self.downgraded_blocks += int((tiers < target_bits).sum())
        self._tag_h[blocks] = tiers
        dev_blocks = torch.from_numpy(blocks).to(self.device)
        self.tag[dev_blocks] = torch.from_numpy(tiers).to(self.device, torch.uint8)
        self.table[dev_blocks] = torch.from_numpy(self._table_h[blocks]).to(self.device, torch.int32)
        self.sealed[dev_blocks] = 0

    def _demote_lru(self, chunks_needed: int, keep: set) -> None:
        """Requantize least recently written full INT4/INT8 blocks to INT2, freeing chunks."""
        cand = np.nonzero((self._tag_h >= 4) & (self._fill_h == self.layout.block_size))[0]
        cand = [b for b in cand[np.argsort(self._last_h[cand], kind="stable")].tolist() if b not in keep]
        victims, freed = [], 0
        for b in cand:
            if freed >= chunks_needed:
                break
            victims.append(b)
            freed += NEXTRA[int(self._tag_h[b])]
        if not victims:
            return
        L = self.layout
        bs, g, h, d = L.block_size, L.k_group, L.num_kv_heads, L.head_size
        vb = torch.tensor(victims, device=self.device)
        blk = vb.repeat_interleave(bs)
        off = torch.arange(bs, device=self.device).repeat(len(victims))
        k, v = self.gather(blk, off)  # dequantized at the current tiers
        kg = k.view(len(victims), bs // g, g, h, d)
        k_scale, k_min = stats_along(kg, 2, dim=2)
        k_codes = quantize_with(kg, k_scale.unsqueeze(2), k_min.unsqueeze(2), 2).view(-1, h, d)
        from echokv_vllm.quant import quantize

        v_codes, v_scale, v_min = quantize(v, 2, L.v_group)
        self.tag[vb] = 2  # INT2 rows live in chunk 0: addressing needs no table
        self.k_scale[vb] = k_scale
        self.k_min[vb] = k_min
        self.v_scale[blk, off] = v_scale
        self.v_min[blk, off] = v_min
        self._write_side(blk, off, 2, k_codes, side=0)
        self._write_side(blk, off, 2, v_codes, side=1)
        for b in victims:
            ne = NEXTRA[int(self._tag_h[b])]
            self._free.extend(self._table_h[b, :ne].tolist())
            self._tag_h[b] = 2
        self.demoted_blocks += len(victims)

    # -- addressing -------------------------------------------------------

    def _locate(self, blk: torch.Tensor, off: torch.Tensor, bits: int):
        """(page, chunk slot, byte offset) of each token's K+V row at ``bits``."""
        L = self.layout
        tpc = L.tokens_per_chunk(bits)
        j = off // tpc
        ids = self.table[blk, (j - 1).clamp_min(0)].long()
        ext = j > 0
        page = torch.where(ext, ids // L.chunks_per_page, blk)
        slot = torch.where(ext, ids % L.chunks_per_page, torch.zeros_like(blk))
        return page, slot, (off % tpc) * L.row_bytes(bits)

    def _write_side(self, blk, off, bits: int, codes: torch.Tensor, side: int) -> None:
        """Pack ``codes [n, H, D]`` into the K (side=0) or V (side=1) half of each row."""
        if blk.numel() == 0:
            return
        page, slot, byte = self._locate(blk, off, bits)
        sb = self.layout.side_bytes(bits)
        cols = (byte + side * sb).unsqueeze(1) + torch.arange(sb, device=self.device)
        self.data[page.unsqueeze(1), slot.unsqueeze(1), cols] = pack(codes, bits).reshape(blk.numel(), -1)

    def _read_side(self, blk, off, bits: int, side: int) -> torch.Tensor:
        page, slot, byte = self._locate(blk, off, bits)
        sb = self.layout.side_bytes(bits)
        cols = (byte + side * sb).unsqueeze(1) + torch.arange(sb, device=self.device)
        rows = self.data[page.unsqueeze(1), slot.unsqueeze(1), cols]
        return unpack(rows, bits).view(blk.numel(), self.layout.num_kv_heads, self.layout.head_size)

    # -- write ------------------------------------------------------------

    def _seal(self, ev: Eviction) -> None:
        """Quantize the ``fill`` tokens of an evicted residual slot against their own stats."""
        if ev.fill <= 0:
            return
        g = self.layout.k_group
        bits = int(self.tag[ev.block])
        ks = self.res_k[ev.slot, : ev.fill].float()
        scale, mn = stats_along(ks, bits, dim=0)
        self.k_scale[ev.block, ev.group] = scale
        self.k_min[ev.block, ev.group] = mn
        blk = torch.full((ev.fill,), ev.block, device=self.device)
        off = ev.group * g + torch.arange(ev.fill, device=self.device)
        self._write_side(blk, off, bits, quantize_with(ks, scale, mn, bits), side=0)
        self.sealed[ev.block] = int(self.sealed[ev.block]) | (1 << ev.group)
        self.sealed_groups += 1

    def apply_plan(self, plan: WritePlan, key: torch.Tensor, value: torch.Tensor, target_bits: int) -> None:
        """Store ``key``/``value`` rows ``[*, H, D]`` as described by ``plan``."""
        from echokv_vllm.kernels.write import finalize_k, write_v

        L = self.layout
        for ev in plan.evictions:  # rare; before the slot is reused below
            self._seal(ev)
        opens = plan.open_np
        if plan.touched_np.size:
            # A write into a block that was never opened (no offset-0 write seen,
            # e.g. vLLM warmup batches): open it rather than write with tier 0.
            fresh = plan.touched_np[self._tag_h[plan.touched_np] == 0]
            if fresh.size:
                opens = np.union1d(opens, fresh)
        if opens.size:
            self.open_blocks(opens, target_bits, protect=plan.touched_np)
        if plan.slots_np.size:
            np.maximum.at(self._fill_h, plan.slots_np // L.block_size, plan.slots_np % L.block_size + 1)
            self._last_h[plan.touched_np] = plan.step
        if plan.res_clear_block.numel():
            self.block_res[plan.res_clear_block] = -1
        if plan.res_set_block.numel():
            self.block_res[plan.res_set_block] = plan.res_set_slot
            self.res_group[plan.res_set_slot] = plan.res_set_group
        if plan.slots.numel() == 0:
            return

        blk = plan.slots // L.block_size
        off = plan.slots % L.block_size
        self._tokens.index_add_(0, self.tag[blk].long(), torch.ones_like(blk))
        write_v(self, value, plan.rows, plan.slots)

        # Finalize before the residual scatter: the planner may hand a slot freed
        # by a completing group to a new group in the same step.
        if plan.n_fin:
            finalize_k(self, key, plan.fin_block, plan.fin_group, plan.fin_res_slot, plan.fin_rows)

        # K tokens of still-open groups go to the residual (masked rows hit the trash row).
        trash = self.res_k.shape[0] - 1
        to_res = plan.tok_res_slot >= 0
        self.res_k[torch.where(to_res, plan.tok_res_slot, trash), off % L.k_group] = key[plan.rows].to(
            self.res_k.dtype
        )
        if plan.n_sealed:
            m = plan.tok_sealed
            b_, o_, r_ = blk[m], off[m], plan.rows[m]
            for bits in SUPPORTED_BITS:
                mb = self.tag[b_] == bits
                if bool(mb.any()):
                    sc = self.k_scale[b_[mb], o_[mb] // L.k_group]
                    mn = self.k_min[b_[mb], o_[mb] // L.k_group]
                    self._write_side(b_[mb], o_[mb], bits, quantize_with(key[r_[mb]], sc, mn, bits), 0)

    def write(self, key: torch.Tensor, value: torch.Tensor, slots: torch.Tensor, target_bits: int) -> None:
        """Quantize and store ``key``/``value`` ``[n, H, D]`` at ``slots`` (-1 = skip)."""
        plan = self.planner.plan(slots.detach().cpu(), self.device)
        self.apply_plan(plan, key, value, target_bits)

    # -- read -------------------------------------------------------------

    def gather(self, blk: torch.Tensor, off: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantized K and V (float32 ``[n, H, D]``) for tokens at (blk, off)."""
        from echokv_vllm.quant import dequantize

        L = self.layout
        n = blk.numel()
        h, d, g = L.num_kv_heads, L.head_size, L.k_group
        k_out = torch.zeros(n, h, d, device=self.device)
        v_out = torch.zeros(n, h, d, device=self.device)
        tags = self.tag[blk]
        grp = off // g
        slot = self.block_res[blk]
        in_res = (slot >= 0) & (self.res_group[slot.clamp_min(0)] == grp)
        for bits in SUPPORTED_BITS:
            m = tags == bits
            if not bool(m.any()):
                continue
            bm, om = blk[m], off[m]
            vc = self._read_side(bm, om, bits, side=1)
            v_out[m] = dequantize(vc, self.v_scale[bm, om], self.v_min[bm, om], L.v_group)
            mk = m & ~in_res
            if bool(mk.any()):
                bk, ok = blk[mk], off[mk]
                kc = self._read_side(bk, ok, bits, side=0)
                k_out[mk] = dequantize_with(kc, self.k_scale[bk, ok // g], self.k_min[bk, ok // g])
        if bool(in_res.any()):
            k_out[in_res] = self.res_k[slot[in_res], off[in_res] % g].float()
        return k_out, v_out

    def tier_histogram(self) -> dict[int, int]:
        """Blocks currently tagged at each tier (includes blocks vLLM has freed)."""
        return {b: int((self.tag == b).sum()) for b in SUPPORTED_BITS}
