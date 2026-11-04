"""Host-side planning of one step's KV writes, shared by every layer.

Everything about a write that does not depend on tensor *values* is known
on the host from the slot mapping alone: which blocks open (offset 0 is
written), which K groups complete in this step, which tokens go to the
residual buffer, when residual slots must be evicted, and where each key
lands in the K_ref ring. ``WritePlanner`` computes that once per step as
numpy arrays indexed by token row; every layer applies the same plan.

``SharedState`` holds what is identical across the layers of one KV cache
group: the planner (residual slot allocator, K_ref ring cursor), the
block -> residual slot map on the device (read by the attention kernels),
and per-block host bookkeeping (fill, recency, tokens written). Slot ids
mean the same thing in every layer (each layer has its own residual buffer
indexed by them), because every layer sees the same sequence of writes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from echokv_vllm._util import to_device


@dataclass
class Eviction:
    slot: int
    block: int
    group: int
    fill: int


def _empty() -> np.ndarray:
    return np.zeros(0, dtype=np.int64)


@dataclass
class WritePlan:
    """One step's writes, as host arrays.

    Per token row (``n_rows`` = rows of the step's key/value tensors):
    ``tok_slot`` (-1 = no write), ``tok_res`` (residual slot for keys of
    still-open groups, else -1), ``tok_sealed`` (1 = key belongs to a group
    sealed by eviction: quantize against its stored stats) and ``tok_kref``
    (K_ref ring row, or -1). ``fin`` has one row per completed K group:
    ``[block, group, residual slot or -1, token row of each position (-1 =
    from the residual)]``.
    """

    step: int
    n_rows: int
    tok_slot: np.ndarray
    tok_res: np.ndarray
    tok_sealed: np.ndarray
    tok_kref: np.ndarray
    fin: np.ndarray  # [f, 3 + k_group]
    evictions: list[Eviction] = field(default_factory=list)  # seal before this step's writes
    open_np: np.ndarray = field(default_factory=_empty)  # blocks (re)opened this step
    touched_np: np.ndarray = field(default_factory=_empty)  # blocks written, their fill and count
    touched_fill: np.ndarray = field(default_factory=_empty)
    touched_count: np.ndarray = field(default_factory=_empty)
    res_map_block: np.ndarray = field(default_factory=_empty)  # block_res[b] = v (-1 = none)
    res_map_value: np.ndarray = field(default_factory=_empty)
    res_group_slot: np.ndarray = field(default_factory=_empty)  # res_group[s] = g
    res_group_value: np.ndarray = field(default_factory=_empty)
    block_size: int = 0
    k_group: int = 0

    @property
    def n_fin(self) -> int:
        return int(self.fin.shape[0])


class WritePlanner:
    def __init__(self, block_size: int, k_group: int, residual_slots: int, kref_window: int = 64) -> None:
        self.bs = block_size
        self.g = k_group
        self.num_slots = residual_slots
        self.free = list(range(residual_slots - 1, -1, -1))
        self.block_slot: dict[int, int] = {}  # block -> slot of its open group
        self.slot_info: dict[int, list[int]] = {}  # slot -> [block, group, fill, last_step]
        self.sealed: dict[int, int] = {}  # block -> bitmask of groups sealed by eviction
        self.step = 0
        # K_ref ring cursor (the rings themselves are per layer).
        self.kref_window = kref_window
        self.kref_pos = 0
        self.kref_fill = 0

    def _release(self, slot: int, res_ops: list) -> None:
        b = self.slot_info.pop(slot)[0]
        self.block_slot.pop(b, None)
        self.free.append(slot)
        res_ops.append((b, -1))

    def plan(self, slot_mapping_cpu: torch.Tensor) -> WritePlan:
        self.step += 1
        bs, g = self.bs, self.g
        sm = slot_mapping_cpu.numpy().astype(np.int64).reshape(-1)
        n_rows = sm.size
        rows = np.nonzero(sm >= 0)[0]
        slots = sm[rows]
        blk = slots // bs
        off = slots % bs
        grp = off // g
        pos = off % g
        n = slots.size

        res_ops: list[tuple[int, int]] = []  # (block, slot or -1), in order
        grp_ops: list[tuple[int, int]] = []  # (slot, group)
        open_blocks = np.unique(blk[off == 0])
        for b in open_blocks.tolist():
            s = self.block_slot.get(b)
            if s is not None:
                self._release(s, res_ops)
            self.sealed.pop(b, None)

        tok_res = np.full(n, -1, dtype=np.int64)
        tok_sealed = np.zeros(n, dtype=np.int64)
        fin = []  # (block, group, res_slot)
        fin_rows = []
        evictions: list[Eviction] = []

        # Group tokens by (block, group), each group's tokens in position order.
        key = blk * (bs // g) + grp
        order = np.lexsort((pos, key))
        ks = key[order]
        starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]]) if n else _empty()
        counts = np.diff(np.r_[starts, n])
        ublk = blk[order[starts]]
        ugrp = grp[order[starts]]

        # Fast path, vectorized: groups whose g tokens are all in this batch (the
        # common prefill case). No residual is involved: earlier tokens would
        # make the count < g.
        fast = counts == g
        if self.sealed:
            fast &= np.array(
                [not (self.sealed.get(int(b), 0) >> int(gr)) & 1 for b, gr in zip(ublk, ugrp)], dtype=bool
            )
        fi = np.flatnonzero(fast)
        fin_fast_rows = rows[order[starts[fi][:, None] + np.arange(g)]] if fi.size else np.zeros((0, g), np.int64)
        for b in ublk[fi].tolist():  # a rewrite of a group that had a residual slot (rare)
            s_ = self.block_slot.get(b)
            if s_ is not None and self.slot_info[s_][1] in set(ugrp[fi][ublk[fi] == b].tolist()):
                self._release(s_, res_ops)

        # Slow path, per group: partial groups, groups completing from the residual,
        # groups sealed by an eviction.
        protected = {self.block_slot[b] for b in np.unique(blk).tolist() if b in self.block_slot}
        for u in np.flatnonzero(~fast).tolist():
            idx = order[starts[u] : starts[u] + counts[u]]
            b, gr = int(ublk[u]), int(ugrp[u])
            p = pos[idx]
            if (self.sealed.get(b, 0) >> gr) & 1:
                tok_sealed[idx] = 1
                continue
            s = self.block_slot.get(b)
            if s is not None and self.slot_info[s][1] != gr:
                s = None  # stale mapping from an earlier group; cannot happen with in-order writes
            if int(p.max()) == g - 1:
                fin.append((b, gr, -1 if s is None else s))
                r = np.full(g, -1, dtype=np.int64)
                r[p] = rows[idx]
                fin_rows.append(r)
                if s is not None:
                    self._release(s, res_ops)
                continue
            if s is None:
                if not self.free:
                    victims = [v for v in self.slot_info if v not in protected]
                    if not victims:
                        raise RuntimeError(f"more than {self.num_slots} open K groups in one step")
                    v = min(victims, key=lambda x: self.slot_info[x][3])
                    vb, vg, vf, _ = self.slot_info[v]
                    evictions.append(Eviction(v, vb, vg, vf))
                    self.sealed[vb] = self.sealed.get(vb, 0) | (1 << vg)
                    self._release(v, res_ops)
                s = self.free.pop()
                self.slot_info[s] = [b, gr, 0, 0]
                self.block_slot[b] = s
                protected.add(s)
                res_ops.append((b, s))
                grp_ops.append((s, gr))
            tok_res[idx] = s
            info = self.slot_info[s]
            info[2] = max(info[2], int(p.max()) + 1)
            info[3] = self.step

        fin_head = np.concatenate(
            [np.stack([ublk[fi], ugrp[fi], np.full(fi.size, -1)], 1).reshape(-1, 3),
             np.asarray(fin, dtype=np.int64).reshape(-1, 3)]
        )
        fin_body = np.concatenate([fin_fast_rows, np.asarray(fin_rows, dtype=np.int64).reshape(-1, g)])
        fin_arr = np.concatenate([fin_head, fin_body], 1).astype(np.int64)

        # Per-block fill (highest offset written + 1) and token count.
        bsorted = blk[order]
        bstarts = np.flatnonzero(np.r_[True, bsorted[1:] != bsorted[:-1]]) if n else _empty()
        touched = bsorted[bstarts]
        touched_fill = np.maximum.reduceat(off[order] + 1, bstarts) if n else _empty()
        touched_count = np.diff(np.r_[bstarts, n]) if n else _empty()

        # The last kref_window keys of the step go to the K_ref ring, in order.
        k = min(n, self.kref_window)
        kref = np.full(n, -1, dtype=np.int64)
        if k:
            kref[n - k :] = (self.kref_pos + np.arange(k)) % self.kref_window
            self.kref_pos = (self.kref_pos + k) % self.kref_window
            self.kref_fill = min(self.kref_window, self.kref_fill + k)

        def per_row(a: np.ndarray) -> np.ndarray:
            out = np.full(n_rows, -1, dtype=np.int64)
            out[rows] = a
            return out

        res_map = dict(res_ops)  # last operation on each block wins
        grp_map = dict(grp_ops)
        return WritePlan(
            step=self.step,
            n_rows=n_rows,
            tok_slot=sm,
            tok_res=per_row(tok_res),
            tok_sealed=np.maximum(per_row(tok_sealed), 0),
            tok_kref=per_row(kref),
            fin=fin_arr,
            evictions=evictions,
            open_np=open_blocks,
            touched_np=touched,
            touched_fill=touched_fill,
            touched_count=touched_count,
            res_map_block=np.fromiter(res_map.keys(), np.int64, len(res_map)),
            res_map_value=np.fromiter(res_map.values(), np.int64, len(res_map)),
            res_group_slot=np.fromiter(grp_map.keys(), np.int64, len(grp_map)),
            res_group_value=np.fromiter(grp_map.values(), np.int64, len(grp_map)),
            block_size=bs,
            k_group=g,
        )


class SharedState:
    """State shared by every layer of one KV cache group (see module docstring)."""

    def __init__(
        self, num_blocks: int, block_size: int, k_group: int, residual_slots: int,
        kref_window: int, device,
    ) -> None:
        self.num_blocks = num_blocks
        self.planner = WritePlanner(block_size, k_group, residual_slots, kref_window)
        self.fill_h = np.zeros(num_blocks, dtype=np.int64)  # highest offset written + 1
        self.last_h = np.zeros(num_blocks, dtype=np.int64)  # step of the latest write
        self.cum_h = np.zeros(num_blocks, dtype=np.int64)  # tokens ever written (stats)
        self.block_res = torch.full((num_blocks,), -1, dtype=torch.long, device=device)
        self.res_group = torch.zeros(residual_slots + 1, dtype=torch.long, device=device)

    def commit(self, plan: WritePlan) -> None:
        """Record the step's writes in the host bookkeeping (after every layer prepared)."""
        t = plan.touched_np
        if t.size:
            self.fill_h[t] = np.maximum(self.fill_h[t], plan.touched_fill)
            self.last_h[t] = plan.step
            self.cum_h[t] += plan.touched_count

    def apply_res_map(self, blocks, values, slots, groups) -> None:
        """Residual map updates (device index tensors) for the step."""
        if blocks.numel():
            self.block_res[blocks] = values
        if slots.numel():
            self.res_group[slots] = groups


@dataclass
class WriteMeta:
    """A plan's device arrays, as the write kernels read them."""

    tok: torch.Tensor  # [4, >= n_rows] int64: slot, residual slot, sealed, K_ref row
    fin: torch.Tensor  # [>= n_fin, 3 + k_group] int64 (see WritePlan.fin)
    fin_count: torch.Tensor  # [1] int64 n_fin: masks a full-grid (graph) finalize
    n_rows: int
    n_fin: int


def stage_plan(plan: WritePlan, device, opened=None) -> tuple[WriteMeta, tuple]:
    """One host-to-device copy of ``plan`` (and of ``opened`` page headers).

    Returns the ``WriteMeta`` and the device tensors for the residual map
    (``SharedState.apply_res_map``) and the headers (``write_headers``; None
    without ``opened``).
    """
    t, f, w = plan.n_rows, plan.n_fin, 3 + plan.k_group
    parts = [
        np.array([f]), plan.tok_slot, plan.tok_res, plan.tok_sealed, plan.tok_kref, plan.fin,
        plan.res_map_block, plan.res_map_value, plan.res_group_slot, plan.res_group_value,
    ]
    if opened is not None:
        parts += [opened[0], opened[1].view(np.int64)]
    flat = [np.asarray(p, dtype=np.int64).reshape(-1) for p in parts]
    dev = to_device(torch.from_numpy(np.concatenate(flat)), device)
    views, o = [], 0
    for a in flat:
        views.append(dev[o : o + a.size])
        o += a.size
    meta = WriteMeta(
        tok=dev[1 : 1 + 4 * t].view(4, t),
        fin=views[5].view(f, w),
        fin_count=views[0],
        n_rows=t,
        n_fin=f,
    )
    headers = (None, None)
    if opened is not None:
        headers = (views[10], views[11].view(-1, 2).view(torch.uint8))
    return meta, (*views[6:10], *headers)
