"""Host-side planning of one step's KV writes.

Everything about a write that does not depend on tensor *values* is known
on the host from the slot mapping alone: which blocks open (offset 0 is
written), which K groups complete in this step, which tokens go to the
residual buffer, and when residual slots must be evicted. ``WritePlanner``
computes that once per step and hands every layer the same ``WritePlan``,
so the per-layer device work needs no host synchronization.

The residual slot allocator lives here, on the host. Slot ids mean the
same thing in every layer (each layer has its own residual buffer indexed
by them), because every layer sees the same sequence of opens and writes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Eviction:
    slot: int
    block: int
    group: int
    fill: int


@dataclass
class WritePlan:
    """Device tensors for one step, plus host counts for branch-free dispatch."""

    slots: torch.Tensor  # [n] valid slots, in batch order
    rows: torch.Tensor  # [n] row of each valid token in the layer's key/value tensors
    open_blocks: torch.Tensor  # [m] blocks (re)opened this step
    tok_res_slot: torch.Tensor  # [n] residual slot for tokens of open groups, else -1
    tok_sealed: torch.Tensor  # [n] bool: token belongs to a group sealed by eviction
    fin_block: torch.Tensor  # [f] completed groups
    fin_group: torch.Tensor
    fin_res_slot: torch.Tensor  # residual slot holding the group's earlier tokens, or -1
    fin_rows: torch.Tensor  # [f, g] row in key of each position's token, -1 = from residual
    res_set_block: torch.Tensor  # block_res/res_group updates for newly assigned slots
    res_set_slot: torch.Tensor
    res_set_group: torch.Tensor
    res_clear_block: torch.Tensor  # blocks whose residual mapping ends this step
    res_clear_slot: torch.Tensor
    evictions: list[Eviction] = field(default_factory=list)  # seal before this step's writes
    block_size: int = 0
    k_group: int = 0
    step: int = 0
    slots_np: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    open_np: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    touched_np: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    touched_fill: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    touched_count: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    res_index: torch.Tensor | None = None  # [n] residual row of each token's K (trash row if none)
    res_pos: torch.Tensor | None = None  # [n] position of each token within its K group
    host_buffer: torch.Tensor | None = None  # keeps the pinned source of the async copy alive
    n_open: int = 0
    n_fin: int = 0
    n_sealed: int = 0


class WritePlanner:
    def __init__(self, block_size: int, k_group: int, residual_slots: int) -> None:
        self.bs = block_size
        self.g = k_group
        self.num_slots = residual_slots
        self.free = list(range(residual_slots - 1, -1, -1))
        self.block_slot: dict[int, int] = {}  # block -> slot of its open group
        self.slot_info: dict[int, list[int]] = {}  # slot -> [block, group, fill, last_step]
        self.sealed: dict[int, int] = {}  # block -> bitmask of groups sealed by eviction
        self.step = 0

    def _release(self, slot: int, clear_b: list, clear_s: list) -> None:
        b = self.slot_info.pop(slot)[0]
        self.block_slot.pop(b, None)
        self.free.append(slot)
        clear_b.append(b)
        clear_s.append(slot)

    def plan(self, slot_mapping_cpu: torch.Tensor, device: torch.device) -> WritePlan:
        self.step += 1
        bs, g = self.bs, self.g
        sm = slot_mapping_cpu.numpy().astype(np.int64)
        rows = np.nonzero(sm >= 0)[0]
        slots = sm[rows]
        blk = slots // bs
        off = slots % bs
        grp = off // g
        pos = off % g
        n = slots.size

        clear_b: list[int] = []
        clear_s: list[int] = []
        open_blocks = np.unique(blk[off == 0])
        for b in open_blocks.tolist():
            s = self.block_slot.get(b)
            if s is not None:
                self._release(s, clear_b, clear_s)
            self.sealed.pop(b, None)

        tok_res = np.full(n, -1, dtype=np.int64)
        tok_sealed = np.zeros(n, dtype=bool)
        fin = []  # (block, group, res_slot)
        fin_rows = []
        set_b, set_s, set_g = [], [], []
        evictions: list[Eviction] = []

        # Group tokens by (block, group), each group's tokens in position order.
        key = blk * (bs // g) + grp
        order = np.lexsort((pos, key))
        ks = key[order]
        starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]]) if n else np.zeros(0, np.int64)
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
                self._release(s_, clear_b, clear_s)

        # Slow path, per group: partial groups, groups completing from the residual,
        # groups sealed by an eviction.
        protected = {self.block_slot[b] for b in np.unique(blk).tolist() if b in self.block_slot}
        for u in np.flatnonzero(~fast).tolist():
            idx = order[starts[u] : starts[u] + counts[u]]
            b, gr = int(ublk[u]), int(ugrp[u])
            p = pos[idx]
            if (self.sealed.get(b, 0) >> gr) & 1:
                tok_sealed[idx] = True
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
                    self._release(s, clear_b, clear_s)
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
                    self._release(v, clear_b, clear_s)
                s = self.free.pop()
                self.slot_info[s] = [b, gr, 0, 0]
                self.block_slot[b] = s
                protected.add(s)
                set_b.append(b)
                set_s.append(s)
                set_g.append(gr)
            tok_res[idx] = s
            info = self.slot_info[s]
            info[2] = max(info[2], int(p.max()) + 1)
            info[3] = self.step

        fin_arr = np.concatenate(
            [np.stack([ublk[fi], ugrp[fi], np.full(fi.size, -1)], 1).reshape(-1, 3),
             np.asarray(fin, dtype=np.int64).reshape(-1, 3)]
        )
        fin_rows_arr = np.concatenate([fin_fast_rows, np.asarray(fin_rows, dtype=np.int64).reshape(-1, g)])

        # Per-block fill (highest offset written + 1), once per step for all layers.
        bsorted = blk[order]
        bstarts = np.flatnonzero(np.r_[True, bsorted[1:] != bsorted[:-1]]) if n else np.zeros(0, np.int64)
        touched = bsorted[bstarts]
        touched_fill = np.maximum.reduceat(off[order] + 1, bstarts) if n else np.zeros(0, np.int64)
        touched_count = np.diff(np.r_[bstarts, n]) if n else np.zeros(0, np.int64)
        # Residual destination of every token: its slot, or the trash row
        # (index num_slots) for tokens whose K goes straight to codes.
        res_index = np.where(tok_res >= 0, tok_res, self.num_slots)
        res_pos = pos

        # One host-to-device copy for every index array of the plan.
        parts = {
            "slots": slots, "rows": rows, "open_blocks": open_blocks, "tok_res_slot": tok_res,
            "tok_sealed": tok_sealed.astype(np.int64), "fin_block": fin_arr[:, 0], "fin_group": fin_arr[:, 1],
            "fin_res_slot": fin_arr[:, 2], "fin_rows": fin_rows_arr.reshape(-1),
            "res_set_block": set_b, "res_set_slot": set_s, "res_set_group": set_g,
            "res_clear_block": clear_b, "res_clear_slot": clear_s,
            "res_index": res_index, "res_pos": res_pos,
        }
        flat = [np.asarray(v, dtype=np.int64).reshape(-1) for v in parts.values()]
        host = torch.from_numpy(np.concatenate(flat)) if flat else torch.zeros(0, dtype=torch.long)
        if torch.device(device).type == "cuda":
            host = host.pin_memory()
        dev = host.to(device, non_blocking=True)
        views, o = {}, 0
        for k, a in zip(parts, flat):
            views[k] = dev[o : o + a.size]
            o += a.size
        views["tok_sealed"] = views["tok_sealed"].bool()
        views["fin_rows"] = views["fin_rows"].view(-1, g)

        return WritePlan(
            slots=views["slots"],
            rows=views["rows"],
            open_blocks=views["open_blocks"],
            tok_res_slot=views["tok_res_slot"],
            tok_sealed=views["tok_sealed"],
            fin_block=views["fin_block"],
            fin_group=views["fin_group"],
            fin_res_slot=views["fin_res_slot"],
            fin_rows=views["fin_rows"],
            res_set_block=views["res_set_block"],
            res_set_slot=views["res_set_slot"],
            res_set_group=views["res_set_group"],
            res_clear_block=views["res_clear_block"],
            res_clear_slot=views["res_clear_slot"],
            evictions=evictions,
            n_open=int(open_blocks.size),
            n_fin=int(fin_arr.shape[0]),
            n_sealed=int(tok_sealed.sum()),
            block_size=bs,
            k_group=g,
            step=self.step,
            slots_np=slots,
            open_np=open_blocks,
            touched_np=touched,
            touched_fill=touched_fill,
            touched_count=touched_count,
            res_index=views["res_index"],
            res_pos=views["res_pos"],
            host_buffer=host,
        )
