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

        # Group tokens by (block, group); vLLM lays a request's tokens out contiguously.
        key = blk * (bs // g) + grp
        order = np.argsort(key, kind="stable")
        bounds = np.nonzero(np.diff(key[order]))[0] + 1
        protected = {self.block_slot[b] for b in np.unique(blk).tolist() if b in self.block_slot}
        for idx in np.split(order, bounds):
            if idx.size == 0:
                continue
            b, gr = int(blk[idx[0]]), int(grp[idx[0]])
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

        def t(x, dtype=torch.long):
            # Contiguous: kernels index these with unit stride (column slices of
            # fin_arr would otherwise keep a stride of 3 through .to(device)).
            arr = np.ascontiguousarray(x, dtype=np.int64 if dtype == torch.long else bool)
            return torch.from_numpy(arr).to(device=device, dtype=dtype).contiguous()

        fin_arr = np.asarray(fin, dtype=np.int64).reshape(-1, 3)
        return WritePlan(
            slots=t(slots),
            rows=t(rows),
            open_blocks=t(open_blocks),
            tok_res_slot=t(tok_res),
            tok_sealed=t(tok_sealed, torch.bool),
            fin_block=t(fin_arr[:, 0]),
            fin_group=t(fin_arr[:, 1]),
            fin_res_slot=t(fin_arr[:, 2]),
            fin_rows=t(np.asarray(fin_rows, dtype=np.int64).reshape(-1, g)),
            res_set_block=t(set_b),
            res_set_slot=t(set_s),
            res_set_group=t(set_g),
            res_clear_block=t(clear_b),
            res_clear_slot=t(clear_s),
            evictions=evictions,
            n_open=int(open_blocks.size),
            n_fin=len(fin),
            n_sealed=int(tok_sealed.sum()),
            block_size=bs,
            k_group=g,
            step=self.step,
            slots_np=slots,
            open_np=open_blocks,
            touched_np=np.unique(blk),
        )
