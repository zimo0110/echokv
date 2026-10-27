"""Slot mapping and sequence lengths on the host, without a GPU sync.

vLLM 0.32 computes ``slot_mapping`` on the GPU (from the block table and
positions), so ``CommonAttentionMetadata`` carries it only as a device
tensor. The write planner needs it on the host, and copying it back waits
for the stream to drain: with async scheduling that is the whole previous
step, so the CPU can no longer prepare step N+1 while the GPU runs step N.

The runner's CPU block table (``BlockTable.block_table.np``) is the source
the GPU copy is made from, and positions follow from ``query_start_loc_cpu``
and ``seq_lens_cpu_upper_bound`` (exact unless async speculative decoding
is on). ``install`` records every ``BlockTable`` vLLM creates; the builder
then finds the one whose device tensor is ``block_table_tensor`` and
computes the same mapping in numpy.
"""

from __future__ import annotations

import weakref

import numpy as np
import torch

_tables: weakref.WeakSet = weakref.WeakSet()


def install() -> None:
    """Record ``BlockTable`` instances as vLLM creates them (idempotent)."""
    try:
        from vllm.v1.worker.block_table import BlockTable
    except ImportError:
        return
    if getattr(BlockTable.__init__, "_echokv", False):
        return
    orig = BlockTable.__init__

    def __init__(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        _tables.add(self)

    __init__._echokv = True
    BlockTable.__init__ = __init__


def _find_table(block_table_tensor: torch.Tensor, block_size: int):
    ptr = block_table_tensor.data_ptr()
    for t in list(_tables):
        buf = getattr(t, "block_table", None)
        if buf is None or buf.gpu.data_ptr() != ptr:
            continue
        if t.block_size != block_size or t.dcp_world_size != 1 or t.pcp_world_size != 1:
            return None
        return t
    return None


def host_lens_and_slots(cm, block_size: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(seq_lens[:num_reqs], slot_mapping[:num_actual_tokens])`` as CPU tensors, or None.

    None when the host copies cannot be trusted (no CPU sequence lengths, an
    unknown block table, context parallelism); the caller then copies the
    device tensors back. Padding tokens map to -1, as on the device.
    """
    seq = cm.seq_lens_cpu_upper_bound
    if seq is None:
        return None
    table = _find_table(cm.block_table_tensor, block_size)
    if table is None:
        return None
    n = cm.num_reqs
    qsl = cm.query_start_loc_cpu[: n + 1].numpy().astype(np.int64)
    seq_np = seq[:n].numpy().astype(np.int64)
    qlen = np.diff(qsl)
    total = int(qsl[-1])
    if total > cm.num_actual_tokens:
        return None
    req = np.repeat(np.arange(n), qlen)
    pos = np.arange(total) - qsl[:-1][req] + (seq_np - qlen)[req]
    slots = np.full(cm.num_actual_tokens, -1, dtype=np.int64)
    slots[:total] = table.block_table.np[req, pos // block_size].astype(np.int64) * block_size + pos % block_size
    return torch.from_numpy(seq_np), torch.from_numpy(slots)
