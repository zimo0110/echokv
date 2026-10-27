"""Slot mapping and sequence lengths on the host, without a GPU sync.

vLLM 0.32 computes ``slot_mapping`` on the GPU (from the block table and
positions), so ``CommonAttentionMetadata`` carries it only as a device
tensor. The write planner needs it on the host, and copying it back waits
for the stream to drain: with async scheduling that is the whole previous
step, so the CPU can no longer prepare step N+1 while the GPU runs step N.

Positions follow from ``query_start_loc_cpu`` and ``seq_lens_cpu_upper_bound``
(exact unless async speculative decoding is on); block ids come from a
host copy of the block table. ``install`` hooks both model runners:

* V1 runner (``BlockTable``): its CPU table ``block_table.np`` is the source
  of the GPU copy, indexed by batch row. We record each instance.
* V2 runner (``BlockTables``, the default on CUDA): tables live on the GPU,
  indexed by request slot, written through ``append_block_ids``. We mirror
  those writes in numpy and map batch rows to slots with ``cm.req_idx``.

The builder finds the table whose device tensor is ``block_table_tensor``
and computes the same mapping in numpy.

Dummy steps (warmup, profiling, CUDA-graph capture, idle data-parallel
ranks) use a slot mapping of all -1 on the device, which writes nothing,
while the block-table rows they point at may belong to live requests. They
are detected (V1: inside ``_dummy_run``; V2: ``get_dummy_slot_mappings``
was called after the last ``compute_slot_mappings``) and left to the device
copy, so EchoKV never writes where vLLM would not.
"""

from __future__ import annotations

import weakref

import numpy as np
import torch

_v1_tables: weakref.WeakSet = weakref.WeakSet()
_v2_tables: weakref.WeakSet = weakref.WeakSet()
_v1_dummy_depth = 0  # > 0 while a V1 runner's _dummy_run is on the stack


def _wrap_init(cls, registry: weakref.WeakSet, after=None) -> None:
    if getattr(cls.__init__, "_echokv", False):
        return
    orig = cls.__init__

    def __init__(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        if after is not None:
            after(self)
        registry.add(self)

    __init__._echokv = True
    cls.__init__ = __init__


def _v2_mirror(self) -> None:
    self._echokv_np = [np.zeros(tuple(b.gpu.shape), dtype=np.int64) for b in self.block_tables]


def install_v2(cls) -> None:
    """Mirror ``cls.append_block_ids`` writes on the host (V2 ``BlockTables``)."""
    _wrap_init(cls, _v2_tables, after=_v2_mirror)
    if getattr(cls.append_block_ids, "_echokv", False):
        return
    orig = cls.append_block_ids

    def append_block_ids(self, req_index, new_block_ids, overwrite):
        starts = [0 if overwrite else int(self.num_blocks.np[i, req_index]) for i in range(self.num_kv_cache_groups)]
        orig(self, req_index, new_block_ids, overwrite)
        for i, start in enumerate(starts):
            ids = new_block_ids[i]
            if self.redirect_writes_to_null_block:
                ids = [0] * len(ids)
            self._echokv_np[i][req_index, start : start + len(ids)] = ids

    append_block_ids._echokv = True
    cls.append_block_ids = append_block_ids
    _wrap_flag(cls, "compute_slot_mappings", False)
    _wrap_flag(cls, "get_dummy_slot_mappings", True)


def _wrap_flag(cls, name: str, dummy: bool) -> None:
    """Record on the instance whether its latest slot mapping is a dummy one."""
    orig = getattr(cls, name)
    if getattr(orig, "_echokv", False):
        return

    def wrapper(self, *args, **kwargs):
        self._echokv_dummy = dummy
        return orig(self, *args, **kwargs)

    wrapper._echokv = True
    setattr(cls, name, wrapper)


def install_v1_dummy(cls) -> None:
    """Track ``cls._dummy_run`` (V1 model runner) so its steps use the device copy."""
    orig = cls._dummy_run
    if getattr(orig, "_echokv", False):
        return

    def _dummy_run(self, *args, **kwargs):
        global _v1_dummy_depth
        _v1_dummy_depth += 1
        try:
            return orig(self, *args, **kwargs)
        finally:
            _v1_dummy_depth -= 1

    _dummy_run._echokv = True
    cls._dummy_run = _dummy_run


def install() -> None:
    """Hook vLLM's block tables so the builder can read them on the host (idempotent)."""
    try:
        from vllm.v1.worker.block_table import BlockTable

        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        _wrap_init(BlockTable, _v1_tables)
        install_v1_dummy(GPUModelRunner)
    except ImportError:
        pass
    try:
        from vllm.v1.worker.gpu.block_table import BlockTables

        install_v2(BlockTables)
    except ImportError:
        pass


def is_dummy_step(cm) -> bool:
    """Whether ``cm`` belongs to one of vLLM's dummy steps (all-PAD slot mapping)."""
    ptr = cm.block_table_tensor.data_ptr()
    if any(getattr(t, "block_table", None) is not None and t.block_table.gpu.data_ptr() == ptr
           for t in list(_v1_tables)):
        return _v1_dummy_depth > 0
    for t in list(_v2_tables):
        if any(inp.data_ptr() == ptr for inp in t.input_block_tables):
            return getattr(t, "_echokv_dummy", False)
    return False


def _find_rows(cm, block_size: int) -> tuple[np.ndarray, np.ndarray | None] | None:
    """(host block table, batch row -> table row map or None) for ``cm``, or None."""
    ptr = cm.block_table_tensor.data_ptr()
    for t in list(_v1_tables):
        buf = getattr(t, "block_table", None)
        if buf is None or buf.gpu.data_ptr() != ptr:
            continue
        if t.block_size != block_size or t.dcp_world_size != 1 or t.pcp_world_size != 1 or _v1_dummy_depth:
            return None
        return t.block_table.np, None
    for t in list(_v2_tables):
        for i, inp in enumerate(t.input_block_tables):
            if inp.data_ptr() != ptr:
                continue
            if (t.block_sizes[i] != block_size or t.cp_size != 1 or cm.req_idx is None
                    or getattr(t, "_echokv_dummy", False)):
                return None
            return t._echokv_np[i], np.asarray(cm.req_idx, dtype=np.int64)
    return None


def host_lens_and_slots(cm, block_size: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(seq_lens[:num_reqs], slot_mapping[:num_actual_tokens])`` as CPU tensors, or None.

    None when the host copies cannot be trusted (a dummy step, no CPU sequence
    lengths, an unknown block table, context parallelism); the caller then copies the
    device tensors back. Padding tokens map to -1, as on the device.
    """
    seq = cm.seq_lens_cpu_upper_bound
    if seq is None:
        return None
    found = _find_rows(cm, block_size)
    if found is None:
        return None
    table, row_map = found
    n = cm.num_reqs
    qsl = cm.query_start_loc_cpu[: n + 1].numpy().astype(np.int64)
    seq_np = seq[:n].numpy().astype(np.int64)
    qlen = np.diff(qsl)
    total = int(qsl[-1])
    if total > cm.num_actual_tokens:
        return None
    req = np.repeat(np.arange(n), qlen)
    pos = np.arange(total) - qsl[:-1][req] + (seq_np - qlen)[req]
    if row_map is not None and req.size and req[-1] >= row_map.size:
        return None
    rows = req if row_map is None else row_map[req]
    slots = np.full(cm.num_actual_tokens, -1, dtype=np.int64)
    slots[:total] = table[rows, pos // block_size].astype(np.int64) * block_size + pos % block_size
    return torch.from_numpy(seq_np), torch.from_numpy(slots)
