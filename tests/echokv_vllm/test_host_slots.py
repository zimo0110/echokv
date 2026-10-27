"""Host slot mapping against a device-style reference, for both runners' block tables."""

from types import SimpleNamespace

import numpy as np
import torch

from echokv_vllm import host_slots
from echokv_vllm.host_slots import host_lens_and_slots, install_v2

BS = 16


def reference_slots(table_rows, qsl, seq, num_tokens):
    """What vLLM's slot-mapping kernel computes, row by row."""
    out = torch.full((num_tokens,), -1, dtype=torch.int64)
    for r in range(len(seq)):
        qs, qe = qsl[r], qsl[r + 1]
        for i, p in enumerate(range(seq[r] - (qe - qs), seq[r])):
            out[qs + i] = int(table_rows[r][p // BS]) * BS + p % BS
    return out


def make_cm(table, qsl, seq, num_tokens, req_idx=None):
    return SimpleNamespace(
        seq_lens_cpu_upper_bound=torch.tensor(seq, dtype=torch.int32),
        block_table_tensor=table,
        num_reqs=len(seq),
        query_start_loc_cpu=torch.tensor(qsl, dtype=torch.int32),
        num_actual_tokens=num_tokens,
        req_idx=req_idx,
    )


class FakeBlockTables:
    """The parts of vLLM's V2 ``BlockTables`` that host_slots relies on."""

    redirect_writes_to_null_block = False

    def __init__(self, max_reqs, max_blocks):
        self.num_kv_cache_groups = 1
        self.block_sizes = [BS]
        self.cp_size = 1
        self.block_tables = [SimpleNamespace(gpu=torch.zeros(max_reqs, max_blocks, dtype=torch.int32))]
        self.num_blocks = SimpleNamespace(np=np.zeros((1, max_reqs), dtype=np.int32))
        self.input_block_tables = [torch.zeros_like(self.block_tables[0].gpu)]

    def append_block_ids(self, req_index, new_block_ids, overwrite):
        start = 0 if overwrite else self.num_blocks.np[0, req_index]
        ids = new_block_ids[0]
        self.block_tables[0].gpu[req_index, start : start + len(ids)] = torch.tensor(ids, dtype=torch.int32)
        self.num_blocks.np[0, req_index] = start + len(ids)

    def gather(self, idx_mapping):
        self.input_block_tables[0][: len(idx_mapping)] = self.block_tables[0].gpu[idx_mapping]
        return self.input_block_tables[0][: len(idx_mapping)]


def test_v2_mirror_tracks_appends_and_slot_reuse():
    install_v2(FakeBlockTables)
    bt = FakeBlockTables(max_reqs=4, max_blocks=8)
    bt.append_block_ids(2, ([7, 3],), overwrite=True)
    bt.append_block_ids(0, ([5],), overwrite=True)
    bt.append_block_ids(2, ([11],), overwrite=False)  # request in slot 2 grows
    bt.append_block_ids(0, ([9, 1],), overwrite=True)  # slot 0 reused by a new request
    # Batch: row 0 = slot 2 decoding at position 40, row 1 = slot 0 prefilling 20 tokens, row 2 padding.
    req_idx = np.array([2, 0])
    table = bt.gather(torch.tensor(req_idx))
    qsl, seq, num_tokens = [0, 1, 21, 21], [41, 20, 0], 24
    cm = make_cm(table, qsl, seq, num_tokens, req_idx)
    lens, slots = host_lens_and_slots(cm, BS)
    assert lens.tolist() == seq
    rows = [[7, 3, 11], [9, 1], []]
    assert torch.equal(slots, reference_slots(rows, qsl, seq, num_tokens))


def test_unknown_table_or_block_size_falls_back():
    install_v2(FakeBlockTables)
    bt = FakeBlockTables(max_reqs=2, max_blocks=4)
    bt.append_block_ids(0, ([1],), overwrite=True)
    table = bt.gather(torch.tensor([0]))
    cm = make_cm(table, [0, 1], [3], 1, np.array([0]))
    assert host_lens_and_slots(cm, BS * 2) is None  # block size mismatch
    assert host_lens_and_slots(make_cm(torch.zeros(1, 4), [0, 1], [3], 1), BS) is None  # not a known table
    cm.seq_lens_cpu_upper_bound = None
    assert host_lens_and_slots(cm, BS) is None


def test_v1_table_indexed_by_batch_row():
    t = SimpleNamespace(
        block_size=BS, dcp_world_size=1, pcp_world_size=1,
        block_table=SimpleNamespace(gpu=torch.zeros(3, 4, dtype=torch.int32), np=np.array([[4, 6, 0, 0], [2, 0, 0, 0], [0] * 4])),
    )
    # The V1 lookup itself runs in test_vllm_e2e (CPU uses the V1 runner); here only the indexing.
    orig = host_slots._find_rows
    host_slots._find_rows = lambda cm, bs: (t.block_table.np, None)
    try:
        qsl, seq = [0, 3, 4], [20, 9]
        cm = make_cm(t.block_table.gpu, qsl, seq, 4)
        _, slots = host_lens_and_slots(cm, BS)
        assert torch.equal(slots, reference_slots(t.block_table.np, qsl, seq, 4))
    finally:
        host_slots._find_rows = orig
