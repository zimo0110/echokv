"""Fused prefill attention over a TieredKVCache: Triton kernel vs PyTorch reference."""

import pytest
import torch

from echokv_vllm.kernels import prefill as pf
from test_decode_kernel import DEVICE, TRITON_OK, build

CASES = [
    # h, d, bs, mem, ctx per request, chunk length per request, tier plan
    (2, 64, 64, 8, [70, 200, 33], [70, 40, 1], [8, 4, 2]),  # full prompt, mid-context chunk, 1 token
    (8, 128, 64, 8, [300, 130], [17, 130], [4, 8]),  # Llama-3.1-8B per-GPU shape
    (1, 128, 32, 4, [97], [50], [8, 8, 4]),  # pool pressure: some blocks open lower
]


def make_batch(case, gqa, seed):
    h, d, bs, mem, ctxs, chunks, plan = case
    store, table, _ = build(h, d, bs, mem, ctxs, plan, seed=seed)
    prefill, req_of, row = [], [], 0
    for r, (ctx, nq) in enumerate(zip(ctxs, chunks)):
        prefill.append((row, row + nq, ctx, 0))
        req_of.append(r)
        row += nq
    query = torch.randn(row, gqa * h, d, device=DEVICE)
    return store, table, prefill, req_of, query


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("gqa", [1, 4, 7])
def test_prefill_kernel_matches_reference(case, gqa):
    if not TRITON_OK:
        pytest.skip("needs CUDA or TRITON_INTERPRET=1")
    store, table, prefill, req_of, query = make_batch(case, gqa, seed=gqa)
    d = query.shape[-1]
    ref = torch.zeros_like(query)
    pf.prefill_attention_torch(store, query, ref, prefill, table, d**-0.5, req_of=req_of)
    got = torch.zeros_like(query)
    pf._prefill_triton(store, query, got, pf.build_tiles(prefill, req_of, gqa, DEVICE), table, d**-0.5)
    torch.testing.assert_close(got, ref, rtol=5e-3, atol=5e-3)  # fp16 dot operands


def test_tiles_cover_every_query_row():
    prefill = [(0, 70, 70, 0), (70, 110, 200, 0), (110, 111, 33, 0)]
    for gqa in (1, 4, 7):
        tiles = pf.build_tiles(prefill, [0, 1, 2], gqa, "cpu")
        covered = sorted(r for q0, n, *_ in tiles.tolist() for r in range(q0, q0 + n))
        assert covered == list(range(111))
        assert all(n <= pf.block_m(gqa) for _, n, *_ in tiles.tolist())
        # Each tile's first query position is ctx - chunk + offset within the chunk.
        assert tiles[0].tolist()[2] == 0 and any(t[2] == 160 for t in tiles.tolist())
