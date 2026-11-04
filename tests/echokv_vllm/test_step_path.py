"""The per-step path a CUDA graph replays: host plan, one upload, masked full-grid kernels.

Under ``TRITON_INTERPRET=1`` (or on CUDA) the writes go through the Triton
kernels; elsewhere through the PyTorch path.
"""

import numpy as np
import pytest
import torch

from echokv_vllm.canary import KRefRing, canary_kl, ring_view
from echokv_vllm.tiered_cache import TieredKVCache, TieredLayout
from echokv_vllm.write_plan import stage_plan

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
H, D, BS, KG, VG = 2, 64, 64, 32, 32


def make_cache(num_blocks=8, mem_bits=8, residual_slots=8):
    layout = TieredLayout(num_kv_heads=H, head_size=D, block_size=BS, mem_bits=mem_bits, k_group=KG, v_group=VG)
    page = BS * TieredLayout.per_token_bytes(H, D, mem_bits, KG, VG)
    pages = torch.zeros(num_blocks, page, dtype=torch.uint8, device=DEVICE)
    return TieredKVCache(layout, pages, residual_slots=residual_slots)


def steps():
    """Two requests: a prefill (A: 40 tokens in block 0, B: 10 in block 3), then
    decode steps that complete groups from the residual and cross into new blocks."""
    yield np.r_[np.arange(40), 3 * BS + np.arange(10)]
    for i in range(60):
        a, b = 40 + i, 10 + i
        yield np.array([[0, 5][a // BS] * BS + a % BS, [3, 6][b // BS] * BS + b % BS])


def test_padded_full_grid_step_matches_plain_write():
    """Padding rows (slot -1, garbage K/V) and full-grid launches masked by the
    device counts, as in a captured decode graph, store exactly the same bytes;
    the K_ref ring holds the last keys in order."""
    plain, graph = make_cache(), make_cache()
    kr_plain = torch.zeros(KG, H, D, device=DEVICE)
    kr_graph = torch.zeros(KG, H, D, device=DEVICE)
    ring = KRefRing(KG)  # a standalone store's K_ref window is one K group
    g = torch.Generator().manual_seed(0)
    for i, sl in enumerate(steps()):
        bits = (8, 4, 2)[(i // 7) % 3]  # tiers change mid-run
        n = sl.size
        k = torch.randn(n + 3, H, D, generator=g)
        k[..., 1] += 30.0
        v = torch.randn(n + 3, H, D, generator=g)
        k, v = k.to(DEVICE), v.to(DEVICE)
        slots = torch.from_numpy(sl).to(DEVICE)
        plain.write(k[:n], v[:n], slots, bits, kref=kr_plain)
        ring.push(k[:n].cpu())

        padded = torch.cat([slots, torch.full((3,), -1, device=DEVICE)])
        plan = graph.planner.plan(padded.cpu())
        opened = graph.prepare(plan, bits)
        graph.shared.commit(plan)
        meta, extra = stage_plan(plan, DEVICE, opened)
        graph.shared.apply_res_map(*extra[:4])
        if opened is not None:
            graph.write_headers(extra[4], extra[5])
        graph.apply(meta, k, v, kref=kr_graph, capturing=True)

    assert torch.equal(plain.pages, graph.pages)
    assert torch.equal(plain.block_res, graph.block_res)
    live = plain.block_res[plain.block_res >= 0]
    assert live.numel() and torch.equal(plain.res_k[live], graph.res_k[live])
    pl = graph.planner
    torch.testing.assert_close(ring_view(kr_graph, pl.kref_pos, pl.kref_fill).cpu(), ring.view(), rtol=0, atol=0)
    torch.testing.assert_close(kr_graph, kr_plain, rtol=0, atol=0)


def test_tokens_written_counts_each_tier():
    cache = make_cache()
    total = 0
    for i, sl in enumerate(steps()):
        k = torch.randn(sl.size, H, D, device=DEVICE)
        cache.write(k, k, torch.from_numpy(sl).to(DEVICE), (8, 4)[i % 2])
        total += sl.size
    tw = cache.tokens_written
    assert sum(tw.values()) == total and tw[8] > 0 and tw[4] > 0
    cache.reset_counters()
    assert sum(cache.tokens_written.values()) == 0


def test_batched_canary_matches_per_layer():
    g = torch.Generator().manual_seed(1)
    q = torch.randn(3, 5, 8, 64, generator=g) * 3
    kref = torch.randn(3, 70, 2, 64, generator=g) * 3  # a partial trailing group is skipped
    batched = canary_kl(q, kref, (2, 4, 8), KG, 64**-0.5)
    each = torch.stack([canary_kl(q[i], kref[i], (2, 4, 8), KG, 64**-0.5) for i in range(3)])
    assert batched.shape == (3, 3)
    torch.testing.assert_close(batched, each, rtol=1e-5, atol=1e-6)
    # Model-dtype inputs (bf16 query rows, or a K_ref ring built in bf16) work too.
    low = canary_kl(q.bfloat16(), kref.bfloat16(), (2, 4, 8), KG, 64**-0.5)
    assert low.dtype == torch.float32 and low.shape == (3, 3)


def test_ring_view_along_a_layer_dim():
    buf = torch.arange(8.0).view(2, 4)  # two layers, window 4
    assert ring_view(buf, 0, 3, dim=1).tolist() == [[0, 1, 2], [4, 5, 6]]
    assert ring_view(buf, 1, 4, dim=1).tolist() == [[1, 2, 3, 0], [5, 6, 7, 4]]


def test_staging_layout():
    backend = pytest.importorskip("echokv_vllm.backend")
    st = backend._Staging(cap=8, nq=2, w=3, tail=16, device="cpu")

    def stage(t, f, big=0):
        decode = (np.array([5, 6]), np.array([0, 1]), np.array([10, 11]))
        tok = tuple(np.arange(t) * 10 + k for k in range(4))
        fin = np.arange(f * 3).reshape(f, 3) + 100
        tail = [np.array([1, 2, 3]), np.zeros(0, np.int64), np.arange(big) + 9]
        return st.stage(f, np.array([3, 0]), decode, tok, fin, tail)

    first = stage(4, 2)
    # (20, 2) does not fit: fresh buffer. (4, 2, 30) fits but its tail does
    # not: the tail is uploaded on its own and the rest stays put.
    for t, f, big in ((4, 2, 1), (6, 1, 1), (20, 2, 1), (4, 2, 30)):
        cnt, can, dec, tok, fin, tail = stage(t, f, big)
        assert cnt.tolist() == [f] and can.tolist() == [3, 0]
        assert [x.tolist() for x in dec] == [[5, 6], [0, 1], [10, 11]]
        assert tok.shape == (4, t) and tok[2].tolist() == (np.arange(t) * 10 + 2).tolist()
        assert fin[:f].reshape(-1).tolist() == list(range(100, 100 + 3 * f))
        assert [x.tolist() for x in tail] == [[1, 2, 3], [], list(range(9, 9 + big))]
        # Graph-visible views keep their addresses while steps fit.
        assert (tok.data_ptr() == first[3].data_ptr()) == (t <= 8)
        assert (fin.data_ptr() == first[4].data_ptr()) == (t <= 8)
