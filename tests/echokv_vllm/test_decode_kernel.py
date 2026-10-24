"""Decode attention over a TieredKVCache: Triton kernel vs PyTorch reference.

The Triton cases run on CUDA, or on CPU under ``TRITON_INTERPRET=1`` (see
docker/triton-interp). Elsewhere only the reference checks run.
"""

import os

import pytest
import torch
import torch.nn.functional as F

from echokv_vllm.kernels import decode as dec
from echokv_vllm.tiered_cache import TieredKVCache, TieredLayout

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRITON_OK = dec.HAS_TRITON and (DEVICE == "cuda" or os.environ.get("TRITON_INTERPRET") == "1")


def build(h, d, bs, mem_bits, seq_lens, tier_plan, seed=0, num_blocks=None):
    """Fill a store with ``len(seq_lens)`` sequences; block i of each opens at tier_plan[i % n]."""
    g = torch.Generator().manual_seed(seed)
    blocks_per_seq = [-(-n // bs) for n in seq_lens]
    num_blocks = num_blocks or sum(blocks_per_seq) + 3
    layout = TieredLayout(num_kv_heads=h, head_size=d, block_size=bs, mem_bits=mem_bits)
    per_tok = TieredLayout.per_token_bytes(h, d, mem_bits)
    pages = torch.zeros(num_blocks, bs * per_tok + 32, dtype=torch.uint8, device=DEVICE)[:, : bs * per_tok]
    store = TieredKVCache(layout, pages, residual_slots=len(seq_lens) + 2)
    perm = torch.randperm(num_blocks, generator=g)
    table = torch.zeros(len(seq_lens), max(blocks_per_seq), dtype=torch.int32)
    nxt = 0
    for s, nb in enumerate(blocks_per_seq):
        table[s, :nb] = perm[nxt : nxt + nb].to(torch.int32)
        nxt += nb
    table = table.to(DEVICE)
    # Write block by block, interleaving sequences, so tiers and residuals mix.
    for bi in range(max(blocks_per_seq)):
        for s, n in enumerate(seq_lens):
            lo, hi = bi * bs, min((bi + 1) * bs, n)
            if lo >= hi:
                continue
            k = torch.randn(hi - lo, h, d, generator=g)
            k[..., 1] += 30.0  # outlier channel
            v = torch.randn(hi - lo, h, d, generator=g)
            slots = table[s, bi].long() * bs + torch.arange(hi - lo, device=DEVICE)
            store.write(k.to(DEVICE), v.to(DEVICE), slots, tier_plan[(bi + s) % len(tier_plan)])
    return store, table, torch.tensor(seq_lens, device=DEVICE)


def naive(store, query, table, lens, scale):
    out = []
    bs = store.layout.block_size
    rep = query.shape[1] // store.layout.num_kv_heads
    for s in range(query.shape[0]):
        n = int(lens[s])
        pos = torch.arange(n, device=DEVICE)
        k, v = store.gather(table[s].long()[pos // bs], pos % bs)
        o = F.scaled_dot_product_attention(
            query[s].float().unsqueeze(1),
            k.repeat_interleave(rep, 1).transpose(0, 1),
            v.repeat_interleave(rep, 1).transpose(0, 1),
            scale=scale,
        )
        out.append(o.squeeze(1))
    return torch.stack(out)


CASES = [
    # h, d, bs, mem_bits, seq_lens, tier plan
    (2, 64, 64, 8, [1, 40, 64, 65, 200], [8, 4, 2]),
    (1, 128, 32, 4, [33, 97, 128], [8, 8, 4]),  # pool runs dry: some blocks downgrade
    (4, 64, 64, 6, [500, 3], [2, 4]),
]


@pytest.mark.parametrize("case", CASES)
def test_batched_reference_matches_per_sequence(case):
    h, d, bs, mem, lens, plan = case
    store, table, lens_t = build(h, d, bs, mem, lens, plan)
    q = torch.randn(len(lens), 4 * h, d, device=DEVICE)
    ref = naive(store, q, table, lens_t, d**-0.5)
    got = dec.decode_attention_torch(store, q, table, lens_t, d**-0.5)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not TRITON_OK, reason="needs CUDA or TRITON_INTERPRET=1")
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("splits", [1, 3])
def test_triton_kernel_matches_reference(case, splits):
    """``splits`` is blocks per split: 1 = many splits, 3 = few."""
    h, d, bs, mem, lens, plan = case
    store, table, lens_t = build(h, d, bs, mem, lens, plan, seed=1)
    for gqa in (1, 4):
        q = torch.randn(len(lens), gqa * h, d, device=DEVICE)
        ref = dec.decode_attention_torch(store, q, table, lens_t, d**-0.5)
        got = dec._decode_triton(store, q, table, lens_t, d**-0.5, splits)
        torch.testing.assert_close(got, ref, rtol=2e-4, atol=2e-4)
    assert store.residual_free < store.res_k.shape[0]  # residual path exercised
    assert len(set(store.tier_histogram()[b] > 0 for b in (2, 4, 8))) >= 1


@pytest.mark.skipif(not TRITON_OK, reason="needs CUDA or TRITON_INTERPRET=1")
def test_triton_kernel_is_batch_invariant():
    h, d, bs, mem, lens, plan = CASES[0]
    store, table, lens_t = build(h, d, bs, mem, lens, plan, seed=2)
    q = torch.randn(len(lens), 2 * h, d, device=DEVICE)
    full = dec._decode_triton(store, q, table, lens_t, d**-0.5, 2)
    for i in range(len(lens)):  # each sequence alone, with a narrower block table
        nb = -(-lens[i] // bs)
        alone = dec._decode_triton(store, q[i : i + 1], table[i : i + 1, :nb], lens_t[i : i + 1], d**-0.5, 2)
        assert torch.equal(alone[0], full[i])
