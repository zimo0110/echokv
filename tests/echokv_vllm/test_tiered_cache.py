import pytest
import torch

from echokv_vllm.quant import (
    dequantize_with,
    fake_quant,
    fake_quant_channels,
    quantize_with,
    stats_along,
)
from echokv_vllm.tiered_cache import TieredKVCache, TieredLayout

H, D, BS, KG, VG = 2, 64, 64, 32, 32


def make_cache(num_blocks, mem_bits, row_pad=0, residual_slots=16):
    layout = TieredLayout(num_kv_heads=H, head_size=D, block_size=BS, mem_bits=mem_bits, k_group=KG, v_group=VG)
    page = BS * TieredLayout.per_token_bytes(H, D, mem_bits, KG, VG)
    assert page >= layout.page_bytes
    # Rows padded past the page model layer-interleaved vLLM layouts, where
    # another layer's page sits between consecutive blocks of this layer.
    stride = page + row_pad
    backing = torch.full((num_blocks * stride + 64,), 0xAB, dtype=torch.uint8)
    pages = torch.as_strided(backing, (num_blocks, page), (stride, 1))
    return TieredKVCache(layout, pages, residual_slots=residual_slots), backing, stride, page


def tokens(n, seed, outlier=True):
    g = torch.Generator().manual_seed(seed)
    k, v = torch.randn(n, H, D, generator=g), torch.randn(n, H, D, generator=g)
    if outlier:
        k[..., 3] += 40.0  # an all-positive outlier channel, like real K
    return k, v


def expected(k, v, bits, filled):
    """Oracle for one block's first ``filled`` tokens stored at ``bits``."""
    ek = fake_quant_channels(k[:filled], bits, KG)
    ev = fake_quant(v[:filled], bits, VG)
    return ek, ev


def check_block(cache, b, k, v, bits, filled=BS):
    assert int(cache.tag[b]) == bits
    pos = torch.arange(filled)
    kg, vg = cache.gather(torch.full((filled,), b), pos)
    ek, ev = expected(k, v, bits, filled)
    torch.testing.assert_close(kg, ek, rtol=0, atol=0)
    torch.testing.assert_close(vg, ev, rtol=0, atol=0)


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_uniform_roundtrip(bits):
    cache, *_ = make_cache(6, mem_bits=8)
    k, v = tokens(6 * BS, 0)
    cache.write(k, v, torch.arange(6 * BS), bits)
    for b in range(6):
        check_block(cache, b, k[b * BS :], v[b * BS :], bits)
    assert cache.residual_free == 16  # whole groups never touch the residual


def test_tier_change_is_forward_only():
    cache, *_ = make_cache(6, mem_bits=8)
    k, v = tokens(6 * BS, 1)
    plan = [8, 2, 4, 8, 2, 4]
    for b, bits in enumerate(plan):
        sl = slice(b * BS, (b + 1) * BS)
        cache.write(k[sl], v[sl], torch.arange(b * BS, (b + 1) * BS), bits)
    for b, bits in enumerate(plan):
        check_block(cache, b, k[b * BS :], v[b * BS :], bits)


def test_decode_one_token_at_a_time():
    cache, *_ = make_cache(3, mem_bits=8)
    k, v = tokens(3 * BS, 2)
    n = 2 * BS + 40  # block 2 ends mid-group: its open group stays in the residual
    for t in range(n):
        target = [4, 8, 2][t % 3]  # mid-block changes must not affect the open block
        cache.write(k[t : t + 1], v[t : t + 1], torch.tensor([t]), target)
        if t == 45:  # group 1 of block 0 half full: read through the residual
            check_block(cache, 0, k, v, 4, filled=46)
    opened_at = [[4, 8, 2][(b * BS) % 3] for b in range(3)]
    assert opened_at == [4, 8, 2]
    check_block(cache, 0, k, v, 4)
    check_block(cache, 1, k[BS:], v[BS:], 8)
    check_block(cache, 2, k[2 * BS :], v[2 * BS :], 2, filled=40)
    assert cache.residual_free == 15


def test_prefill_then_decode_completes_group_from_residual():
    cache, *_ = make_cache(2, mem_bits=8)
    k, v = tokens(BS, 3)
    cache.write(k[:20], v[:20], torch.arange(20), 4)  # partial first group
    cache.write(k[20:50], v[20:50], torch.arange(20, 50), 4)  # completes group 0
    cache.write(k[50:], v[50:], torch.arange(50, BS), 4)
    check_block(cache, 0, k, v, 4)


def test_pool_exhaustion_downgrades_without_overlap():
    n = 12
    cache, backing, stride, page = make_cache(n, mem_bits=4, row_pad=48)
    assert cache.pool_size == n  # one spare chunk per page
    k, v = tokens(n * BS, 4)
    cache.write(k, v, torch.arange(n * BS), 8)
    tiers = [int(t) for t in cache.tag]
    assert tiers == [8] * 4 + [2] * 8  # INT8 takes 3 spares each
    assert cache.downgraded_blocks == 8 and cache.pool_free == 0
    for b, bits in enumerate(tiers):
        check_block(cache, b, k[b * BS :], v[b * BS :], bits)
    for b in range(n):
        assert (backing[b * stride + page : (b + 1) * stride] == 0xAB).all()


def test_pool_pressure_demotes_lru_full_block():
    cache, *_ = make_cache(4, mem_bits=4)
    k0, v0 = tokens(BS, 5)
    cache.write(k0, v0, torch.arange(BS), 8)  # takes 3 of 4 spares
    k1, v1 = tokens(BS, 6)
    cache.write(k1, v1, torch.arange(BS, 2 * BS), 8)  # needs 3, 1 left: demote block 0
    assert int(cache.tag[1]) == 8 and cache.demoted_blocks == 1
    check_block(cache, 1, k1, v1, 8)
    # Block 0 now holds the INT2 requantization of its INT8 contents.
    kg, vg = cache.gather(torch.zeros(BS, dtype=torch.long), torch.arange(BS))
    ek, ev = expected(k0, v0, 8, BS)
    torch.testing.assert_close(kg, fake_quant_channels(ek, 2, KG), rtol=0, atol=0)
    torch.testing.assert_close(vg, fake_quant(ev, 2, VG), rtol=0, atol=0)
    assert int(cache.tag[0]) == 2 and cache.pool_free == 1


def test_pool_pressure_without_victims_opens_lower():
    cache, *_ = make_cache(4, mem_bits=4)
    cache.write(*tokens(40, 5), torch.arange(40), 8)  # block 0 still filling: not demotable
    cache.write(*tokens(BS, 6), torch.arange(BS, 2 * BS), 8)  # 1 spare left: INT4 fits
    assert int(cache.tag[0]) == 8 and int(cache.tag[1]) == 4 and cache.demoted_blocks == 0


def test_reopen_returns_chunks_and_residual():
    cache, *_ = make_cache(4, mem_bits=8)
    full = cache.pool_free
    k, v = tokens(4 * BS, 7)
    cache.write(k[: 3 * BS + 10], v[: 3 * BS + 10], torch.arange(3 * BS + 10), 8)
    assert cache.pool_free == full - 4 * 3 and cache.residual_free == 15
    # vLLM frees block 3 (a finished request's partial block) and block 0, then reuses them.
    k2, v2 = tokens(BS, 8)
    cache.write(k2, v2, torch.arange(3 * BS, 4 * BS), 2)
    cache.write(k2, v2, torch.arange(0, BS), 2)
    assert cache.pool_free == full - 2 * 3 and cache.residual_free == 16
    check_block(cache, 3, k2, v2, 2)
    check_block(cache, 1, k[BS:], v[BS:], 8)


def test_interleaved_requests_partial_blocks():
    cache, *_ = make_cache(10, mem_bits=6)
    a = torch.arange(3 * BS, 3 * BS + 21)
    b = torch.arange(7 * BS, 7 * BS + 37)
    k, v = tokens(a.numel() + b.numel(), 9)
    order = torch.randperm(a.numel() + b.numel(), generator=torch.Generator().manual_seed(0))
    slots = torch.cat([a, b])
    # One batch, rows in arbitrary order (vLLM batches requests side by side).
    cache.write(k[order], v[order], slots[order], 4)
    check_block(cache, 3, k[: a.numel()], v[: a.numel()], 4, filled=21)
    check_block(cache, 7, k[a.numel() :], v[a.numel() :], 4, filled=37)


def test_residual_eviction_seals_lru_group():
    cache, *_ = make_cache(6, mem_bits=8, residual_slots=2)
    k, v = tokens(6 * BS, 10)
    for b in range(3):  # three open groups, two slots
        cache.write(k[b * BS : b * BS + 5], v[b * BS : b * BS + 5], torch.arange(b * BS, b * BS + 5), 8)
    assert cache.sealed_groups == 1 and int(cache.sealed[0]) == 1
    # Block 0's request was alive after all: its group keeps filling against sealed stats.
    cache.write(k[5:BS], v[5:BS], torch.arange(5, BS), 8)
    kg, vg = cache.gather(torch.zeros(BS, dtype=torch.long), torch.arange(BS))
    torch.testing.assert_close(vg, fake_quant(v[:BS], 8, VG), rtol=0, atol=0)
    # Sealed group: stats from the 5 tokens present at eviction, later tokens clamped to them.
    scale, mn = stats_along(k[:5], 8, dim=0)
    sealed = dequantize_with(quantize_with(k[:KG], scale, mn, 8), scale, mn)
    torch.testing.assert_close(kg[:KG], sealed, rtol=0, atol=0)
    # The next group was never evicted and is quantized normally.
    torch.testing.assert_close(kg[KG:], fake_quant_channels(k[KG:BS], 8, KG), rtol=0, atol=0)
    check_block(cache, 1, k[BS:], v[BS:], 8, filled=5)
    check_block(cache, 2, k[2 * BS :], v[2 * BS :], 8, filled=5)


def test_layout_rejects_bad_configs():
    with pytest.raises(ValueError):
        TieredLayout(num_kv_heads=2, head_size=64, block_size=8, mem_bits=4)
    with pytest.raises(ValueError):
        TieredLayout(num_kv_heads=2, head_size=64, block_size=32, mem_bits=5)
    with pytest.raises(ValueError):
        TieredLayout(num_kv_heads=2, head_size=64, block_size=48, mem_bits=4, k_group=32)


@pytest.mark.parametrize("h,d,bs,mem", [(8, 128, 32, 4), (2, 64, 64, 8), (1, 128, 256, 2), (4, 96, 32, 6)])
def test_declared_bytes_cover_layout(h, d, bs, mem):
    layout = TieredLayout(num_kv_heads=h, head_size=d, block_size=bs, mem_bits=mem)
    assert bs * TieredLayout.per_token_bytes(h, d, mem) >= layout.page_bytes
