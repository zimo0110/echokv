"""Unit tests for core.block_allocator.

Covers allocator behavior and a local statement of INV-1 (no
per-block auxiliary buffer) and INV-2 (no rewrite API) at the
module-surface level. The full cross-module statement of
INV-1..INV-5 lives in tests/test_invariants.py.
"""

from __future__ import annotations

import pytest

from core.block_allocator import (
    BlockAllocator,
    OutOfMemory,
    PhysicalTokenBlock,
    Precision,
    SUPPORTED_PRECISIONS,
)


def _make(pool=(4, 8, 4)):
    return BlockAllocator(
        pool_sizes={
            Precision.INT2: pool[0],
            Precision.INT4: pool[1],
            Precision.INT8: pool[2],
        }
    )


def test_construction_requires_all_precisions():
    with pytest.raises(ValueError):
        BlockAllocator(pool_sizes={Precision.INT4: 8})


def test_construction_rejects_negative_pool():
    with pytest.raises(ValueError):
        BlockAllocator(
            pool_sizes={
                Precision.INT2: 0,
                Precision.INT4: -1,
                Precision.INT8: 0,
            }
        )


def test_allocate_and_free_roundtrip():
    alloc = _make()
    blocks = alloc.allocate_with_precision(
        layer_id=0, precision=Precision.INT4, num_blocks=3
    )
    assert len(blocks) == 3
    assert alloc.num_free(Precision.INT4) == 5
    assert alloc.num_allocated(Precision.INT4) == 3
    for b in blocks:
        assert b.precision == Precision.INT4
        assert b.layer_id == 0
        assert b.ref_count == 1
        alloc.free(b)
    assert alloc.num_free(Precision.INT4) == 8
    assert alloc.num_allocated() == 0


def test_allocate_at_each_precision():
    alloc = _make()
    for p in SUPPORTED_PRECISIONS:
        b = alloc.allocate_with_precision(
            layer_id=1, precision=p, num_blocks=1
        )[0]
        assert b.precision == p


def test_out_of_memory_raises():
    alloc = _make(pool=(0, 2, 0))
    alloc.allocate_with_precision(
        layer_id=0, precision=Precision.INT4, num_blocks=2
    )
    with pytest.raises(OutOfMemory):
        alloc.allocate_with_precision(
            layer_id=0, precision=Precision.INT4, num_blocks=1
        )


def test_out_of_memory_does_not_silently_downgrade():
    """Allocator must not fall back to another pool — the caller
    (or the controller) is responsible for any downgrade decision.
    """
    alloc = _make(pool=(4, 0, 4))
    with pytest.raises(OutOfMemory):
        alloc.allocate_with_precision(
            layer_id=0, precision=Precision.INT4, num_blocks=1
        )
    # the INT2 and INT8 pools are untouched.
    assert alloc.num_free(Precision.INT2) == 4
    assert alloc.num_free(Precision.INT8) == 4


def test_refcount_prevents_premature_free():
    alloc = _make()
    b = alloc.allocate_with_precision(
        layer_id=0, precision=Precision.INT8, num_blocks=1
    )[0]
    alloc.incref(b)
    assert b.ref_count == 2
    alloc.free(b)
    assert b.ref_count == 1
    assert alloc.num_allocated() == 1
    alloc.free(b)
    assert alloc.num_allocated() == 0


def test_free_rejects_unknown_block():
    alloc = _make()
    rogue = PhysicalTokenBlock(
        block_id=9999, precision=Precision.INT4, layer_id=0, ref_count=1
    )
    with pytest.raises(ValueError):
        alloc.free(rogue)


def test_kv_utilization_tracks_allocations():
    alloc = _make(pool=(2, 2, 2))
    assert alloc.kv_utilization() == 0.0
    blocks = alloc.allocate_with_precision(
        layer_id=0, precision=Precision.INT4, num_blocks=2
    )
    assert alloc.kv_utilization() == pytest.approx(2.0 / 6.0)
    for b in blocks:
        alloc.free(b)
    assert alloc.kv_utilization() == 0.0


def test_inv1_no_per_block_auxiliary_fields():
    """INV-1 local form: PhysicalTokenBlock exposes only the minimal
    fields the allocator needs. No FP8 shadow, no refresh buffer,
    no "original precision" etc.
    """
    allowed = {"block_id", "precision", "layer_id", "ref_count"}
    present = {
        name for name in vars(
            PhysicalTokenBlock(
                block_id=0, precision=Precision.INT4, layer_id=0
            )
        )
    }
    # hidden dunder fields from dataclass internals are not in vars()
    assert present == allowed, (
        f"unexpected fields on PhysicalTokenBlock: {present - allowed}"
    )


def test_inv2_no_rewrite_api_on_allocator():
    """INV-2 local form: BlockAllocator does not expose any entry
    point named refresh/repair/retroactive/rewrite."""
    forbidden = {"refresh", "repair", "retroactive", "rewrite", "migrate"}
    public = {
        name for name in dir(BlockAllocator) if not name.startswith("_")
    }
    assert not (public & forbidden), (
        f"forbidden rewrite API on BlockAllocator: {public & forbidden}"
    )
