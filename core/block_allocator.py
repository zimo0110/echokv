"""Precision-aware block allocator for EchoKV.

Three disjoint free-list pools (INT2 / INT4 / INT8) back paged KV
allocation. The controller emits a target precision per layer; this
allocator serves that target at block-granularity for *new* allocations
only.

Invariants enforced in this module
----------------------------------
INV-1  No per-block auxiliary buffer. A block carries only its
       precision tag plus data the upstream paging system already
       tracks. The precision tag is stored in-line on the block
       object; nothing is allocated per block beyond that.

INV-2  No in-place rewrite of an existing block. The public API
       exposes ``allocate_with_precision`` and ``free``. Migration
       between pools is not supported; any "change precision" intent
       is realised by freeing an old block and allocating a new one
       at the new precision. There is intentionally no
       ``refresh``/``repair``/``retroactive``/``rewrite`` entry
       point. ``tests/test_invariants.py`` enforces this by
       introspecting the public surface.

This implementation is a pure-Python reference. The production path
will wrap vLLM V1's block manager; keeping a reference here lets us
unit-test invariants in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional


class Precision(IntEnum):
    """KV block precision levels.

    Values are the bit-width; used both as enum identity and as the
    numeric precision when sizing pools.
    """

    INT2 = 2
    INT4 = 4
    INT8 = 8


SUPPORTED_PRECISIONS = (Precision.INT2, Precision.INT4, Precision.INT8)


@dataclass
class PhysicalTokenBlock:
    """A single paged KV block.

    The only extra state beyond upstream vLLM's PhysicalTokenBlock is
    the ``precision`` tag. This tag is intrinsic to where the block
    lives (which pool), not an auxiliary buffer. It costs 2 bits per
    block in the block table and is not charged against the KV pool
    (INV-1).
    """

    block_id: int
    precision: Precision
    layer_id: int
    # ``ref_count`` mirrors upstream semantics: the allocator does not
    # free a block while it is still referenced by a sequence.
    ref_count: int = 0


class OutOfMemory(RuntimeError):
    """Raised when the requested pool has no free blocks."""


class BlockAllocator:
    """Three-pool precision-aware allocator.

    Pool sizes are fixed at construction time. Dynamic resizing of
    pools is deferred — see REPORT.md § Open Questions (adaptive pool
    sizing risks thrashing).
    """

    def __init__(self, pool_sizes: Dict[Precision, int]) -> None:
        for p in SUPPORTED_PRECISIONS:
            if p not in pool_sizes:
                raise ValueError(f"pool_sizes missing precision {p.name}")
            if pool_sizes[p] < 0:
                raise ValueError(
                    f"pool size for {p.name} must be non-negative"
                )

        self._pool_sizes: Dict[Precision, int] = dict(pool_sizes)
        self._free_lists: Dict[Precision, List[PhysicalTokenBlock]] = {
            p: [] for p in SUPPORTED_PRECISIONS
        }
        self._allocated: Dict[int, PhysicalTokenBlock] = {}

        next_id = 0
        for p in SUPPORTED_PRECISIONS:
            for _ in range(pool_sizes[p]):
                blk = PhysicalTokenBlock(
                    block_id=next_id, precision=p, layer_id=-1
                )
                self._free_lists[p].append(blk)
                next_id += 1
        self._total_blocks = next_id

    # ------------------------------------------------------------------
    # public API — allocate / free only. Intentionally no rewrite path.
    # ------------------------------------------------------------------

    def allocate_with_precision(
        self,
        layer_id: int,
        precision: Precision,
        num_blocks: int = 1,
    ) -> List[PhysicalTokenBlock]:
        """Allocate ``num_blocks`` blocks at ``precision`` for ``layer_id``.

        Raises :class:`OutOfMemory` if the corresponding pool cannot
        satisfy the request. The caller decides how to respond
        (typically: fall back to a lower precision via the controller,
        or trigger preemption). This allocator does not silently
        downgrade — doing so would violate the controller's contract.
        """
        if precision not in SUPPORTED_PRECISIONS:
            raise ValueError(f"unsupported precision: {precision}")
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")

        pool = self._free_lists[precision]
        if len(pool) < num_blocks:
            raise OutOfMemory(
                f"pool {precision.name} has {len(pool)} free blocks, "
                f"requested {num_blocks}"
            )

        out: List[PhysicalTokenBlock] = []
        for _ in range(num_blocks):
            blk = pool.pop()
            blk.layer_id = layer_id
            blk.ref_count = 1
            self._allocated[blk.block_id] = blk
            out.append(blk)
        return out

    def free(self, block: PhysicalTokenBlock) -> None:
        """Decrement ref count; return to its native pool at zero."""
        if block.block_id not in self._allocated:
            raise ValueError(
                f"block {block.block_id} is not currently allocated"
            )
        if block.ref_count <= 0:
            raise ValueError(
                f"block {block.block_id} has non-positive ref_count "
                f"{block.ref_count}"
            )
        block.ref_count -= 1
        if block.ref_count == 0:
            del self._allocated[block.block_id]
            block.layer_id = -1
            self._free_lists[block.precision].append(block)

    def incref(self, block: PhysicalTokenBlock) -> None:
        """Increment ref count (for copy-on-write / prefix sharing)."""
        if block.block_id not in self._allocated:
            raise ValueError(
                f"block {block.block_id} is not currently allocated"
            )
        block.ref_count += 1

    # ------------------------------------------------------------------
    # inspection helpers — read-only.
    # ------------------------------------------------------------------

    def num_free(self, precision: Precision) -> int:
        return len(self._free_lists[precision])

    def num_allocated(self, precision: Optional[Precision] = None) -> int:
        if precision is None:
            return len(self._allocated)
        return sum(
            1
            for b in self._allocated.values()
            if b.precision == precision
        )

    def kv_utilization(self) -> float:
        """Fraction of total blocks currently allocated, [0, 1]."""
        if self._total_blocks == 0:
            return 0.0
        return len(self._allocated) / self._total_blocks

    @property
    def total_blocks(self) -> int:
        return self._total_blocks

    @property
    def pool_sizes(self) -> Dict[Precision, int]:
        return dict(self._pool_sizes)
