"""SLO-Coupled Precision Budget — interface stubs.

Maps SLO pressure (p99 TPOT vs. configured target) into a scalar
``precision_budget`` ∈ [0, 1] consumed by the PID controller. Under
burst load, a low budget relaxes per-layer setpoints → new blocks
allocate at lower precision → KV pool shrinks → batch grows.

This module is also the TP-correctness barrier. The scheduler is the
single point of truth for which requests run next; before admitting a
new decode step, it broadcasts the current b_ℓ vector to all TP ranks
over the existing global barrier. That is what makes INV-3 hold on a
single node with NVLink-connected GPUs. Multi-node TP is out of
scope (see ``ARCHITECTURE.md § 4.3``).

Invariants
----------
INV-3  No asynchronous b_ℓ update across TP ranks. The broadcast
       point is this module's ``broadcast_precision`` call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .block_allocator import Precision


DEFAULT_TARGET_P99_TPOT_MS: float = 50.0  # SLO for Llama-3.1-8B
DEFAULT_MIN_BUDGET: float = 0.1
DEFAULT_MAX_BUDGET: float = 1.0


@dataclass
class SLOBudget:
    """Rolling SLO tracker → precision budget scalar.

    Parameters
    ----------
    target_p99_tpot_ms:
        SLO target for per-output-token time. Model-dependent.
    min_budget, max_budget:
        Clamp range; 0.1..1.0 keeps the system responsive without
        collapsing all layers to INT2.
    """

    target_p99_tpot_ms: float = DEFAULT_TARGET_P99_TPOT_MS
    min_budget: float = DEFAULT_MIN_BUDGET
    max_budget: float = DEFAULT_MAX_BUDGET

    _current_p99_ms: Optional[float] = None
    _current_budget: float = DEFAULT_MAX_BUDGET

    def update_p99(self, p99_tpot_ms: float) -> None:
        """Record a fresh p99 TPOT measurement (from vLLM metrics)."""
        raise NotImplementedError

    def budget(self) -> float:
        """Current precision budget ∈ [min_budget, max_budget]."""
        raise NotImplementedError

    def broadcast_precision(
        self,
        levels: Sequence[Precision],
        tp_world_size: int,
    ) -> None:
        """Broadcast the current per-layer precision vector to all TP ranks.

        Must be called from the scheduler thread at a step boundary,
        before admitting the next decode step. This is the point
        where INV-3 is enforced.
        """
        raise NotImplementedError
