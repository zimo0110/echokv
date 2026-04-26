"""SLO-Coupled Precision Budget — TPOT-pressure → precision_budget signal.

Maps SLO pressure (rolling p99 TPOT vs configured target) into a scalar
``precision_budget`` ∈ ``[min_budget, max_budget]`` ⊆ [0, 1] consumed by
the PID controller. Under burst load, latency rises → budget falls →
controller's τ_eff rises → new blocks allocate at lower precision → KV
pool shrinks → batch grows.

This module is also the **TP-correctness barrier** (INV-3). The PID
controller emits decisions but does not commit them; the SLO budget
calls ``broadcast_precision`` synchronously at a step boundary, which
invokes the registered scheduler callback. The callback is the single
broadcast point: every TP rank either observes the new precision
vector before its next decode step or doesn't see it at all. Because
precision applies only to *new* allocations (INV-2), in-flight blocks
are unaffected and no cross-rank reconciliation is needed.

Single-node TP (≤8 GPUs, NVLink) is in scope; multi-node TP is
explicitly out of scope (see ``ARCHITECTURE.md`` § 4.3).

Invariants
----------
INV-3  ``broadcast_precision`` calls the registered scheduler callback
       synchronously. Tests assert (a) the callback fires on every
       call, (b) it observes the exact levels passed in, and (c) no
       state is mutated between emit and callback. The mock scheduler
       in ``tests/test_invariants.py`` uses this contract.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, List, Optional, Sequence

from .block_allocator import Precision


DEFAULT_TARGET_P99_TPOT_MS: float = 50.0  # SLO for Llama-3.1-8B
DEFAULT_MIN_BUDGET: float = 0.1
DEFAULT_MAX_BUDGET: float = 1.0
DEFAULT_WINDOW_SIZE: int = 200  # TPOT samples; ~200 decode steps


# Scheduler callback signature: (levels, tp_world_size) → None.
# The callback runs synchronously in the caller's thread; it must
# complete before broadcast_precision returns.
BroadcastCallback = Callable[[List[Precision], int], None]


def _percentile(samples: Sequence[float], q: float) -> float:
    """Simple percentile — q ∈ [0, 1]. Avoids numpy dep."""
    if not samples:
        raise ValueError("empty samples")
    s = sorted(samples)
    if len(s) == 1:
        return s[0]
    rank = q * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


@dataclass
class SLOBudget:
    """Rolling-window SLO tracker → precision budget scalar.

    The rolling window holds the last ``window_size`` raw TPOT samples
    via ``update_tpot_sample``; ``budget()`` computes the p99 over the
    window and maps the ratio ``p99 / target`` to a budget in
    ``[min_budget, max_budget]``. Alternatively a pre-computed p99 can
    be passed via ``update_p99``; both paths converge on the same
    ``_current_p99_ms`` and recompute the budget.

    Mapping (intentionally simple, monotone, sweepable):

        ratio = p99 / target
        if ratio ≤ 1:   budget = max_budget          (well under SLO)
        if ratio ≥ 2:   budget = min_budget          (severe overrun)
        else:           linear interp between the two

    The mapping's exact shape is a knob — what matters is that it is
    monotone non-increasing in ratio so tighter SLO pressure yields a
    lower budget, and that the controller's τ_eff is correspondingly
    monotone non-increasing in budget.
    """

    target_p99_tpot_ms: float = DEFAULT_TARGET_P99_TPOT_MS
    min_budget: float = DEFAULT_MIN_BUDGET
    max_budget: float = DEFAULT_MAX_BUDGET
    window_size: int = DEFAULT_WINDOW_SIZE

    _samples: Deque[float] = field(default_factory=deque)
    _current_p99_ms: Optional[float] = None
    _current_budget: float = DEFAULT_MAX_BUDGET
    _broadcast_cb: Optional[BroadcastCallback] = None
    _broadcasts: int = 0  # for INV-3 tests: every emit fires a broadcast

    def __post_init__(self) -> None:
        if self.target_p99_tpot_ms <= 0:
            raise ValueError("target_p99_tpot_ms must be > 0")
        if not (0.0 <= self.min_budget <= self.max_budget <= 1.0):
            raise ValueError("0 <= min_budget <= max_budget <= 1 required")
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        self._samples = deque(maxlen=self.window_size)

    # ------------------------------------------------------------------
    # measurement ingest
    # ------------------------------------------------------------------

    def update_tpot_sample(self, tpot_ms: float) -> None:
        """Append one raw TPOT measurement to the rolling window.

        Called once per (request, decode-step) pair from the serving
        layer; in tests it's driven by synthetic latency series.
        """
        if tpot_ms < 0:
            raise ValueError("tpot_ms must be >= 0")
        self._samples.append(float(tpot_ms))
        self._recompute_from_window()

    def update_p99(self, p99_tpot_ms: float) -> None:
        """Record a fresh pre-computed p99 TPOT measurement.

        Equivalent endpoint for callers that already maintain their
        own latency stats (e.g. vLLM metrics) and just want to push
        the rolled-up p99 into the budget calculation. The rolling
        window is bypassed; the budget is recomputed directly.
        """
        if p99_tpot_ms < 0:
            raise ValueError("p99_tpot_ms must be >= 0")
        self._current_p99_ms = float(p99_tpot_ms)
        self._current_budget = self._budget_from_p99(self._current_p99_ms)

    def _recompute_from_window(self) -> None:
        if not self._samples:
            return
        self._current_p99_ms = _percentile(self._samples, 0.99)
        self._current_budget = self._budget_from_p99(self._current_p99_ms)

    def _budget_from_p99(self, p99_ms: float) -> float:
        ratio = p99_ms / self.target_p99_tpot_ms
        if ratio <= 1.0:
            b = self.max_budget
        elif ratio >= 2.0:
            b = self.min_budget
        else:
            # linear from max_budget at ratio=1 to min_budget at ratio=2
            t = ratio - 1.0  # in [0, 1]
            b = self.max_budget + t * (self.min_budget - self.max_budget)
        return max(self.min_budget, min(self.max_budget, b))

    # ------------------------------------------------------------------
    # read path
    # ------------------------------------------------------------------

    def budget(self) -> float:
        """Current precision budget ∈ [min_budget, max_budget]."""
        return self._current_budget

    def current_p99_ms(self) -> Optional[float]:
        return self._current_p99_ms

    @property
    def num_samples(self) -> int:
        return len(self._samples)

    @property
    def num_broadcasts(self) -> int:
        """Total ``broadcast_precision`` calls. Used by INV-3 tests to
        verify every controller emit is delivered to the scheduler."""
        return self._broadcasts

    # ------------------------------------------------------------------
    # broadcast (INV-3 enforcement point)
    # ------------------------------------------------------------------

    def register_broadcast_callback(
        self, callback: Optional[BroadcastCallback]
    ) -> None:
        """Wire the scheduler callback. Pass ``None`` to clear."""
        self._broadcast_cb = callback

    def broadcast_precision(
        self,
        levels: Sequence[Precision],
        tp_world_size: int,
    ) -> None:
        """Synchronously broadcast the per-layer precision vector to
        all TP ranks via the registered scheduler callback.

        INV-3 contract: the callback must complete before this method
        returns. The scheduler is expected to hold the global barrier
        across the broadcast so no rank admits the next decode step
        between emit and delivery. If no callback is registered this
        is a no-op other than the broadcast counter — useful in unit
        tests of the controller in isolation.
        """
        if tp_world_size < 1:
            raise ValueError("tp_world_size must be >= 1")
        levels_list = list(levels)
        self._broadcasts += 1
        if self._broadcast_cb is not None:
            self._broadcast_cb(levels_list, tp_world_size)
