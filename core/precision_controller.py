"""PID Precision Controller — per-layer hysteresis-guarded controller.

Reads per-layer EMA δ̂_ℓ from :class:`core.canary_estimator.CanaryEstimator`
and emits per-layer target precisions ``b_ℓ ∈ {INT2, INT4, INT8}``. New
allocations from the block allocator pick up the latest level
(``controller.level(layer_id)``); existing blocks are never modified
(INV-2). Decisions are broadcast through the scheduler via
:class:`core.slo_budget.SLOBudget` (INV-3) — this module returns the
decisions; the SLO budget delivers them at a step boundary.

v1 implementation
-----------------
Per-layer P controller with a three-state hysteresis ladder over INT2 /
INT4 / INT8. PI/PID is a follow-up (REPORT.md § Open Questions).

Update law (per layer per tick)
-------------------------------
Inputs:
    δ̂           per-layer EMA of canary δ (already smoothed by the
                estimator)
    u           kv_utilization ∈ [0, 1] from the allocator
    b           precision_budget ∈ [0, 1] from the SLO budget

Compute the effective setpoint:
    τ_eff = τ · g(u, b)
    g(u, b) = 1 + 1.5 · u + 1.5 · (1 − b)

g is monotone non-decreasing in u (high utilisation relaxes the setpoint
so the controller is less eager to upgrade) and monotone non-increasing
in b (tight budget tightens the constraint against allocating more
INT8 blocks). g(0, 1) = 1 (baseline) and g(1, 0) = 4 (worst-case
pressure).

Per layer, with hysteresis band ``H``:
    if   δ̂ > τ_eff           → upgrade   (INT2 → INT4 → INT8)
    elif δ̂ < τ_eff · (1 − H)  → downgrade (INT8 → INT4 → INT2)
    else                      → hold

A stall counter prevents pathological flapping just outside the band:
after a transition, a layer holds its new level for at least
``min_dwell_ticks`` ticks before another transition is allowed.

Invariants
----------
INV-2  Decisions emitted here apply only to *new* allocations. The
       controller exposes ``level()`` (read) and ``tick()`` (advance);
       no surface name on this class implies a rewrite of an existing
       block. ``tests/test_invariants.py`` enforces the public-surface
       audit.
INV-3  Broadcast happens via :class:`SLOBudget.broadcast_precision`;
       this module returns decisions but does not commit them. The
       SLO budget calls a registered scheduler callback that
       synchronously notifies all TP ranks before any rank admits the
       next decode step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Sequence

from .block_allocator import Precision


DEFAULT_SETPOINT: float = 0.01  # τ; swept {0.001,0.005,0.01,0.02,0.05}
DEFAULT_TICK_INTERVAL: int = 100  # N_c decode steps
DEFAULT_HYSTERESIS_BAND: float = 0.20  # 20% band around each boundary
DEFAULT_MIN_DWELL_TICKS: int = 2  # post-transition lockout, in ticks


_LADDER: List[Precision] = [Precision.INT2, Precision.INT4, Precision.INT8]
_INDEX: dict = {p: i for i, p in enumerate(_LADDER)}


class ControllerDirection(IntEnum):
    """Direction of a proposed level change."""

    HOLD = 0
    UPGRADE = 1
    DOWNGRADE = -1


@dataclass
class PrecisionDecision:
    """What the controller wants the allocator to do for a layer."""

    layer_id: int
    previous: Precision
    target: Precision
    direction: ControllerDirection
    tick: int


@dataclass
class PIDPrecisionController:
    """Per-layer hysteresis-guarded precision controller.

    Parameters
    ----------
    num_layers:
        Number of transformer layers.
    setpoint:
        Target error τ. A layer with δ̂ > τ_eff is a candidate for
        upgrade; δ̂ < τ_eff · (1 − band) is a candidate for downgrade.
    tick_interval:
        Controller cadence, in decode steps. Informational here — the
        caller decides when to call ``tick()``; this field documents
        the design cadence.
    hysteresis_band:
        Relative width of the deadband between upgrade and downgrade
        thresholds; prevents oscillation at the boundary.
    min_dwell_ticks:
        Post-transition lockout. A layer that just changed level holds
        the new level for at least this many ticks before another
        transition is allowed.
    initial_level:
        Starting precision for every layer. Default INT4 — KIVI-4
        baseline. The controller upgrades or downgrades from here.
    """

    num_layers: int
    setpoint: float = DEFAULT_SETPOINT
    tick_interval: int = DEFAULT_TICK_INTERVAL
    hysteresis_band: float = DEFAULT_HYSTERESIS_BAND
    min_dwell_ticks: int = DEFAULT_MIN_DWELL_TICKS
    initial_level: Precision = Precision.INT4

    _levels: List[Precision] = field(default_factory=list)
    _last_change_tick: List[int] = field(default_factory=list)
    _tick: int = 0

    def __post_init__(self) -> None:
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if self.setpoint <= 0:
            raise ValueError("setpoint must be > 0")
        if not (0.0 <= self.hysteresis_band < 1.0):
            raise ValueError("hysteresis_band must be in [0, 1)")
        if self.min_dwell_ticks < 0:
            raise ValueError("min_dwell_ticks must be >= 0")
        if not self._levels:
            self._levels = [self.initial_level] * self.num_layers
        if not self._last_change_tick:
            # Sentinel: a value that always passes the dwell check on
            # the first tick so layers can move out of the initial
            # level immediately if signal is already strong.
            self._last_change_tick = [-(self.min_dwell_ticks + 1)] * self.num_layers

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def reset(self, initial_level: Optional[Precision] = None) -> None:
        """Initialise all layers to ``initial_level`` and reset tick."""
        level = initial_level if initial_level is not None else self.initial_level
        self._levels = [level] * self.num_layers
        self._last_change_tick = [
            -(self.min_dwell_ticks + 1)
        ] * self.num_layers
        self._tick = 0

    def effective_setpoint(
        self,
        kv_utilization: float,
        precision_budget: float,
    ) -> float:
        """Compute τ_eff from τ, kv_utilization, and precision_budget.

        Monotone non-decreasing in ``kv_utilization`` (relax under
        pressure); monotone non-increasing in ``precision_budget``
        (tight budget → high τ_eff → harder to upgrade → save memory).
        """
        u = max(0.0, min(1.0, float(kv_utilization)))
        b = max(0.0, min(1.0, float(precision_budget)))
        g = 1.0 + 1.5 * u + 1.5 * (1.0 - b)
        return self.setpoint * g

    def tick(
        self,
        delta_hat: Sequence[float],
        kv_utilization: float,
        precision_budget: float,
    ) -> List[PrecisionDecision]:
        """Advance the controller by one tick.

        Returns the list of decisions for this tick; the scheduler is
        responsible for broadcasting them (INV-3) before the next
        decode step. Decisions affect only future allocations (INV-2).
        """
        if len(delta_hat) != self.num_layers:
            raise ValueError(
                f"delta_hat has {len(delta_hat)} entries, "
                f"expected {self.num_layers}"
            )
        self._tick += 1
        tau_eff = self.effective_setpoint(kv_utilization, precision_budget)
        tau_down = tau_eff * (1.0 - self.hysteresis_band)

        decisions: List[PrecisionDecision] = []
        for layer_id in range(self.num_layers):
            current = self._levels[layer_id]
            d_hat = float(delta_hat[layer_id])
            since_last = self._tick - self._last_change_tick[layer_id]

            direction = ControllerDirection.HOLD
            target = current

            if since_last >= self.min_dwell_ticks:
                if d_hat > tau_eff:
                    upgraded = self._step_up(current)
                    if upgraded is not current:
                        target = upgraded
                        direction = ControllerDirection.UPGRADE
                elif d_hat < tau_down:
                    downgraded = self._step_down(current)
                    if downgraded is not current:
                        target = downgraded
                        direction = ControllerDirection.DOWNGRADE

            if direction != ControllerDirection.HOLD:
                self._levels[layer_id] = target
                self._last_change_tick[layer_id] = self._tick

            decisions.append(
                PrecisionDecision(
                    layer_id=layer_id,
                    previous=current,
                    target=target,
                    direction=direction,
                    tick=self._tick,
                )
            )
        return decisions

    def level(self, layer_id: int) -> Precision:
        """Current target precision for ``layer_id``."""
        if layer_id < 0 or layer_id >= self.num_layers:
            raise ValueError(
                f"layer_id {layer_id} out of range [0, {self.num_layers})"
            )
        return self._levels[layer_id]

    def snapshot(self) -> List[Precision]:
        """Copy of the full per-layer level vector."""
        return list(self._levels)

    @property
    def current_tick(self) -> int:
        return self._tick

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @staticmethod
    def _step_up(p: Precision) -> Precision:
        i = _INDEX[p]
        return _LADDER[min(i + 1, len(_LADDER) - 1)]

    @staticmethod
    def _step_down(p: Precision) -> Precision:
        i = _INDEX[p]
        return _LADDER[max(i - 1, 0)]
