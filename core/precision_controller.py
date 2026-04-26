"""PID Precision Controller — interface stubs.

Consumes per-layer EMA δ̂ from :mod:`core.canary_estimator`; produces
per-layer target precision b_ℓ ∈ {2, 4, 8} for the block allocator.

Runs on a Python ``threading.Thread`` that ticks every ``N_c``
decode steps. Updates are staged, not applied directly — the
scheduler applies them at the next global barrier to preserve INV-3.

Invariants
----------
INV-2  Emitted precision applies to **new** block allocations only.
       This module never issues an operation that touches an
       existing block.

INV-3  Under TP, updates must be broadcast through the scheduler
       before any rank starts the next decode step. This module
       exposes an update *intent*; the scheduler is responsible for
       the broadcast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Sequence

from .block_allocator import Precision


DEFAULT_SETPOINT: float = 0.01  # τ; swept {0.001,0.005,0.01,0.02,0.05}
DEFAULT_TICK_INTERVAL: int = 100  # N_c decode steps
DEFAULT_HYSTERESIS_BAND: float = 0.20  # 20% band around each boundary


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

    v1 implementation is P-only; PI/PID are a follow-up (see
    ``REPORT.md § Open Questions``).

    Parameters
    ----------
    num_layers:
        Number of transformer layers.
    setpoint:
        Target error τ. A layer with δ̂ > τ_eff is a candidate for
        upgrade; δ̂ < τ_eff · (1 - band) is a candidate for downgrade.
    tick_interval:
        Controller cadence, in decode steps.
    hysteresis_band:
        Relative width of the deadband between upgrade and downgrade
        thresholds; prevents oscillation.
    """

    num_layers: int
    setpoint: float = DEFAULT_SETPOINT
    tick_interval: int = DEFAULT_TICK_INTERVAL
    hysteresis_band: float = DEFAULT_HYSTERESIS_BAND

    _levels: List[Precision] = field(default_factory=list)
    _tick: int = 0

    def reset(self, initial_level: Precision = Precision.INT4) -> None:
        """Initialise all layers to ``initial_level``."""
        raise NotImplementedError

    def effective_setpoint(
        self,
        kv_utilization: float,
        precision_budget: float,
    ) -> float:
        """Compute τ_eff from τ, kv_utilization, and precision_budget.

        Monotone non-decreasing in ``kv_utilization`` (relax under
        pressure); monotone non-increasing in ``precision_budget``
        (tight budget → tight setpoint).
        """
        raise NotImplementedError

    def tick(
        self,
        delta_hat: Sequence[float],
        kv_utilization: float,
        precision_budget: float,
    ) -> List[PrecisionDecision]:
        """Advance the controller by one tick.

        Returns the list of decisions for this tick; the scheduler is
        responsible for broadcasting them before the next decode
        step (INV-3). Decisions affect only future allocations
        (INV-2).
        """
        raise NotImplementedError

    def level(self, layer_id: int) -> Precision:
        """Current target precision for ``layer_id``."""
        raise NotImplementedError
