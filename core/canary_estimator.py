"""Shadow Canary Estimator — per-layer KL divergence accumulator.

The canary kernel (``kernels.canary_attention``) writes one KL
divergence scalar δ(ℓ) per canary step into a pinned scratchpad.
This module owns:

  1. The scratchpad ring buffer — a per-layer list of raw CanarySample
     records, capped at ``scratchpad_depth`` entries (oldest evicted).
  2. The per-layer EMA δ̂_ℓ — the smooth signal the PID controller reads.
  3. The read path — ``ema(layer_id)`` and ``snapshot()``.

Invariants
----------
INV-1  No per-block auxiliary buffer. The estimator accumulates
       per-layer statistics; it never allocates state proportional to
       the number of KV blocks.

INV-4  When ``is_canary`` is ``False``, ``observe()`` is a strict
       no-op: nothing written, no state mutated. The kernel side
       takes the standard FlashInfer fast path, and this Python
       module adds zero overhead.
"""

from __future__ import annotations

import random
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Deque, List, Optional


DEFAULT_CANARY_RATE: float = 1.0 / 64.0
DEFAULT_EMA_ALPHA: float = 0.1
DEFAULT_SCRATCHPAD_DEPTH: int = 4096  # samples per layer before wrap


@dataclass
class CanarySample:
    """One canary observation: layer, step, KL value."""

    layer_id: int
    step: int
    kl: float


class CanaryEstimator:
    """Per-layer canary error accumulator.

    Thread-safety: ``observe()`` is called from the decode thread (or
    multiple TP threads in production). ``ema()`` / ``snapshot()`` are
    called from the PID controller thread. A single lock serialises
    both paths. The lock is never held across a kernel call — it only
    guards the in-memory EMA vector and scratchpad lists.

    Parameters
    ----------
    num_layers:
        Number of transformer layers in the model.
    canary_rate:
        Probability that a given decode step is promoted to a canary
        step. Default 1/64. When ``is_canary_step()`` is used by the
        caller, they draw independently; this value is informational.
    ema_alpha:
        Weight on the *new* sample in the EMA update:
        ``δ̂ = (1-α)·δ̂_prev + α·δ_new``.  Default 0.1 (equivalent
        to a 9-step half-life). Not yet ablated — see REPORT.md §
        Open Questions.
    scratchpad_depth:
        Maximum raw samples retained per layer. Ring semantics: oldest
        entry is dropped when the cap is reached.
    """

    def __init__(
        self,
        num_layers: int,
        canary_rate: float = DEFAULT_CANARY_RATE,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        scratchpad_depth: int = DEFAULT_SCRATCHPAD_DEPTH,
    ) -> None:
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if not (0.0 < canary_rate <= 1.0):
            raise ValueError("canary_rate must be in (0, 1]")
        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError("ema_alpha must be in (0, 1]")
        if scratchpad_depth < 1:
            raise ValueError("scratchpad_depth must be >= 1")

        self.num_layers = num_layers
        self.canary_rate = canary_rate
        self.ema_alpha = ema_alpha
        self.scratchpad_depth = scratchpad_depth

        self._lock = threading.Lock()
        self._ema: List[float] = [0.0] * num_layers
        self._scratchpad: Dict[int, Deque[CanarySample]] = {
            i: deque(maxlen=scratchpad_depth) for i in range(num_layers)
        }
        self._initialized: List[bool] = [False] * num_layers

    def reset(self) -> None:
        """Zero all per-layer state. Call at conversation start."""
        with self._lock:
            self._ema = [0.0] * self.num_layers
            for q in self._scratchpad.values():
                q.clear()
            self._initialized = [False] * self.num_layers

    def is_canary_step(self, rng: Optional[random.Random] = None) -> bool:
        """Sample whether the current decode step should be a canary step.

        Callers can pass a seeded ``random.Random`` for determinism in
        tests; the production path may use any random source with the
        same Bernoulli(p_c) distribution.
        """
        r = (rng or random).random()
        return r < self.canary_rate

    def observe(
        self,
        layer_id: int,
        step: int,
        kl: Optional[float],
        *,
        is_canary: bool,
    ) -> None:
        """Record a kernel observation.

        When ``is_canary`` is ``False`` this is a strict no-op (INV-4):
        no state is mutated, no lock acquired.

        When ``is_canary`` is ``True``, ``kl`` must be a finite
        non-negative float. The EMA is updated and the raw sample is
        appended to the scratchpad ring.

        Parameters
        ----------
        layer_id:
            Layer index, must be in ``[0, num_layers)``.
        step:
            Global decode step counter at the time of the observation.
        kl:
            KL(ȧ_ℓ ‖ ā_ℓ) from the canary kernel. Must be ≥ 0 and
            finite when ``is_canary`` is ``True``.
        is_canary:
            If ``False``, this call is a no-op (INV-4).
        """
        if not is_canary:
            return  # INV-4: strict no-op

        if layer_id < 0 or layer_id >= self.num_layers:
            raise ValueError(
                f"layer_id {layer_id} out of range [0, {self.num_layers})"
            )
        if kl is None:
            raise ValueError("kl must be provided when is_canary=True")
        if not isinstance(kl, (int, float)) or kl != kl:  # NaN check
            raise ValueError(f"kl must be a finite float, got {kl!r}")
        if kl < 0:
            raise ValueError(f"KL divergence must be non-negative, got {kl}")

        kl_f = float(kl)
        with self._lock:
            if self._initialized[layer_id]:
                self._ema[layer_id] = (
                    (1.0 - self.ema_alpha) * self._ema[layer_id]
                    + self.ema_alpha * kl_f
                )
            else:
                # Cold start: seed the EMA with the first observation
                # directly rather than blending from zero, which would
                # create a fictitiously low initial estimate.
                self._ema[layer_id] = kl_f
                self._initialized[layer_id] = True
            self._scratchpad[layer_id].append(
                CanarySample(layer_id=layer_id, step=step, kl=kl_f)
            )

    def ema(self, layer_id: int) -> float:
        """Return δ̂_ℓ — the PID controller's view of per-layer error.

        Returns 0.0 if no canary sample has been observed for this
        layer yet (pre-warm-up behaviour — the controller will stay at
        its default precision until the first canary fires).
        """
        if layer_id < 0 or layer_id >= self.num_layers:
            raise ValueError(
                f"layer_id {layer_id} out of range [0, {self.num_layers})"
            )
        with self._lock:
            return self._ema[layer_id]

    def snapshot(self) -> List[float]:
        """Return a copy of the full per-layer EMA vector (length = num_layers)."""
        with self._lock:
            return list(self._ema)

    def scratchpad_samples(self, layer_id: int) -> List[CanarySample]:
        """Return a copy of the scratchpad ring for one layer (for analysis)."""
        if layer_id < 0 or layer_id >= self.num_layers:
            raise ValueError(
                f"layer_id {layer_id} out of range [0, {self.num_layers})"
            )
        with self._lock:
            return list(self._scratchpad[layer_id])

    def num_samples(self, layer_id: int) -> int:
        """Total canary samples recorded for this layer (capped at scratchpad_depth)."""
        if layer_id < 0 or layer_id >= self.num_layers:
            raise ValueError(
                f"layer_id {layer_id} out of range [0, {self.num_layers})"
            )
        with self._lock:
            return len(self._scratchpad[layer_id])
