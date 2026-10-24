"""Per-layer tier controller: EMA of canary δ̂ per tier, lowest tier under τ."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from echokv_vllm.quant import SUPPORTED_BITS


@dataclass
class ControllerParams:
    tau0: float = 0.01
    l0: int = 4096
    tau_scaling: str = "none"  # "none" or "sqrt"
    alpha: float = 0.1
    init_bits: int = 8
    fixed_bits: int | None = None
    frozen: bool = False


@dataclass
class LayerController:
    """Tier decision for one attention layer.

    Each canary fire updates ``ema[b]`` for every tier, then (unless frozen
    or pinned) picks the lowest tier whose EMA is below τ, falling back to
    the highest tier. The rule is re-evaluated on every fire, so a layer can
    move up or down.

    With ``tau_scaling="sqrt"``, ``τ(L) = τ0 · sqrt(max(L, L0) / L0)``: the
    budget loosens with context length beyond L0 and never tightens below τ0.
    """

    params: ControllerParams
    bits: int = 0
    ema: dict[int, float | None] = field(default_factory=lambda: {b: None for b in SUPPORTED_BITS})
    fires: int = 0
    transitions: int = 0

    def __post_init__(self) -> None:
        if not self.bits:
            self.bits = self.params.fixed_bits or self.params.init_bits

    def tau(self, ctx_len: float) -> float:
        p = self.params
        if p.tau_scaling == "sqrt":
            return p.tau0 * math.sqrt(max(ctx_len, p.l0) / p.l0)
        return p.tau0

    def observe(self, deltas: dict[int, float], ctx_len: float) -> None:
        a = self.params.alpha
        for b, d in deltas.items():
            prev = self.ema[b]
            self.ema[b] = d if prev is None else a * d + (1.0 - a) * prev
        self.fires += 1
        if self.params.frozen or self.params.fixed_bits is not None:
            return
        tau = self.tau(ctx_len)
        new = max(SUPPORTED_BITS)
        for b in sorted(SUPPORTED_BITS):
            e = self.ema[b]
            if e is not None and e < tau:
                new = b
                break
        if new != self.bits:
            self.transitions += 1
            self.bits = new
