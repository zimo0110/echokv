"""Write-Ahead Precision (WAP) — SinkPredictor interface stubs.

WAP predicts attention-sink (heavy-hitter) blocks *before* they are
written, so the allocator can serve them at INT8 from first write. No
reconstruction, no information loss, no INV-2 violation.

Signals
-------
Positional prior
    Positions 0..3 are empirically attention sinks in every modern
    decoder-only LLM we have measured. They are promoted to INT8
    unconditionally. This requires no warm-up.

Running accumulator
    After a warm-up window W=256 tokens, maintain an EMA of the
    attention weight w_{ℓ,i} paid by queries to each key/value
    position. If w_{ℓ,i} > θ=0.15, promote blocks containing that
    position's KV to INT8.

Decision rule (from REPORT.md)
------------------------------
WAP ships as Component 4 iff EchoKV-4C improves RULER@128K by
≥0.5pt over EchoKV-3C at ≤15% memory overhead. Otherwise: negative
result, reported in Discussion, paper ships as three-component.

Invariants
----------
INV-2  WAP emits allocator hints for *new* blocks only. It never
       touches an existing block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from core.block_allocator import Precision


DEFAULT_WARMUP_WINDOW: int = 256  # W
DEFAULT_HEAVY_THRESHOLD: float = 0.15  # θ
DEFAULT_POSITIONAL_PRIOR_RANGE: Tuple[int, int] = (0, 4)  # positions 0..3


@dataclass
class SinkPredictor:
    """Per-layer heavy-hitter predictor.

    Parameters
    ----------
    num_layers:
        Number of transformer layers.
    warmup_window:
        ``W`` — tokens before the accumulator may promote positions.
        At 32K+ context this is <1% of the input; positional priors
        still apply during warm-up.
    heavy_threshold:
        ``θ`` — EMA attention-weight threshold for promotion.
    positional_prior_range:
        Half-open ``[lo, hi)`` of positions always allocated at INT8.
    """

    num_layers: int
    warmup_window: int = DEFAULT_WARMUP_WINDOW
    heavy_threshold: float = DEFAULT_HEAVY_THRESHOLD
    positional_prior_range: Tuple[int, int] = DEFAULT_POSITIONAL_PRIOR_RANGE

    _ema_attn_weight: Dict[Tuple[int, int], float] = field(default_factory=dict)
    _tokens_seen: int = 0

    def observe_attention(
        self,
        layer_id: int,
        attention_weights: List[float],
    ) -> None:
        """Record one decode step's attention distribution.

        ``attention_weights[i]`` is the weight the current query paid
        to key position ``i``.
        """
        raise NotImplementedError

    def suggested_precision(
        self,
        layer_id: int,
        position: int,
        default: Precision = Precision.INT4,
    ) -> Precision:
        """Return the precision WAP recommends for a new block.

        Positional prior: positions in ``positional_prior_range`` →
        INT8. Accumulator: after warm-up, positions with EMA weight
        above ``heavy_threshold`` → INT8. Otherwise ``default``.
        """
        raise NotImplementedError
