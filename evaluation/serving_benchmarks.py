"""Serving benchmark runner — trace replay.

Traces: ShareGPT (Poisson, 70% target utilization), LMSYS-Chat-1M
(production trace), BurstGPT (bursty — key test for SLO Budget).

Metrics: p99 TPOT, throughput, Goodput@SLO.

Status: stub.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


TRACES: List[str] = ["sharegpt", "lmsys_chat_1m", "burstgpt"]


@dataclass
class ServingRunConfig:
    system: str
    model: str
    trace: str
    target_utilization: float = 0.7
    slo_p99_tpot_ms: float = 50.0


def run_serving_trace(config: ServingRunConfig) -> Dict[str, float]:
    """Replay a trace and return the metrics row for REPORT.md."""
    raise NotImplementedError
