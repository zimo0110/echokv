"""Accuracy benchmark runner.

Coverage: RULER@{32K, 64K, 128K}, GSM8K-CoT 8-shot, MATH 4-shot,
LiveCodeBench, LongBench. Numbers land in REPORT.md § Benchmarks
only after the run completes (INV-5).

Status: stub.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


BENCHMARKS: List[str] = [
    "ruler_32k",
    "ruler_64k",
    "ruler_128k",
    "gsm8k_cot_8shot",
    "math_4shot",
    "livecodebench",
    "longbench",
]


@dataclass
class AccuracyRunConfig:
    system: str  # e.g. "echokv_3c", "echokv_4c", "kivi_4", ...
    model: str
    benchmarks: List[str]


def run_accuracy_suite(config: AccuracyRunConfig) -> None:
    """Run all requested accuracy benchmarks for a system/model pair."""
    raise NotImplementedError
