"""Session 14 — multi-needle RULER sweep.

The Session-13 calibration confirmed per-layer δ̂ structure exists at
long context (7.3× spread at 128K), but single-needle RULER + INT4 +
Llama-3.1-8B is too robust to demonstrate any closed-loop accuracy
win: KIVI-4 already saturates at 1.000. The next-cheapest experiment
the critic recommended is **multi-needle RULER**: insert 5 labeled
needles at distinct positions and force the decoder to attend to all
5 simultaneously when it answers the question.

Hypothesis
----------
INT4 group-32 KV compression introduces small per-key errors that are
amortized when attention is sharp on one position (single needle). On
multi-needle, attention has to remain sharp on multiple positions
during decode; the per-key noise should accumulate and force one or
more needles to be retrieved incorrectly. If FP16 stays at ~1.0 while
static_int4 drops below ~0.9 on any context, that's the workload
EchoKV-3C has been waiting for.

Usage
-----
::

    FLASHINFER_DISABLE_VERSION_CHECK=1 \\
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \\
    CUDA_VISIBLE_DEVICES=0 \\
    /opt/conda/envs/scalable/bin/python \\
        -m evaluation.serving.run_session14_multineedle

By default this runs FP16 + static_int4 only — the cheap diagnostic.
Pass ``--configs fp16 static_int4 echokv_3c`` once a break has been
observed to also evaluate the closed-loop controller on the failing
config.

Output
------
``results/serving/session14_multineedle.json`` — full per-cell blob.
Per-prompt scores are fractions in {0, 0.2, 0.4, 0.6, 0.8, 1.0}.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.motivation.workloads import _ruler_multineedle_sample, Request
from evaluation.serving.run_session8 import (
    DEFAULT_MODEL,
    RunResult,
    run_one,
    warmup,
)
from evaluation.serving.run_session11_sweep import make_harness


log = logging.getLogger("session14")


def make_multineedle_prompts(
    n: int,
    seed: int,
    target_tokens: int,
    n_needles: int,
) -> list[Request]:
    return [
        _ruler_multineedle_sample(
            i, seed=seed, target_tokens=target_tokens, n_needles=n_needles,
        )
        for i in range(n)
    ]


def run_cell(
    llm,
    *,
    context: int,
    n_prompts: int,
    n_needles: int,
    max_new_tokens: int,
    config: str,
    seed: int,
    tick_interval: int,
    observability_rate: float,
    tau: float = 0.01,
    tau_budget: float = 0.05,
    multi_precision: bool = False,
    warmup_n: int = 8,
) -> dict:
    """Run one (context, config) cell on multi-needle RULER.

    For ``echokv_3c`` we use the same 2-phase warmup/measurement
    protocol as ``run_session11_sweep.run_cell``. For static configs
    we run a single phase.
    """
    from core.block_allocator import Precision
    from evaluation.serving.canary_vllm import CanaryHarness

    prompts = make_multineedle_prompts(
        n=n_prompts, seed=seed, target_tokens=context, n_needles=n_needles,
    )

    if config != "echokv_3c":
        harness = make_harness(
            llm, config,
            seed=seed,
            tick_interval=tick_interval,
            observability_rate=observability_rate,
        )
        label = f"{config}_{context}_n{n_needles}"
        if harness is not None:
            harness.install()
        try:
            result: RunResult = run_one(
                llm, prompts,
                label=label,
                max_new_tokens=max_new_tokens,
                harness=harness,
            )
        finally:
            if harness is not None:
                harness.remove()
        blob = asdict(result)
        blob["config"] = config
        blob["context"] = context
        blob["n_prompts"] = n_prompts
        blob["n_needles"] = n_needles
        return blob

    # echokv_3c — 2-phase protocol.
    log.info("    [warmup] running %d prompts at observability rate %.3f",
             warmup_n, observability_rate)
    warmup_prompts = make_multineedle_prompts(
        n=warmup_n, seed=seed + 100, target_tokens=context, n_needles=n_needles,
    )
    warmup_harness = CanaryHarness(
        llm,
        canary_rate=observability_rate,
        tau=tau,
        tick_interval=tick_interval,
        tier_application=False,
        freeze_controller=False,
        multi_precision=multi_precision,
        tau_budget=tau_budget,
        seed=seed,
    )
    warmup_harness.install()
    try:
        run_one(
            llm, warmup_prompts,
            label=f"echokv_warmup_{context}_n{n_needles}",
            max_new_tokens=max_new_tokens,
            harness=warmup_harness,
        )
    finally:
        warmup_harness.remove()
    converged_levels = warmup_harness.controller.snapshot()
    log.info("    [warmup] converged tier vector: %s",
             [int(l) for l in converged_levels])

    measure_harness = CanaryHarness(
        llm,
        canary_rate=1e-6,
        tau=tau,
        tick_interval=tick_interval,
        tier_application=True,
        freeze_controller=True,
        multi_precision=multi_precision,
        tau_budget=tau_budget,
        seed=seed + 1,
    )
    for li, lvl in enumerate(converged_levels):
        measure_harness.controller._levels[li] = Precision(int(lvl))

    measure_harness.install()
    try:
        result = run_one(
            llm, prompts,
            label=f"echokv_3c_{context}_n{n_needles}",
            max_new_tokens=max_new_tokens,
            harness=measure_harness,
        )
    finally:
        measure_harness.remove()

    blob = asdict(result)
    blob["config"] = config
    blob["context"] = context
    blob["n_prompts"] = n_prompts
    blob["n_needles"] = n_needles
    blob["warmup_converged_tiers"] = [int(l) for l in converged_levels]
    return blob


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-model-len", type=int, default=131_072)
    p.add_argument("--max-new-tokens", type=int, default=128,
                   help="must accommodate 5 'label=NNNNNN' lines plus slack")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--n-needles", type=int, default=5)
    p.add_argument("--tick-interval", type=int, default=10)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--tau-budget", type=float, default=0.05)
    p.add_argument("--multi-precision", action="store_true")
    p.add_argument("--observability-rate", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=Path("results/serving/session14_multineedle.json"))
    p.add_argument("--contexts", type=int, nargs="+",
                   default=[32_000, 64_000, 128_000])
    p.add_argument("--n-per-context", type=int, nargs="+",
                   default=[30, 30, 30])
    p.add_argument("--configs", nargs="+",
                   default=["fp16", "static_int4"],
                   choices=["fp16", "static_int4", "static_int2", "echokv_3c"])
    a = p.parse_args(argv)

    if len(a.contexts) != len(a.n_per_context):
        p.error("--contexts and --n-per-context must have the same length")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    from vllm import LLM

    log.info("loading %s (max_model_len=%d)", a.model, a.max_model_len)
    t0 = time.perf_counter()
    llm = LLM(
        model=a.model,
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=a.gpu_memory_utilization,
        max_model_len=a.max_model_len,
        enable_prefix_caching=False,
    )
    log.info("loaded in %.1fs", time.perf_counter() - t0)

    smallest = min(a.contexts)
    warmup_prompts = make_multineedle_prompts(
        n=4, seed=a.seed, target_tokens=smallest, n_needles=a.n_needles,
    )
    warmup(llm, warmup_prompts, a.max_new_tokens)

    all_cells: list[dict] = []
    for context, n_prompts in zip(a.contexts, a.n_per_context):
        for config in a.configs:
            log.info(
                "=== context=%d  config=%s  n=%d  n_needles=%d ===",
                context, config, n_prompts, a.n_needles,
            )
            cell = run_cell(
                llm,
                context=context,
                n_prompts=n_prompts,
                n_needles=a.n_needles,
                max_new_tokens=a.max_new_tokens,
                config=config,
                seed=a.seed,
                tick_interval=a.tick_interval,
                observability_rate=a.observability_rate,
                tau=a.tau,
                tau_budget=a.tau_budget,
                multi_precision=a.multi_precision,
            )
            all_cells.append(cell)
            log.info(
                "    acc=%.3f  wall=%.1fs",
                cell["accuracy"], cell["wall_clock_s"],
            )

    summary = {
        "session": "20260428_14",
        "model": a.model,
        "max_new_tokens": a.max_new_tokens,
        "n_needles": a.n_needles,
        "observability_rate": a.observability_rate,
        "tick_interval": a.tick_interval,
        "cells": all_cells,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=2))
    log.info("wrote %s", a.out)

    print()
    print("=" * 88)
    print(f"Session 14 — multi-needle RULER ({a.n_needles} needles)")
    print("=" * 88)
    by_ctx: dict[int, dict[str, dict]] = {}
    for cell in all_cells:
        by_ctx.setdefault(cell["context"], {})[cell["config"]] = cell

    cfg_order = a.configs
    print(f"{'context':>10}  ", end="")
    for cfg in cfg_order:
        print(f"{cfg:>20}  ", end="")
    print()
    for context in sorted(by_ctx.keys()):
        print(f"{context:>10}  ", end="")
        row = by_ctx[context]
        for cfg in cfg_order:
            cell = row.get(cfg, {})
            acc = cell.get("accuracy", float("nan"))
            print(f"{acc:>20.3f}  ", end="")
        print()
    print("=" * 88)

    fp16_baseline = lambda ctx: by_ctx[ctx].get("fp16", {}).get("accuracy", 0.0)
    broken_baselines = []
    for ctx in by_ctx:
        for cfg in ("static_int4", "static_int2"):
            cell_acc = by_ctx[ctx].get(cfg, {}).get("accuracy", None)
            if cell_acc is None:
                continue
            if cell_acc < fp16_baseline(ctx) - 0.05:
                broken_baselines.append((ctx, cfg, cell_acc, fp16_baseline(ctx)))
    if broken_baselines:
        print("VERDICT: at least one static baseline degrades meaningfully:")
        for ctx, cfg, acc, fp16_acc in broken_baselines:
            print(f"  - {cfg} @ {ctx}: acc={acc:.3f} "
                  f"(FP16={fp16_acc:.3f}, Δ={acc - fp16_acc:+.3f})")
    else:
        print("VERDICT: no static baseline degrades on multi-needle RULER "
              "at any tested context.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
