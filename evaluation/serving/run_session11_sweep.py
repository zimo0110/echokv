"""Session 11 — RULER context sweep.

The critic's session-11 priority: find a workload where static KIVI-4
actually fails. Without one, EchoKV has no Pareto win to claim.
Concrete plan: RULER at 32K / 64K / 128K, three configurations per
context — FP16 baseline, static INT4 (every layer forced to INT4 via
``force_bits=4``), and EchoKV-3C (controller-driven mix of
INT2/INT4/INT8). Find the (context, model) combo where INT4 degrades
from FP16's accuracy. EchoKV-3C is then expected to recover most of
the lost accuracy at lower memory than FP16.

Honest risk: Llama-3.1-8B with KIVI-style group-32 INT4 is
empirically robust, especially on a needle-retrieval task whose
attention pattern is sharp on a single position. If RULER doesn't
break at any context length, we fall back to LongBench (HotpotQA,
2WikiMultihopQA) or escalate to a larger model. Both are session 12+
work.

Usage
-----
::

    FLASHINFER_DISABLE_VERSION_CHECK=1 \\
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \\
    CUDA_VISIBLE_DEVICES=0 \\
    /opt/conda/envs/scalable/bin/python -m evaluation.serving.run_session11_sweep

Output
------
``results/serving/session11_sweep.json`` with the full per-(context,
config) accuracy + memory projection. INV-5 preserved: human commits
the real numbers to REPORT.md after reading the JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.serving.run_session8 import (
    DEFAULT_MODEL,
    RunResult,
    make_ruler_prompts,
    run_one,
    warmup,
)


log = logging.getLogger("session11")


# --------------------------------------------------------------------------
# Harness factory: returns the right CanaryHarness (or None) per config.
# --------------------------------------------------------------------------


def make_harness(
    llm: Any,
    config: str,
    *,
    seed: int,
    tick_interval: int,
    observability_rate: float,
    tau: float = 0.01,
    initial_level: Optional[Any] = None,
) -> Optional[Any]:
    """Build the per-config harness.

    - ``fp16``         → None (no canary, no tier application).
    - ``static_int4``  → harness with ``force_bits=4``, ``tier_application=True``,
                         ``canary_rate`` near zero (no canary fires; we just
                         use the tier-application path to apply uniform INT4).
    - ``echokv_3c``    → harness with full canary at observability rate +
                         tier_application; controller drives the mix.
    """
    from evaluation.serving.canary_vllm import CanaryHarness

    if config == "fp16":
        return None
    if config == "static_int4":
        return CanaryHarness(
            llm,
            canary_rate=1e-6,            # essentially never fires
            tick_interval=tick_interval,
            tier_application=True,
            tier_application_quant_v=True,
            force_bits=4,                # uniform INT4 across all layers
            seed=seed,
        )
    if config == "static_int2":
        return CanaryHarness(
            llm,
            canary_rate=1e-6,
            tick_interval=tick_interval,
            tier_application=True,
            tier_application_quant_v=True,
            force_bits=2,                # uniform INT2 across all layers
            seed=seed,
        )
    if config == "echokv_3c":
        return CanaryHarness(
            llm,
            canary_rate=observability_rate,
            tick_interval=tick_interval,
            tier_application=True,
            tier_application_quant_v=True,
            seed=seed,
        )
    raise ValueError(f"unknown config: {config}")


# --------------------------------------------------------------------------
# One (context, config) cell
# --------------------------------------------------------------------------


def run_cell(
    llm: Any,
    *,
    context: int,
    n_prompts: int,
    max_new_tokens: int,
    config: str,
    seed: int,
    tick_interval: int,
    observability_rate: float,
    tau: float = 0.01,
    initial_level_int: Optional[int] = None,
    principled_initial_tier: bool = False,
    adaptive_tau: bool = False,
    adaptive_tau_percentile: float = 50.0,
    adaptive_tau_scale: float = 1.0,
    multi_precision: bool = False,
    tau_budget: float = 0.05,
    warmup_n: int = 8,
) -> dict:
    """Run one (context, config) cell.

    For ``echokv_3c`` we use a 2-phase protocol:
      1. Warmup phase: install harness with full canary (observability_rate)
         + ``tier_application=False``; run a small batch of prompts to let
         the controller observe δ̂ and converge on a tier vector. The
         tier vector + per-layer EMA are captured.
      2. Measurement phase: install a NEW harness with ``tier_application=True``,
         ``freeze_controller=True``, and the converged levels seeded from
         phase 1. Run the full ``n_prompts`` measurement batch. Every
         prompt sees the same tier vector — no mid-run tier flapping.

    For ``fp16`` / ``static_int4`` / ``static_int2`` the cell runs in a
    single phase (no warmup needed).
    """
    from core.block_allocator import Precision
    from evaluation.serving.canary_vllm import CanaryHarness

    prompts = make_ruler_prompts(n=n_prompts, seed=seed, target_tokens=context)

    if config != "echokv_3c":
        harness = make_harness(
            llm, config,
            seed=seed,
            tick_interval=tick_interval,
            observability_rate=observability_rate,
        )
        label = f"{config}_{context}"
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
        return blob

    # echokv_3c — 2-phase protocol.
    # Phase 1: warmup — observe δ̂, let controller converge.
    log.info("    [warmup] running %d prompts at observability rate %.3f",
             warmup_n, observability_rate)
    warmup_prompts = make_ruler_prompts(
        n=warmup_n, seed=seed + 100, target_tokens=context,
    )
    from evaluation.serving.canary_vllm import initial_tier_for_context

    if principled_initial_tier:
        initial_level_obj = initial_tier_for_context(context)
    elif initial_level_int is not None:
        initial_level_obj = Precision(initial_level_int)
    else:
        initial_level_obj = None
    warmup_harness = CanaryHarness(
        llm,
        canary_rate=observability_rate,
        tau=tau,
        initial_level=initial_level_obj,
        tick_interval=tick_interval,
        tier_application=False,           # warmup is observation only
        freeze_controller=False,
        adaptive_tau=adaptive_tau,
        adaptive_tau_percentile=adaptive_tau_percentile,
        adaptive_tau_scale=adaptive_tau_scale,
        multi_precision=multi_precision,
        tau_budget=tau_budget,
        seed=seed,
    )
    warmup_harness.install()
    try:
        run_one(
            llm, warmup_prompts,
            label=f"echokv_warmup_{context}",
            max_new_tokens=max_new_tokens,
            harness=warmup_harness,
        )
    finally:
        warmup_harness.remove()
    converged_levels = warmup_harness.controller.snapshot()
    log.info("    [warmup] converged tier vector: %s",
             [int(l) for l in converged_levels])

    # Phase 2: measurement — frozen tier, tier_application on.
    measure_harness = CanaryHarness(
        llm,
        canary_rate=1e-6,                 # essentially no canary fires (already converged)
        tau=tau,
        initial_level=initial_level_obj,
        tick_interval=tick_interval,
        tier_application=True,            # apply tiers to K and V
        freeze_controller=True,           # tier vector stays put
        multi_precision=multi_precision,
        tau_budget=tau_budget,
        seed=seed + 1,
    )
    # Seed the measurement controller's level vector.
    for li, lvl in enumerate(converged_levels):
        measure_harness.controller._levels[li] = Precision(int(lvl))

    measure_harness.install()
    try:
        result = run_one(
            llm, prompts,
            label=f"echokv_3c_{context}",
            max_new_tokens=max_new_tokens,
            harness=measure_harness,
        )
    finally:
        measure_harness.remove()

    blob = asdict(result)
    blob["config"] = config
    blob["context"] = context
    blob["n_prompts"] = n_prompts
    blob["warmup_converged_tiers"] = [int(l) for l in converged_levels]
    return blob


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-model-len", type=int, default=131_072)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--tick-interval", type=int, default=10,
                   help="for short runs the controller needs a fast cadence")
    p.add_argument("--tau", type=float, default=0.01,
                   help="controller setpoint (default 0.01). Lower → easier "
                        "to upgrade, harder to downgrade — useful at long "
                        "context where δ̂ is small in absolute terms.")
    p.add_argument("--initial-level", type=int, default=None, choices=[2, 4, 8],
                   help="controller's per-layer initial precision. Default "
                        "INT4 (the architectural KIVI baseline). INT8 is a "
                        "useful starting point at long context where the "
                        "default τ would otherwise cause aggressive downgrade.")
    p.add_argument("--principled-initial-tier", action="store_true",
                   help="derive initial_level from prompt length per the "
                        "ARCHITECTURE.md § 9.6 cost-asymmetry rule "
                        "(<512 → INT2, 512-8K → INT4, ≥8K → INT8). "
                        "Overrides --initial-level if both are set.")
    p.add_argument("--adaptive-tau", action="store_true",
                   help="recompute τ from the observed δ̂_ℓ distribution "
                        "before each controller tick (Path A from "
                        "Open Question 12). τ_adaptive = "
                        "percentile(δ̂_ℓ, p) × scale.")
    p.add_argument("--adaptive-tau-percentile", type=float, default=50.0)
    p.add_argument("--adaptive-tau-scale", type=float, default=1.0)
    p.add_argument("--multi-precision", action="store_true",
                   help="use the neighbours-only multi-precision canary "
                        "(ARCHITECTURE.md § 9.8). Probes per-tier δ̂ and "
                        "uses argmin-with-budget instead of scalar τ.")
    p.add_argument("--tau-budget", type=float, default=0.05,
                   help="divergence budget for multi-precision controller. "
                        "From τ_budget calibration (4K probe): static-INT4 "
                        "EMA(δ̂_4) mean ≈ 0.022 (acc=1.0), static-INT2 "
                        "EMA(δ̂_2) mean ≈ 0.36 (acc=0.05). 0.05 lies cleanly "
                        "in the gap.")
    p.add_argument("--observability-rate", type=float, default=0.25,
                   help=("canary rate for the EchoKV-3C config — needs to be "
                         "high enough that EMA differentiates per-layer "
                         "signal in a small N. Production rate is 1/64; "
                         "for this measurement we observe more aggressively."))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=Path("results/serving/session11_sweep.json"))
    p.add_argument("--contexts", type=int, nargs="+",
                   default=[32_000, 64_000, 128_000])
    p.add_argument("--n-per-context", type=int, nargs="+",
                   default=[50, 30, 20],
                   help="prompt counts per context (must match --contexts length)")
    p.add_argument("--configs", nargs="+",
                   default=["fp16", "static_int4", "echokv_3c"],
                   choices=["fp16", "static_int4", "static_int2", "echokv_3c"])
    a = p.parse_args(argv)

    if len(a.contexts) != len(a.n_per_context):
        p.error("--contexts and --n-per-context must have the same length")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    from vllm import LLM  # lazy import — env vars must be set before

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

    # Warmup at smallest context.
    smallest = min(a.contexts)
    warmup_prompts = make_ruler_prompts(n=4, seed=a.seed, target_tokens=smallest)
    warmup(llm, warmup_prompts, a.max_new_tokens)

    # Run the full grid.
    all_cells: list[dict] = []
    for context, n_prompts in zip(a.contexts, a.n_per_context):
        for config in a.configs:
            log.info(
                "=== context=%d  config=%s  n=%d ===", context, config, n_prompts,
            )
            cell = run_cell(
                llm,
                context=context,
                n_prompts=n_prompts,
                max_new_tokens=a.max_new_tokens,
                config=config,
                seed=a.seed,
                tick_interval=a.tick_interval,
                observability_rate=a.observability_rate,
                tau=a.tau,
                initial_level_int=a.initial_level,
                principled_initial_tier=a.principled_initial_tier,
                adaptive_tau=a.adaptive_tau,
                adaptive_tau_percentile=a.adaptive_tau_percentile,
                adaptive_tau_scale=a.adaptive_tau_scale,
                multi_precision=a.multi_precision,
                tau_budget=a.tau_budget,
            )
            all_cells.append(cell)
            log.info(
                "    acc=%.3f  wall=%.1fs",
                cell["accuracy"], cell["wall_clock_s"],
            )

    summary = {
        "session": "20260426_11",
        "model": a.model,
        "max_new_tokens": a.max_new_tokens,
        "observability_rate": a.observability_rate,
        "tick_interval": a.tick_interval,
        "cells": all_cells,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=2))
    log.info("wrote %s", a.out)

    # Pretty-print: configs as columns, contexts as rows. Per cell:
    # accuracy and (if available) memory savings vs FP16 from
    # memory_projection.
    print()
    print("=" * 88)
    print(f"Session 11 — RULER context sweep")
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
            mp = (cell.get("canary_report") or {}).get("memory_projection") or {}
            save = mp.get("savings_vs_baseline_pct")
            if save is None and cfg == "fp16":
                save_s = "  (baseline)"
            elif save is None:
                save_s = ""
            else:
                save_s = f"  {save:+.1f}% vs FP16"
            print(f"{f'{acc:.3f}{save_s}':>20}  ", end="")
        print()
    print("=" * 88)

    # Verdict: did we find a baseline that breaks at this context?
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
            print(f"  - {cfg} @ {ctx}: acc={acc:.3f} (FP16={fp16_acc:.3f}, Δ={acc - fp16_acc:+.3f})")
        # If echokv_3c exists for any of these, check if it recovers.
        for ctx, cfg, acc, fp16_acc in broken_baselines:
            ek_acc = by_ctx[ctx].get("echokv_3c", {}).get("accuracy")
            if ek_acc is not None:
                print(f"  EchoKV-3C @ {ctx}: acc={ek_acc:.3f}  "
                      f"(recovers {ek_acc - acc:+.3f} of the loss vs {cfg})")
    else:
        print("VERDICT: no static baseline degrades on RULER at any tested context.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
