"""τ_budget calibration via static-tier probes.

Runs RULER prompts at static INT2, INT4, INT8 (fixed via ``force_bits``)
with the multi-precision canary observing in background. Extracts the
per-tier EMA distributions under each static configuration. The natural
τ_budget for a paper-defensible default is somewhere between:

  - the highest EMA(δ̂_8) seen under static INT8 (which keeps full
    accuracy — defines an upper bound on "safe" divergence at INT8),
  - the lowest EMA(δ̂_2) seen under static INT2 (which fails — defines
    the divergence level the canary should reject)

Picks a τ_budget that lands between the two, justifies the choice, and
records the calibration in ``results/serving/tau_budget_calibration.json``.

Usage::

    FLASHINFER_DISABLE_VERSION_CHECK=1 \\
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \\
    CUDA_VISIBLE_DEVICES=0 \\
    /opt/conda/envs/scalable/bin/python -m evaluation.serving.calibrate_tau_budget
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.block_allocator import Precision
from evaluation.motivation.workloads import _ruler_sample
from evaluation.serving.canary_vllm import CanaryHarness
from evaluation.serving.run_session8 import DEFAULT_MODEL


log = logging.getLogger("calibrate_tau_budget")


def run_static_tier_probe(
    llm: Any,
    *,
    bits: int,
    n_prompts: int,
    target_tokens: int,
    max_new_tokens: int,
    canary_rate: float,
    seed: int,
) -> dict:
    """Run RULER under a forced static tier with the multi-precision canary
    on. Returns per-tier EMA stats.
    """
    from vllm import SamplingParams

    prompts = [
        _ruler_sample(i, seed=seed, target_tokens=target_tokens)
        for i in range(n_prompts)
    ]
    h = CanaryHarness(
        llm,
        canary_rate=canary_rate,
        multi_precision=True,
        tau_budget=0.02,                # placeholder; we're not using the controller's decisions
        tier_application=True,
        tier_application_quant_v=True,
        force_bits=bits,                # uniform static precision
        freeze_controller=True,         # no tier changes during the probe
        monotonicity_check=True,
        seed=seed,
    )
    h.install()
    try:
        sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        out = llm.generate([r.prompt for r in prompts], sampling_params=sp, use_tqdm=False)
    finally:
        h.remove()

    n_correct = 0
    for r, o in zip(prompts, out):
        text = o.outputs[0].text
        if r.score_fn:
            try:
                if r.score_fn(text):
                    n_correct += 1
            except Exception:
                pass

    rpt = h.report()
    em = rpt["mp_ema_per_tier_per_layer"] or {}
    per_tier_stats: dict = {}
    for tier_int in (2, 4, 8):
        vec = em.get(tier_int, [])
        nz = [v for v in vec if v > 0]
        per_tier_stats[tier_int] = {
            "n_layers_with_signal": len(nz),
            "ema_min": min(nz) if nz else None,
            "ema_p50": (
                sorted(nz)[len(nz) // 2] if nz else None
            ),
            "ema_mean": statistics.fmean(nz) if nz else None,
            "ema_max": max(nz) if nz else None,
            "ema_per_layer": list(vec),
        }
    return {
        "static_bits": bits,
        "n_prompts": n_prompts,
        "accuracy": n_correct / max(1, n_prompts),
        "canary_fires": rpt["canary_fires"],
        "monotonicity_violations": rpt["mp_monotonicity_violations"],
        "per_tier_stats": per_tier_stats,
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--target-tokens", type=int, default=4_000,
                   help="RULER prompt size in tokens. 4K is fast and "
                        "sufficient for τ_budget calibration.")
    p.add_argument("--n-prompts", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--canary-rate", type=float, default=0.5,
                   help="high rate so EMA converges quickly")
    p.add_argument("--max-model-len", type=int, default=8_192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=Path("results/serving/tau_budget_calibration.json"))
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from vllm import LLM

    log.info("loading %s", a.model)
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

    summary: dict = {"model": a.model, "target_tokens": a.target_tokens, "n_prompts": a.n_prompts,
                     "probes": []}

    for bits in (8, 4, 2):
        log.info("=== static_INT%d, n=%d ===", bits, a.n_prompts)
        cell = run_static_tier_probe(
            llm,
            bits=bits, n_prompts=a.n_prompts,
            target_tokens=a.target_tokens, max_new_tokens=a.max_new_tokens,
            canary_rate=a.canary_rate, seed=a.seed,
        )
        summary["probes"].append(cell)
        st = cell["per_tier_stats"]
        log.info(
            "    acc=%.3f  monotonicity_violations=%d",
            cell["accuracy"], cell["monotonicity_violations"],
        )
        for t_int in (2, 4, 8):
            s = st[t_int]
            log.info(
                "      EMA(δ̂_INT%d): min=%s p50=%s mean=%s max=%s",
                t_int,
                f"{s['ema_min']:.5f}" if s["ema_min"] else "NA",
                f"{s['ema_p50']:.5f}" if s["ema_p50"] else "NA",
                f"{s['ema_mean']:.5f}" if s["ema_mean"] else "NA",
                f"{s['ema_max']:.5f}" if s["ema_max"] else "NA",
            )

    # Calibration: pick τ_budget between
    #   (highest EMA(δ̂_8) under static INT8 with acc=1.0)  ← upper bound on "safe"
    #   (lowest  EMA(δ̂_2) under static INT2 with acc<1.0)  ← lower bound on "broken"
    int8_probe = next(p for p in summary["probes"] if p["static_bits"] == 8)
    int2_probe = next(p for p in summary["probes"] if p["static_bits"] == 2)

    safe_upper = int8_probe["per_tier_stats"][8]["ema_max"]   # INT8 keeps acc; this is the safe ceiling
    broken_lower = int2_probe["per_tier_stats"][2]["ema_min"]  # INT2 breaks; this is the broken floor

    if safe_upper is None or broken_lower is None:
        recommended = None
        rationale = "could not derive bounds (missing EMA values)"
    elif safe_upper >= broken_lower:
        # Distributions overlap — no clean τ_budget exists at this level.
        recommended = (safe_upper + broken_lower) / 2
        rationale = (
            f"distributions overlap: EMA(δ̂_8 under INT8).max={safe_upper:.5f} "
            f">= EMA(δ̂_2 under INT2).min={broken_lower:.5f}; "
            f"using midpoint as best-effort"
        )
    else:
        # Clean gap: pick geometric mean of safe_upper and broken_lower.
        recommended = (safe_upper * broken_lower) ** 0.5
        rationale = (
            f"safe_upper={safe_upper:.5f} (INT8 max), "
            f"broken_lower={broken_lower:.5f} (INT2 min); "
            f"τ_budget = √(safe_upper · broken_lower) lands inside the gap"
        )

    summary["calibration"] = {
        "safe_upper_int8_max": safe_upper,
        "broken_lower_int2_min": broken_lower,
        "recommended_tau_budget": recommended,
        "rationale": rationale,
    }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 80)
    print("τ_budget calibration — RULER@{:d}K, n={}".format(
        a.target_tokens // 1000, a.n_prompts,
    ))
    print("=" * 80)
    print(f"{'static':<10}  {'acc':<8}  {'EMA(δ̂_2)':<10}  {'EMA(δ̂_4)':<10}  {'EMA(δ̂_8)':<10}")
    for cell in summary["probes"]:
        st = cell["per_tier_stats"]
        bits = cell["static_bits"]
        emas = []
        for t_int in (2, 4, 8):
            mean = st[t_int]["ema_mean"]
            emas.append(f"{mean:.5f}" if mean else "NA       ")
        print(f"INT{bits:<7}  {cell['accuracy']:<8.3f}  {emas[0]:<10}  {emas[1]:<10}  {emas[2]:<10}")
    print()
    print(f"Recommended τ_budget = {recommended}")
    print(f"  rationale: {rationale}")
    print()
    print(f"  → use this value in CanaryHarness(tau_budget=...)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
