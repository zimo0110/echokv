"""τ_budget calibration on 70B + GSM8K-Hard via TP-aware static probes.

Counterpart to ``calibrate_tau_budget.py`` (which only handles 8B+RULER
in-process). Probes static INT2/INT4/INT8 on Llama-3.1-70B at TP=4 with
the multi-precision canary observing in shadow mode, and reports per-
tier per-layer EMA distributions.

The motivation: Session 14 Phase 3 ran EchoKV-3C on 70B GSM8K-Hard at
``τ_budget=0.05`` and got a tier vector of {INT4: 77, INT2: 3} —
no INT8 upgrades and a ~2pp regression vs static INT4. The question
this calibration answers:

- Scenario A: ``δ̂_4`` max > 0.02 on some layers → meaningful per-layer
  variance; tighten τ_budget to trigger INT8 upgrades.
- Scenario B: ``δ̂_4`` max < 0.02 everywhere → INT4 damage is diffuse;
  no per-layer controller can fix the 2.2pp loss.
- Scenario C: ``δ̂_2`` min near 0.05 → tighter budget cleanly rejects
  the boundary INT2 layers but doesn't unlock INT8 upgrades.

Usage::

    VLLM_ALLOW_INSECURE_SERIALIZATION=1 \\
    FLASHINFER_DISABLE_VERSION_CHECK=1 \\
    CUDA_VISIBLE_DEVICES=4,5,6,7 \\
    /opt/conda/envs/scalable/bin/python \\
        -m evaluation.serving.calibrate_tau_budget_70b_gsm8k_hard
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.environ["PYTHONPATH"] = (
    f"{_REPO_ROOT}:" + os.environ.get("PYTHONPATH", "")
)

from evaluation.motivation.workloads import _gsm8k_hard_stream
from evaluation.serving.run_session8 import run_one
from evaluation.serving.run_session14_gsm8k_hard import (
    materialize_chat_prompts,
)
from evaluation.serving.canary_tp import TPCanaryHandle


log = logging.getLogger("calibrate_70b_gsm8k_hard")
DEFAULT_70B = "/mnt/nvme_tmp6/llama-70b"


def _consolidate_per_layer(
    per_rank_per_tier_delta: list[dict],
) -> dict[int, list[float]]:
    """Mean across TP ranks of per-layer EMA, per tier.

    Each rank's snapshot has a ``per_tier_delta`` dict like
    ``{2: [...], 4: [...], 8: [...]}`` keyed by tier-int (or by string
    after JSON round-trip). With TP=4, each rank sees a different head
    subset for each layer, so per-layer δ̂ is a noisy estimate from
    that rank's local view; averaging tightens the per-layer signal.
    Layers with no observed signal (EMA == 0) are excluded from the
    mean for that layer; if no rank observed a layer, the consolidated
    EMA is None.
    """
    # Normalize keys (snapshot_per_tier may return int or string keys).
    def _key(t):
        return int(t) if isinstance(t, str) else t

    consolidated: dict[int, list[float]] = {}
    n_layers = None
    for d in per_rank_per_tier_delta:
        for t_raw, vec in d.items():
            t = _key(t_raw)
            if n_layers is None:
                n_layers = len(vec)

    for tier in (2, 4, 8):
        rank_vecs: list[list[float]] = []
        for d in per_rank_per_tier_delta:
            for t_raw, vec in d.items():
                if _key(t_raw) == tier:
                    rank_vecs.append(vec)
                    break
        per_layer: list[float] = []
        for li in range(n_layers or 0):
            vals = [
                rv[li] for rv in rank_vecs
                if li < len(rv) and rv[li] > 0.0
            ]
            if vals:
                per_layer.append(sum(vals) / len(vals))
            else:
                per_layer.append(0.0)  # 0.0 is the no-signal sentinel
        consolidated[tier] = per_layer
    return consolidated


def run_static_probe(
    llm: Any,
    *,
    bits: int,
    prompts: list,
    max_new_tokens: int,
    canary_rate: float,
    seed: int,
) -> dict:
    """Force the static tier to ``bits`` (no controller), run prompts,
    and capture the multi-precision canary's per-tier EMA per rank."""
    handle = TPCanaryHandle(
        llm,
        harness_kwargs={
            "canary_rate": canary_rate,
            "tau": 0.01,
            "tick_interval": 10,
            "tier_application": True,
            "tier_application_quant_v": True,
            "force_bits": bits,
            "freeze_controller": True,
            "multi_precision": True,
            "tau_budget": 0.02,    # placeholder; controller is frozen
            "monotonicity_check": True,
            "seed": seed,
        },
    )
    handle.install()
    try:
        result = run_one(
            llm, prompts,
            label=f"calib_static_int{bits}_70b_gsm8k_hard",
            max_new_tokens=max_new_tokens,
            harness=None,
        )
        snaps = handle.snapshot()
    finally:
        handle.remove()

    per_rank_mp = [s.get("per_tier_delta") or {} for s in snaps]
    consolidated = _consolidate_per_layer(per_rank_mp)

    per_tier_stats: dict[int, dict] = {}
    for tier in (2, 4, 8):
        vec = consolidated.get(tier, [])
        nz = [v for v in vec if v > 0.0]
        per_tier_stats[tier] = {
            "n_layers_with_signal": len(nz),
            "ema_min": min(nz) if nz else None,
            "ema_p50": (sorted(nz)[len(nz) // 2] if nz else None),
            "ema_mean": statistics.fmean(nz) if nz else None,
            "ema_max": max(nz) if nz else None,
            "ema_per_layer": list(vec),
        }
    return {
        "static_bits": bits,
        "n_prompts": len(prompts),
        "accuracy": result.accuracy,
        "wall_clock_s": result.wall_clock_s,
        "per_rank_canary_fires": [
            s.get("stats", {}).get("canary_fires", 0) for s in snaps
        ],
        "per_tier_stats": per_tier_stats,
    }


def _tau_budget_recommendation(probes: list[dict]) -> tuple[Optional[float], str]:
    int8 = next(p for p in probes if p["static_bits"] == 8)
    int2 = next(p for p in probes if p["static_bits"] == 2)
    safe_upper = int8["per_tier_stats"][8]["ema_max"]
    broken_lower = int2["per_tier_stats"][2]["ema_min"]
    if safe_upper is None or broken_lower is None:
        return None, "missing EMA values; no recommendation"
    if safe_upper >= broken_lower:
        recommended = (safe_upper + broken_lower) / 2
        return recommended, (
            f"distributions overlap (safe_upper={safe_upper:.5f}, "
            f"broken_lower={broken_lower:.5f}); midpoint as best-effort"
        )
    recommended = (safe_upper * broken_lower) ** 0.5
    return recommended, (
        f"clean gap: safe_upper={safe_upper:.5f}, "
        f"broken_lower={broken_lower:.5f}; geometric mean used"
    )


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_70B)
    p.add_argument("--tensor-parallel-size", type=int, default=4)
    p.add_argument("--max-model-len", type=int, default=4_096)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-num-seqs", type=int, default=8)
    p.add_argument("--n-prompts", type=int, default=20)
    p.add_argument(
        "--canary-rate", type=float, default=0.5,
        help="high rate so per-tier EMA converges in 20 prompts",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out", type=Path,
        default=Path(
            "results/serving/session14_70b/"
            "tau_budget_calib_gsm8k_hard.json"
        ),
    )
    a = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    from vllm import LLM

    log.info("loading %s tp=%d", a.model, a.tensor_parallel_size)
    t0 = time.perf_counter()
    llm = LLM(
        model=a.model,
        tensor_parallel_size=a.tensor_parallel_size,
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=a.gpu_memory_utilization,
        max_model_len=a.max_model_len,
        enable_prefix_caching=False,
        max_num_seqs=a.max_num_seqs,
    )
    log.info("loaded in %.1fs", time.perf_counter() - t0)

    raw = list(_gsm8k_hard_stream(n=a.n_prompts, seed=a.seed))
    prompts = materialize_chat_prompts(llm, raw)

    summary = {
        "model": a.model,
        "tensor_parallel_size": a.tensor_parallel_size,
        "n_prompts": a.n_prompts,
        "max_new_tokens": a.max_new_tokens,
        "max_num_seqs": a.max_num_seqs,
        "canary_rate": a.canary_rate,
        "probes": [],
    }

    # Order matters: INT8 first (highest accuracy), then INT4, then INT2
    # (so any cumulative state degradation only affects the noisiest tier).
    for bits in (8, 4, 2):
        log.info("=== static_INT%d  n=%d ===", bits, a.n_prompts)
        cell = run_static_probe(
            llm, bits=bits, prompts=prompts,
            max_new_tokens=a.max_new_tokens,
            canary_rate=a.canary_rate, seed=a.seed,
        )
        summary["probes"].append(cell)
        st = cell["per_tier_stats"]
        log.info(
            "    acc=%.3f  wall=%.1fs  fires/rank=%s",
            cell["accuracy"], cell["wall_clock_s"],
            cell["per_rank_canary_fires"],
        )
        for t_int in (2, 4, 8):
            s = st[t_int]
            log.info(
                "      EMA(δ̂_%d): min=%s p50=%s mean=%s max=%s "
                "(n=%d layers)",
                t_int,
                f"{s['ema_min']:.5f}" if s["ema_min"] else "NA",
                f"{s['ema_p50']:.5f}" if s["ema_p50"] else "NA",
                f"{s['ema_mean']:.5f}" if s["ema_mean"] else "NA",
                f"{s['ema_max']:.5f}" if s["ema_max"] else "NA",
                s["n_layers_with_signal"],
            )

    rec, rationale = _tau_budget_recommendation(summary["probes"])
    summary["recommendation"] = {
        "tau_budget": rec,
        "rationale": rationale,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=2))
    log.info("wrote %s", a.out)

    # Pretty-print
    print()
    print("=" * 80)
    print(f"τ_budget calibration — 70B + GSM8K-Hard, n={a.n_prompts}")
    print("=" * 80)
    print(f"{'static':<10} {'acc':<6} "
          f"{'δ̂_2 min/mean/max':<28} "
          f"{'δ̂_4 min/mean/max':<28} "
          f"{'δ̂_8 max':<10}")
    for cell in summary["probes"]:
        st = cell["per_tier_stats"]
        bits = cell["static_bits"]

        def _fmt(s):
            if s["ema_min"] is None:
                return "NA"
            return (f"{s['ema_min']:.4f}/{s['ema_mean']:.4f}/"
                    f"{s['ema_max']:.4f}")

        d2 = _fmt(st[2])
        d4 = _fmt(st[4])
        d8 = (f"{st[8]['ema_max']:.5f}"
              if st[8]['ema_max'] else "NA")
        print(f"INT{bits:<7} {cell['accuracy']:<6.3f} "
              f"{d2:<28} {d4:<28} {d8:<10}")
    print()
    print(f"Recommendation: τ_budget = {rec}")
    print(f"  rationale: {rationale}")
    print()
    print("Scenario interpretation:")
    int4_probe = next(p for p in summary["probes"] if p["static_bits"] == 4)
    delta4_max = int4_probe["per_tier_stats"][4]["ema_max"]
    if delta4_max and delta4_max > 0.02:
        print(f"  Scenario A: δ̂_4 max = {delta4_max:.4f} > 0.02 — "
              "meaningful per-layer variance; INT8 upgrades available.")
    elif delta4_max:
        print(f"  Scenario B: δ̂_4 max = {delta4_max:.4f} < 0.02 — "
              "INT4 damage is diffuse; no per-layer controller can "
              "fix the static_int4 loss.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
