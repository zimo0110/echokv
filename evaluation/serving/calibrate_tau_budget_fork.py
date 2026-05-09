"""Fork-side tau_budget calibrator (Phase G(e) follow-up #1).

Why this exists
---------------
The upstream ``calibrate_tau_budget*.py`` scripts run the canary
under shadow-mode (FP16 cache + simulated K/V quantization at
attention input). That gives a δ̂ distribution against an FP16
reference. The fork has a *real* KIVI cache, so the canary's
neighbour-tier δ̂ is measured against an already-quantized reference —
fundamentally a different signal regime (G(e) finding: tier-8 δ̂ on
the fork is ∼100× smaller than on shadow-mode). The paper's
``tau_budget=0.02`` was calibrated for shadow-mode and keeps INT4
always-safe on the fork; the controller never differentiates.

This calibrator runs the canary under the **fork's real KIVI cache**
at each probe tier in turn (INT8, INT4, INT2 active). For each probe
it dumps the per-layer per-tier δ̂ EMA distribution. The output is a
recommended ``tau_budget`` that bisects the active-tier-EMA
distribution at the chosen heterogeneity level.

The fork-port differs from upstream in two ways:

  1. **No tier_application / no force_bits** — the cache is *really*
     at the probe tier (boot at ``int8_kivi`` for max headroom, swap
     active bits via ``swap_tier_vector_to_model`` per probe). Shadow
     mode would double-apply the noise.
  2. **Probe tier is set before each canary observation pass**,
     not via the controller. We measure the canary's view of the
     world while the cache is locked at INT8 / INT4 / INT2 in turn.

Output (JSON):

  - ``probes``: per-probe-tier dict with per-layer per-target-tier δ̂
    EMA stats (min, p25, p50, p75, max, per_layer list).
  - ``recommendations``: per-probe-tier ``tau_budget`` candidates that
    bisect the per-target-tier EMA distribution at given quantiles
    (for "how heterogeneous do you want the controller to land").

Usage
-----
::

    CUDA_VISIBLE_DEVICES=0,1,2,3 \\
    VLLM_ALLOW_INSECURE_SERIALIZATION=1 \\
    FLASHINFER_DISABLE_VERSION_CHECK=1 \\
    python evaluation/serving/calibrate_tau_budget_fork.py \\
      --model /mnt/nvme_tmp6/llama-70b --tp 4 \\
      --workload gsm8k_hard --n-prompts 20
"""
from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional

_FORK = "/mnt/nvme_tmp10/nazar_vllm_fork"
_REPO = str(Path(__file__).resolve().parents[2])
if _FORK not in sys.path:
    sys.path.insert(0, _FORK)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.environ["PYTHONPATH"] = f"{_REPO}:{_FORK}:" + os.environ.get("PYTHONPATH", "")

log = logging.getLogger("calibrate_tau_budget_fork")

DEFAULT_8B = (
    "/mnt/nvme_tmp/nazar_hf/hub/"
    "models--NousResearch--Meta-Llama-3.1-8B-Instruct/"
    "snapshots/d10aef7999a2b5ba950ab3974312feeedbfe0b77"
)
DEFAULT_70B = "/mnt/nvme_tmp6/llama-70b"


def _consolidate_per_layer(per_rank_mp: list[dict]) -> dict:
    """Per-layer mean δ̂ across ranks (each rank measures on its own
    head subset under TP). Mirrors the upstream calibrator."""
    out: dict = {}
    if not per_rank_mp:
        return out
    tiers = sorted({int(t) for d in per_rank_mp for t in d.keys()})
    for tier in tiers:
        n_layers_per_rank = [
            len(d.get(str(tier), [])) for d in per_rank_mp
        ]
        n = max(n_layers_per_rank) if n_layers_per_rank else 0
        per_layer: list[float] = []
        for li in range(n):
            vals = []
            for d in per_rank_mp:
                lst = d.get(str(tier)) or d.get(tier) or []
                if li < len(lst) and lst[li] > 0.0:
                    vals.append(lst[li])
            per_layer.append(sum(vals) / len(vals) if vals else 0.0)
        out[tier] = per_layer
    return out


def _tier_stats(vec: list[float]) -> dict:
    nz = sorted(v for v in vec if v > 0.0)
    if not nz:
        return {
            "n_layers_with_signal": 0,
            "ema_min": None, "ema_p25": None, "ema_p50": None,
            "ema_p75": None, "ema_mean": None, "ema_max": None,
            "ema_per_layer": list(vec),
        }
    n = len(nz)
    return {
        "n_layers_with_signal": n,
        "ema_min": nz[0],
        "ema_p25": nz[n // 4],
        "ema_p50": nz[n // 2],
        "ema_p75": nz[3 * n // 4],
        "ema_mean": statistics.fmean(nz),
        "ema_max": nz[-1],
        "ema_per_layer": list(vec),
    }


def make_workload_requests(workload: str, n: int, seed: int) -> list:
    if workload == "gsm8k_hard":
        from evaluation.motivation.workloads import _gsm8k_hard_stream
        return list(_gsm8k_hard_stream(n=n, seed=seed))
    if workload == "hotpotqa":
        from evaluation.motivation.workloads import _hotpotqa_stream
        return list(_hotpotqa_stream(n=n, seed=seed, level="hard"))
    raise ValueError(f"unknown workload {workload!r}")


def materialize_chat_prompts(llm: Any, requests: list) -> list:
    from evaluation.motivation.workloads import Request as WReq
    tok = llm.get_tokenizer()
    out = []
    for r in requests:
        if r.messages is None:
            out.append(r)
            continue
        templated = tok.apply_chat_template(
            r.messages, tokenize=False, add_generation_prompt=True,
        )
        out.append(WReq(
            workload=r.workload, request_id=r.request_id,
            prompt=templated, reference=r.reference,
            score_fn=r.score_fn, messages=r.messages,
        ))
    return out


def run_probe(
    llm: Any, *,
    probe_bits: int,
    prompts: list,
    max_new_tokens: int,
    canary_rate: float,
    seed: int,
) -> dict:
    """Lock the engine's active bits to ``probe_bits`` via
    ``swap_tier_vector_to_model``, install the canary in observation
    mode (multi_precision=True), run the probe prompts, return the
    per-tier δ̂ snapshot."""
    from vllm import SamplingParams
    from evaluation.serving.canary_tp import TPCanaryHandle
    from evaluation.serving.fork_tier import (
        build_uniform_tier_vector, read_buffer_bits_from_model,
        read_tier_vector_from_model, swap_tier_vector_to_model,
    )

    rank0_buffer = llm.apply_model(read_buffer_bits_from_model)[0]
    num_layers = len(rank0_buffer)
    if probe_bits > max(rank0_buffer):
        raise ValueError(
            f"probe_bits={probe_bits} exceeds buffer_bits={max(rank0_buffer)}; "
            "boot the engine at the highest probe tier (e.g. int8_kivi)."
        )

    target = build_uniform_tier_vector(num_layers, probe_bits)
    log.info("PROBE bits=%d (uniform across %d layers)", probe_bits, num_layers)
    llm.apply_model(
        functools.partial(swap_tier_vector_to_model, bits_per_layer=target)
    )
    rank_active = llm.apply_model(read_tier_vector_from_model)
    for ri, vec in enumerate(rank_active):
        if any(b != probe_bits for b in vec):
            raise RuntimeError(
                f"rank {ri} mismatch after swap: expected {probe_bits}, "
                f"got set({sorted(set(vec))})"
            )

    handle = TPCanaryHandle(
        llm,
        harness_kwargs={
            "canary_rate": canary_rate,
            "tau": 0.01,
            "tick_interval": 10,
            "tier_application": False,    # the cache IS already at probe_bits
            "freeze_controller": True,    # don't let the controller move tiers
            "multi_precision": True,
            "tau_budget": 0.02,           # ignored when controller is frozen
            "decode_only_canary": True,
            "use_canary_stream": True,
            "seed": seed,
        },
    )
    handle.install()
    try:
        sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        t0 = time.perf_counter()
        outputs = llm.generate(
            [r.prompt for r in prompts],
            sampling_params=sp, use_tqdm=False,
        )
        wall = time.perf_counter() - t0
        snaps = handle.snapshot()
    finally:
        handle.remove()

    n_correct = 0.0
    for r, o in zip(prompts, outputs):
        n_correct += float(r.score_fn(o.outputs[0].text))
    accuracy = n_correct / max(1, len(prompts))

    per_rank_mp = [s.get("per_tier_delta") or {} for s in snaps]
    consolidated = _consolidate_per_layer(per_rank_mp)
    per_tier_stats = {tier: _tier_stats(vec) for tier, vec in consolidated.items()}

    return {
        "probe_bits": probe_bits,
        "n_prompts": len(prompts),
        "accuracy": accuracy,
        "wall_clock_s": wall,
        "per_rank_canary_fires": [
            s.get("stats", {}).get("canary_fires", 0) for s in snaps
        ],
        "per_tier_stats": per_tier_stats,
    }


def recommend_tau_budgets(probes: list[dict]) -> dict:
    """Compute fork-tuned ``tau_budget`` recommendations.

    Strategy: the controller's ``argmin-with-budget`` rule prefers
    downgrade if safe. To force *upgrade*-direction differentiation
    on the fork at a given probe tier, ``tau_budget`` must be tight
    enough that some layers' ``probe_bits``-tier EMA exceeds it.
    The recommended budgets bisect the per-layer probe-tier EMA
    distribution at common quantiles (40/60, 50/50, 60/40 splits).
    """
    out = {}
    for p in probes:
        bits = p["probe_bits"]
        ts = p["per_tier_stats"].get(bits) or p["per_tier_stats"].get(str(bits))
        if not ts:
            continue
        per_layer = sorted(v for v in ts.get("ema_per_layer", []) if v > 0.0)
        if not per_layer:
            continue
        n = len(per_layer)
        # tau_budget thresholds: budget below k-th value → ≥ k layers
        # have probe-tier EMA > budget → those layers fail at probe-tier
        # → controller falls through to next-larger tier (upgrade).
        # Splits of the form "k_low INT_probe + (n-k_low) INT_higher".
        recs = {}
        for split in (0.30, 0.40, 0.50, 0.60, 0.70):
            k = max(1, min(n - 1, int(round(split * n))))
            # Geometric mean of value at index k-1 and value at index k
            # (placing the threshold *between* them) is more robust
            # than picking either endpoint.
            lo = per_layer[k - 1]
            hi = per_layer[k]
            recs[f"split_{int(split*100)}_lower_int{bits}"] = {
                "tau_budget": (lo * hi) ** 0.5,
                "k_layers_below_budget": k,
                "n_layers_with_signal": n,
            }
        out[f"probe_int{bits}"] = recs
    return out


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_8B,
                   help=f"Model path. For 70B pass {DEFAULT_70B!r}.")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--workload", default="gsm8k_hard",
                   choices=["gsm8k_hard", "hotpotqa"])
    p.add_argument("--max-model-len", type=int, default=4_096)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-num-seqs", type=int, default=4)
    p.add_argument("--n-prompts", type=int, default=20,
                   help="Calibration prompts per probe tier.")
    p.add_argument(
        "--canary-rate", type=float, default=0.5,
        help="High rate (default 0.5) so per-tier EMA converges in "
             "n_prompts. Production canary is 1/64.",
    )
    p.add_argument(
        "--probe-tiers", type=int, nargs="+", default=[8, 4, 2],
        help="Active bit-widths to probe. Buffer is always int8_kivi "
             "(max headroom).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out", type=Path,
        default=Path("results/serving/calibrate_tau_budget_fork.json"),
    )
    a = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    from vllm import LLM

    log.info(
        "loading %s tp=%d (kv_cache_dtype=int8_kivi for max-bits buffer)",
        a.model, a.tp,
    )
    t0 = time.perf_counter()
    llm = LLM(
        model=a.model,
        tensor_parallel_size=a.tp,
        dtype="bfloat16",
        kv_cache_dtype="int8_kivi",
        block_size=16,
        enforce_eager=True,
        gpu_memory_utilization=a.gpu_memory_utilization,
        max_model_len=a.max_model_len,
        enable_prefix_caching=False,
        max_num_seqs=a.max_num_seqs,
    )
    log.info("loaded in %.1fs", time.perf_counter() - t0)

    raw = make_workload_requests(a.workload, n=a.n_prompts, seed=a.seed)
    prompts = materialize_chat_prompts(llm, raw)

    probes = []
    for bits in a.probe_tiers:
        log.info("=" * 64)
        result = run_probe(
            llm, probe_bits=bits, prompts=prompts,
            max_new_tokens=a.max_new_tokens,
            canary_rate=a.canary_rate, seed=a.seed,
        )
        log.info(
            "PROBE int%d: acc=%.3f wall=%.1fs canary_fires=%s",
            bits, result["accuracy"], result["wall_clock_s"],
            result["per_rank_canary_fires"],
        )
        for tier in (2, 4, 8):
            st = result["per_tier_stats"].get(tier) or result["per_tier_stats"].get(str(tier))
            if st and st["n_layers_with_signal"] > 0:
                log.info(
                    "  tier %d EMA: n=%d min=%.3e p50=%.3e mean=%.3e max=%.3e",
                    tier, st["n_layers_with_signal"],
                    st["ema_min"], st["ema_p50"], st["ema_mean"], st["ema_max"],
                )
        probes.append(result)

    recommendations = recommend_tau_budgets(probes)

    log.info("=" * 64)
    log.info("RECOMMENDATIONS (fork-tuned tau_budget candidates)")
    for probe_key, recs in recommendations.items():
        log.info("  %s:", probe_key)
        for split_key, rec in recs.items():
            log.info(
                "    %-30s tau_budget=%.3e (k=%d/%d below budget)",
                split_key, rec["tau_budget"],
                rec["k_layers_below_budget"], rec["n_layers_with_signal"],
            )

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "config": {
            "model": a.model, "tp": a.tp, "workload": a.workload,
            "max_model_len": a.max_model_len,
            "max_new_tokens": a.max_new_tokens,
            "n_prompts": a.n_prompts,
            "canary_rate": a.canary_rate,
            "probe_tiers": a.probe_tiers, "seed": a.seed,
        },
        "probes": probes,
        "recommendations": recommendations,
    }, indent=2))
    log.info("wrote %s", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
