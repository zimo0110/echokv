"""Lane 1(b): 3-phase EchoKV harness on the vLLM fork.

Mirrors the upstream-vLLM ``run_session14_70b_*_echokv`` 3-phase
pattern (warmup → consolidate → measure) but drives the fork's *real*
KIVI cache via ``swap_tier_vector_to_model`` instead of shadow-mode
``tier_application``. The whole loop runs in **one engine** thanks to
F.3.1 — no subprocess restart between warmup and measurement.

Three-phase protocol
--------------------
1. **Warmup** (small batch): boot at ``int8_kivi`` (max headroom),
   start every layer at active=INT4 via ``swap_tier_vector_to_model``,
   install ``CanaryHarness`` in observation-only mode
   (``tier_application=False``: the cache is already real KIVI, so
   shadow-quantizing K/V before attention would double-apply noise).
   Run warmup prompts with ``multi_precision=True`` so the canary
   measures δ̂ at the active tier and its neighbours; controller picks
   per-layer tiers.

2. **Consolidate**: per-layer max across TP ranks (most conservative
   — the rank that wants more precision wins). Same logic as upstream.

3. **Measurement**: ``swap_tier_vector_to_model(consolidated)`` —
   same engine. Re-install harness with ``freeze_controller=True``.
   Run measurement prompts; report accuracy, per-layer tier
   distribution, memory projection.

Differences vs upstream
-----------------------
  * No shadow-mode ``tier_application``: the cache is already KIVI,
    so the controller's decision is realised at swap time, not by
    pre-attention quantization.
  * Engine boots at ``int8_kivi`` (max bits) so the buffer can host
    any layer at INT2 / INT4 / INT8 without restart.
  * ``swap_tier_vector_to_model`` between warmup and measurement
    replaces the upstream "seed tiers via controller._levels"
    pattern — the bits change in the fork's actual KV cache.

Why this is small-n by default
------------------------------
The upstream Session 14 harness runs n=100 prompts at 70B for a
single Pareto cell. This driver targets Llama-3.1-8B at small n
(default 30) as a *demonstration* that the 3-phase loop works on the
fork end-to-end. Scaling to paper-grade n on Llama-70B is the
follow-up; this is the smallest unit that proves the wiring.

Usage
-----
::

    CUDA_VISIBLE_DEVICES=4 \\
    VLLM_ALLOW_INSECURE_SERIALIZATION=1 \\
    FLASHINFER_DISABLE_VERSION_CHECK=1 \\
    python evaluation/serving/run_phaseG_3phase_fork.py \\
        --workload gsm8k_hard --n-prompts 30 --warmup-n 8
"""
from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

_FORK = "/mnt/nvme_tmp10/nazar_vllm_fork"
_REPO = str(Path(__file__).resolve().parents[2])
if _FORK not in sys.path:
    sys.path.insert(0, _FORK)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.environ["PYTHONPATH"] = (
    f"{_REPO}:{_FORK}:" + os.environ.get("PYTHONPATH", "")
)

log = logging.getLogger("phaseG_3phase_fork")

DEFAULT_8B = (
    "/mnt/nvme_tmp/nazar_hf/hub/"
    "models--NousResearch--Meta-Llama-3.1-8B-Instruct/"
    "snapshots/d10aef7999a2b5ba950ab3974312feeedbfe0b77"
)
DEFAULT_70B = "/mnt/nvme_tmp6/llama-70b"


def consolidate_tiers(
    per_rank_snapshots: list[dict],
    *,
    rule: str = "max",
) -> list[int]:
    """Combine per-rank tier vectors into a single per-layer vector.

    Phase G(e) addendum 2 finding: the upstream ``max`` rule
    (per-layer max — "rank that wants more precision wins") collapses
    the fork's heterogeneous per-rank decisions into mostly-uniform
    INT8 because TP head-subsetting decorrelates rank-level picks
    near ``tau_budget``. Softer rules ("median-low", "vote-low",
    "mean-round") preserve the per-rank differentiation; on
    Llama-70B GSM8K-Hard at ``tau_budget=3.2e-3`` they all converge
    to ~39 INT4 + 41 INT8 (vs 5 INT4 + 75 INT8 under ``max``).

    Rules:
      * ``max``        — per-layer max; upstream-compat.
      * ``min``        — per-layer min; aggressive downgrade.
      * ``median-low`` — per-layer ``statistics.median_low``; ties
        break low.
      * ``median-high``— per-layer ``statistics.median_high``; ties
        break high.
      * ``mean-round`` — per-layer rounded mean; nearest of {2,4,8}.
      * ``vote-low``   — per-layer majority vote; ties break low.
      * ``vote-high``  — per-layer majority vote; ties break high.
    """
    import statistics
    from collections import Counter
    n = per_rank_snapshots[0]["num_layers"]
    cols = [
        [s["tier_vector"][li] for s in per_rank_snapshots]
        for li in range(n)
    ]
    if rule == "max":
        return [max(c) for c in cols]
    if rule == "min":
        return [min(c) for c in cols]
    if rule == "median-low":
        return [statistics.median_low(c) for c in cols]
    if rule == "median-high":
        return [statistics.median_high(c) for c in cols]
    if rule == "mean-round":
        valid = (2, 4, 8)
        return [
            min(valid, key=lambda t, c=c: abs(t - sum(c) / len(c)))
            for c in cols
        ]
    if rule == "vote-low" or rule == "vote-high":
        out = []
        for c in cols:
            counts = Counter(c)
            top = max(counts.values())
            winners = sorted([t for t, k in counts.items() if k == top])
            out.append(winners[0] if rule == "vote-low" else winners[-1])
        return out
    raise ValueError(f"unknown consolidate rule {rule!r}")


def consolidate_deltas(per_rank_snapshots: list[dict]) -> list[float]:
    """Per-layer mean δ̂ across ranks (each rank measures on its own
    head subset under TP)."""
    n = per_rank_snapshots[0]["num_layers"]
    out: list[float] = []
    for li in range(n):
        vals = [
            s["per_layer_delta"][li] for s in per_rank_snapshots
            if s["per_layer_delta"][li] > 0.0
        ]
        out.append(sum(vals) / len(vals) if vals else 0.0)
    return out


def memory_projection(tiers: list[int]) -> dict:
    n = len(tiers)
    avg_bits = sum(tiers) / n
    return {
        "num_layers": n,
        "avg_bits": avg_bits,
        "savings_vs_fp16_pct": 100.0 * (1.0 - avg_bits / 16.0),
        "savings_vs_kivi4_pct": 100.0 * (1.0 - avg_bits / 4.0),
        "tier_counts": dict(Counter(tiers)),
    }


def materialize_chat_prompts(llm: Any, requests: list) -> list:
    """Apply the tokenizer's chat template in place. Mirrors the
    upstream helper but lives here so this driver is self-contained."""
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
        out.append(
            WReq(
                workload=r.workload,
                request_id=r.request_id,
                prompt=templated,
                reference=r.reference,
                score_fn=r.score_fn,
                messages=r.messages,
            )
        )
    return out


def make_workload_requests(workload: str, n: int, seed: int) -> list:
    if workload == "gsm8k_hard":
        from evaluation.motivation.workloads import _gsm8k_hard_stream
        return list(_gsm8k_hard_stream(n=n, seed=seed))
    if workload == "hotpotqa":
        from evaluation.motivation.workloads import _hotpotqa_stream
        return list(_hotpotqa_stream(n=n, seed=seed, level="hard"))
    if workload == "ruler":
        from evaluation.motivation.workloads import _ruler_sample
        # Shorter than upstream's RULER@4K — at 4K tokens with 80
        # layers, observability_rate=0.25, the canary's per-prefill
        # work overflows vLLM's RPC sample_tokens timeout (~60 s).
        # 1K tokens matches the typical GSM8K-Hard prompt length and
        # keeps the diagnostic comparison apples-to-apples on
        # prompt-distribution-vs-content rather than length.
        return [_ruler_sample(i, seed=seed, target_tokens=1_000) for i in range(n)]
    raise ValueError(f"unknown workload {workload!r}")


def run_one(llm, prompts, *, max_new_tokens: int, label: str) -> dict:
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    t0 = time.perf_counter()
    outputs = llm.generate(
        [r.prompt for r in prompts],
        sampling_params=sp, use_tqdm=False,
    )
    wall = time.perf_counter() - t0
    n_correct = 0.0
    total_in = 0
    total_out = 0
    per_prompt = []
    for r, o in zip(prompts, outputs):
        completion = o.outputs[0].text
        correct = float(r.score_fn(completion))
        n_correct += correct
        total_in += len(o.prompt_token_ids)
        total_out += len(o.outputs[0].token_ids)
        per_prompt.append({
            "request_id": r.request_id,
            "reference": r.reference,
            "completion_tokens": len(o.outputs[0].token_ids),
            "completion": completion,
            "correct": correct,
        })
    return {
        "label": label,
        "n_prompts": len(prompts),
        "n_correct": n_correct,
        "accuracy": n_correct / max(1, len(prompts)),
        "wall_clock_s": wall,
        "total_in_tokens": total_in,
        "total_out_tokens": total_out,
        "tps_in": total_in / wall if wall > 0 else 0.0,
        "tps_out": total_out / wall if wall > 0 else 0.0,
        "per_prompt": per_prompt,
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_8B)
    ap.add_argument("--workload", default="gsm8k_hard",
                    choices=["gsm8k_hard", "hotpotqa", "ruler"])
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-num-seqs", type=int, default=4)
    ap.add_argument("--n-prompts", type=int, default=30)
    ap.add_argument(
        "--warmup-n", type=int, default=32,
        help="Warmup prompts per rank. Phase G(e) finding: at "
             "warmup_n=8 the per-rank EMAs land at wildly different "
             "values across runs (canary fires stochastically, "
             "warmup is short). Bumping to 32 gives the EMA enough "
             "samples per layer to converge near the true value.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--observability-rate", type=float, default=0.25,
        help="canary rate during warmup; production is 1/64.",
    )
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--tick-interval", type=int, default=10)
    ap.add_argument("--tau-budget", type=float, default=0.02)
    ap.add_argument(
        "--per-rank-seed", action="store_true", default=True,
        help="Each TP rank seeds its canary RNG with base_seed + "
             "rank_id (default on). Collapses one source of warmup "
             "nondeterminism. Disable with --no-per-rank-seed for "
             "upstream-compat.",
    )
    ap.add_argument(
        "--no-per-rank-seed", action="store_false", dest="per_rank_seed",
    )
    ap.add_argument(
        "--warmup-grace-ticks", type=int, default=0,
        help="Controller HOLDs every layer for the first N ticks so "
             "the per-tier EMA can absorb samples against the active "
             "substrate before the rule fires on transient early "
             "values. EMAs continue to update during grace. Phase G(e) "
             "addendum 11: upgrade-only with no grace cascades because "
             "first-tick δ̂_4 is high (early samples dominate); a grace "
             "period lets the EMA stabilise before any decision is "
             "committed. With ema_alpha=0.1 the half-life is ~7 "
             "samples; 20 ticks of grace ≈ 3 half-lives.",
    )
    ap.add_argument(
        "--warmup-only-upgrades", action="store_true",
        help="During warmup, the controller is allowed to upgrade tiers "
             "(higher bit-width) but downgrades are suppressed. Breaks "
             "the cache-state feedback loop where downgrades shrink "
             "subsequent δ̂ measurements (Phase G(e) addendum 10: "
             "active-vs-frozen δ̂ ratio is ~5× at tiers 4/8). The "
             "consolidation step still applies normally.",
    )
    ap.add_argument(
        "--freeze-warmup-controller", action="store_true",
        help="Freeze the controller during warmup. Layer tiers stay at "
             "the initial value (set by --initial-active) for the whole "
             "warmup; canary observes per-layer per-tier δ̂ but doesn't "
             "trigger transitions. Used for the contamination diagnostic "
             "(Phase G(e) addendum 8): comparing per-tier δ̂ at the same "
             "active tier under different upstream tier histories tells "
             "us whether the canary's reading is contaminated by other "
             "layers' tier states.",
    )
    ap.add_argument("--multi-precision", action="store_true", default=True)
    ap.add_argument(
        "--initial-active", type=int, default=4, choices=[2, 4, 8],
        help="Active bit-width to start warmup at. Buffer is always "
             "INT8 (cache_dtype=int8_kivi); active is set via "
             "swap_tier_vector_to_model after init.",
    )
    ap.add_argument(
        "--consolidate-rule", default="vote-low",
        choices=["max", "min", "median-low", "median-high",
                 "mean-round", "vote-low", "vote-high"],
        help="How to combine per-rank tier vectors into one. Default "
             "``vote-low`` — per-layer majority vote, ties break "
             "low — preserves the conservative spirit of upstream's "
             "``max`` rule (a layer upgrades to INT8 only if a "
             "majority of ranks decide it) while surviving the rank-"
             "decorrelation from the fork's real-INT4 cache. Pass "
             "``--consolidate-rule max`` for upstream-compat.",
    )
    ap.add_argument(
        "--force-pareto-tiers", action="store_true",
        help="Diagnostic: ignore the controller's consolidated tiers "
             "and inject a fixed heterogeneous tier vector (alternating "
             "INT4/INT8) for phase 3. Useful for verifying the "
             "swap-and-measure half of the loop on workloads where the "
             "controller doesn't differentiate (e.g. 8B+GSM8K-Hard at "
             "default τ); the paper's actual Pareto sweep uses the "
             "real consolidated vector.",
    )
    ap.add_argument(
        "--out", type=Path,
        default=Path("results/serving/phaseG_3phase_fork.json"),
    )
    a = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    from vllm import LLM
    from evaluation.serving.canary_tp import TPCanaryHandle
    from evaluation.serving.fork_tier import (
        build_uniform_tier_vector,
        read_buffer_bits_from_model,
        read_tier_vector_from_model,
        swap_tier_vector_to_model,
    )

    log.info(
        "loading %s (tp=%d, kv_cache_dtype=int8_kivi for max headroom)",
        a.model, a.tp,
    )
    t0 = time.perf_counter()
    llm = LLM(
        model=a.model,
        tensor_parallel_size=a.tp,
        dtype="bfloat16",
        kv_cache_dtype="int8_kivi",   # buffer = max bits
        block_size=16,
        enforce_eager=True,
        gpu_memory_utilization=a.gpu_memory_utilization,
        max_model_len=a.max_model_len,
        enable_prefix_caching=False,
        max_num_seqs=a.max_num_seqs,
    )
    load_s = time.perf_counter() - t0
    log.info("loaded in %.1fs", load_s)

    # Read buffer_bits to confirm full INT8 headroom; log num_layers.
    rank_buf_bits = llm.apply_model(read_buffer_bits_from_model)
    num_layers = len(rank_buf_bits[0])
    if any(b != 8 for b in rank_buf_bits[0]):
        raise RuntimeError(
            f"expected uniform INT8 buffers, got {set(rank_buf_bits[0])}"
        )
    log.info("buffer_bits: uniform INT8 across %d layers", num_layers)

    # Build prompts.
    raw_warmup = make_workload_requests(
        a.workload, n=a.warmup_n, seed=a.seed + 100,
    )
    raw_measure = make_workload_requests(
        a.workload, n=a.n_prompts, seed=a.seed,
    )
    warmup_prompts = materialize_chat_prompts(llm, raw_warmup)
    measure_prompts = materialize_chat_prompts(llm, raw_measure)

    # Set the warmup-phase active bits via runtime swap (buffer stays
    # INT8). The canary's multi-precision mode then measures δ̂ at the
    # active tier's neighbours.
    initial_vec = build_uniform_tier_vector(num_layers, a.initial_active)
    log.info("setting warmup active bits = INT%d (buffer remains INT8)",
             a.initial_active)
    llm.apply_model(
        functools.partial(swap_tier_vector_to_model,
                          bits_per_layer=initial_vec),
    )

    # ---- Phase 1: warmup with canary observing ----
    log.info(
        "=== phase 1: warmup (n=%d, active=INT%d, observability_rate=%.2f) ===",
        a.warmup_n, a.initial_active, a.observability_rate,
    )
    warmup_handle = TPCanaryHandle(
        llm,
        harness_kwargs={
            "canary_rate": a.observability_rate,
            "tau": a.tau,
            "tick_interval": a.tick_interval,
            # Observation-only on the fork — cache is already real
            # KIVI; tier_application would shadow-double-quantize.
            "tier_application": False,
            "freeze_controller": a.freeze_warmup_controller,
            "multi_precision": a.multi_precision,
            "tau_budget": a.tau_budget,
            "decode_only_canary": True,
            "use_canary_stream": True,
            "seed": a.seed,
            # Per-rank seed offset: collapse one source of warmup
            # nondeterminism. Each rank seeds its canary RNG with
            # base_seed + rank_id; runs at the same a.seed are
            # reproducible.
            "per_rank_seed": a.per_rank_seed,
            "upgrade_only": a.warmup_only_upgrades,
            "grace_ticks": a.warmup_grace_ticks,
        },
    )
    warmup_handle.install()
    try:
        warmup_result = run_one(
            llm, warmup_prompts,
            label=f"warmup_{a.workload}",
            max_new_tokens=a.max_new_tokens,
        )
        warmup_snapshots = warmup_handle.snapshot()
    finally:
        warmup_handle.remove()

    log.info("warmup acc=%.3f wall=%.1fs",
             warmup_result["accuracy"], warmup_result["wall_clock_s"])

    # ---- Phase 2: consolidate ----
    consolidated = consolidate_tiers(warmup_snapshots, rule=a.consolidate_rule)
    log.info("consolidate rule: %r", a.consolidate_rule)
    consolidated_delta = consolidate_deltas(warmup_snapshots)
    proj = memory_projection(consolidated)
    log.info("consolidated tiers (counts=%s, avg_bits=%.2f)",
             proj["tier_counts"], proj["avg_bits"])
    log.info("savings vs FP16: %.1f%%; savings vs static-INT4: %.1f%%",
             proj["savings_vs_fp16_pct"], proj["savings_vs_kivi4_pct"])

    if a.force_pareto_tiers:
        # Diagnostic override: alternate INT4/INT8 layer-by-layer.
        # Lets us verify the swap-and-measure plumbing on workloads
        # where the controller doesn't naturally differentiate.
        forced = [(8 if i % 2 == 0 else 4) for i in range(num_layers)]
        log.info("forcing diagnostic alternating-4-8 tier vector "
                 "(real consolidated was %s)", proj["tier_counts"])
        consolidated = forced
        proj = memory_projection(consolidated)

    is_uniform_int4 = all(t == 4 for t in consolidated)
    if is_uniform_int4:
        log.info("controller converged to uniform INT4 — no Pareto win "
                 "to test on this workload at these settings. Stopping "
                 "after warmup.")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({
            "config": vars(a) | {"out": str(a.out), "model": a.model},
            "phase": "warmup_only",
            "warmup_result": warmup_result,
            "warmup_snapshots": warmup_snapshots,
            "consolidated_tiers": consolidated,
            "consolidated_delta": consolidated_delta,
            "memory_projection": proj,
        }, indent=2))
        log.info("wrote %s", a.out)
        return 0

    # ---- Phase 3: measurement (swap to consolidated, run full batch) ----
    log.info("=== phase 3: measurement (n=%d, swap_tier_vector_to_model) ===",
             a.n_prompts)
    llm.apply_model(
        functools.partial(swap_tier_vector_to_model,
                          bits_per_layer=consolidated),
    )

    # Verify the swap took on every rank.
    post_swap = llm.apply_model(read_tier_vector_from_model)
    for ri, vec in enumerate(post_swap):
        if vec != consolidated:
            raise RuntimeError(
                f"rank {ri} did not pick up consolidated tiers; "
                f"expected {consolidated}, got {vec}"
            )
    log.info("    consolidated swap verified on all %d ranks", len(post_swap))

    # Re-install harness in frozen-tier mode (for the per-prompt
    # canary record only; controller is frozen so tiers don't drift).
    measure_handle = TPCanaryHandle(
        llm,
        harness_kwargs={
            "canary_rate": 1e-6,           # essentially off during measurement
            "tau": a.tau,
            "tick_interval": a.tick_interval,
            "tier_application": False,
            "freeze_controller": True,
            "multi_precision": a.multi_precision,
            "tau_budget": a.tau_budget,
            "seed": a.seed + 1,
            "per_rank_seed": a.per_rank_seed,
        },
    )
    measure_handle.install()
    try:
        measure_result = run_one(
            llm, measure_prompts,
            label=f"measure_{a.workload}_echokv",
            max_new_tokens=a.max_new_tokens,
        )
        measure_snapshots = measure_handle.snapshot()
    finally:
        measure_handle.remove()

    log.info("measurement acc=%.3f wall=%.1fs",
             measure_result["accuracy"], measure_result["wall_clock_s"])

    # ---- Persist combined record ----
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "config": vars(a) | {"out": str(a.out), "model": a.model},
        "phase": "full_3phase",
        "warmup_result": warmup_result,
        "warmup_snapshots": warmup_snapshots,
        "consolidated_tiers": consolidated,
        "consolidated_delta": consolidated_delta,
        "memory_projection": proj,
        "measure_result": measure_result,
        "measure_snapshots": measure_snapshots,
    }, indent=2))
    log.info("wrote %s", a.out)

    print()
    print("=" * 72)
    print(f"Lane 1(b) 3-phase fork: workload={a.workload!r} "
          f"n={a.n_prompts} (warmup_n={a.warmup_n})")
    print("=" * 72)
    print(f"  warmup       acc={warmup_result['accuracy']:.3f} "
          f"wall={warmup_result['wall_clock_s']:.1f}s")
    print(f"  consolidated tiers: {proj['tier_counts']} "
          f"(avg_bits={proj['avg_bits']:.2f}, "
          f"savings vs FP16: {proj['savings_vs_fp16_pct']:.1f}%, "
          f"vs static-INT4: {proj['savings_vs_kivi4_pct']:.1f}%)")
    print(f"  measurement  acc={measure_result['accuracy']:.3f} "
          f"wall={measure_result['wall_clock_s']:.1f}s")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
