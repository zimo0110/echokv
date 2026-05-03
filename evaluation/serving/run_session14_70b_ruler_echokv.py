"""Session 14 / Phase 3 — EchoKV-3C closed-loop on 70B RULER@128K.

The Pareto test. Phase 2 found a workload (70B + RULER@128K at large
n) where static INT4 measurably loses ~2.2 accuracy points vs FP16
on hard reasoning. This driver runs EchoKV-3C — closed-loop per-layer
quantization driven by the canary signal — to test whether the
controller can recover some of that loss at lower memory cost than
uniform INT8.

Two-phase protocol
------------------
1. **Warmup** (small batch, default 8 prompts): canary observes per-
   layer δ̂ at high observability rate, controller updates per-layer
   tiers. ``tier_application=True`` so the controller's K/V quant
   decisions actually take effect during warmup.
2. **Consolidate**: each TP worker's controller has been making
   independent decisions on its slice of attention heads (TP shards
   heads, not layers, so each rank's per-layer δ̂ is computed on a
   different head subset). Consolidate to one tier vector:
   per-layer max across ranks (the most conservative — the rank
   that wants more precision wins). This avoids a "per-rank-per-
   head precision mosaic" during measurement.
3. **Measurement** (full batch, default 100 prompts): same harness
   but ``freeze_controller=True`` and the consolidated tier vector
   seeded on every rank. Every prompt sees the same tier vector.

Decision tree
-------------
- If the consolidated tier vector is **uniform INT4**, EchoKV-3C
  has converged to the static-INT4 baseline; stop and write the
  result. Per-tier τ_budget refinement might help but is the next
  experiment, not this one.
- If **non-uniform**, run the full measurement and report acc + the
  per-layer tier distribution + memory projection.

Usage
-----
::

    VLLM_ALLOW_INSECURE_SERIALIZATION=1 \\
    FLASHINFER_DISABLE_VERSION_CHECK=1 \\
    CUDA_VISIBLE_DEVICES=4,5,6,7 \\
    /opt/conda/envs/scalable/bin/python \\
        -m evaluation.serving.run_session14_70b_ruler_echokv \\
        --n-prompts 100
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.environ["PYTHONPATH"] = (
    f"{_REPO_ROOT}:" + os.environ.get("PYTHONPATH", "")
)

from evaluation.serving.run_session8 import (
    RunResult, make_ruler_prompts, run_one, warmup,
)
from evaluation.serving.canary_tp import TPCanaryHandle


log = logging.getLogger("session14_70b_echokv")
DEFAULT_70B = "/mnt/nvme_tmp6/llama-70b"


def consolidate_tiers(per_rank_snapshots: list[dict]) -> list[int]:
    """Per-layer max across ranks. The rank that wants the highest
    precision on a given layer wins. Ties → that precision."""
    n = per_rank_snapshots[0]["num_layers"]
    out: list[int] = []
    for li in range(n):
        out.append(max(s["tier_vector"][li] for s in per_rank_snapshots))
    return out


def consolidate_deltas(per_rank_snapshots: list[dict]) -> list[float]:
    """Per-layer mean δ̂ across ranks (each rank measured on its own
    head subset; averaging gives a less noisy per-layer estimate
    than any single rank). Returns 0.0 for layers where no canary
    fired on any rank."""
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
    """Average bits per layer + memory savings vs FP16 (16 bits)."""
    n = len(tiers)
    avg_bits = sum(tiers) / n
    bits_per_count = Counter(tiers)
    return {
        "num_layers": n,
        "avg_bits": avg_bits,
        "savings_vs_fp16_pct": 100.0 * (1.0 - avg_bits / 16.0),
        "savings_vs_kivi4_pct": 100.0 * (1.0 - avg_bits / 4.0),
        "tier_counts": dict(bits_per_count),
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_70B)
    p.add_argument("--tensor-parallel-size", type=int, default=4)
    p.add_argument("--max-model-len", type=int, default=131_072)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-num-seqs", type=int, default=1)
    p.add_argument("--n-prompts", type=int, default=20)
    p.add_argument("--warmup-n", type=int, default=8)
    p.add_argument("--context", type=int, default=128_000,
                   help="approximate prompt length in tokens (RULER haystack)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--observability-rate", type=float, default=0.25,
        help="canary rate during warmup; production is 1/64 but for "
             "warmup we want the EMA to converge quickly",
    )
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--tick-interval", type=int, default=10)
    p.add_argument(
        "--multi-precision", action="store_true",
        help="use neighbours-only multi-precision canary (probes per-tier "
             "δ̂ directly). The scalar canary collapsed to uniform INT2 on "
             "70B RULER@128K at warmup_n=8 because per-step δ̂_4 was small; "
             "multi-precision tracks δ̂_2 separately so the controller "
             "doesn't downgrade past INT4 when INT2 is unsafe.",
    )
    p.add_argument(
        "--tau-budget", type=float, default=0.05,
        help="divergence budget for the multi-precision controller. From "
             "Session 13 calibration on 8B+RULER, τ_budget=0.05 lies "
             "cleanly between static-INT4 EMA mean (≈0.022, acc=1.0) and "
             "static-INT2 EMA mean (≈0.36, acc=0.05).",
    )
    p.add_argument(
        "--absolute-canary", action="store_true",
        help="use the sliding-window absolute-KL sensor (ARCHITECTURE.md § "
             "9.9). Decouples the canary measurement from cache state and "
             "from batch size T. Fixes the T=1 INT2 cascade observed in "
             "Phase-4 max_num_seqs=1 runs.",
    )
    p.add_argument(
        "--kref-window-size", type=int, default=64,
        help="K_ref sliding-window size when --absolute-canary is on.",
    )
    p.add_argument(
        "--out", type=Path,
        default=Path("results/serving/session14_70b/echokv_3c_n100.json"),
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

    # RULER prompts are raw strings (no chat template), so we can use
    # them directly without ``materialize_chat_prompts``.
    target_tokens = a.context
    warmup_prompts = make_ruler_prompts(
        n=4, seed=a.seed + 9999, target_tokens=target_tokens,
    )
    warmup_phase_prompts = make_ruler_prompts(
        n=a.warmup_n, seed=a.seed + 100, target_tokens=target_tokens,
    )
    measurement_prompts = make_ruler_prompts(
        n=a.n_prompts, seed=a.seed, target_tokens=target_tokens,
    )

    # Engine warmup (one-shot prompt to pay compile cost).
    warmup(llm, warmup_prompts, a.max_new_tokens)

    # Phase 1: warmup with canary observing + tier_application on.
    # The controller updates per-layer tiers as δ̂ accumulates.
    log.info("=== phase 1: warmup (n=%d, observability=%.2f) ===",
             a.warmup_n, a.observability_rate)
    warmup_handle = TPCanaryHandle(
        llm,
        harness_kwargs={
            "canary_rate": a.observability_rate,
            "tau": a.tau,
            "tick_interval": a.tick_interval,
            "tier_application": True,
            "tier_application_quant_v": True,
            "freeze_controller": False,
            "multi_precision": a.multi_precision,
            "tau_budget": a.tau_budget,
            "absolute_canary": a.absolute_canary,
            "kref_window_size": a.kref_window_size,
            "seed": a.seed,
        },
    )
    warmup_handle.install()
    try:
        warmup_result = run_one(
            llm, warmup_phase_prompts,
            label="echokv_warmup_70b_ruler",
            max_new_tokens=a.max_new_tokens,
            harness=None,
        )
        warmup_snapshots = warmup_handle.snapshot()
    finally:
        warmup_handle.remove()

    # Phase 2: consolidate
    consolidated = consolidate_tiers(warmup_snapshots)
    consolidated_delta = consolidate_deltas(warmup_snapshots)
    log.info("    consolidated tier vector: %s", consolidated)
    log.info("    consolidated δ̂ (mean across ranks): %s",
             [f"{d:.4f}" for d in consolidated_delta])

    is_uniform_int4 = all(t == 4 for t in consolidated)
    proj = memory_projection(consolidated)
    log.info("    memory projection: avg_bits=%.2f savings_vs_kivi4=%.1f%%",
             proj["avg_bits"], proj["savings_vs_kivi4_pct"])
    log.info("    tier counts: %s", proj["tier_counts"])

    if is_uniform_int4:
        log.info("STOP — controller converged to uniform INT4. "
                 "EchoKV-3C is operationally equivalent to static "
                 "KIVI-4 on this workload. No measurement run.")
        summary = {
            "session": "20260429_14_70b_echokv_3c",
            "model": a.model,
            "tensor_parallel_size": a.tensor_parallel_size,
            "n_prompts": a.n_prompts,
            "warmup_n": a.warmup_n,
            "max_num_seqs": a.max_num_seqs,
            "phase": "warmup_only",
            "warmup_tiers_per_rank": [
                s["tier_vector"] for s in warmup_snapshots
            ],
            "warmup_deltas_per_rank": [
                s["per_layer_delta"] for s in warmup_snapshots
            ],
            "consolidated_tiers": consolidated,
            "consolidated_delta": consolidated_delta,
            "is_uniform_int4": True,
            "memory_projection": proj,
        }
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(summary, indent=2))
        log.info("wrote %s", a.out)
        return 0

    # Phase 3: measurement with the consolidated tier vector frozen.
    log.info("=== phase 3: measurement (n=%d, frozen tiers) ===", a.n_prompts)
    measure_handle = TPCanaryHandle(
        llm,
        harness_kwargs={
            "canary_rate": 1e-6,                 # essentially never fires
            "tau": a.tau,
            "tick_interval": a.tick_interval,
            "tier_application": True,
            "tier_application_quant_v": True,
            "freeze_controller": True,
            "multi_precision": a.multi_precision,
            "tau_budget": a.tau_budget,
            "absolute_canary": a.absolute_canary,
            "kref_window_size": a.kref_window_size,
            "seed": a.seed + 1,
        },
    )
    measure_handle.install()
    try:
        measure_handle.seed_tiers(consolidated)
        # Sanity check: read back to confirm the seed took on every rank.
        seeded_snaps = measure_handle.snapshot()
        for ri, s in enumerate(seeded_snaps):
            if s["tier_vector"] != consolidated:
                raise RuntimeError(
                    f"rank {ri} did not pick up consolidated tiers. "
                    f"expected {consolidated}, got {s['tier_vector']}"
                )
        log.info("    seeded tiers verified on all %d ranks",
                 len(seeded_snaps))
        measure_result: RunResult = run_one(
            llm, measurement_prompts,
            label="echokv_3c_70b_ruler",
            max_new_tokens=a.max_new_tokens,
            harness=None,
        )
        measure_snapshots = measure_handle.snapshot()
    finally:
        measure_handle.remove()

    log.info("    echokv_3c: acc=%.3f wall=%.1fs",
             measure_result.accuracy, measure_result.wall_clock_s)

    summary = {
        "session": "20260429_14_70b_echokv_3c",
        "model": a.model,
        "tensor_parallel_size": a.tensor_parallel_size,
        "n_prompts": a.n_prompts,
        "warmup_n": a.warmup_n,
        "max_num_seqs": a.max_num_seqs,
        "phase": "measurement_complete",
        "warmup_tiers_per_rank": [
            s["tier_vector"] for s in warmup_snapshots
        ],
        "warmup_deltas_per_rank": [
            s["per_layer_delta"] for s in warmup_snapshots
        ],
        # Per-tier per-layer EMA at end of warmup — load-bearing for the
        # convergence diagnostic (does the production EMA match the
        # static calibration?). Only present when ``multi_precision=True``.
        "warmup_per_tier_delta_per_rank": [
            s.get("per_tier_delta") for s in warmup_snapshots
        ],
        "warmup_stats_per_rank": [
            s["stats"] for s in warmup_snapshots
        ],
        "consolidated_tiers": consolidated,
        "consolidated_delta": consolidated_delta,
        "memory_projection": proj,
        "is_uniform_int4": False,
        "measurement": {
            **asdict(measure_result),
            "config": "echokv_3c",
        },
        "measurement_snapshots": measure_snapshots,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=2))
    log.info("wrote %s", a.out)

    print()
    print("=" * 64)
    print(f"Session 14 / 70B EchoKV-3C ({a.n_prompts} prompts)")
    print("=" * 64)
    print(f"consolidated tiers (per-layer max across {a.tensor_parallel_size} ranks):")
    print(f"  {consolidated}")
    print(f"tier counts: {proj['tier_counts']}")
    print(f"avg bits:    {proj['avg_bits']:.2f}  "
          f"(savings vs FP16: {proj['savings_vs_fp16_pct']:.1f}%, "
          f"vs KIVI-4: {proj['savings_vs_kivi4_pct']:+.1f}%)")
    print(f"accuracy:    {measure_result.accuracy:.3f}")
    print("=" * 64)
    # Look up the matching reference cell (FP16 / static_int4 at the
    # same n) and run a paired McNemar test if the file exists.
    _print_paired_comparison(
        ech_scores={r["request_id"]: r["correct"]
                    for r in measure_result.per_request},
        n=a.n_prompts,
        max_num_seqs=a.max_num_seqs,
    )
    return 0


def _find_reference_file(n: int, max_num_seqs: int) -> Optional[Path]:
    """Best-effort lookup of FP16+INT4 reference cells at the same n
    and protocol (max_num_seqs). Returns ``None`` if no matching file
    is found — caller should print a soft note rather than fail."""
    base = Path("results/serving/session14_70b")
    candidates = [
        # n=200 mns=1 publication runs are split across two files
        (base / "publication" / "fp16_n200_seq1.json", n == 200 and max_num_seqs == 1),
        # The phase-2 batched n-sweep is one file with three cells
        (base / f"ruler_n{n}.json", max_num_seqs != 1),
        # Try the smaller-n batched files as fallback
        (base / "ruler.json", n == 20),
    ]
    for path, _hint in candidates:
        if path.exists():
            return path
    return None


def _print_paired_comparison(
    *, ech_scores: dict, n: int, max_num_seqs: int,
) -> None:
    """If a reference file exists at the matching (n, mns) protocol,
    print a paired McNemar comparison. Otherwise print a soft note."""
    ref_path = _find_reference_file(n, max_num_seqs)
    if ref_path is None:
        print(f"No paired reference found for n={n} mns={max_num_seqs}; "
              f"skipping comparison.")
        print("=" * 64)
        return

    print(f"Paired comparison vs reference cells in:")
    print(f"  {ref_path}")
    blob = json.loads(ref_path.read_text())

    def _scores(b: dict, cfg: str) -> Optional[dict]:
        # ruler driver: top-level "cells", each with config + per_request
        if "cells" in b:
            for c in b["cells"]:
                if c.get("config") == cfg:
                    return {r["request_id"]: r["correct"]
                            for r in c["per_request"]}
            return None
        # publication driver: single config per file
        if "measurement" in b:
            m = b["measurement"]
            return {r["request_id"]: r["correct"] for r in m["per_request"]}
        return None

    fp = _scores(blob, "fp16")
    i4 = _scores(blob, "static_int4")
    # If the n=200 mns=1 path was used, FP16 and INT4 are in separate
    # files. Try to find INT4 alongside.
    if fp is not None and i4 is None:
        sib = ref_path.parent / "int4_n200_seq1.json"
        if sib.exists():
            i4_blob = json.loads(sib.read_text())
            i4 = _scores(i4_blob, "static_int4") or _scores(i4_blob, "fp16")

    import math as _math

    def _mcnemar(a: dict, b: dict) -> tuple:
        common = sorted(set(a) & set(b))
        bw = sum(1 for k in common if b[k] < a[k])
        cw = sum(1 for k in common if b[k] > a[k])
        nd = bw + cw
        if nd > 1:
            kk = max(bw, cw)
            p_one = sum(_math.comb(nd, i) * 0.5 ** nd for i in range(kk, nd + 1))
            p = min(2 * p_one, 1.0)
        else:
            p = 1.0
        delta = (cw - bw) / len(common) if common else 0.0
        return bw, cw, p, delta, len(common)

    print(f"  {'comparison':<22} {'b/c':<10} {'McNemar p':<11} {'paired Δ':>9}")
    print("  " + "-" * 56)
    if fp:
        bw, cw, p, d, n_paired = _mcnemar(fp, ech_scores)
        print(f"  {'EchoKV-3C vs FP16':<22} {bw}/{cw:<8} {p:<11.4f} {d:+.4f}")
    if i4:
        bw, cw, p, d, n_paired = _mcnemar(i4, ech_scores)
        print(f"  {'EchoKV-3C vs INT4':<22} {bw}/{cw:<8} {p:<11.4f} {d:+.4f}")
    print("=" * 64)


if __name__ == "__main__":
    sys.exit(main())
