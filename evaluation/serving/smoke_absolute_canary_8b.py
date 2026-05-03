"""Smoke test for the absolute-canary sensor on 8B + TP=2.

Verifies four claims before committing to a 70B re-run with the new
sensor (per the critic's discipline in ARCHITECTURE § 9.9):

1. **Magnitude.** With ``absolute_canary=True``, per-layer
   δ_absolute(b) is non-trivial (> some small floor) — the canary is
   producing real signal.
2. **max_num_seqs invariance.** Running the same prompts at
   ``max_num_seqs=1`` and ``max_num_seqs=8`` produces comparable
   per-layer δ distributions. The legacy canary collapses to 0 at
   ``max_num_seqs=1`` (T=1 in the per-step attention); the absolute
   canary should not.
3. **Cross-tier monotonicity.** δ(2) ≥ δ(4) ≥ δ(8) on every layer.
   This holds by construction (more aggressive quantization → more
   divergence) but verifying empirically catches buffer / quantization
   bugs.
4. **Cross-seed stability.** Two runs with different seeds produce
   similar per-layer δ rankings — the signal reflects layer-intrinsic
   sensitivity, not transient cache dynamics.

Run with::

    VLLM_ALLOW_INSECURE_SERIALIZATION=1 \\
    FLASHINFER_DISABLE_VERSION_CHECK=1 \\
    CUDA_VISIBLE_DEVICES=4,5 \\
    /opt/conda/envs/scalable/bin/python \\
        -m evaluation.serving.smoke_absolute_canary_8b
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.environ["PYTHONPATH"] = (
    f"{_REPO_ROOT}:" + os.environ.get("PYTHONPATH", "")
)

from evaluation.motivation.workloads import _ruler_sample
from evaluation.serving.canary_tp import TPCanaryHandle
from evaluation.serving.run_session8 import DEFAULT_MODEL


log = logging.getLogger("smoke_absolute_canary_8b")


def _avg_per_layer(per_rank_snapshots: list[dict], tier_key) -> list[float]:
    """Mean across ranks of per-tier per-layer EMA (skipping zeros).

    The per_tier_delta dict's keys may be Precision enum (in-memory),
    int (after some round-trips), or string (after JSON). Accept any
    of the three by trying int / str conversions.
    """
    if not per_rank_snapshots or per_rank_snapshots[0].get("per_tier_delta") is None:
        return []
    target_int = int(tier_key)
    vecs: list[list[float]] = []
    for s in per_rank_snapshots:
        d = s["per_tier_delta"]
        # Try int, str, then exact key match.
        v = d.get(target_int) or d.get(str(target_int))
        if v is None:
            for k, val in d.items():
                if int(k) == target_int:
                    v = val
                    break
        vecs.append(v or [])
    if not vecs or not vecs[0]:
        return []
    n = len(vecs[0])
    out: list[float] = []
    for li in range(n):
        vals = [v[li] for v in vecs if v[li] > 0.0]
        out.append(sum(vals) / len(vals) if vals else 0.0)
    return out


def _run_once(
    llm: Any,
    *,
    label: str,
    seed: int,
    max_num_seqs: int,
    n_prompts: int = 10,
) -> dict:
    """Install absolute-canary harness, run n prompts, snapshot."""
    from vllm import SamplingParams

    handle = TPCanaryHandle(
        llm,
        harness_kwargs={
            "canary_rate": 0.5,
            "tau": 0.01,
            "tick_interval": 10,
            "tier_application": False,
            "freeze_controller": True,    # don't change tiers; we only
                                          # care about the sensor's signal
            "multi_precision": True,
            "tau_budget": 0.05,
            "absolute_canary": True,
            "kref_window_size": 64,
            "seed": seed,
        },
    )
    counts = handle.install()
    log.info("[%s] hooks per rank: %s", label, counts)

    prompts = [
        _ruler_sample(i, seed=seed, target_tokens=2048).prompt
        for i in range(n_prompts)
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=16)
    t0 = time.perf_counter()
    llm.generate(prompts, sp, use_tqdm=False)
    wall = time.perf_counter() - t0

    snaps = handle.snapshot()
    handle.remove()

    per_layer_d2 = _avg_per_layer(snaps, "2")
    per_layer_d4 = _avg_per_layer(snaps, "4")
    per_layer_d8 = _avg_per_layer(snaps, "8")
    fires = [s.get("stats", {}).get("canary_fires", 0) for s in snaps]
    return {
        "label": label,
        "seed": seed,
        "max_num_seqs": max_num_seqs,
        "wall_s": wall,
        "fires_per_rank": fires,
        "delta_2_per_layer": per_layer_d2,
        "delta_4_per_layer": per_layer_d4,
        "delta_8_per_layer": per_layer_d8,
    }


def _summarize(label: str, run: dict) -> None:
    nz4 = [v for v in run["delta_4_per_layer"] if v > 0.0]
    nz2 = [v for v in run["delta_2_per_layer"] if v > 0.0]
    nz8 = [v for v in run["delta_8_per_layer"] if v > 0.0]
    if nz4:
        log.info(
            "  [%s] δ_2 [n=%d max=%.5f mean=%.5f]  "
            "δ_4 [n=%d max=%.5f mean=%.5f]  "
            "δ_8 [n=%d max=%.5f mean=%.5f]",
            label,
            len(nz2), max(nz2) if nz2 else 0, (sum(nz2)/len(nz2)) if nz2 else 0,
            len(nz4), max(nz4) if nz4 else 0, (sum(nz4)/len(nz4)) if nz4 else 0,
            len(nz8), max(nz8) if nz8 else 0, (sum(nz8)/len(nz8)) if nz8 else 0,
        )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    from vllm import LLM

    runs: list[dict] = []

    # Run two LLM instances with different max_num_seqs to test invariance.
    # Sharing one LLM across configs requires re-loading; cheaper to do
    # both in sequence with different LLM objects.
    for max_num_seqs in (1, 8):
        log.info("=== loading 8B TP=2 max_num_seqs=%d ===", max_num_seqs)
        t0 = time.perf_counter()
        llm = LLM(
            model=DEFAULT_MODEL,
            tensor_parallel_size=2,
            dtype="bfloat16",
            enforce_eager=True,
            gpu_memory_utilization=0.85,
            max_model_len=4_096,
            enable_prefix_caching=False,
            max_num_seqs=max_num_seqs,
        )
        log.info("loaded in %.1fs", time.perf_counter() - t0)

        for seed in (0, 7):
            label = f"mns={max_num_seqs}_seed={seed}"
            run = _run_once(
                llm, label=label, seed=seed,
                max_num_seqs=max_num_seqs, n_prompts=10,
            )
            runs.append(run)
            _summarize(label, run)

        # vLLM doesn't have a clean shutdown API in this version; rely
        # on Python GC. The next iteration constructs a fresh LLM.
        del llm
        import gc
        gc.collect()

    # ------------------------------------------------------------------
    # Acceptance checks
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("Absolute-canary smoke test on 8B + TP=2")
    print("=" * 72)

    # Group runs for the cross-seed and cross-mns checks.
    by_mns: dict[int, list[dict]] = {}
    for r in runs:
        by_mns.setdefault(r["max_num_seqs"], []).append(r)

    # Check 1 — magnitude. Every run should have non-trivial δ_4.
    print("Check 1 — magnitude (any δ_4 above 1e-6 floor):")
    pass_1 = True
    for r in runs:
        max_d4 = max((v for v in r["delta_4_per_layer"] if v > 0), default=0.0)
        ok = max_d4 > 1e-6
        pass_1 = pass_1 and ok
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {r['label']:<22}  max δ_4 = {max_d4:.6f}")

    # Check 2 — max_num_seqs invariance.
    # Compare seed=0 runs across mns=1 vs mns=8.
    s0_mns1 = next((r for r in runs if r["max_num_seqs"] == 1 and r["seed"] == 0), None)
    s0_mns8 = next((r for r in runs if r["max_num_seqs"] == 8 and r["seed"] == 0), None)
    print("Check 2 — max_num_seqs invariance (seed=0, mns=1 vs mns=8):")
    pass_2 = False
    if s0_mns1 and s0_mns8:
        max1 = max((v for v in s0_mns1["delta_4_per_layer"] if v > 0), default=0.0)
        max8 = max((v for v in s0_mns8["delta_4_per_layer"] if v > 0), default=0.0)
        ratio = max(max1, max8) / max(min(max1, max8), 1e-9)
        ok = ratio < 5.0  # signals within 5× of each other
        pass_2 = ok
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] mns=1 max δ_4 = {max1:.6f}, "
              f"mns=8 max δ_4 = {max8:.6f}, ratio = {ratio:.2f}")

    # Check 3 — cross-tier monotonicity (every layer, every run).
    print("Check 3 — cross-tier monotonicity δ_2 ≥ δ_4 ≥ δ_8:")
    pass_3 = True
    for r in runs:
        n_layers = len(r["delta_4_per_layer"])
        viol = 0
        for li in range(n_layers):
            d2 = r["delta_2_per_layer"][li]
            d4 = r["delta_4_per_layer"][li]
            d8 = r["delta_8_per_layer"][li]
            # Allow tiny FP slack
            if d2 + 1e-7 < d4 or d4 + 1e-7 < d8:
                viol += 1
        ok = viol == 0
        pass_3 = pass_3 and ok
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {r['label']:<22}  violations = {viol} / {n_layers}")

    # Check 4 — cross-seed stability (mns fixed, two seeds give similar rankings).
    print("Check 4 — cross-seed Spearman rank correlation (mns=8, seeds 0 vs 7):")
    pass_4 = False
    s7_mns8 = next((r for r in runs if r["max_num_seqs"] == 8 and r["seed"] == 7), None)
    if s0_mns8 and s7_mns8:
        a = s0_mns8["delta_4_per_layer"]
        b = s7_mns8["delta_4_per_layer"]
        n = len(a)
        # Spearman: rank then Pearson on ranks
        def _rank(xs):
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            ranks = [0.0] * len(xs)
            for r, i in enumerate(order):
                ranks[i] = float(r)
            return ranks
        ra, rb = _rank(a), _rank(b)
        mean_a = sum(ra) / n
        mean_b = sum(rb) / n
        num = sum((ra[i] - mean_a) * (rb[i] - mean_b) for i in range(n))
        den_a = (sum((x - mean_a) ** 2 for x in ra)) ** 0.5
        den_b = (sum((x - mean_b) ** 2 for x in rb)) ** 0.5
        rho = num / (den_a * den_b) if den_a * den_b > 0 else 0.0
        ok = rho > 0.5
        pass_4 = ok
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] Spearman ρ = {rho:.3f}  (threshold 0.50)")

    out_path = Path("results/serving/smoke_absolute_canary_8b.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "runs": runs,
        "checks": {
            "magnitude": pass_1,
            "mns_invariance": pass_2,
            "monotonicity": pass_3,
            "cross_seed_stability": pass_4,
        },
    }, indent=2))
    log.info("wrote %s", out_path)

    print()
    overall = pass_1 and pass_2 and pass_3 and pass_4
    print(f"OVERALL: {'PASS — proceed to 70B re-run' if overall else 'FAIL — diagnose before 70B'}")
    print("=" * 72)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
