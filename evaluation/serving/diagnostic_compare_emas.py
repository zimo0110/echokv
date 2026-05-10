"""Phase G(e) addendum 8 diagnostic: contamination check.

Compares per-layer per-tier δ̂ EMAs across three warmup configurations:

  * Run A: uniform INT4, controller frozen. All upstream stays at INT4.
  * Run B: initial INT8, controller active. Mixed final tier vector.
  * Run C: uniform INT8, controller frozen. All upstream stays at INT8.

Hypothesis: contamination — at layer ℓ, the canary's δ̂_b reading
depends on what tier other layers are at. If the canary were
contamination-free, δ̂_b at layer ℓ would be a function of layer ℓ's
weights and the workload only.

Diagnostic: compare per-layer δ̂_b (at each measurement tier b ∈
{2,4,8}) across the three runs. Large differences (>10×) at the same
layer are evidence of contamination.

We can't perfectly isolate "upstream tier" from "active tier" with
the existing infrastructure (the canary measures at whatever active
tier the layer is currently at), but the three frozen-tier runs give
us controlled cells:

  * Run A vs Run C: same workload, different uniform tier. Different
    upstream AND different active. Both confounded.
  * Within Run B: layers that drifted INT8→INT4 vs layers that
    stayed at INT8. Within-run, different upstream contexts with
    correlated active tiers.

Cleanest contamination evidence: at a fixed measurement tier b,
compare δ̂_b distributions across runs. If δ̂_4 distribution differs
strongly between Run A (all-INT4 cache) and Run C (all-INT8 cache),
the canary's measurement is being influenced by upstream cache state
even though both runs measure the same hypothetical "what-if-INT4"
counterfactual.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def load_run(path: str) -> dict:
    d = json.load(open(path))
    snap = d['warmup_snapshots'][0]
    return {
        'path': path,
        'tier_vector': snap['tier_vector'],
        'per_tier_delta': {int(k): v for k, v in (snap.get('per_tier_delta') or {}).items()},
        'memory_projection': d.get('memory_projection', {}),
    }


def per_tier_stats(deltas: list[float]) -> dict:
    nz = sorted(v for v in deltas if v > 0)
    if not nz:
        return None
    n = len(nz)
    return {
        'n': n, 'min': nz[0], 'p25': nz[n // 4], 'p50': nz[n // 2],
        'p75': nz[3 * n // 4], 'max': nz[-1],
        'mean': statistics.fmean(nz),
    }


def main() -> int:
    runs = {
        'A_int4_frozen': '/home/nazar/echokv/results/serving/phaseG_diag_runA_int4_frozen.json',
        'B_int8_active': '/home/nazar/echokv/results/serving/phaseG_pareto_fork_70b_init8_tau3e4.json',
        'C_int8_frozen': '/home/nazar/echokv/results/serving/phaseG_diag_runC_int8_frozen.json',
    }
    data = {}
    for name, path in runs.items():
        if not Path(path).is_file():
            print(f"MISSING: {path}")
            return 1
        data[name] = load_run(path)

    print("=" * 78)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 78)
    for name, run in data.items():
        from collections import Counter
        tier_counts = Counter(run['tier_vector'])
        c = {b: tier_counts.get(b, 0) for b in (2, 4, 8)}
        print(f"  {name}: final tier vector = {c}, avg_bits = "
              f"{run['memory_projection'].get('avg_bits'):.2f}")
    print()

    print("Per-tier δ̂ summary (each run, all layers):")
    print(f"  {'run':<16} {'tier':>4} {'p25':>11} {'p50':>11} {'p75':>11}")
    print("-" * 78)
    for name, run in data.items():
        for tier in (2, 4, 8):
            stats = per_tier_stats(run['per_tier_delta'].get(tier, []))
            if stats:
                print(f"  {name:<16} {tier:>4} "
                      f"{stats['p25']:>11.3e} {stats['p50']:>11.3e} {stats['p75']:>11.3e}")
    print()

    # Pairwise ratio: at each measurement tier, compute median ratio
    # of δ̂ across runs. Large ratios = strong upstream-context dependence.
    print("Median-of-per-layer-ratios (pairwise, at each measurement tier):")
    print(f"  {'pair':<26} {'tier':>4} {'p50 ratio':>12} {'verdict':>20}")
    print("-" * 78)
    pairs = [('A_int4_frozen', 'C_int8_frozen'),
             ('A_int4_frozen', 'B_int8_active'),
             ('C_int8_frozen', 'B_int8_active')]
    for a, b in pairs:
        ra = data[a]['per_tier_delta']
        rb = data[b]['per_tier_delta']
        for tier in (2, 4, 8):
            la = ra.get(tier, [])
            lb = rb.get(tier, [])
            if not la or not lb:
                continue
            n = min(len(la), len(lb))
            ratios = []
            for li in range(n):
                if la[li] > 0 and lb[li] > 0:
                    ratios.append(la[li] / lb[li])
            if not ratios:
                continue
            ratios.sort()
            mratio = ratios[len(ratios) // 2]
            inv = 1.0 / mratio if mratio > 0 else float('inf')
            extreme = max(mratio, inv)
            if extreme >= 10:
                verdict = "STRONG"
            elif extreme >= 3:
                verdict = "AMBIGUOUS"
            else:
                verdict = "WEAK"
            print(f"  {a:>12}/{b:<13} {tier:>4} "
                  f"{mratio:>12.2f} {verdict:>20}  (n={len(ratios)})")
    print()
    print("Interpretation:")
    print("  STRONG (≥10×): contamination confirmed. Activation buffer fix justified.")
    print("  AMBIGUOUS (3-10×): contamination present but not dominant.")
    print("  WEAK (<3×): contamination minor; investigate other causes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
