#!/usr/bin/env bash
# Empirical test of the τ(L) = τ₀ · √(L/L₀) scaling derived from
# Pinsker's inequality + attention-sparsity assumption.
#
# Hypothesis: at RULER@128K (L=128000), the right τ is
#   τ_eff = τ_base · √(128000/4000) = 0.022 · 5.66 ≈ 0.124
# where τ_base = 0.022 was calibrated on 8B+RULER@4K (L=4000).
#
# Expected outcome:
#   - tier vector mostly INT4 (no INT8 upgrades — δ_4 max=0.051 < 0.124)
#   - acc = 1.000 (matches FP16/INT4 baselines, both also 1.000)
#   - confirms √L scaling holds, gives the paper §4.X material
#
# If acc<1.000 with this τ → scaling factor is too loose (some layers
# really do need INT8 even at long context); refine.
# If tier vector still has many INT8 layers → scaling factor is too
# tight (need a different exponent than 0.5).
#
# Run as: bash scripts/06_ruler128k_echokv_tau_sqrtL.sh
# Or in tmux:
#   tmux new -s ruler-tau 'bash scripts/06_ruler128k_echokv_tau_sqrtL.sh'
#
# Expected wall: ~10 min (warmup ~3 min + measurement n=20 mns=1 ~7 min).

set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

STAMP=$(date -u +%Y%m%d_%H%M%S)
LOG="/tmp/ruler128k_echokv_sqrtL_${STAMP}.log"
OUT="results/serving/session14_70b/echokv3c_abscanary_tau124_ruler128k_n20_mns1.json"

# Pinsker-derived τ for L=128000: τ = 0.022 · √(128000/4000) ≈ 0.124
TAU=0.124

echo "================================================================"
echo "EchoKV-3C abs-canary @ RULER@128K with √L-scaled τ"
echo "  τ_base (4K) = 0.022"
echo "  L          = 128000  (RULER context)"
echo "  L_base     = 4000    (8B smoke calibration context)"
echo "  τ_eff      = ${TAU}  (= 0.022 × √(128000/4000) ≈ 0.022 × 5.66)"
echo "  CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "  log: ${LOG}"
echo "  out: ${OUT}"
echo "  start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

/opt/conda/envs/scalable/bin/python \
    -m evaluation.serving.run_session14_70b_ruler_echokv \
    --n-prompts 20 \
    --warmup-n 8 \
    --max-num-seqs 1 \
    --max-new-tokens 32 \
    --context 128000 \
    --max-model-len 131072 \
    --multi-precision \
    --absolute-canary \
    --kref-window-size 64 \
    --tau-budget "${TAU}" \
    --seed 0 \
    --out "${OUT}" 2>&1 | tee "${LOG}"

echo
echo "================================================================"
echo "Done at $(date -u +%Y-%m-%dT%H:%M:%SZ)."
echo
echo "Hypothesis check: tier vector should be mostly INT4 (no INT8)"
echo "                  and acc=1.000 (matches static baselines)"
/opt/conda/envs/scalable/bin/python << EOF
import json
import math
import statistics
from pathlib import Path

p = Path("${OUT}")
b = json.load(open(p))
ech_m = b["measurement"]
proj = b["memory_projection"]
tiers = b["consolidated_tiers"]

# Phase-2 reference baselines (RULER@128K n=20 mns=1)
phase2 = json.load(open("results/serving/session14_70b/diag.json"))
fp16_cell = next(c for c in phase2["cells"] if c["config"] == "fp16")
int4_cell = next(c for c in phase2["cells"] if c["config"] == "static_int4")
fp16 = {r["request_id"]: r["correct"] for r in fp16_cell["per_request"]}
int4 = {r["request_id"]: r["correct"] for r in int4_cell["per_request"]}
ech = {r["request_id"]: r["correct"] for r in ech_m["per_request"]}

print(f"  FP16  (Phase-2):      acc={fp16_cell['accuracy']:.3f}  bits=16.00")
print(f"  static_int4 (Phase-2): acc={int4_cell['accuracy']:.3f}  bits=4.00")
print(f"  EchoKV-3C @ τ=${TAU}: acc={ech_m['accuracy']:.3f}  bits={proj['avg_bits']:.2f}  "
      f"savings_vs_fp16={proj['savings_vs_fp16_pct']:.1f}%  "
      f"tier_counts={proj['tier_counts']}")

# Compare tier vector to the prior τ=0.022 run if it exists
prior = Path("results/serving/session14_70b/echokv3c_abscanary_tau022_ruler128k_n20_mns1.json")
if prior.exists():
    pb = json.load(open(prior))
    pp = pb["memory_projection"]
    print()
    print(f"  (prior τ=0.022 run: bits={pp['avg_bits']:.2f}  "
          f"tiers={pp['tier_counts']})")

# Sanity: count INT8 layers
n_int8 = tiers.count(8)
n_int4 = tiers.count(4)
n_int2 = tiers.count(2)
print()
print(f"  Tier breakdown: {n_int4} INT4 / {n_int8} INT8 / {n_int2} INT2")
if n_int8 == 0:
    print("  ✓ HYPOTHESIS CONFIRMED: zero INT8 upgrades (workload-adaptive √L scaling works)")
elif n_int8 < 5:
    print(f"  ✓ HYPOTHESIS LARGELY CONFIRMED: only {n_int8} INT8 upgrades (close to expected)")
elif n_int8 < 30:
    print(f"  ~ PARTIAL: {n_int8} INT8 upgrades (less than τ=0.022's 71, but not 'easy workload')")
else:
    print(f"  ✗ HYPOTHESIS REFUTED: still {n_int8} INT8 upgrades — √L scaling is too tight")
EOF
echo "================================================================"
