#!/usr/bin/env bash
# Continuously-dynamic measurement protocol on HotpotQA + RULER@128K,
# matching the GSM8K-Hard run from scripts/08. With this script, all
# three workloads have a paired (frozen-tier, continuous-dynamic) cell
# at the same τ-budget the headline used.
#
# Why: scripts/08 showed that on GSM8K-Hard the dynamic protocol
# downgrades 28 of 80 layers from INT8 to INT4 during measurement,
# ending at 4.75 bits (vs frozen-tier's 5.45 bits) at FP16-equivalent
# accuracy. We need to know whether that pattern generalises:
#
#   - HotpotQA: warmup picks {INT4:69, INT8:11} = 4.55 bits frozen.
#     If dynamic mode finds even more downgrades, we get sub-INT4
#     average bits. If it stays flat, the frozen vector is already
#     near-optimal.
#
#   - RULER@128K: warmup at τ=0.124 picks {INT4:79, INT2:1} = 3.98
#     bits frozen. The controller has very little upgrade budget
#     left; the question is whether dynamic mode admits more INT2
#     layers as the long-context EMAs settle.
#
# Run as:        bash scripts/09_hotpotqa_ruler_continuous_dynamic.sh
# Run in tmux:   tmux new -s cd-rest 'bash scripts/09_hotpotqa_ruler_continuous_dynamic.sh'
#
# Expected wall:
#   HotpotQA (n=500 mns=8):    ~2 min
#   RULER@128K (n=20 mns=1):   ~7 min
#   Total:                     ~10 min

set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

STAMP=$(date -u +%Y%m%d_%H%M%S)
LOG_DIR="/tmp/cont_dyn_${STAMP}"
mkdir -p "${LOG_DIR}"

PY="/opt/conda/envs/scalable/bin/python"

OUT_HOT="results/serving/session14_70b/hotpotqa_n500_mns8/echokv3c_abscanary_tau022_n500_mns8_continuous_dynamic.json"
OUT_RUL="results/serving/session14_70b/echokv3c_abscanary_tau124_ruler128k_n20_mns1_continuous_dynamic.json"
mkdir -p "$(dirname "${OUT_HOT}")" "$(dirname "${OUT_RUL}")"

echo "================================================================"
echo "Continuous-dynamic protocol on HotpotQA + RULER@128K"
echo "  HotpotQA   : τ=0.022, n=500 mns=8"
echo "  RULER@128K : τ=0.124 (√L-scaled), n=20 mns=1"
echo "  CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "  log dir = ${LOG_DIR}"
echo "  start   = $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

# ---------------------------- HotpotQA ----------------------------------
echo "----------------------------------------------------------------"
echo "[1/2] HotpotQA echokv-3c, τ=0.022, --continuous-dynamic"
echo "  out: ${OUT_HOT}"
echo "  start: $(date -u +%H:%M:%S)"
echo "----------------------------------------------------------------"
"${PY}" -m evaluation.serving.run_session14_70b_hotpotqa_echokv \
    --n-prompts 500 \
    --warmup-n 16 \
    --multi-precision \
    --absolute-canary \
    --kref-window-size 64 \
    --tau-budget 0.022 \
    --continuous-dynamic \
    --max-num-seqs 8 \
    --max-new-tokens 64 \
    --seed 0 \
    --out "${OUT_HOT}" 2>&1 | tee "${LOG_DIR}/hotpotqa.log"

# --------------------------- RULER@128K ---------------------------------
echo
echo "----------------------------------------------------------------"
echo "[2/2] RULER@128K echokv-3c, τ=0.124, --continuous-dynamic"
echo "  out: ${OUT_RUL}"
echo "  start: $(date -u +%H:%M:%S)"
echo "----------------------------------------------------------------"
"${PY}" -m evaluation.serving.run_session14_70b_ruler_echokv \
    --n-prompts 20 \
    --warmup-n 8 \
    --max-num-seqs 1 \
    --max-new-tokens 32 \
    --context 128000 \
    --max-model-len 131072 \
    --multi-precision \
    --absolute-canary \
    --kref-window-size 64 \
    --tau-budget 0.124 \
    --continuous-dynamic \
    --seed 0 \
    --out "${OUT_RUL}" 2>&1 | tee "${LOG_DIR}/ruler128k.log"

# ------------------------- comparison summary ---------------------------
echo
echo "================================================================"
echo "Done at $(date -u +%Y-%m-%dT%H:%M:%SZ)."
echo
echo "Frozen vs continuous-dynamic comparison (all three workloads):"
"${PY}" << 'EOF'
import json
import math
from collections import Counter
from pathlib import Path

def per_req(blob):
    m = blob['measurement'] if 'measurement' in blob else blob['cells'][0]
    return {r['request_id']: r['correct'] for r in m['per_request']}, m['accuracy'], m.get('wall_clock_s')

def mcnemar(a, b):
    common = set(a) & set(b)
    bw = sum(1 for k in common if b[k] < a[k])
    cw = sum(1 for k in common if b[k] > a[k])
    n = bw + cw
    if n == 0:
        return 0, 0, 1.0, 0.0
    k = max(bw, cw)
    p_one = sum(math.comb(n, i) * 0.5**n for i in range(k, n+1))
    p_two = min(2 * p_one, 1.0) if n > 1 else 1.0
    delta = (cw - bw) / max(len(common), 1)
    return bw, cw, p_two, delta

def consol_max(per_rank):
    if not per_rank:
        return None
    L = len(per_rank[0])
    return [max(per_rank[r][li] for r in range(len(per_rank))) for li in range(L)]

def counts(t):
    return dict(sorted(Counter(t).items()))

def bits(t):
    return sum(t) / len(t)

def report(name, frozen_path, dyn_path):
    print(f"---- {name} -----------------------------------------------")
    fz = json.load(open(frozen_path))
    dy = json.load(open(dyn_path))
    fz_scores, fz_acc, fz_wall = per_req(fz)
    dy_scores, dy_acc, dy_wall = per_req(dy)
    fz_proj = fz['memory_projection']
    dy_proj = dy['memory_projection']
    dy_seeded = dy.get('consolidated_tiers')
    dy_final_pr = dy.get('final_tiers_per_rank')
    dy_final_consol = consol_max(dy_final_pr) if dy_final_pr else None

    print(f"  frozen  : acc={fz_acc:.3f}  bits(seeded)={fz_proj['avg_bits']:.2f}  "
          f"tiers={fz_proj['tier_counts']}  wall={fz_wall:.0f}s")
    print(f"  dynamic : acc={dy_acc:.3f}  bits(seeded)={dy_proj['avg_bits']:.2f}  "
          f"tiers(seeded)={dy_proj['tier_counts']}  wall={dy_wall:.0f}s")
    if dy_final_consol is not None:
        # Per-rank summary
        per_rank_bits = [bits(pr) for pr in dy_final_pr]
        per_rank_counts = [counts(pr) for pr in dy_final_pr]
        print(f"  dynamic final (consolidated): bits={bits(dy_final_consol):.2f}  "
              f"tiers={counts(dy_final_consol)}")
        print(f"    per-rank bits: " + ", ".join(f"r{r}={b:.2f}" for r, b in enumerate(per_rank_bits)))
        n_changes = sum(1 for li in range(len(dy_seeded)) if dy_final_consol[li] != dy_seeded[li])
        print(f"    drift during measurement: {n_changes} of {len(dy_seeded)} consolidated layer slots changed")

    bw, cw, p, d = mcnemar(fz_scores, dy_scores)
    print(f"  paired McNemar dyn vs frozen: Δ={d:+.4f}  (b/c={bw}/{cw}, p={p:.4f})")
    print()

# GSM8K-Hard
report(
    "GSM8K-Hard (n=500 mns=8, τ=0.022)",
    "results/serving/session14_70b/echokv3c_abscanary_tau022_n500_mns8.json",
    "results/serving/session14_70b/echokv3c_abscanary_tau022_n500_mns8_continuous_dynamic.json",
)

# HotpotQA
report(
    "HotpotQA (n=500 mns=8, τ=0.022)",
    "results/serving/session14_70b/hotpotqa_n500_mns8/echokv3c_abscanary_tau022_n500_mns8.json",
    "results/serving/session14_70b/hotpotqa_n500_mns8/echokv3c_abscanary_tau022_n500_mns8_continuous_dynamic.json",
)

# RULER@128K
report(
    "RULER@128K (n=20 mns=1, τ=0.124 √L-scaled)",
    "results/serving/session14_70b/echokv3c_abscanary_tau124_ruler128k_n20_mns1.json",
    "results/serving/session14_70b/echokv3c_abscanary_tau124_ruler128k_n20_mns1_continuous_dynamic.json",
)
EOF
echo "================================================================"
