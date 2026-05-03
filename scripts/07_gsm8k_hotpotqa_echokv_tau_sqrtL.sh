#!/usr/bin/env bash
# Re-run GSM8K-Hard and HotpotQA with the √L-scaled τ-budget derived
# in PAPER_DRAFT.md §4.X, paired against the existing τ=0.022 cells.
#
# Pinsker formula (§4.X): τ(L) = τ₀ · √(L/L₀), with τ₀ = 0.022 at
# L₀ = 4 000 (the 8B + RULER@4K calibration probe).
#
# Per-workload L (prompt-context length in tokens):
#   GSM8K-Hard:  L ≈ 400   → τ = 0.022 · √(400/4000)   ≈ 0.007
#   HotpotQA:    L ≈ 1500  → τ = 0.022 · √(1500/4000)  ≈ 0.013
#
# Both L values are < L₀ = 4 000, so the formula gives τ < τ₀ — i.e.
# a *tighter* per-tier threshold than the calibration value. The
# controller will therefore admit fewer layers at INT4 and upgrade
# more to INT8 vs the τ=0.022 runs. Concretely we expect:
#
#   GSM8K-Hard τ=0.007 (vs τ=0.022 → {INT4:51, INT8:29}, 5.45 bits):
#     - tier shifts toward more INT8 (e.g. {INT4:~30, INT8:~50})
#     - avg bits goes UP (e.g. ~6.5–7.0)
#     - accuracy: still matches FP16 (already matched at τ=0.022)
#
#   HotpotQA   τ=0.013 (vs τ=0.022 → {INT4:69, INT8:11}, 4.55 bits):
#     - tier shifts toward more INT8 (e.g. {INT4:~50, INT8:~30})
#     - avg bits goes UP (e.g. ~5.5–6.0)
#     - accuracy: still matches FP16 (INT4 already sufficient)
#
# What this run tells us about the formula:
#   - If accuracy stays at FP16 baseline on both workloads → the
#     formula over-tightens at L < L₀ (paper claim "use τ₀ as a
#     floor at small L" is correct).
#   - If accuracy IMPROVES on GSM8K-Hard → the τ₀=0.022 calibration
#     was a touch loose for short-prompt reasoning.
#   - If accuracy DROPS → something is wrong with our derivation
#     (the √L scaling does not extend to short L).
#
# Run as:        bash scripts/07_gsm8k_hotpotqa_echokv_tau_sqrtL.sh
# Run in tmux:   tmux new -s tau-sqrtL 'bash scripts/07_gsm8k_hotpotqa_echokv_tau_sqrtL.sh'
#
# Expected total wall: ~60–70 min on 70B + TP=4
#   GSM8K-Hard echokv (n=500 mns=8):  ~40–50 min
#   HotpotQA   echokv (n=500 mns=8):  ~20 min

set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

STAMP=$(date -u +%Y%m%d_%H%M%S)
LOG_DIR="/tmp/tau_sqrtL_${STAMP}"
mkdir -p "${LOG_DIR}"

PY="/opt/conda/envs/scalable/bin/python"

# Pinsker-derived τ for each workload (see header for derivation).
TAU_GSM=0.007       # = 0.022 · √(400/4000)
TAU_HOT=0.013       # = 0.022 · √(1500/4000)

OUT_GSM="results/serving/session14_70b/echokv3c_abscanary_tau007_n500_mns8.json"
OUT_HOT="results/serving/session14_70b/hotpotqa_n500_mns8/echokv3c_abscanary_tau013_n500_mns8.json"
mkdir -p "$(dirname "${OUT_GSM}")" "$(dirname "${OUT_HOT}")"

echo "================================================================"
echo "EchoKV-3C abs-canary @ √L-scaled τ — paired re-run"
echo "  τ_base (L₀=4000)        = 0.022"
echo "  GSM8K-Hard (L≈400)  τ   = ${TAU_GSM}"
echo "  HotpotQA   (L≈1500) τ   = ${TAU_HOT}"
echo "  CUDA_VISIBLE_DEVICES    = ${CUDA_VISIBLE_DEVICES}"
echo "  log dir                 = ${LOG_DIR}"
echo "  start                   = $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

# ---------------------------- GSM8K-Hard --------------------------------
echo "----------------------------------------------------------------"
echo "[1/2] GSM8K-Hard echokv-3c, τ=${TAU_GSM} (n=500 mns=8)"
echo "  out: ${OUT_GSM}"
echo "  start: $(date -u +%H:%M:%S)"
echo "----------------------------------------------------------------"
"${PY}" -m evaluation.serving.run_session14_70b_gsm8k_hard_echokv \
    --n-prompts 500 \
    --warmup-n 16 \
    --multi-precision \
    --absolute-canary \
    --kref-window-size 64 \
    --tau-budget "${TAU_GSM}" \
    --max-num-seqs 8 \
    --seed 0 \
    --out "${OUT_GSM}" 2>&1 | tee "${LOG_DIR}/gsm8k_hard.log"

# ----------------------------- HotpotQA ---------------------------------
echo
echo "----------------------------------------------------------------"
echo "[2/2] HotpotQA echokv-3c, τ=${TAU_HOT} (n=500 mns=8)"
echo "  out: ${OUT_HOT}"
echo "  start: $(date -u +%H:%M:%S)"
echo "----------------------------------------------------------------"
"${PY}" -m evaluation.serving.run_session14_70b_hotpotqa_echokv \
    --n-prompts 500 \
    --warmup-n 16 \
    --multi-precision \
    --absolute-canary \
    --kref-window-size 64 \
    --tau-budget "${TAU_HOT}" \
    --max-num-seqs 8 \
    --max-new-tokens 64 \
    --seed 0 \
    --out "${OUT_HOT}" 2>&1 | tee "${LOG_DIR}/hotpotqa.log"

# ------------------------- comparison summary ---------------------------
echo
echo "================================================================"
echo "Done at $(date -u +%Y-%m-%dT%H:%M:%SZ)."
echo
echo "Paired comparison: √L-scaled τ vs the prior τ=0.022 cells."
"${PY}" << EOF
import json
import math
from pathlib import Path

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

def per_req(blob, key='measurement'):
    m = blob[key] if key in blob else blob['cells'][0]
    return {r['request_id']: r['correct'] for r in m['per_request']}, m['accuracy']

# --- GSM8K-Hard ---
print("---- GSM8K-Hard (n=500 mns=8) -----------------------------------")
ph2 = json.load(open('results/serving/session14_70b/gsm8k_hard_n500.json'))
fp16_cell = next(c for c in ph2['cells'] if c['config'] == 'fp16')
int4_cell = next(c for c in ph2['cells'] if c['config'] == 'static_int4')
fp16 = {r['request_id']: r['correct'] for r in fp16_cell['per_request']}
int4 = {r['request_id']: r['correct'] for r in int4_cell['per_request']}

prev_path = Path('results/serving/session14_70b/echokv3c_abscanary_tau022_n500_mns8.json')
prev_blob = json.load(open(prev_path)) if prev_path.exists() else None
new_blob = json.load(open('${OUT_GSM}'))

if prev_blob is not None:
    prev_scores, prev_acc = per_req(prev_blob)
    prev_proj = prev_blob['memory_projection']
    print(f"  τ=0.022 (prior): acc={prev_acc:.3f}  bits={prev_proj['avg_bits']:.2f}  "
          f"tiers={prev_proj['tier_counts']}")
new_scores, new_acc = per_req(new_blob)
new_proj = new_blob['memory_projection']
print(f"  τ=${TAU_GSM} (√L) : acc={new_acc:.3f}  bits={new_proj['avg_bits']:.2f}  "
      f"savings_vs_fp16={new_proj['savings_vs_fp16_pct']:.1f}%  "
      f"tiers={new_proj['tier_counts']}")
print(f"  FP16   (ref)   : acc={fp16_cell['accuracy']:.3f}")
print(f"  INT4   (ref)   : acc={int4_cell['accuracy']:.3f}")
for ref, ref_name in [(fp16, 'FP16'), (int4, 'INT4')]:
    bw, cw, p, d = mcnemar(ref, new_scores)
    print(f"  τ=${TAU_GSM} vs {ref_name:<4}  Δ={d:+.4f}  (b/c={bw}/{cw}, p={p:.4f})")
if prev_blob is not None:
    bw, cw, p, d = mcnemar(prev_scores, new_scores)
    print(f"  τ=${TAU_GSM} vs τ=0.022  Δ={d:+.4f}  (b/c={bw}/{cw}, p={p:.4f})")

# --- HotpotQA ---
print()
print("---- HotpotQA (n=500 mns=8) -------------------------------------")
hbase = Path('results/serving/session14_70b/hotpotqa_n500_mns8')
def load_hot(name):
    p = hbase / f'{name}_n500_mns8.json'
    if not p.exists():
        return None, None
    b = json.load(open(p))
    if 'measurement' in b:
        m = b['measurement']
    else:
        m = b['cells'][0]
    return {r['request_id']: r['correct'] for r in m['per_request']}, m['accuracy']

fp16_h, fp16_ha = load_hot('fp16')
int4_h, int4_ha = load_hot('int4')
int8_h, int8_ha = load_hot('int8')
prev_path = hbase / 'echokv3c_abscanary_tau022_n500_mns8.json'
prev_blob = json.load(open(prev_path)) if prev_path.exists() else None
new_blob = json.load(open('${OUT_HOT}'))

if prev_blob is not None:
    prev_scores, prev_acc = per_req(prev_blob)
    prev_proj = prev_blob['memory_projection']
    print(f"  τ=0.022 (prior): acc={prev_acc:.3f}  bits={prev_proj['avg_bits']:.2f}  "
          f"tiers={prev_proj['tier_counts']}")
new_scores, new_acc = per_req(new_blob)
new_proj = new_blob['memory_projection']
print(f"  τ=${TAU_HOT} (√L) : acc={new_acc:.3f}  bits={new_proj['avg_bits']:.2f}  "
      f"savings_vs_fp16={new_proj['savings_vs_fp16_pct']:.1f}%  "
      f"tiers={new_proj['tier_counts']}")
if fp16_ha is not None:
    print(f"  FP16   (ref)   : acc={fp16_ha:.3f}")
if int4_ha is not None:
    print(f"  INT4   (ref)   : acc={int4_ha:.3f}")
if int8_ha is not None:
    print(f"  INT8   (ref)   : acc={int8_ha:.3f}")
for ref, ref_name in [(fp16_h, 'FP16'), (int4_h, 'INT4'), (int8_h, 'INT8')]:
    if ref is None:
        continue
    bw, cw, p, d = mcnemar(ref, new_scores)
    print(f"  τ=${TAU_HOT} vs {ref_name:<4}  Δ={d:+.4f}  (b/c={bw}/{cw}, p={p:.4f})")
if prev_blob is not None:
    bw, cw, p, d = mcnemar(prev_scores, new_scores)
    print(f"  τ=${TAU_HOT} vs τ=0.022  Δ={d:+.4f}  (b/c={bw}/{cw}, p={p:.4f})")

print()
print("Hypothesis check:")
print("  - acc unchanged on both workloads + bits go UP → √L formula")
print("    over-tightens at L < L₀; floor-at-τ₀ framing in §4.X is correct.")
print("  - acc IMPROVES on GSM8K-Hard → τ₀=0.022 was slightly loose for")
print("    short-prompt reasoning.")
print("  - acc DROPS on either workload → something is off with the")
print("    derivation at small L (re-investigate the aggregation step).")
EOF
echo "================================================================"
