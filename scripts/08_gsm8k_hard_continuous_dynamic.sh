#!/usr/bin/env bash
# Continuously-dynamic measurement protocol on GSM8K-Hard, paired against
# the existing frozen-tier τ=0.022 cell at the same n=500 mns=8 protocol.
#
# Why this run exists. Phase-5 / Phase-6 measurements set
# canary_rate=1e-6 and freeze_controller=True during the measurement
# phase: the consolidated tier vector from warmup is seeded onto every
# rank and held fixed for all 500 measurement prompts. That gives
# clean paired McNemar (every prompt sees the same cache config) but
# overstates what "closed-loop runtime KV quantization" actually
# exercises. The architecture (INV-2) supports continuously-dynamic
# operation: canary at production rate p_c=1/64, controller live, tier
# changes apply to new block allocations only. We have just never run
# the measurement that way.
#
# This script does. If accuracy + bit-budget match the frozen-tier
# cell within paired McNemar noise, the paper's "closed-loop / dynamic"
# framing is honest: warmup-frozen serving and continuously-dynamic
# serving are the same product, just measured under two protocols.
#
# Run as:        bash scripts/08_gsm8k_hard_continuous_dynamic.sh
# Run in tmux:   tmux new -s cont-dyn 'bash scripts/08_gsm8k_hard_continuous_dynamic.sh'
#
# Expected wall: ~45–55 min on 70B + TP=4 (slightly longer than the
#   frozen-tier ~40 min because the canary fires at p_c=1/64 throughout
#   measurement instead of essentially never; +0.09% TPOT × 500 prompts).

set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

STAMP=$(date -u +%Y%m%d_%H%M%S)
LOG="/tmp/gsm8k_hard_cont_dyn_${STAMP}.log"
OUT="results/serving/session14_70b/echokv3c_abscanary_tau022_n500_mns8_continuous_dynamic.json"
mkdir -p "$(dirname "${OUT}")"

PY="/opt/conda/envs/scalable/bin/python"

echo "================================================================"
echo "EchoKV-3C abs-canary @ GSM8K-Hard, n=500 mns=8, CONTINUOUS-DYNAMIC"
echo "  protocol     = canary_rate=1/64, freeze_controller=False"
echo "  τ_budget     = 0.022 (same as the frozen-tier headline cell)"
echo "  paired vs    : echokv3c_abscanary_tau022_n500_mns8.json (frozen)"
echo "  CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "  log: ${LOG}"
echo "  out: ${OUT}"
echo "  start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

"${PY}" -m evaluation.serving.run_session14_70b_gsm8k_hard_echokv \
    --n-prompts 500 \
    --warmup-n 16 \
    --multi-precision \
    --absolute-canary \
    --kref-window-size 64 \
    --tau-budget 0.022 \
    --continuous-dynamic \
    --max-num-seqs 8 \
    --seed 0 \
    --out "${OUT}" 2>&1 | tee "${LOG}"

echo
echo "================================================================"
echo "Done at $(date -u +%Y-%m-%dT%H:%M:%SZ)."
echo
echo "Continuous-dynamic vs frozen-tier comparison:"
"${PY}" << EOF
import json
import math
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

ph2 = json.load(open('results/serving/session14_70b/gsm8k_hard_n500.json'))
fp16_cell = next(c for c in ph2['cells'] if c['config'] == 'fp16')
int4_cell = next(c for c in ph2['cells'] if c['config'] == 'static_int4')
fp16 = {r['request_id']: r['correct'] for r in fp16_cell['per_request']}
int4 = {r['request_id']: r['correct'] for r in int4_cell['per_request']}

frozen = json.load(open('results/serving/session14_70b/echokv3c_abscanary_tau022_n500_mns8.json'))
fz_scores, fz_acc, fz_wall = per_req(frozen)
fz_proj = frozen['memory_projection']
fz_consol = frozen.get('consolidated_tiers')

dyn = json.load(open('${OUT}'))
dy_scores, dy_acc, dy_wall = per_req(dyn)
dy_proj = dyn['memory_projection']
dy_consol = dyn.get('consolidated_tiers')
dy_final = dyn.get('final_tiers_per_rank')
dy_proto = dyn.get('measurement_protocol', '?')

print(f"  FP16          : acc={fp16_cell['accuracy']:.3f}")
print(f"  INT4          : acc={int4_cell['accuracy']:.3f}")
print(f"  EchoKV (frozen): acc={fz_acc:.3f}  bits={fz_proj['avg_bits']:.2f}  "
      f"savings={fz_proj['savings_vs_fp16_pct']:.1f}%  "
      f"tiers={fz_proj['tier_counts']}  wall={fz_wall:.0f}s")
print(f"  EchoKV ({dy_proto}): acc={dy_acc:.3f}  bits={dy_proj['avg_bits']:.2f}  "
      f"savings={dy_proj['savings_vs_fp16_pct']:.1f}%  "
      f"tiers={dy_proj['tier_counts']}  wall={dy_wall:.0f}s")
print()

# Tier-vector drift: did the controller change anything during measurement?
if dy_consol is not None and dy_final:
    drift_layers = []
    for ri, fin in enumerate(dy_final):
        for li, t in enumerate(fin):
            if t != dy_consol[li]:
                drift_layers.append((ri, li, dy_consol[li], t))
    if drift_layers:
        print(f"  Tier drift during measurement: {len(drift_layers)} (rank, layer, seeded→final) changes:")
        for ri, li, before, after in drift_layers[:20]:
            print(f"    rank{ri} layer{li}: {before}→{after}")
        if len(drift_layers) > 20:
            print(f"    … {len(drift_layers)-20} more")
    else:
        print("  Tier drift during measurement: none (controller stayed at the seeded vector)")

print()
print("Paired McNemar:")
for ref, ref_name, cand, cand_name in [
    (fp16, 'FP16',           dy_scores, 'EchoKV (dyn)'),
    (int4, 'INT4',            dy_scores, 'EchoKV (dyn)'),
    (fz_scores, 'EchoKV(fzn)', dy_scores, 'EchoKV (dyn)'),
]:
    bw, cw, p, d = mcnemar(ref, cand)
    print(f"  {cand_name:<14} vs {ref_name:<11} Δ={d:+.4f}  (b/c={bw}/{cw}, p={p:.4f})")

print()
print("Hypothesis check:")
print("  - dyn matches frozen within p>0.20 + bits within 0.5 → architecture")
print("    is honestly closed-loop. Paper §6.5 framing stands.")
print("  - dyn drifts toward static_INT4 (more INT4) → controller is not")
print("    adding value in steady state; warmup tiers were the win.")
print("  - dyn drifts toward more INT8 → controller adds conservatism over")
print("    time; might want to dampen the EMA.")
EOF
echo "================================================================"
