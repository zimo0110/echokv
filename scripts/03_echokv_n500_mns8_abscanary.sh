#!/usr/bin/env bash
# EchoKV-3C with absolute canary at n=500, max_num_seqs=8 — paired
# against the existing Phase-2 n=500 mns=8 baselines on GSM8K-Hard.
#
# Why mns=8 not mns=1: the existing n=500 baselines (FP16=0.488,
# INT4=0.466, McNemar p=0.052 for INT4 vs FP16) were both run at
# mns=8. To do a paired three-way comparison against those baselines
# we need EchoKV-3C at the same protocol. mns=1 would force a
# protocol mismatch.
#
# The absolute-canary K_ref buffer is independent of cache state and
# of batch size (T_key = 64 always), so the sensor side is clean
# regardless of mns. The mns=8 batch-composition noise affects the
# inference-path matmul symmetrically across all three configs.
#
# Run as: bash scripts/03_echokv_n500_mns8_abscanary.sh
# Or in tmux:
#   tmux new -s n500 'bash scripts/03_echokv_n500_mns8_abscanary.sh'

set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

STAMP=$(date -u +%Y%m%d_%H%M%S)
LOG="/tmp/echokv_n500_mns8_${STAMP}.log"
OUT="results/serving/session14_70b/echokv3c_abscanary_tau022_n500_mns8.json"
mkdir -p "$(dirname "${OUT}")"

echo "================================================================"
echo "EchoKV-3C abs-canary @ n=500, max_num_seqs=8 (paired vs Phase-2)"
echo "  CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "  log: ${LOG}"
echo "  out: ${OUT}"
echo "  expected wall: ~40-50 min"
echo "  start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

/opt/conda/envs/scalable/bin/python \
    -m evaluation.serving.run_session14_70b_gsm8k_hard_echokv \
    --n-prompts 500 \
    --warmup-n 16 \
    --multi-precision \
    --absolute-canary \
    --kref-window-size 64 \
    --tau-budget 0.022 \
    --max-num-seqs 8 \
    --seed 0 \
    --out "${OUT}" 2>&1 | tee "${LOG}"

echo
echo "================================================================"
echo "Done at $(date -u +%Y-%m-%dT%H:%M:%SZ)."
echo
echo "Result + paired comparison vs Phase-2 baselines at the same protocol:"
/opt/conda/envs/scalable/bin/python << EOF
import json
import math
from pathlib import Path

# Load EchoKV-3C result
ech_blob = json.loads(Path("${OUT}").read_text())
ech_m = ech_blob["measurement"]
proj = ech_blob["memory_projection"]
ech_scores = {r["request_id"]: r["correct"] for r in ech_m["per_request"]}
ech_acc = ech_m["accuracy"]

# Load existing Phase-2 baselines (FP16 + INT4 cells in the same JSON)
ph2 = json.loads(Path("results/serving/session14_70b/gsm8k_hard_n500.json").read_text())
fp16_cell = next(c for c in ph2["cells"] if c["config"] == "fp16")
int4_cell = next(c for c in ph2["cells"] if c["config"] == "static_int4")
fp16 = {r["request_id"]: r["correct"] for r in fp16_cell["per_request"]}
int4 = {r["request_id"]: r["correct"] for r in int4_cell["per_request"]}

print(f"  FP16  (Phase-2 baseline): acc={fp16_cell['accuracy']:.3f}")
print(f"  INT4  (Phase-2 baseline): acc={int4_cell['accuracy']:.3f}")
print(f"  EchoKV-3C abs-canary    : acc={ech_acc:.3f}  "
      f"avg_bits={proj['avg_bits']:.2f}  "
      f"savings_vs_fp16={proj['savings_vs_fp16_pct']:.1f}%  "
      f"tier_counts={proj['tier_counts']}")

def mcnemar(a, b):
    common = set(a) & set(b)
    bw = sum(1 for k in common if b[k] < a[k])
    cw = sum(1 for k in common if b[k] > a[k])
    n = bw + cw
    if n == 0:
        return 0, 0, 1.0
    k = max(bw, cw)
    p_one = sum(math.comb(n, i) * 0.5**n for i in range(k, n+1))
    p_two = min(2 * p_one, 1.0) if n > 1 else 1.0
    return bw, cw, p_two

bw, cw, p = mcnemar(fp16, ech_scores)
print(f"  EchoKV vs FP16 (paired): {bw} worse / {cw} better, McNemar p={p:.4f}")
bw, cw, p = mcnemar(int4, ech_scores)
print(f"  EchoKV vs INT4 (paired): {bw} worse / {cw} better, McNemar p={p:.4f}")
EOF
echo "================================================================"
