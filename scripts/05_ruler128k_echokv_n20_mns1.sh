#!/usr/bin/env bash
# EchoKV-3C with absolute canary on RULER@128K at n=20, mns=1 —
# matches the existing Phase-2 RULER@128K protocol so we can pair
# against the FP16 and INT4 cells already saved in
# results/serving/session14_70b/diag.json.
#
# Workload: single-needle haystack at 128K context. Both static
# baselines saturate at 1.000 accuracy at this size, so the
# interesting quantity is what tier vector the controller picks
# (we expect mostly INT4 because there's no INT4 deficit to recover).
#
# Why n=20 mns=1 and not n=500 mns=8: 128K-context prefill on 70B+TP=4
# is ~18s/prompt (sequential) or ~80-100s per mns=8 batch. n=500 would
# be 5-6 hours total for 4 cells. n=20 mns=1 reuses the existing
# Phase-2 baselines; the bit-budget the controller picks stabilises
# during warmup regardless of measurement n.
#
# Run as: bash scripts/05_ruler128k_echokv_n20_mns1.sh
# Or in tmux:
#   tmux new -s ruler-ekv 'bash scripts/05_ruler128k_echokv_n20_mns1.sh'

set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

STAMP=$(date -u +%Y%m%d_%H%M%S)
LOG="/tmp/ruler128k_echokv_${STAMP}.log"
OUT="results/serving/session14_70b/echokv3c_abscanary_tau022_ruler128k_n20_mns1.json"
mkdir -p "$(dirname "${OUT}")"

echo "================================================================"
echo "EchoKV-3C abs-canary @ RULER@128K, n=20, max_num_seqs=1"
echo "(paired against Phase-2 FP16/INT4 in diag.json)"
echo "  CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "  log: ${LOG}"
echo "  out: ${OUT}"
echo "  expected wall: ~7-10 min"
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
    --tau-budget 0.022 \
    --seed 0 \
    --out "${OUT}" 2>&1 | tee "${LOG}"

echo
echo "================================================================"
echo "Done at $(date -u +%Y-%m-%dT%H:%M:%SZ)."
echo
echo "Result + paired comparison vs Phase-2 RULER@128K baselines:"
/opt/conda/envs/scalable/bin/python << EOF
import json
import math
from pathlib import Path

ech_blob = json.loads(Path("${OUT}").read_text())
ech_m = ech_blob["measurement"]
proj = ech_blob["memory_projection"]
ech_scores = {r["request_id"]: r["correct"] for r in ech_m["per_request"]}
ech_acc = ech_m["accuracy"]

ph2 = json.loads(Path("results/serving/session14_70b/diag.json").read_text())
fp16_cell = next(c for c in ph2["cells"] if c["config"] == "fp16")
int4_cell = next(c for c in ph2["cells"] if c["config"] == "static_int4")
fp16 = {r["request_id"]: r["correct"] for r in fp16_cell["per_request"]}
int4 = {r["request_id"]: r["correct"] for r in int4_cell["per_request"]}

print(f"  FP16  (Phase-2 baseline): acc={fp16_cell['accuracy']:.3f}  bits=16.00")
print(f"  INT4  (Phase-2 baseline): acc={int4_cell['accuracy']:.3f}  bits=4.00")
print(f"  EchoKV-3C abs-canary    : acc={ech_acc:.3f}  "
      f"bits={proj['avg_bits']:.2f}  "
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
