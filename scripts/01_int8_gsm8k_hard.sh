#!/usr/bin/env bash
# static_int8 GSM8K-Hard at n=200, max_num_seqs=1 — finishes the
# 4-baseline column on the demonstration workload.
#
# This restarts the run that was killed mid-flight before bed.
# Expected wall time: ~50-60 minutes on 70B + TP=4.
#
# Run as: bash scripts/01_int8_gsm8k_hard.sh
# Or in tmux:
#   tmux new -s int8-gsm 'bash scripts/01_int8_gsm8k_hard.sh'
#   # Detach: Ctrl-b then d
#   # Reattach later: tmux attach -t int8-gsm

set -euo pipefail
cd "$(dirname "$0")/.."

# 70B requires TP=4 — pin to GPUs 4-7.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

STAMP=$(date -u +%Y%m%d_%H%M%S)
LOG="/tmp/run_int8_gsm8k_hard_${STAMP}.log"
OUT_DIR="results/serving/session14_70b/publication"
OUT="${OUT_DIR}/int8_n200_seq1.json"
mkdir -p "${OUT_DIR}"

echo "================================================================"
echo "Run 1 — static_int8 GSM8K-Hard, n=200, max_num_seqs=1"
echo "  model: 70B, TP=4 (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "  log: ${LOG}"
echo "  out: ${OUT}"
echo "  expected wall: ~50-60 minutes"
echo "  start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

/opt/conda/envs/scalable/bin/python \
    -m evaluation.serving.run_session14_70b_gsm8k_hard \
    --n-prompts 200 \
    --configs static_int8 \
    --max-num-seqs 1 \
    --seed 0 \
    --out "${OUT}" 2>&1 | tee "${LOG}"

echo
echo "================================================================"
echo "Done at $(date -u +%Y-%m-%dT%H:%M:%SZ)."
echo
echo "Result:"
/opt/conda/envs/scalable/bin/python -c "
import json
b = json.load(open('${OUT}'))
for c in b.get('cells', []):
    print(f'  {c[\"config\"]}: acc={c[\"accuracy\"]:.3f} wall={c[\"wall_clock_s\"]:.1f}s')
"
echo "================================================================"
