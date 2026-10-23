#!/usr/bin/env bash
# Smoke test: HotpotQA driver on 8B + TP=2 with 4 prompts.
#
# Validates the driver clone before committing 70B compute. Should
# complete in ~2 minutes (load ~30s + 4 prompts × ~5s + cleanup).
#
# Run as: bash scripts/00_smoke_hotpotqa.sh
# Or in tmux:
#   tmux new -s smoke-hotpot 'bash scripts/00_smoke_hotpotqa.sh'

set -euo pipefail
cd "$(dirname "$0")/.."

# Use GPUs 4,5 (TP=2 for 8B). Adjust if these are busy.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

STAMP=$(date -u +%Y%m%d_%H%M%S)
LOG="/tmp/smoke_hotpotqa_${STAMP}.log"
OUT="results/serving/smoke_hotpotqa_8b.json"

echo "================================================================"
echo "Smoke test: HotpotQA on Llama-3.1-8B, TP=2, n=4 (FP16 only)"
echo "  CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "  log: ${LOG}"
echo "  out: ${OUT}"
echo "================================================================"

DEFAULT_8B="/mnt/nvme_tmp/nazar_hf/hub/models--NousResearch--Meta-Llama-3.1-8B-Instruct/snapshots/d10aef7999a2b5ba950ab3974312feeedbfe0b77"

# tee-pipe so the user sees progress in tmux + we have the log.
/opt/conda/envs/scalable/bin/python \
    -m evaluation.serving.run_session14_70b_hotpotqa \
    --model "${DEFAULT_8B}" \
    --tensor-parallel-size 2 \
    --max-model-len 4096 \
    --max-new-tokens 64 \
    --max-num-seqs 4 \
    --n-prompts 4 \
    --configs fp16 \
    --gpu-memory-utilization 0.85 \
    --out "${OUT}" 2>&1 | tee "${LOG}"

echo
echo "================================================================"
echo "Smoke acc (should be > 0 — even partial means scoring is wired):"
/opt/conda/envs/scalable/bin/python -c "
import json
b = json.load(open('${OUT}'))
for c in b.get('cells', []):
    print(f'  {c[\"config\"]}: acc={c[\"accuracy\"]:.3f} ({sum(r[\"correct\"] for r in c[\"per_request\"]):.0f}/{c[\"n_prompts\"]})')
"
echo "================================================================"
