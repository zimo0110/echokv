#!/usr/bin/env bash
# Fork FP16 vs fork static INT4 paired-McNemar on Llama-70B GSM8K-Hard, n=1000.
#
# Companion to scripts/20_*.sh (n=500). The n=500 run gave Δ=+0.020 with
# p=0.123 (n.s.), borderline between Outcome 1 ("no meaningful deficit")
# and Outcome 3 ("ambiguous"). Doubling the sample size halves the
# binomial SE (0.022 → 0.015) — should resolve whether the small INT4
# deficit on the fork is real.
#
# Run as: bash scripts/21_fork_fp16_vs_int4_mcnemar_n1000.sh
# Or in tmux:
#   tmux new -s mcnemar1k 'bash scripts/21_fork_fp16_vs_int4_mcnemar_n1000.sh'
#   # Detach: Ctrl-b then d
#   # Reattach: tmux attach -t mcnemar1k
#
# Expected wall: ~50-70 min on 70B + TP=4 (two ~25-min n=1000 generation
# phases plus two engine inits at ~40s each).

set -euo pipefail
cd "$(dirname "$0")/.."

# 70B requires TP=4. Default to GPUs 0-3 (Jason's caspo training has been
# living on 4-7). Override with CUDA_VISIBLE_DEVICES=… if needed.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

STAMP=$(date -u +%Y%m%d_%H%M%S)
LOG="/tmp/fork_fp16_vs_int4_mcnemar_n1000_${STAMP}.log"
OUT_DIR="results/serving"
mkdir -p "${OUT_DIR}"

N_PROMPTS=1000
FP16_OUT="${OUT_DIR}/fork_fp16_70b_gsm8k_hard_n${N_PROMPTS}_per_prompt.json"
INT4_OUT="${OUT_DIR}/fork_static_int4_70b_gsm8k_hard_n${N_PROMPTS}_per_prompt.json"
MCNEMAR_OUT="${OUT_DIR}/fork_fp16_vs_int4_70b_gsm8k_hard_n${N_PROMPTS}_mcnemar.json"

echo "================================================================"
echo "Fork FP16 vs Fork static INT4 paired-McNemar  (Llama-70B, n=${N_PROMPTS})"
echo "  workload: GSM8K-Hard, seed=0, mns=8, max_new_tokens=512"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  log: ${LOG}"
echo "  outputs:"
echo "    ${FP16_OUT}"
echo "    ${INT4_OUT}"
echo "    ${MCNEMAR_OUT}"
echo "  expected wall: ~50-70 min"
echo "  start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

/opt/conda/envs/scalable/bin/python \
    -m evaluation.serving.run_fork_fp16_vs_int4_mcnemar \
    --n-prompts "${N_PROMPTS}" \
    2>&1 | tee "${LOG}"

echo
echo "================================================================"
echo "Done at $(date -u +%Y-%m-%dT%H:%M:%SZ)."
echo
echo "McNemar summary:"
/opt/conda/envs/scalable/bin/python -c "
import json
r = json.load(open('${MCNEMAR_OUT}'))
print(f\"  Acc(FP16) = {r['acc_a']:.4f}\")
print(f\"  Acc(INT4) = {r['acc_b']:.4f}\")
print(f\"  Δ (FP16 − INT4) = {r['delta_acc']:+.4f}\")
print(f\"  Discordant: FP16 wins={r['a_wins']}, INT4 wins={r['b_wins']}\")
print(f\"  Both right: {r['both_right']}, both wrong: {r['both_wrong']}\")
print(f\"  McNemar χ² (Yates) = {r['chi2_yates']:.3f}, p = {r['p_value']:.4g}\")
"
echo "================================================================"
