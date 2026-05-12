#!/usr/bin/env bash
# Qwen2.5-72B RULER@128K sweep: FP16 + KIVI-INT4 + EchoKV.
# Reads τ_0 from results/serving/qwen72b/tau_budget.txt and scales it by
# sqrt(128K/4K) ≈ 5.66 per the paper's context-aware budget rule.
#
# RULER is slow (5–6 GPU-hours per cell at n=500), so we run n=20 mns=1
# matching the paper protocol. Static INT8 omitted (paper didn't run it).
#
# Run:    bash scripts/17_qwen_ruler.sh
# Tmux:   tmux new -s qwen-ruler 'bash scripts/17_qwen_ruler.sh'

set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

PY="${PY:-/opt/conda/envs/scalable/bin/python}"
MODEL_QWEN="${MODEL_QWEN:-/mnt/nvme_tmp10/qwen2.5-72b}"
GPUS="${GPUS:-0,1,2,3}"
TP="${TP:-4}"
N_PROMPTS="${N_PROMPTS:-20}"
WARMUP_N="${WARMUP_N:-8}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
CONTEXT="${CONTEXT:-128000}"

OUT_DIR="${REPO_ROOT}/results/serving/qwen72b"
LOG="${OUT_DIR}/ruler_128k.log"
TAU_FILE="${OUT_DIR}/tau_budget.txt"
mkdir -p "${OUT_DIR}"

[ -d "${MODEL_QWEN}" ] || { echo "ERROR: model dir not found: ${MODEL_QWEN}"; exit 1; }
[ -f "${TAU_FILE}" ]   || { echo "ERROR: ${TAU_FILE} missing — run scripts/17_qwen_calibrate.sh first"; exit 1; }
TAU0=$(cat "${TAU_FILE}")
TAU_RULER=$("${PY}" -c "import math; print(f'{${TAU0} * math.sqrt(128000/4000):.5f}')")

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

echo "================================================================"
echo "Qwen2.5-72B RULER@128K, n=${N_PROMPTS} mns=${MAX_NUM_SEQS}"
echo "  τ_0(@4K)         = ${TAU0}"
echo "  τ(@128K)= τ_0·√32 = ${TAU_RULER}"
echo "================================================================"

echo "[$(date -u +%H:%M:%S)] FP16 + KIVI-INT4 (--phase fp16_int4)" | tee -a "${LOG}"
"${PY}" -m evaluation.serving.run_session14_70b_ruler \
    --model "${MODEL_QWEN}" \
    --tensor-parallel-size "${TP}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --n-prompts "${N_PROMPTS}" \
    --context "${CONTEXT}" \
    --phase fp16_int4 \
    --out "${OUT_DIR}/ruler_128k_static.json" 2>&1 | tee -a "${LOG}"

echo "[$(date -u +%H:%M:%S)] EchoKV (continuous-dynamic, τ=${TAU_RULER})" | tee -a "${LOG}"
"${PY}" -m evaluation.serving.run_session14_70b_ruler_echokv \
    --model "${MODEL_QWEN}" \
    --tensor-parallel-size "${TP}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --n-prompts "${N_PROMPTS}" \
    --warmup-n "${WARMUP_N}" \
    --context "${CONTEXT}" \
    --multi-precision \
    --absolute-canary \
    --kref-window-size 64 \
    --tau-budget "${TAU_RULER}" \
    --continuous-dynamic \
    --out "${OUT_DIR}/ruler_128k_echokv.json" 2>&1 | tee -a "${LOG}"

echo "[$(date -u +%H:%M:%S)] DONE RULER@128K" | tee -a "${LOG}"
