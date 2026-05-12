#!/usr/bin/env bash
# Qwen2.5-72B HotpotQA sweep: FP16 + KIVI-INT4 + Static INT8 + EchoKV.
# Reads τ_0 from results/serving/qwen72b/tau_budget.txt (run calibrate first).
#
# Run:    bash scripts/17_qwen_hotpotqa.sh
# Tmux:   tmux new -s qwen-hotpot 'bash scripts/17_qwen_hotpotqa.sh'

set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

PY="${PY:-/opt/conda/envs/scalable/bin/python}"
MODEL_QWEN="${MODEL_QWEN:-/mnt/nvme_tmp10/qwen2.5-72b}"
GPUS="${GPUS:-0,1,2,3}"
TP="${TP:-4}"
N_PROMPTS="${N_PROMPTS:-500}"
WARMUP_N="${WARMUP_N:-16}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

OUT_DIR="${REPO_ROOT}/results/serving/qwen72b"
LOG="${OUT_DIR}/hotpotqa.log"
TAU_FILE="${OUT_DIR}/tau_budget.txt"
mkdir -p "${OUT_DIR}"

[ -d "${MODEL_QWEN}" ] || { echo "ERROR: model dir not found: ${MODEL_QWEN}"; exit 1; }
[ -f "${TAU_FILE}" ]   || { echo "ERROR: ${TAU_FILE} missing — run scripts/17_qwen_calibrate.sh first"; exit 1; }
TAU=$(cat "${TAU_FILE}")

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

echo "================================================================"
echo "Qwen2.5-72B HotpotQA sweep, n=${N_PROMPTS}, mns=${MAX_NUM_SEQS}, τ_0=${TAU}"
echo "================================================================"

echo "[$(date -u +%H:%M:%S)] FP16 + KIVI-INT4" | tee -a "${LOG}"
"${PY}" -m evaluation.serving.run_session14_70b_hotpotqa \
    --model "${MODEL_QWEN}" \
    --tensor-parallel-size "${TP}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --n-prompts "${N_PROMPTS}" \
    --configs fp16 static_int4 \
    --out "${OUT_DIR}/hotpotqa_static.json" 2>&1 | tee -a "${LOG}"

echo "[$(date -u +%H:%M:%S)] Static INT8" | tee -a "${LOG}"
"${PY}" -m evaluation.serving.run_session14_70b_hotpotqa \
    --model "${MODEL_QWEN}" \
    --tensor-parallel-size "${TP}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --n-prompts "${N_PROMPTS}" \
    --configs static_int8 \
    --out "${OUT_DIR}/hotpotqa_int8.json" 2>&1 | tee -a "${LOG}"

echo "[$(date -u +%H:%M:%S)] EchoKV (continuous-dynamic, τ_0=${TAU})" | tee -a "${LOG}"
"${PY}" -m evaluation.serving.run_session14_70b_hotpotqa_echokv \
    --model "${MODEL_QWEN}" \
    --tensor-parallel-size "${TP}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --n-prompts "${N_PROMPTS}" \
    --warmup-n "${WARMUP_N}" \
    --multi-precision \
    --absolute-canary \
    --kref-window-size 64 \
    --tau-budget "${TAU}" \
    --continuous-dynamic \
    --out "${OUT_DIR}/hotpotqa_echokv.json" 2>&1 | tee -a "${LOG}"

echo "[$(date -u +%H:%M:%S)] DONE HotpotQA" | tee -a "${LOG}"
