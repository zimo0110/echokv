#!/usr/bin/env bash
# RULER@128K at n=100 on both Llama-3.1-70B and Qwen2.5-72B.
#
# Sequential on the same 4-GPU lane: Llama first, then Qwen. Each model
# runs 3 cells (FP16 + KIVI-INT4 + EchoKV). Static INT8 omitted (paper
# didn't include it for RULER, and n=20 already saturated at 1.000 for
# all configs we did include).
#
# τ values:
#   Llama-3.1-70B  τ_0 = 0.022   → τ(@128K) = τ_0·√(128K/4K) = 0.124
#   Qwen2.5-72B    τ_0 from results/serving/qwen72b/tau_budget.txt
#                  → τ(@128K) = τ_0·√32
#
# Estimated wall: ~3 hr per cell at n=100 → ~9 hr per model → ~18 hr total.
# This is overnight territory; expect to launch and check in the morning.
#
# Pre-req: results/serving/qwen72b/tau_budget.txt must exist
#          (run scripts/17_qwen_calibrate.sh first).
#          Qwen2.5-72B-Instruct must have YaRN scaling in config.json
#          (we patched it earlier; see config.json.bak for the original).
#
# Run:    bash scripts/18_ruler_n100_both_models.sh
# Tmux:   tmux new -s ruler-n100 'bash scripts/18_ruler_n100_both_models.sh; echo "[done]"; read'
# Watch:  tail -f results/serving/{session14_70b,qwen72b}/ruler_n100*.log

set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

PY="${PY:-/opt/conda/envs/scalable/bin/python}"
GPUS="${GPUS:-0,1,2,3}"
TP="${TP:-4}"
N_PROMPTS="${N_PROMPTS:-100}"
WARMUP_N="${WARMUP_N:-8}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
CONTEXT="${CONTEXT:-128000}"

MODEL_LLAMA="${MODEL_LLAMA:-/mnt/nvme_tmp6/llama-70b}"
MODEL_QWEN="${MODEL_QWEN:-/mnt/nvme_tmp10/qwen2.5-72b}"

TAU_LLAMA="${TAU_LLAMA:-0.022}"
TAU_FILE_QWEN="${REPO_ROOT}/results/serving/qwen72b/tau_budget.txt"

OUT_LLAMA="${REPO_ROOT}/results/serving/session14_70b"
OUT_QWEN="${REPO_ROOT}/results/serving/qwen72b"
mkdir -p "${OUT_LLAMA}" "${OUT_QWEN}"

# ─── Pre-flight ───
[ -d "${MODEL_LLAMA}" ] || { echo "ERROR: Llama model dir not found: ${MODEL_LLAMA}"; exit 1; }
[ -d "${MODEL_QWEN}" ]  || { echo "ERROR: Qwen model dir not found: ${MODEL_QWEN}"; exit 1; }
[ -f "${TAU_FILE_QWEN}" ] || { echo "ERROR: ${TAU_FILE_QWEN} missing — run scripts/17_qwen_calibrate.sh first"; exit 1; }
[ -x "${PY}" ] || { echo "ERROR: python not executable at ${PY}"; exit 1; }

TAU_QWEN_0=$(cat "${TAU_FILE_QWEN}")
TAU_LLAMA_RULER=$("${PY}" -c "import math; print(f'{${TAU_LLAMA} * math.sqrt(${CONTEXT}/4000):.5f}')")
TAU_QWEN_RULER=$("${PY}" -c "import math; print(f'{${TAU_QWEN_0} * math.sqrt(${CONTEXT}/4000):.5f}')")

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

echo "================================================================"
echo "RULER@128K n=${N_PROMPTS} — Llama-3.1-70B + Qwen2.5-72B"
echo "  GPUs ${GPUS}  TP=${TP}  mns=${MAX_NUM_SEQS}  context=${CONTEXT}"
echo "  Llama τ_0=${TAU_LLAMA}     → τ(@${CONTEXT}) = ${TAU_LLAMA_RULER}"
echo "  Qwen  τ_0=${TAU_QWEN_0}    → τ(@${CONTEXT}) = ${TAU_QWEN_RULER}"
echo "================================================================"

# ───────────────────────────────────────────────────────────────────────
# Phase 1: Llama-3.1-70B
# ───────────────────────────────────────────────────────────────────────
LOG_LLAMA="${OUT_LLAMA}/ruler_n${N_PROMPTS}.log"
echo "[$(date -u +%H:%M:%S)] Llama RULER@128K FP16 + KIVI-INT4" | tee -a "${LOG_LLAMA}"
"${PY}" -m evaluation.serving.run_session14_70b_ruler \
    --model "${MODEL_LLAMA}" \
    --tensor-parallel-size "${TP}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --n-prompts "${N_PROMPTS}" \
    --context "${CONTEXT}" \
    --phase fp16_int4 \
    --out "${OUT_LLAMA}/ruler_n${N_PROMPTS}_static.json" 2>&1 | tee -a "${LOG_LLAMA}"

echo "[$(date -u +%H:%M:%S)] Llama RULER@128K EchoKV (τ=${TAU_LLAMA_RULER})" | tee -a "${LOG_LLAMA}"
"${PY}" -m evaluation.serving.run_session14_70b_ruler_echokv \
    --model "${MODEL_LLAMA}" \
    --tensor-parallel-size "${TP}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --n-prompts "${N_PROMPTS}" \
    --warmup-n "${WARMUP_N}" \
    --context "${CONTEXT}" \
    --multi-precision \
    --absolute-canary \
    --kref-window-size 64 \
    --tau-budget "${TAU_LLAMA_RULER}" \
    --continuous-dynamic \
    --out "${OUT_LLAMA}/ruler_n${N_PROMPTS}_echokv.json" 2>&1 | tee -a "${LOG_LLAMA}"

echo "[$(date -u +%H:%M:%S)] Llama RULER@128K DONE" | tee -a "${LOG_LLAMA}"

# ───────────────────────────────────────────────────────────────────────
# Phase 2: Qwen2.5-72B
# ───────────────────────────────────────────────────────────────────────
LOG_QWEN="${OUT_QWEN}/ruler_n${N_PROMPTS}.log"
echo "[$(date -u +%H:%M:%S)] Qwen RULER@128K FP16 + KIVI-INT4" | tee -a "${LOG_QWEN}"
"${PY}" -m evaluation.serving.run_session14_70b_ruler \
    --model "${MODEL_QWEN}" \
    --tensor-parallel-size "${TP}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --n-prompts "${N_PROMPTS}" \
    --context "${CONTEXT}" \
    --phase fp16_int4 \
    --out "${OUT_QWEN}/ruler_n${N_PROMPTS}_static.json" 2>&1 | tee -a "${LOG_QWEN}"

echo "[$(date -u +%H:%M:%S)] Qwen RULER@128K EchoKV (τ=${TAU_QWEN_RULER})" | tee -a "${LOG_QWEN}"
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
    --tau-budget "${TAU_QWEN_RULER}" \
    --continuous-dynamic \
    --out "${OUT_QWEN}/ruler_n${N_PROMPTS}_echokv.json" 2>&1 | tee -a "${LOG_QWEN}"

echo "[$(date -u +%H:%M:%S)] Qwen RULER@128K DONE" | tee -a "${LOG_QWEN}"

echo
echo "================================================================"
echo "ALL DONE [$(date -u +%Y-%m-%dT%H:%M:%SZ)]"
echo "  Llama: ${OUT_LLAMA}/ruler_n${N_PROMPTS}_{static,echokv}.json"
echo "  Qwen:  ${OUT_QWEN}/ruler_n${N_PROMPTS}_{static,echokv}.json"
echo "================================================================"
