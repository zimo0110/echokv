#!/usr/bin/env bash
# Qwen2.5-72B τ-Bench sweep (airline + retail): FP16 + INT4 + INT8 + EchoKV.
# Reads τ_0 from results/serving/qwen72b/tau_budget.txt.
#
# τ-Bench is multi-turn agentic, so wall is dominated by tool-calling and
# decode steps. Per-split n=50 gives n=90 paired McNemar after merge.
#
# Note: TP=8 in the paper to fit the agent + tool harness on one node;
# this script defaults to TP=4 (single node) — override with TP=8 GPUS=0,1,2,3,4,5,6,7
# if you have 8×H100.
#
# Run:    bash scripts/17_qwen_taubench.sh
# Tmux:   tmux new -s qwen-tau 'bash scripts/17_qwen_taubench.sh'

set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

PY="${PY:-/opt/conda/envs/scalable/bin/python}"
MODEL_QWEN="${MODEL_QWEN:-/mnt/nvme_tmp10/qwen2.5-72b}"
GPUS="${GPUS:-0,1,2,3}"
TP="${TP:-4}"
N_TASKS="${N_TASKS:-50}"
WARMUP_N="${WARMUP_N:-5}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_TURNS="${MAX_TURNS:-15}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

OUT_DIR="${REPO_ROOT}/results/serving/qwen72b"
LOG="${OUT_DIR}/taubench.log"
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
echo "Qwen2.5-72B τ-Bench sweep (airline + retail), n=${N_TASKS}/split, τ_0=${TAU}"
echo "================================================================"

run_static() {
    local env="$1" cfg="$2" out="$3"
    echo "[$(date -u +%H:%M:%S)] τ-Bench/${env}  ${cfg}" | tee -a "${LOG}"
    "${PY}" -m evaluation.serving.run_session15_70b_taubench \
        --model "${MODEL_QWEN}" \
        --tensor-parallel-size "${TP}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --max-turns "${MAX_TURNS}" \
        --n-tasks "${N_TASKS}" \
        --env "${env}" \
        --configs "${cfg}" \
        --out "${out}" 2>&1 | tee -a "${LOG}"
}

run_echokv() {
    local env="$1" out="$2"
    echo "[$(date -u +%H:%M:%S)] τ-Bench/${env}  EchoKV (τ=${TAU})" | tee -a "${LOG}"
    "${PY}" -m evaluation.serving.run_session15_70b_taubench_echokv \
        --model "${MODEL_QWEN}" \
        --tensor-parallel-size "${TP}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --max-turns "${MAX_TURNS}" \
        --n-tasks "${N_TASKS}" \
        --env "${env}" \
        --warmup-n "${WARMUP_N}" \
        --multi-precision \
        --absolute-canary \
        --kref-window-size 64 \
        --tau-budget "${TAU}" \
        --continuous-dynamic \
        --out "${out}" 2>&1 | tee -a "${LOG}"
}

# Airline split
run_static  airline fp16        "${OUT_DIR}/taubench_airline_fp16.json"
run_static  airline static_int4 "${OUT_DIR}/taubench_airline_int4.json"
run_static  airline static_int8 "${OUT_DIR}/taubench_airline_int8.json"
run_echokv  airline             "${OUT_DIR}/taubench_airline_echokv.json"

# Retail split
run_static  retail  fp16        "${OUT_DIR}/taubench_retail_fp16.json"
run_static  retail  static_int4 "${OUT_DIR}/taubench_retail_int4.json"
run_static  retail  static_int8 "${OUT_DIR}/taubench_retail_int8.json"
run_echokv  retail              "${OUT_DIR}/taubench_retail_echokv.json"

echo "[$(date -u +%H:%M:%S)] DONE τ-Bench" | tee -a "${LOG}"
