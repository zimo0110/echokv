#!/usr/bin/env bash
# Terminal-Bench launcher — Phase 11 candidate (PAPER_DRAFT §7.2).
#
# Status: **FP16 baseline cell only**. The quantized cells (INT4 /
# INT8 / NVFP4 / EchoKV-3C) are blocked on vLLM-server-side hook
# integration; see the "Blocked cells" section at the bottom.
#
# What this script does:
#   1. Installs terminal-bench (uv tool / pip) if missing.
#   2. Pulls the Terminal-Bench-Core v0.1.1 task set (the leaderboard).
#   3. Spins up vLLM's OpenAI-compatible server with Llama-3.1-70B at
#      TP=8, BF16 (i.e. FP16-equivalent KV cache).
#   4. Waits for the endpoint to be ready (polls /v1/models).
#   5. Runs tb run --agent terminus --model openai/llama-3.1-70b
#      against http://localhost:${VLLM_PORT}/v1.
#   6. Saves results to results/serving/session16_70b/terminalbench_fp16/.
#   7. Tears down the vLLM server.
#
# Run via:
#   tmux new -s tbench 'bash scripts/12_terminal_bench_launch.sh'
#
# Wall: ~1–3 hours depending on how many tasks the agent gives up on
# (Terminal-Bench-Core v0.1.1 has ~100 tasks; published Llama-70B
# scores are ~5–15% pass@1).

set -euo pipefail
cd "$(dirname "$0")/.."

# ----------------------------- Config -----------------------------
MODEL_PATH="${MODEL_PATH:-/mnt/nvme_tmp6/llama-70b}"
MODEL_NAME="${MODEL_NAME:-llama-3.1-70b}"
TP_SIZE="${TP_SIZE:-8}"
VLLM_PORT="${VLLM_PORT:-8000}"
DATASET="${DATASET:-terminal-bench-core==0.1.1}"
N_CONCURRENT="${N_CONCURRENT:-1}"   # tasks-in-parallel; >1 = more Docker containers
N_ATTEMPTS="${N_ATTEMPTS:-1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

PY="/opt/conda/envs/scalable/bin/python"
PIP="/opt/conda/envs/scalable/bin/pip"

STAMP=$(date -u +%Y%m%d_%H%M%S)
LOG_DIR="/tmp/tbench_${STAMP}"
OUT_DIR="results/serving/session16_70b/terminalbench_fp16_${STAMP}"
VLLM_LOG="${LOG_DIR}/vllm_serve.log"
TB_LOG="${LOG_DIR}/tb_run.log"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

echo "================================================================"
echo "Terminal-Bench launch (FP16 cell)"
echo "  model         = ${MODEL_PATH} (TP=${TP_SIZE})"
echo "  served as     = ${MODEL_NAME} on http://localhost:${VLLM_PORT}/v1"
echo "  dataset       = ${DATASET}"
echo "  log dir       = ${LOG_DIR}"
echo "  out dir       = ${OUT_DIR}"
echo "  start         = $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

# ----------------------- Prerequisite checks ----------------------
if ! command -v docker > /dev/null; then
    echo "ERROR: docker not found in PATH. Terminal-Bench requires Docker." >&2
    exit 1
fi
if ! docker ps > /dev/null 2>&1; then
    echo "ERROR: docker is installed but the daemon isn't reachable / no permission." >&2
    exit 1
fi

# terminal-bench requires Python 3.12+; install via `uv tool` which
# manages its own python toolchain. The clone lives at /tmp/terminal-bench
# (refreshed if missing).
if [[ ! -d /tmp/terminal-bench ]]; then
    git clone --depth=1 https://github.com/laude-institute/terminal-bench /tmp/terminal-bench
fi
if ! command -v uv > /dev/null; then
    echo "ERROR: uv not on PATH. Install from https://docs.astral.sh/uv/." >&2
    exit 1
fi
TB_BIN="${HOME}/.local/bin/tb"
if [[ ! -x "${TB_BIN}" ]]; then
    echo "Installing terminal-bench via uv (Python 3.12)..."
    uv tool install /tmp/terminal-bench --python 3.12 2>&1 | tail -3
fi
[[ -x "${TB_BIN}" ]] || { echo "ERROR: tb CLI still not found at ${TB_BIN}"; exit 1; }
echo "  tb binary     = ${TB_BIN}"

# ------------------------- Spin up vLLM server --------------------
echo
echo "[1/3] starting vLLM OpenAI server (FP16, BF16 weights)..."
echo "    log: ${VLLM_LOG}"

"${PY}" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --dtype bfloat16 \
    --enforce-eager \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --port "${VLLM_PORT}" \
    --host 127.0.0.1 \
    > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
echo "    vLLM PID=${VLLM_PID}"

cleanup() {
    echo
    echo "tearing down vLLM (PID=${VLLM_PID})..."
    kill -TERM "${VLLM_PID}" 2>/dev/null || true
    sleep 5
    kill -KILL "${VLLM_PID}" 2>/dev/null || true
    pkill -KILL -f "Worker_TP\|EngineCore" 2>/dev/null || true
    sleep 2
    nvidia-smi --query-gpu=index,memory.used --format=csv | head -2
}
trap cleanup EXIT INT TERM

# ----------------------- Wait for endpoint ------------------------
echo "[2/3] waiting for endpoint to become healthy (max 5 min)..."
for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
        echo "    ready after ${i} polls (~$((i*5))s)"
        break
    fi
    sleep 5
    if [[ $i -eq 60 ]]; then
        echo "    ERROR: endpoint never came up. Tail of vLLM log:" >&2
        tail -30 "${VLLM_LOG}" >&2
        exit 1
    fi
done

# ----------------------- Run Terminal-Bench -----------------------
echo
echo "[3/3] running Terminal-Bench..."
echo "    log: ${TB_LOG}"

OPENAI_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1" \
OPENAI_API_KEY="dummy-not-checked" \
"${TB_BIN}" run \
    --agent terminus \
    --model "openai/${MODEL_NAME}" \
    --dataset "${DATASET}" \
    --n-concurrent "${N_CONCURRENT}" \
    --n-attempts "${N_ATTEMPTS}" \
    --output-path "${OUT_DIR}" \
    2>&1 | tee "${TB_LOG}"

# ----------------------- Summary ----------------------------------
echo
echo "================================================================"
echo "Terminal-Bench FP16 cell complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  results dir: ${OUT_DIR}"
ls -la "${OUT_DIR}" | head -8
echo
"${PY}" -c "
import json
from pathlib import Path
out = Path('${OUT_DIR}')
# Terminal-Bench drops a results JSON; locate it
for p in sorted(out.rglob('*.json')):
    try:
        b = json.loads(p.read_text())
        if isinstance(b, dict) and 'tasks' in b:
            tot = len(b['tasks'])
            ok = sum(1 for t in b['tasks'] if t.get('passed'))
            print(f'  {p.relative_to(out)}: pass@1 = {ok}/{tot} = {ok/max(tot,1):.3f}')
    except Exception:
        pass
" 2>/dev/null || true
echo "================================================================"

# ===================================================================
# Blocked cells (INT4 / INT8 / NVFP4 / EchoKV-3C frozen + dynamic)
# ===================================================================
#
# To run the quantized cells we need to install our forward-pre-hooks
# inside the vLLM server. The standard vllm.entrypoints.openai.api_server
# constructs an LLM internally and doesn't expose apply_model() to user
# code. Two integration paths, in increasing order of effort:
#
#   1. Subclass the api_server entrypoint:
#      - Build a custom server module that imports api_server, calls
#        apply_model with our hook installer between LLM construction
#        and serve(), then defers to api_server's main loop.
#      - Estimate: 50-100 LOC + testing. ~1 day.
#
#   2. Add a vLLM plugin hook:
#      - vLLM has a plugin entrypoint mechanism for model-modifying
#        code. Register install_static_int4_hooks / install_nvfp4_hooks
#        as plugins, then vllm serve --quantization-plugin our.module
#        installs them before serving.
#      - Estimate: 1-2 days; involves vLLM plugin API.
#
# For Phase 11 (Terminal-Bench full sweep) we recommend path 1 because
# it's contained and matches what we already do in the in-process
# harness. The script structure for the quantized cells would be:
#
#   for cfg in fp16 static_int4 static_int8 nvfp4 echokv_frozen echokv_dynamic; do
#       PY -m evaluation.serving.vllm_server_with_hooks \
#           --hook-config ${cfg} \
#           --port ${VLLM_PORT} ... &
#       wait_for_endpoint
#       OPENAI_BASE_URL=... tb run --agent terminus --model openai/${MODEL_NAME} \
#           --output-path results/serving/session16_70b/terminalbench_${cfg}_${STAMP}
#       teardown_vllm
#   done
#
# See PAPER_DRAFT §7.2 — Terminal-Bench is listed there as a sandbox-
# blocked benchmark. Docker is no longer the blocker (it's installed
# and works); the remaining blocker is the vLLM-server hook integration.
