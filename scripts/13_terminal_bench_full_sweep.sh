#!/usr/bin/env bash
# Terminal-Bench full sweep — drives the custom hook-aware server
# (evaluation/serving/vllm_server_with_hooks.py) across all 6 cells:
#   FP16, INT4, INT8, NVFP4, EchoKV-3C frozen, EchoKV-3C dynamic.
#
# Each cell:
#   1. Spins up the custom server with --config <cfg> on GPUs.
#   2. Waits for /v1/models to come up.
#   3. Runs `tb run --agent terminus --model openai/...` against it.
#   4. Saves results under results/serving/session16_70b/terminalbench_<cfg>_<stamp>/.
#   5. Tears down the server (SIGTERM → SIGKILL fallback).
#
# Run in tmux:
#   tmux new -s tbench-full 'bash scripts/13_terminal_bench_full_sweep.sh'
#
# Wall: ~8-12 hours total (each cell ~1-2 h on Terminal-Bench-Core
# v0.1.1's ~100 tasks; EchoKV cells add ~5-10 min of warmup).

set -euo pipefail
cd "$(dirname "$0")/.."

# ----------------------------- Config -----------------------------
MODEL_PATH="${MODEL_PATH:-/mnt/nvme_tmp6/llama-70b}"
MODEL_NAME="${MODEL_NAME:-llama-3.1-70b}"
TP_SIZE="${TP_SIZE:-8}"
VLLM_PORT="${VLLM_PORT:-8000}"
DATASET="${DATASET:-terminal-bench-core==0.1.1}"
N_CONCURRENT="${N_CONCURRENT:-1}"
N_ATTEMPTS="${N_ATTEMPTS:-1}"

# Restrict to a few tasks for smoke testing if SMOKE_TASK_IDS is set.
SMOKE_TASK_IDS="${SMOKE_TASK_IDS:-}"

# Comma-separated list; default = all 6 cells.
CELLS="${CELLS:-fp16,static_int4,static_int8,nvfp4,echokv_3c_frozen,echokv_3c_dynamic}"

# EchoKV warmup file — defaults to a small synthetic prompt set we
# generate the first time this script runs (see ensure_warmup_jsonl).
ECHOKV_WARMUP_JSONL="${ECHOKV_WARMUP_JSONL:-results/serving/session16_70b/echokv_warmup_prompts.jsonl}"
ECHOKV_TAU_BUDGET="${ECHOKV_TAU_BUDGET:-0.022}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

PY="/opt/conda/envs/scalable/bin/python"
PIP="/opt/conda/envs/scalable/bin/pip"

STAMP=$(date -u +%Y%m%d_%H%M%S)
LOG_DIR="/tmp/tbench_full_${STAMP}"
RESULTS_BASE="results/serving/session16_70b"
mkdir -p "${LOG_DIR}" "${RESULTS_BASE}"

echo "================================================================"
echo "Terminal-Bench full 6-cell sweep"
echo "  cells           = ${CELLS}"
echo "  dataset         = ${DATASET}"
echo "  log dir         = ${LOG_DIR}"
echo "  start           = $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

# ----------------------- Prerequisite checks ----------------------
if ! command -v docker > /dev/null || ! docker ps > /dev/null 2>&1; then
    echo "ERROR: docker not reachable. Terminal-Bench requires Docker." >&2
    exit 1
fi
# terminal-bench requires Python 3.12+; install via uv.
if [[ ! -d /tmp/terminal-bench ]]; then
    git clone --depth=1 https://github.com/laude-institute/terminal-bench /tmp/terminal-bench
fi
TB_BIN="${HOME}/.local/bin/tb"
if [[ ! -x "${TB_BIN}" ]]; then
    if ! command -v uv > /dev/null; then
        echo "ERROR: uv not on PATH" >&2; exit 1
    fi
    uv tool install /tmp/terminal-bench --python 3.12 2>&1 | tail -3
fi
[[ -x "${TB_BIN}" ]] || { echo "ERROR: tb CLI not found at ${TB_BIN}"; exit 1; }

# ----------------- Synthesize EchoKV warmup if missing ------------
ensure_warmup_jsonl() {
    if [[ -f "${ECHOKV_WARMUP_JSONL}" ]]; then return 0; fi
    echo "Generating synthetic EchoKV warmup prompts: ${ECHOKV_WARMUP_JSONL}"
    mkdir -p "$(dirname "${ECHOKV_WARMUP_JSONL}")"
    "${PY}" - "${ECHOKV_WARMUP_JSONL}" << 'EOF'
import json, sys
out = sys.argv[1]
prompts = [
    "Write a bash one-liner that finds the 5 largest files under /var/log.",
    "What's the difference between `kill -TERM` and `kill -KILL`? Answer in 3 sentences.",
    "Explain how to set up a TLS-terminating nginx reverse proxy for a backend on port 8080.",
    "I have a python service that's leaking memory. Walk me through how to diagnose it with py-spy.",
    "Sketch a shell script that backs up /home to /backup, rotating 7 days of snapshots.",
]
with open(out, "w") as f:
    for p in prompts:
        f.write(json.dumps({"messages": [
            {"role": "system", "content": "You are a helpful Linux systems engineer. Be concise."},
            {"role": "user", "content": p},
        ]}) + "\n")
print(f"  wrote {len(prompts)} warmup prompts")
EOF
}
ensure_warmup_jsonl

# ----------------------- Per-cell runner --------------------------
launch_server_for_cell() {
    local cfg="$1"
    local log="$2"

    local extra_args=()
    if [[ "${cfg}" == "echokv_3c_frozen" || "${cfg}" == "echokv_3c_dynamic" ]]; then
        extra_args+=( "--echokv-warmup-jsonl" "${ECHOKV_WARMUP_JSONL}"
                      "--tau-budget" "${ECHOKV_TAU_BUDGET}" )
    fi

    "${PY}" -m evaluation.serving.vllm_server_with_hooks \
        --model "${MODEL_PATH}" \
        --served-model-name "${MODEL_NAME}" \
        --tensor-parallel-size "${TP_SIZE}" \
        --max-model-len 16384 \
        --gpu-memory-utilization 0.85 \
        --port "${VLLM_PORT}" \
        --host 127.0.0.1 \
        --config "${cfg}" \
        "${extra_args[@]}" \
        > "${log}" 2>&1 &
    echo $!
}

wait_for_endpoint() {
    local timeout=600  # 10 min — EchoKV warmup adds time
    for i in $(seq 1 $((timeout / 5))); do
        if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        sleep 5
    done
    return 1
}

teardown_server() {
    local pid="$1"
    kill -TERM "${pid}" 2>/dev/null || true
    sleep 8
    kill -KILL "${pid}" 2>/dev/null || true
    pkill -KILL -f "Worker_TP\|EngineCore" 2>/dev/null || true
    sleep 3
}

run_cell() {
    local cfg="$1"
    local cell_log="${LOG_DIR}/server_${cfg}.log"
    local tb_log="${LOG_DIR}/tb_${cfg}.log"
    local out="${RESULTS_BASE}/terminalbench_${cfg}_${STAMP}"
    mkdir -p "${out}"

    echo "================================================================"
    echo "Cell: ${cfg}  →  ${out}"
    echo "  start = $(date -u +%H:%M:%S)"
    echo "================================================================"

    local pid
    pid=$(launch_server_for_cell "${cfg}" "${cell_log}")
    echo "  vLLM PID=${pid}, log=${cell_log}"

    if ! wait_for_endpoint; then
        echo "  ERROR: endpoint never came up. Tail of server log:" >&2
        tail -40 "${cell_log}" >&2
        teardown_server "${pid}"
        return 1
    fi
    echo "  endpoint healthy"

    # Record the served config metadata
    curl -s "http://127.0.0.1:${VLLM_PORT}/v1/echokv/status" \
        > "${out}/echokv_status.json" || true

    local tb_args=( "run"
        "--agent" "terminus"
        "--model" "openai/${MODEL_NAME}"
        "--dataset" "${DATASET}"
        "--n-concurrent" "${N_CONCURRENT}"
        "--n-attempts" "${N_ATTEMPTS}"
        "--output-path" "${out}"
    )
    if [[ -n "${SMOKE_TASK_IDS}" ]]; then
        for tid in ${SMOKE_TASK_IDS//,/ }; do
            tb_args+=( "--task-id" "${tid}" )
        done
    fi

    OPENAI_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1" \
    OPENAI_API_KEY="dummy-not-checked" \
    "${TB_BIN}" "${tb_args[@]}" 2>&1 | tee "${tb_log}" || true

    teardown_server "${pid}"

    echo "  cell ${cfg} done at $(date -u +%H:%M:%S)"
    echo
}

# ----------------------- Drive the sweep --------------------------
trap 'pkill -KILL -f "Worker_TP\|EngineCore\|vllm_server_with_hooks" 2>/dev/null || true' EXIT INT TERM

for cell in ${CELLS//,/ }; do
    run_cell "${cell}"
done

# ----------------------- Cross-cell summary -----------------------
echo "================================================================"
echo "All cells complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo
echo "Per-cell summary:"
"${PY}" - "${RESULTS_BASE}" "${STAMP}" << 'EOF'
import json, sys
from pathlib import Path
base, stamp = Path(sys.argv[1]), sys.argv[2]
for cell_dir in sorted(base.glob(f"terminalbench_*_{stamp}")):
    name = cell_dir.name.replace(f"_{stamp}", "").replace("terminalbench_", "")
    pass_at_1 = None
    n_total = None
    for p in sorted(cell_dir.rglob("*.json")):
        try:
            b = json.loads(p.read_text())
            if isinstance(b, dict) and "tasks" in b:
                n_total = len(b["tasks"])
                pass_at_1 = sum(1 for t in b["tasks"] if t.get("passed"))
                break
        except Exception:
            pass
    if pass_at_1 is not None:
        print(f"  {name:<22}: pass@1 = {pass_at_1}/{n_total} = {pass_at_1/max(n_total,1):.3f}")
    else:
        print(f"  {name:<22}: (no results JSON found)")
EOF
echo "================================================================"
