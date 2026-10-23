#!/usr/bin/env bash
# 8B validation + hyperparameter ablation — concrete-recommendation launcher.
#
# Why this script exists.
#   Two reviewer concerns are not yet addressed in the paper:
#     (1) Single model — every result is on Llama-3.1-70B-Instruct.
#     (2) No (alpha, p_c) sensitivity — the canary EMA decay alpha and the
#         production canary rate p_c are both fixed without an ablation.
#   The 4-hour fix:
#     A) Run the 4-config headline on Llama-3.1-8B-Instruct GSM8K-Hard
#        (FP16, KIVI-INT4, Static INT8, EchoKV) at n=500 mns=8. ~1h on TP=2.
#     B) Same on HotpotQA. ~1h on TP=2.
#     C) Ablation: alpha in {0.05, 0.10, 0.20} x p_c in {1/32, 1/64, 1/128}
#        on 8B GSM8K-Hard at n=200, split across 2 parallel TP=2 lanes.
#
# Layout (edit GPU_* below if you have a different GPU count):
#   tmux session "8b-gsm8k"           GPUs 0,1
#   tmux session "8b-hotpot"          GPUs 2,3
#   tmux session "8b-ablation-a"      GPUs 4,5  (5 sweep configs)
#   tmux session "8b-ablation-b"      GPUs 6,7  (4 sweep configs)
#
# Resumable: if any of those sessions are already alive (e.g. you previously
# launched and one survived), this script will skip them and only start the
# missing ones.
#
# Run:    bash scripts/16_8b_validation_run.sh
# Watch:  tmux attach -t 8b-gsm8k       (Ctrl-b d to detach)
# Logs:   tail -f /tmp/8b_validation_*/<job>.log

set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
PY="${PY:-/opt/conda/envs/scalable/bin/python}"
MODEL_8B="${MODEL_8B:-/mnt/nvme_tmp/nazar_hf/hub/models--NousResearch--Meta-Llama-3.1-8B-Instruct/snapshots/d10aef7999a2b5ba950ab3974312feeedbfe0b77}"

GPU_GSM8K="${GPU_GSM8K:-0,1}"
GPU_HOTPOT="${GPU_HOTPOT:-2,3}"
GPU_ABLATION_LANE_A="${GPU_ABLATION_LANE_A:-4,5}"
GPU_ABLATION_LANE_B="${GPU_ABLATION_LANE_B:-6,7}"

TP_SIZE="${TP_SIZE:-2}"
HEADLINE_N="${HEADLINE_N:-500}"
ABLATION_N="${ABLATION_N:-200}"
WARMUP_N="${WARMUP_N:-16}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_MODEL_LEN_GSM8K="${MAX_MODEL_LEN_GSM8K:-4096}"
MAX_MODEL_LEN_HOTPOT="${MAX_MODEL_LEN_HOTPOT:-8192}"

TAU_BUDGET="${TAU_BUDGET:-0.022}"
ALPHA_DEFAULT="${ALPHA_DEFAULT:-0.10}"
PC_DEFAULT="${PC_DEFAULT:-0.015625}"   # 1/64

ALPHAS=("0.05" "0.10" "0.20")
PCS=("0.03125" "0.015625" "0.0078125") # 1/32, 1/64, 1/128

# ---------------------------------------------------------------------------
# Output dirs (reuse the existing run dir if there is exactly one, else stamp)
# ---------------------------------------------------------------------------
existing_dirs=( /tmp/8b_validation_*/ )
if [ ${#existing_dirs[@]} -eq 1 ] && [ -d "${existing_dirs[0]}" ] && [ -z "${FORCE_NEW_LOG_DIR:-}" ]; then
    LOG_DIR="${existing_dirs[0]%/}"
    STAMP="$(basename "${LOG_DIR}" | sed 's/^8b_validation_//')"
    echo "Reusing existing log dir: ${LOG_DIR}"
else
    STAMP="$(date -u +%Y%m%d_%H%M%S)"
    LOG_DIR="/tmp/8b_validation_${STAMP}"
    mkdir -p "${LOG_DIR}"
    echo "Created log dir: ${LOG_DIR}"
fi
OUT_DIR="${REPO_ROOT}/results/serving/session16_8b"
ABL_DIR="${OUT_DIR}/ablation"
mkdir -p "${OUT_DIR}" "${ABL_DIR}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
write_runner() {
    # write_runner <output_path> <devices> <body...>
    local out="$1"; shift
    local devices="$1"; shift
    {
        echo '#!/usr/bin/env bash'
        echo 'set -euo pipefail'
        echo "cd ${REPO_ROOT}"
        echo 'export VLLM_ALLOW_INSECURE_SERIALIZATION=1'
        echo 'export FLASHINFER_DISABLE_VERSION_CHECK=1'
        echo "export PYTHONPATH=${REPO_ROOT}:\${PYTHONPATH:-}"
        echo "export CUDA_VISIBLE_DEVICES=${devices}"
        echo
        cat
    } > "${out}" <<EOF_END
$*
EOF_END
    chmod +x "${out}"
}

start_or_skip() {
    # start_or_skip <session_name> <runner_script>
    local sess="$1"
    local script="$2"
    if tmux has-session -t "${sess}" 2>/dev/null; then
        echo "  [skip] tmux session '${sess}' already exists"
        return 0
    fi
    tmux new-session -d -s "${sess}" "bash '${script}'; echo; echo '*** ${sess} finished. press enter to close.'; read -r _"
    echo "  [start] tmux session '${sess}' -> bash ${script}"
}

# ---------------------------------------------------------------------------
# Build the runner scripts
# ---------------------------------------------------------------------------
GSM8K_RUNNER="${LOG_DIR}/run_gsm8k.sh"
cat > "${GSM8K_RUNNER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd ${REPO_ROOT}
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTHONPATH=${REPO_ROOT}:\${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${GPU_GSM8K}
LOG="${LOG_DIR}/gsm8k.log"

echo "=== [\$(date -u +%H:%M:%S)] 8B GSM8K-Hard / FP16 + KIVI-INT4 (n=${HEADLINE_N}) ===" | tee -a "\${LOG}"
${PY} -m evaluation.serving.run_session14_70b_gsm8k_hard \\
    --model "${MODEL_8B}" \\
    --tensor-parallel-size ${TP_SIZE} \\
    --max-model-len ${MAX_MODEL_LEN_GSM8K} \\
    --max-num-seqs ${MAX_NUM_SEQS} \\
    --n-prompts ${HEADLINE_N} \\
    --configs fp16 static_int4 \\
    --out "${OUT_DIR}/gsm8k_hard_static.json" 2>&1 | tee -a "\${LOG}"

echo "=== [\$(date -u +%H:%M:%S)] 8B GSM8K-Hard / Static INT8 (n=${HEADLINE_N}) ===" | tee -a "\${LOG}"
${PY} -m evaluation.serving.run_session14_70b_gsm8k_hard \\
    --model "${MODEL_8B}" \\
    --tensor-parallel-size ${TP_SIZE} \\
    --max-model-len ${MAX_MODEL_LEN_GSM8K} \\
    --max-num-seqs ${MAX_NUM_SEQS} \\
    --n-prompts ${HEADLINE_N} \\
    --configs static_int8 \\
    --out "${OUT_DIR}/gsm8k_hard_int8.json" 2>&1 | tee -a "\${LOG}"

echo "=== [\$(date -u +%H:%M:%S)] 8B GSM8K-Hard / EchoKV (n=${HEADLINE_N}) ===" | tee -a "\${LOG}"
${PY} -m evaluation.serving.run_session14_70b_gsm8k_hard_echokv \\
    --model "${MODEL_8B}" \\
    --tensor-parallel-size ${TP_SIZE} \\
    --max-model-len ${MAX_MODEL_LEN_GSM8K} \\
    --max-num-seqs ${MAX_NUM_SEQS} \\
    --n-prompts ${HEADLINE_N} \\
    --warmup-n ${WARMUP_N} \\
    --multi-precision \\
    --absolute-canary \\
    --kref-window-size 64 \\
    --tau-budget ${TAU_BUDGET} \\
    --ema-alpha ${ALPHA_DEFAULT} \\
    --measurement-canary-rate ${PC_DEFAULT} \\
    --continuous-dynamic \\
    --out "${OUT_DIR}/gsm8k_hard_echokv.json" 2>&1 | tee -a "\${LOG}"

echo "=== [\$(date -u +%H:%M:%S)] 8B GSM8K-Hard ALL DONE ===" | tee -a "\${LOG}"
EOF
chmod +x "${GSM8K_RUNNER}"

HOTPOT_RUNNER="${LOG_DIR}/run_hotpot.sh"
cat > "${HOTPOT_RUNNER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd ${REPO_ROOT}
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTHONPATH=${REPO_ROOT}:\${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${GPU_HOTPOT}
LOG="${LOG_DIR}/hotpot.log"

echo "=== [\$(date -u +%H:%M:%S)] 8B HotpotQA / FP16 + KIVI-INT4 (n=${HEADLINE_N}) ===" | tee -a "\${LOG}"
${PY} -m evaluation.serving.run_session14_70b_hotpotqa \\
    --model "${MODEL_8B}" \\
    --tensor-parallel-size ${TP_SIZE} \\
    --max-model-len ${MAX_MODEL_LEN_HOTPOT} \\
    --max-num-seqs ${MAX_NUM_SEQS} \\
    --n-prompts ${HEADLINE_N} \\
    --configs fp16 static_int4 \\
    --out "${OUT_DIR}/hotpotqa_static.json" 2>&1 | tee -a "\${LOG}"

echo "=== [\$(date -u +%H:%M:%S)] 8B HotpotQA / Static INT8 (n=${HEADLINE_N}) ===" | tee -a "\${LOG}"
${PY} -m evaluation.serving.run_session14_70b_hotpotqa \\
    --model "${MODEL_8B}" \\
    --tensor-parallel-size ${TP_SIZE} \\
    --max-model-len ${MAX_MODEL_LEN_HOTPOT} \\
    --max-num-seqs ${MAX_NUM_SEQS} \\
    --n-prompts ${HEADLINE_N} \\
    --configs static_int8 \\
    --out "${OUT_DIR}/hotpotqa_int8.json" 2>&1 | tee -a "\${LOG}"

echo "=== [\$(date -u +%H:%M:%S)] 8B HotpotQA / EchoKV (n=${HEADLINE_N}) ===" | tee -a "\${LOG}"
${PY} -m evaluation.serving.run_session14_70b_hotpotqa_echokv \\
    --model "${MODEL_8B}" \\
    --tensor-parallel-size ${TP_SIZE} \\
    --max-model-len ${MAX_MODEL_LEN_HOTPOT} \\
    --max-num-seqs ${MAX_NUM_SEQS} \\
    --n-prompts ${HEADLINE_N} \\
    --warmup-n ${WARMUP_N} \\
    --multi-precision \\
    --absolute-canary \\
    --kref-window-size 64 \\
    --tau-budget ${TAU_BUDGET} \\
    --ema-alpha ${ALPHA_DEFAULT} \\
    --measurement-canary-rate ${PC_DEFAULT} \\
    --continuous-dynamic \\
    --out "${OUT_DIR}/hotpotqa_echokv.json" 2>&1 | tee -a "\${LOG}"

echo "=== [\$(date -u +%H:%M:%S)] 8B HotpotQA ALL DONE ===" | tee -a "\${LOG}"
EOF
chmod +x "${HOTPOT_RUNNER}"

# ---------------------------------------------------------------------------
# Ablation lanes
# ---------------------------------------------------------------------------
build_ablation_runner() {
    local out="$1"; shift
    local devices="$1"; shift
    local logfile="$1"; shift
    local pairs=("$@")
    {
        echo '#!/usr/bin/env bash'
        echo 'set -euo pipefail'
        echo "cd ${REPO_ROOT}"
        echo 'export VLLM_ALLOW_INSECURE_SERIALIZATION=1'
        echo 'export FLASHINFER_DISABLE_VERSION_CHECK=1'
        echo "export PYTHONPATH=${REPO_ROOT}:\${PYTHONPATH:-}"
        echo "export CUDA_VISIBLE_DEVICES=${devices}"
        echo "LOG='${logfile}'"
        echo
        for pair in "${pairs[@]}"; do
            local alpha="${pair%%:*}"
            local pc="${pair##*:}"
            local pc_tag; pc_tag="$(echo "${pc}" | tr '.' 'p')"
            local out_json="${ABL_DIR}/echokv_a${alpha//./p}_pc${pc_tag}.json"
            echo "echo \"=== [\$(date -u +%H:%M:%S)] ablation alpha=${alpha} pc=${pc} ===\" | tee -a \"\${LOG}\""
            cat <<RUNJOB
${PY} -m evaluation.serving.run_session14_70b_gsm8k_hard_echokv \\
    --model "${MODEL_8B}" \\
    --tensor-parallel-size ${TP_SIZE} \\
    --max-model-len ${MAX_MODEL_LEN_GSM8K} \\
    --max-num-seqs ${MAX_NUM_SEQS} \\
    --n-prompts ${ABLATION_N} \\
    --warmup-n ${WARMUP_N} \\
    --multi-precision \\
    --absolute-canary \\
    --kref-window-size 64 \\
    --tau-budget ${TAU_BUDGET} \\
    --ema-alpha ${alpha} \\
    --measurement-canary-rate ${pc} \\
    --continuous-dynamic \\
    --out "${out_json}" 2>&1 | tee -a "\${LOG}"
RUNJOB
        done
        echo "echo \"=== [\$(date -u +%H:%M:%S)] ablation lane DONE on devices ${devices} ===\" | tee -a \"\${LOG}\""
    } > "${out}"
    chmod +x "${out}"
}

ALL_PAIRS=()
for a in "${ALPHAS[@]}"; do
    for pc in "${PCS[@]}"; do
        ALL_PAIRS+=("${a}:${pc}")
    done
done
LANE_A_PAIRS=()
LANE_B_PAIRS=()
for i in "${!ALL_PAIRS[@]}"; do
    if (( i % 2 == 0 )); then
        LANE_A_PAIRS+=("${ALL_PAIRS[$i]}")
    else
        LANE_B_PAIRS+=("${ALL_PAIRS[$i]}")
    fi
done

ABL_A_RUNNER="${LOG_DIR}/run_ablation_a.sh"
ABL_B_RUNNER="${LOG_DIR}/run_ablation_b.sh"
build_ablation_runner "${ABL_A_RUNNER}" "${GPU_ABLATION_LANE_A}" "${LOG_DIR}/ablation_lane_a.log" "${LANE_A_PAIRS[@]}"
build_ablation_runner "${ABL_B_RUNNER}" "${GPU_ABLATION_LANE_B}" "${LOG_DIR}/ablation_lane_b.log" "${LANE_B_PAIRS[@]}"

# ---------------------------------------------------------------------------
# Sanity
# ---------------------------------------------------------------------------
command -v tmux >/dev/null || { echo "ERROR: tmux not found"; exit 1; }
[ -x "${PY}" ] || { echo "ERROR: python not executable at ${PY}"; exit 1; }
[ -d "${MODEL_8B}" ] || { echo "ERROR: 8B model dir not found: ${MODEL_8B}"; exit 1; }

# ---------------------------------------------------------------------------
# Launch (skips sessions that are already alive)
# ---------------------------------------------------------------------------
echo "================================================================"
echo "8B validation + ablation launcher (${STAMP})"
echo "  model: ${MODEL_8B}"
echo "  TP:    ${TP_SIZE}"
echo "  out:   ${OUT_DIR}"
echo "  logs:  ${LOG_DIR}"
echo "  GPUs: gsm8k=${GPU_GSM8K}  hotpot=${GPU_HOTPOT}  ablA=${GPU_ABLATION_LANE_A}  ablB=${GPU_ABLATION_LANE_B}"
echo "  ablation lane A pairs: ${LANE_A_PAIRS[*]}"
echo "  ablation lane B pairs: ${LANE_B_PAIRS[*]}"
echo "================================================================"

start_or_skip 8b-gsm8k       "${GSM8K_RUNNER}"
start_or_skip 8b-hotpot      "${HOTPOT_RUNNER}"
start_or_skip 8b-ablation-a  "${ABL_A_RUNNER}"
start_or_skip 8b-ablation-b  "${ABL_B_RUNNER}"

echo
tmux ls
echo
echo "Attach:  tmux attach -t 8b-gsm8k    (Ctrl-b d to detach)"
echo "         tmux attach -t 8b-hotpot"
echo "         tmux attach -t 8b-ablation-a"
echo "         tmux attach -t 8b-ablation-b"
echo
echo "Tail:    tail -f ${LOG_DIR}/*.log"
