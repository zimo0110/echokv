#!/usr/bin/env bash
# Run all 4 Qwen workload sweeps split across 2 parallel TP=4 lanes.
#
# Lane A  (GPUs 0,1,2,3):  RULER@128K  →  HotpotQA      (~5.5 h estimated)
# Lane B  (GPUs 4,5,6,7):  τ-Bench     →  GSM8K-Hard    (~4.5 h estimated)
#
# Pairs the long workloads with short ones to roughly balance wall time.
# Each lane runs its two workloads sequentially in one tmux session, so
# only one process touches its GPUs at a time (avoids OOM contention).
#
# Pre-req: scripts/17_qwen_calibrate.sh has produced
#          results/serving/qwen72b/tau_budget.txt
#
# Run:   bash scripts/17_qwen_sweep_parallel.sh
# Watch: tmux ls
#        tmux attach -t qwen-lane-a    (Ctrl-b d to detach)
#        tmux attach -t qwen-lane-b
# Tail:  tail -f /home/nazar/echokv/results/serving/qwen72b/*.log

set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

# ─── Tunables ───
GPUS_LANE_A="${GPUS_LANE_A:-0,1,2,3}"
GPUS_LANE_B="${GPUS_LANE_B:-4,5,6,7}"
TP="${TP:-4}"
MODEL_QWEN="${MODEL_QWEN:-/mnt/nvme_tmp10/qwen2.5-72b}"

# ─── Pre-flight ───
TAU_FILE="${REPO_ROOT}/results/serving/qwen72b/tau_budget.txt"
[ -f "${TAU_FILE}" ] || { echo "ERROR: ${TAU_FILE} missing — run scripts/17_qwen_calibrate.sh first"; exit 1; }
[ -d "${MODEL_QWEN}" ] || { echo "ERROR: model dir not found: ${MODEL_QWEN}"; exit 1; }
command -v tmux >/dev/null || { echo "ERROR: tmux not found"; exit 1; }
for sess in qwen-lane-a qwen-lane-b; do
    if tmux has-session -t "${sess}" 2>/dev/null; then
        echo "ERROR: tmux session '${sess}' already exists. Kill it first:"
        echo "  tmux kill-session -t ${sess}"
        exit 1
    fi
done

# ─── Build per-lane runner scripts ───
LANE_DIR="/tmp/qwen_sweep_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "${LANE_DIR}"

write_lane() {
    local out="$1"; shift
    local devices="$1"; shift
    local lane_name="$1"; shift
    {
        echo '#!/usr/bin/env bash'
        echo 'set -euo pipefail'
        echo "cd ${REPO_ROOT}"
        echo "export MODEL_QWEN='${MODEL_QWEN}'"
        echo "export GPUS='${devices}'"
        echo "export TP=${TP}"
        echo
        echo "echo '================================================================'"
        echo "echo '  ${lane_name} starting on GPUs ${devices}, TP=${TP}'"
        echo "echo '  \$(date -u +%Y-%m-%dT%H:%M:%SZ)'"
        echo "echo '================================================================'"
        echo
        for wkl_script in "$@"; do
            echo "echo"
            echo "echo '>>>  [\$(date -u +%H:%M:%S)]  ${lane_name}: ${wkl_script}  <<<'"
            echo "bash '${REPO_ROOT}/scripts/${wkl_script}' || { echo 'FAIL: ${wkl_script}'; exit 1; }"
        done
        echo
        echo "echo '================================================================'"
        echo "echo '  ${lane_name} ALL DONE  \$(date -u +%Y-%m-%dT%H:%M:%SZ)'"
        echo "echo '================================================================'"
    } > "${out}"
    chmod +x "${out}"
}

write_lane "${LANE_DIR}/lane_a.sh" "${GPUS_LANE_A}" "Lane A" \
    17_qwen_ruler.sh \
    17_qwen_hotpotqa.sh

write_lane "${LANE_DIR}/lane_b.sh" "${GPUS_LANE_B}" "Lane B" \
    17_qwen_taubench.sh \
    17_qwen_gsm8k_hard.sh

# ─── Launch ───
echo "================================================================"
echo "Qwen2.5-72B 4-workload sweep, 2 parallel TP=${TP} lanes"
echo "  τ_0   = $(cat "${TAU_FILE}")"
echo "  Lane A on GPUs ${GPUS_LANE_A}: RULER@128K → HotpotQA"
echo "  Lane B on GPUs ${GPUS_LANE_B}: τ-Bench    → GSM8K-Hard"
echo "  runners: ${LANE_DIR}/"
echo "================================================================"

tmux new-session -d -s qwen-lane-a "bash '${LANE_DIR}/lane_a.sh'; echo; echo '*** Lane A finished. press enter to close.'; read -r _"
tmux new-session -d -s qwen-lane-b "bash '${LANE_DIR}/lane_b.sh'; echo; echo '*** Lane B finished. press enter to close.'; read -r _"

echo
tmux ls
echo
echo "Attach:  tmux attach -t qwen-lane-a    (Ctrl-b d to detach)"
echo "         tmux attach -t qwen-lane-b"
echo "Tail:    tail -f ${REPO_ROOT}/results/serving/qwen72b/*.log"
