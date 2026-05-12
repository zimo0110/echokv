#!/usr/bin/env bash
# τ_0 calibration on Qwen2.5-72B-Instruct.
#
# Runs the same per-tier static probe used for the 70B Llama calibration
# in the paper, but on Qwen so each layer's INT4 EMA reflects Qwen's own
# quantization sensitivity. Sequential RULER@4K then GSM8K-Hard so we can
# verify τ_0 is stable across calibration workloads (paper used RULER@4K).
#
# Output:  results/serving/qwen72b/tau_budget_calib.txt   <- one float, used by workload runners
#          results/serving/qwen72b/calib_ruler_4k.json
#          results/serving/qwen72b/calib_gsm8k_hard.json
#
# Pre-req: Qwen2.5-72B-Instruct weights on disk. If you don't have them yet:
#   huggingface-cli download Qwen/Qwen2.5-72B-Instruct --local-dir /mnt/nvme_tmp10/qwen2.5-72b
# Then export MODEL_QWEN to that path (default below assumes that location).
#
# Run:    bash scripts/17_qwen_calibrate.sh
# Tmux:   tmux new -s qwen-calib 'bash scripts/17_qwen_calibrate.sh'

set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

PY="${PY:-/opt/conda/envs/scalable/bin/python}"
MODEL_QWEN="${MODEL_QWEN:-/mnt/nvme_tmp10/qwen2.5-72b}"
GPUS="${GPUS:-0,1,2,3}"
TP="${TP:-4}"
N_PROMPTS="${N_PROMPTS:-20}"

OUT_DIR="${REPO_ROOT}/results/serving/qwen72b"
LOG="${OUT_DIR}/calib.log"
mkdir -p "${OUT_DIR}"

[ -d "${MODEL_QWEN}" ] || { echo "ERROR: model dir not found: ${MODEL_QWEN}"; exit 1; }
[ -x "${PY}" ] || { echo "ERROR: python not executable at ${PY}"; exit 1; }

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

echo "================================================================"
echo "τ_0 calibration on Qwen2.5-72B-Instruct"
echo "  model  = ${MODEL_QWEN}"
echo "  TP     = ${TP}  (GPUs ${GPUS})"
echo "  n      = ${N_PROMPTS}  (per probe)"
echo "  out    = ${OUT_DIR}"
echo "================================================================"

# ─── RULER@4K calibration (matches paper recipe) ───
echo "[$(date -u +%H:%M:%S)] RULER@4K calibration ..." | tee -a "${LOG}"
"${PY}" -m evaluation.serving.calibrate_tau_budget \
    --model "${MODEL_QWEN}" \
    --tensor-parallel-size "${TP}" \
    --target-tokens 4000 \
    --n-prompts "${N_PROMPTS}" \
    --max-new-tokens 24 \
    --max-model-len 8192 \
    --canary-rate 0.5 \
    --out "${OUT_DIR}/calib_ruler_4k.json" 2>&1 | tee -a "${LOG}"

# ─── GSM8K-Hard calibration (cross-check) ───
echo "[$(date -u +%H:%M:%S)] GSM8K-Hard calibration ..." | tee -a "${LOG}"
"${PY}" -m evaluation.serving.calibrate_tau_budget_70b_gsm8k_hard \
    --model "${MODEL_QWEN}" \
    --tensor-parallel-size "${TP}" \
    --max-model-len 4096 \
    --max-num-seqs 8 \
    --n-prompts "${N_PROMPTS}" \
    --canary-rate 0.5 \
    --out "${OUT_DIR}/calib_gsm8k_hard.json" 2>&1 | tee -a "${LOG}"

# ─── Extract τ_0 = layer-averaged δ̂_4 EMA mean from RULER@4K static-INT4 probe ───
TAU_FILE="${OUT_DIR}/tau_budget.txt"
"${PY}" - <<EOF | tee -a "${LOG}"
import json, sys
ruler = json.load(open("${OUT_DIR}/calib_ruler_4k.json"))
gsm   = json.load(open("${OUT_DIR}/calib_gsm8k_hard.json"))

def int4_mean(blob):
    p = next(p for p in blob["probes"] if p["static_bits"] == 4)
    return p["per_tier_stats"]["4"]["ema_mean"] if "4" in p["per_tier_stats"] else p["per_tier_stats"][4]["ema_mean"]

t_ruler = int4_mean(ruler)
t_gsm   = int4_mean(gsm)
print(f"RULER@4K     τ_0 = {t_ruler:.5f}")
print(f"GSM8K-Hard   τ_0 = {t_gsm:.5f}")
print(f"ratio        = {t_gsm/t_ruler:.3f}")
# Use RULER@4K as canonical (matches paper's calibration)
with open("${TAU_FILE}", "w") as f:
    f.write(f"{t_ruler:.5f}\n")
print(f"wrote ${TAU_FILE}: {t_ruler:.5f}")
EOF

echo
echo "================================================================"
echo "Calibration done.  τ_0 written to ${TAU_FILE}"
echo "Sanity: if RULER and GSM8K-Hard τ_0 differ by >2× the controller may"
echo "behave differently between the workloads — investigate before sweeping."
echo "================================================================"
