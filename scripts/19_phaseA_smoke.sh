#!/usr/bin/env bash
# Phase A smoke test — run kernels/native_int4_kv parity tests on a real H100.
#
# What this validates (from tests/test_native_int4_kv.py):
#   - Random uniform inputs (float32, float16, bfloat16) × group_size {16, 32, 64}
#   - Heavy-tailed inputs with 1% outliers at 8× scale
#   - Constant-valued groups (span=0 edge case, folded-into-scale encoding)
#   - Realistic K/V tensor shapes (T=128, num_kv_heads=8, head_dim=128, bf16)
#   - Input dim not divisible by group_size → clean ValueError
#   - CPU input → clean ValueError
#   - Output dtype matches input dtype
#   - LEVELS == 15 (sanity)
# All compared bit-for-bit (≤1e-5 abs diff) against
# evaluation.motivation.kivi.simulate_quantization.
#
# Pre-req: at least one H100 with >1 GB free. Check first:
#   nvidia-smi --query-gpu=index,memory.free --format=csv
#
# Run:    GPU=0 bash scripts/19_phaseA_smoke.sh
# Tmux:   tmux new -s phaseA 'GPU=0 bash scripts/19_phaseA_smoke.sh; read'

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-/opt/conda/envs/scalable/bin/python}"
GPU="${GPU:-0}"

[ -x "${PY}" ] || { echo "ERROR: python not executable at ${PY}"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi not found"; exit 1; }

# Sanity: confirm chosen GPU exists
if ! nvidia-smi --query-gpu=index --format=csv,noheader | grep -qx "${GPU}"; then
    echo "ERROR: GPU ${GPU} not found. Available:"
    nvidia-smi --query-gpu=index,name,memory.free --format=csv
    exit 1
fi

FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU}")
echo "================================================================"
echo "Phase A smoke test (kernel parity)"
echo "  GPU       = ${GPU}  (${FREE_MIB} MiB free)"
echo "  module    = kernels.native_int4_kv"
echo "  reference = evaluation.motivation.kivi.simulate_quantization"
echo "================================================================"

if [ "${FREE_MIB}" -lt 512 ]; then
    echo "WARNING: only ${FREE_MIB} MiB free on GPU ${GPU}; tests may OOM."
    echo "         Bail out (Ctrl-C) or proceed."
    sleep 3
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo
echo "[$(date -u +%H:%M:%S)] running pytest -v ..."
"${PY}" -m pytest tests/test_native_int4_kv.py -v --tb=short --color=yes

echo
echo "================================================================"
echo "Phase A smoke PASSED  [$(date -u +%H:%M:%S)]"
echo "Next: scripts/20_phaseB_*.sh  (write-path Triton kernel)"
echo "================================================================"
