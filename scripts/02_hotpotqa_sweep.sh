#!/usr/bin/env bash
# HotpotQA 4-config Pareto sweep at n=200, max_num_seqs=1.
#
# Runs FP16 → static_int4 → static_int8 → EchoKV-3C absolute-canary
# sequentially in 4 separate processes (each loads vLLM once; static-
# int hooks can't coexist with FP16 in one process). Each cell's JSON
# is saved independently, so a mid-sweep failure leaves earlier cells
# usable.
#
# Expected per-cell wall (rough, depends on prompt length distribution):
#   FP16:        ~15-25 min
#   static_int4: ~20-35 min
#   static_int8: ~20-35 min
#   EchoKV-3C:   ~20-35 min
# Total sweep: ~2-2.5 hours.
#
# Run as: bash scripts/02_hotpotqa_sweep.sh
# Or in tmux:
#   tmux new -s hotpot 'bash scripts/02_hotpotqa_sweep.sh'

set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

STAMP=$(date -u +%Y%m%d_%H%M%S)
LOG_DIR="/tmp/hotpotqa_sweep_${STAMP}"
mkdir -p "${LOG_DIR}"
OUT_DIR="results/serving/session14_70b/hotpotqa"
mkdir -p "${OUT_DIR}"

PY="/opt/conda/envs/scalable/bin/python"

run_baseline() {
    local cfg="$1"
    local name="$2"
    local out="${OUT_DIR}/${name}_n200_seq1.json"
    local log="${LOG_DIR}/${name}.log"
    echo "----------------------------------------------------------------"
    echo "Cell ${name}: $(date -u +%H:%M:%S) → ${out}"
    echo "----------------------------------------------------------------"
    "${PY}" -m evaluation.serving.run_session14_70b_hotpotqa \
        --n-prompts 200 \
        --configs "${cfg}" \
        --max-num-seqs 1 \
        --seed 0 \
        --out "${out}" 2>&1 | tee "${log}"
    "${PY}" -c "
import json
b = json.load(open('${out}'))
for c in b.get('cells', []):
    print(f'  {c[\"config\"]}: acc={c[\"accuracy\"]:.3f} wall={c[\"wall_clock_s\"]:.1f}s')
"
}

run_echokv3c() {
    local out="${OUT_DIR}/echokv3c_abscanary_tau022_n200_seq1.json"
    local log="${LOG_DIR}/echokv3c.log"
    echo "----------------------------------------------------------------"
    echo "Cell echokv3c (abs-canary, τ=0.022): $(date -u +%H:%M:%S) → ${out}"
    echo "----------------------------------------------------------------"
    "${PY}" -m evaluation.serving.run_session14_70b_hotpotqa_echokv \
        --n-prompts 200 --warmup-n 16 \
        --multi-precision --absolute-canary --kref-window-size 64 \
        --tau-budget 0.022 \
        --max-num-seqs 1 --seed 0 \
        --out "${out}" 2>&1 | tee "${log}"
    "${PY}" -c "
import json
b = json.load(open('${out}'))
m = b.get('measurement', {})
proj = b.get('memory_projection', {})
print(f'  echokv_3c: acc={m.get(\"accuracy\", float(\"nan\")):.3f} wall={m.get(\"wall_clock_s\", 0):.1f}s '
      f'avg_bits={proj.get(\"avg_bits\", float(\"nan\")):.2f} '
      f'tier_counts={proj.get(\"tier_counts\")}')
"
}

echo "================================================================"
echo "HotpotQA 4-config Pareto sweep — n=200, max_num_seqs=1"
echo "  CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "  log dir: ${LOG_DIR}"
echo "  out dir: ${OUT_DIR}"
echo "  start:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

run_baseline fp16          fp16
run_baseline static_int4   int4
run_baseline static_int8   int8
run_echokv3c

echo
echo "================================================================"
echo "All cells complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo
echo "Final HotpotQA Pareto table:"
"${PY}" -c "
import json, glob
files = sorted(glob.glob('${OUT_DIR}/*.json'))
for f in files:
    b = json.load(open(f))
    if 'cells' in b:
        for c in b['cells']:
            print(f'  {c[\"config\"]:<14}  acc={c[\"accuracy\"]:.3f}  wall={c[\"wall_clock_s\"]:>7.1f}s')
    elif 'measurement' in b:
        m = b['measurement']
        proj = b.get('memory_projection', {})
        print(f'  echokv_3c       acc={m[\"accuracy\"]:.3f}  wall={m[\"wall_clock_s\"]:>7.1f}s  '
              f'avg_bits={proj.get(\"avg_bits\", float(\"nan\")):.2f}  '
              f'tier_counts={proj.get(\"tier_counts\")}')
"
echo "================================================================"
