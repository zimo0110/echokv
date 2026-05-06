#!/usr/bin/env bash
# τ-Bench airline (50 tasks) full static-baseline sweep:
#   FP16 → INT4 → INT8 → NVFP4
#
# Each cell needs a fresh vLLM process because the static-quant hooks
# are sticky in-process. ~50 min wall per cell × 4 cells ≈ ~3.5 hours
# total at 70B + TP=8.
#
# Run via:
#   bash scripts/10_taubench_static_sweep.sh
# or in tmux for SSH durability:
#   tmux new -s taubench 'bash scripts/10_taubench_static_sweep.sh'

set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

STAMP=$(date -u +%Y%m%d_%H%M%S)
LOG_DIR="/tmp/taubench_sweep_${STAMP}"
OUT_DIR="results/serving/session15_70b"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

PY="/opt/conda/envs/scalable/bin/python"

run_cell() {
    local cfg="$1"
    local out_name="$2"
    local out="${OUT_DIR}/taubench_airline_${out_name}.json"
    local log="${LOG_DIR}/${out_name}.log"
    echo "================================================================"
    echo "Cell: ${cfg}  →  ${out}"
    echo "  start: $(date -u +%H:%M:%S)"
    echo "================================================================"
    "${PY}" -m evaluation.serving.run_session15_70b_taubench \
        --tensor-parallel-size 8 \
        --max-model-len 16384 \
        --n-tasks 50 \
        --max-turns 15 \
        --max-new-tokens 512 \
        --max-num-seqs 8 \
        --configs "${cfg}" \
        --seed 0 \
        --out "${out}" 2>&1 | tee "${log}"
    "${PY}" -c "
import json
b = json.load(open('${out}'))
c = b['cells'][0]
print(f'  ${cfg}: acc={c[\"accuracy\"]:.3f} n={c[\"n_prompts\"]} '
      f'total_decode={c[\"total_output_tokens\"]} wall={c[\"wall_clock_s\"]:.0f}s')
"
}

echo "================================================================"
echo "τ-Bench airline static-baseline sweep — n=50, max-turns=15"
echo "  CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "  log dir = ${LOG_DIR}"
echo "  out dir = ${OUT_DIR}"
echo "  start   = $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

run_cell fp16        fp16_n50
run_cell static_int4 int4_n50
run_cell static_int8 int8_n50
run_cell nvfp4       nvfp4_n50

echo
echo "================================================================"
echo "All cells complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo
echo "Aggregate summary (paired McNemar of EchoKV-relevant comparisons):"
"${PY}" << 'EOF'
import json, math
from pathlib import Path

base = Path("results/serving/session15_70b")

cells = {}
for cfg, name in [("FP16", "fp16_n50"), ("INT4", "int4_n50"),
                  ("INT8", "int8_n50"), ("NVFP4", "nvfp4_n50")]:
    p = base / f"taubench_airline_{name}.json"
    if not p.exists():
        print(f"  {cfg:<6}: missing")
        continue
    b = json.load(open(p))
    c = b["cells"][0]
    cells[cfg] = c
    print(f"  {cfg:<6}: acc={c['accuracy']:.3f}  n={c['n_prompts']}  "
          f"total_decode={c['total_output_tokens']}  wall={c['wall_clock_s']:.0f}s")

print()
def per_req(c):
    return {r["request_id"]: r["correct"] for r in c["per_request"]}

def mcnemar(a, b):
    common = set(a) & set(b)
    bw = sum(1 for k in common if b[k] < a[k])
    cw = sum(1 for k in common if b[k] > a[k])
    n = bw + cw
    if n == 0: return 0, 0, 1.0, 0.0
    k = max(bw, cw)
    p = sum(math.comb(n, i) * 0.5**n for i in range(k, n+1))
    return bw, cw, min(2*p, 1.0), (cw - bw) / max(len(common), 1)

print("Paired McNemar:")
if "FP16" in cells:
    fp = per_req(cells["FP16"])
    for ref_name in ["INT4", "INT8", "NVFP4"]:
        if ref_name not in cells: continue
        bw, cw, p, d = mcnemar(fp, per_req(cells[ref_name]))
        print(f"  {ref_name:<5} vs FP16   Δ={d:+.4f}  (b/c={bw}/{cw}, p={p:.4f})")
EOF
echo "================================================================"
