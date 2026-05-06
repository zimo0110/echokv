#!/usr/bin/env bash
# τ-Bench airline EchoKV-3C sweep — frozen-tier then continuous-dynamic.
# Companion to scripts/10 (the static FP16/INT4/INT8/NVFP4 baselines).
#
# Both cells run the 3-phase warmup → consolidate → measure protocol:
#   - Phase 1: first 5 airline tasks with canary observing at p_c=0.25,
#     tier_application=True, freeze_controller=False (controller adapts)
#   - Phase 2: per-layer max-across-ranks tier consolidation
#   - Phase 3: remaining 45 tasks measured. Frozen vs --continuous-dynamic
#     differ in whether the canary stays live + controller stays unfrozen.
#
# Run via:
#   tmux new -s echokv-tau 'bash scripts/11_taubench_echokv_sweep.sh'

set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

STAMP=$(date -u +%Y%m%d_%H%M%S)
LOG_DIR="/tmp/taubench_echokv_${STAMP}"
OUT_DIR="results/serving/session15_70b"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

PY="/opt/conda/envs/scalable/bin/python"

run_echokv_cell() {
    local protocol_flag="$1"
    local out_name="$2"
    local out="${OUT_DIR}/taubench_airline_${out_name}.json"
    local log="${LOG_DIR}/${out_name}.log"
    echo "================================================================"
    echo "Cell: ${out_name}  →  ${out}"
    echo "  start: $(date -u +%H:%M:%S)"
    echo "================================================================"
    "${PY}" -m evaluation.serving.run_session15_70b_taubench_echokv \
        --tensor-parallel-size 8 \
        --max-model-len 16384 \
        --n-tasks 50 \
        --warmup-n 5 \
        --max-turns 15 \
        --max-new-tokens 512 \
        --max-num-seqs 8 \
        --multi-precision \
        --absolute-canary \
        --kref-window-size 64 \
        --tau-budget 0.022 \
        --observability-rate 0.25 \
        --tick-interval 10 \
        ${protocol_flag} \
        --seed 0 \
        --out "${out}" 2>&1 | tee "${log}"
    "${PY}" -c "
import json
b = json.load(open('${out}'))
m = b['measurement']
proj = b['memory_projection']
print(f'  echokv_3c (${out_name}):')
print(f'    measurement acc={m[\"accuracy\"]:.3f} n={m[\"n_prompts\"]} '
      f'wall={m[\"wall_clock_s\"]:.0f}s')
print(f'    avg_bits (consolidated) = {proj[\"avg_bits\"]:.2f}')
print(f'    tier_counts = {proj[\"tier_counts\"]}')
"
}

echo "================================================================"
echo "τ-Bench airline EchoKV-3C sweep — n=50 (5 warmup + 45 measure)"
echo "  CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "  log dir = ${LOG_DIR}"
echo "  out dir = ${OUT_DIR}"
echo "  start   = $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

run_echokv_cell ""                       echokv_frozen_n50
run_echokv_cell "--continuous-dynamic"   echokv_dynamic_n50

echo
echo "================================================================"
echo "Both cells complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo
echo "Cross-cell summary (EchoKV vs the static baselines on the same n=50):"
"${PY}" << 'EOF'
import json, math
from collections import Counter
from pathlib import Path

base = Path("results/serving/session15_70b")
cells = {}
for cfg, name in [("FP16",      "fp16_n50"),
                  ("INT4",      "int4_n50"),
                  ("INT8",      "int8_n50"),
                  ("NVFP4",     "nvfp4_n50"),
                  ("EchoKV-frz","echokv_frozen_n50"),
                  ("EchoKV-dyn","echokv_dynamic_n50")]:
    p = base / f"taubench_airline_{name}.json"
    if not p.exists():
        print(f"  {cfg:<11}: missing")
        continue
    b = json.load(open(p))
    if "measurement" in b:
        c = b["measurement"]
        c["bits"] = b["memory_projection"]["avg_bits"]
        c["tiers"] = b["memory_projection"]["tier_counts"]
    else:
        c = b["cells"][0]
    cells[cfg] = c

print(f"  {'cell':<12} {'acc':>5} {'n':>3} {'avg_decode':>11}")
for cfg, c in cells.items():
    avg_d = sum(r["decode_tokens"] for r in c["per_request"]) / max(len(c["per_request"]), 1)
    print(f"  {cfg:<12} {c['accuracy']:>5.3f} {len(c['per_request']):>3}    {avg_d:>8.0f}"
          + (f"  bits={c.get('bits', '-')}" if c.get('bits') else "")
          + (f"  tiers={c.get('tiers')}" if c.get('tiers') else ""))

def per_req(c):
    return {r["request_id"]: int(r["correct"]) for r in c["per_request"]}

def mcnemar(a, b):
    common = set(a) & set(b)
    bw = sum(1 for k in common if b[k] < a[k])
    cw = sum(1 for k in common if b[k] > a[k])
    n = bw + cw
    if n == 0: return 0, 0, 1.0, 0.0
    k = max(bw, cw)
    p = sum(math.comb(n, i) * 0.5**n for i in range(k, n+1))
    return bw, cw, min(2*p, 1.0), (cw - bw) / max(len(common), 1)

print()
print("Paired McNemar (EchoKV cells vs FP16 / INT4 / NVFP4):")
if "FP16" in cells:
    fp = per_req(cells["FP16"])
    for cell_name in ["EchoKV-frz", "EchoKV-dyn"]:
        if cell_name not in cells: continue
        e = per_req(cells[cell_name])
        for ref_name, ref in [("FP16", fp), ("INT4", per_req(cells.get("INT4", {"per_request": []}))),
                               ("NVFP4", per_req(cells.get("NVFP4", {"per_request": []})))]:
            if not ref: continue
            bw, cw, p, d = mcnemar(ref, e)
            print(f"  {cell_name} vs {ref_name:<6}  Δ={d:+.4f}  (b/c={bw}/{cw}, p={p:.4f})")

# Trajectory-divergence metrics: Hamming distance vs FP16
def action_seq(per_req_list, task_id):
    for r in per_req_list:
        if r["request_id"] == task_id:
            return [(a["name"], tuple(sorted(a["kwargs"].keys())))
                    for a in r["actions"]]
    return []

def hamming(a, b):
    n = max(len(a), len(b))
    if n == 0: return 0
    matches = sum(1 for x, y in zip(a, b) if x == y)
    return n - matches

print()
print("Mean per-task action-sequence Hamming distance vs FP16:")
fp_pr = cells["FP16"]["per_request"]
task_ids = [r["request_id"] for r in fp_pr]
for cfg in ["INT4", "INT8", "NVFP4", "EchoKV-frz", "EchoKV-dyn"]:
    if cfg not in cells: continue
    other_pr = cells[cfg]["per_request"]
    distances = [hamming(action_seq(fp_pr, tid), action_seq(other_pr, tid))
                 for tid in task_ids]
    n_identical = sum(1 for d in distances if d == 0)
    print(f"  {cfg:<11}: mean_hamming = {sum(distances)/max(len(distances), 1):.2f}  "
          f"identical_traces = {n_identical}/{len(distances)}")
EOF
echo "================================================================"
