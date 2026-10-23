#!/usr/bin/env bash
# τ-Bench RETAIL EchoKV-3C sweep — frozen + continuous-dynamic.
# Companion to scripts/14 (retail static baselines) and scripts/11
# (airline EchoKV).
#
# Phase-1 warmup uses the first 5 retail tasks (canary at p_c=0.25,
# tier_application=True, controller adapts). Phase-3 measurement
# runs on the remaining 110 tasks. With airline's existing 45
# measurement tasks + retail's 110, EchoKV has 155 paired tasks
# total — same n as the static baselines after merging.
#
# Run via:
#   tmux new -s ekv-retail 'bash scripts/15_taubench_retail_echokv_sweep.sh'

set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
RETAIL_N_TASKS="${RETAIL_N_TASKS:-50}"
RETAIL_WARMUP_N="${RETAIL_WARMUP_N:-5}"

STAMP=$(date -u +%Y%m%d_%H%M%S)
LOG_DIR="/tmp/taubench_retail_echokv_${STAMP}"
OUT_DIR="results/serving/session15_70b"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

PY="/opt/conda/envs/scalable/bin/python"

run_echokv_cell() {
    local protocol_flag="$1"
    local out_name="$2"
    local out="${OUT_DIR}/taubench_retail_${out_name}.json"
    local log="${LOG_DIR}/${out_name}.log"
    echo "================================================================"
    echo "Cell: ${out_name}  →  ${out}"
    echo "  start: $(date -u +%H:%M:%S)"
    echo "================================================================"
    "${PY}" -m evaluation.serving.run_session15_70b_taubench_echokv \
        --tensor-parallel-size 8 \
        --max-model-len 16384 \
        --env retail \
        --n-tasks "${RETAIL_N_TASKS}" \
        --warmup-n "${RETAIL_WARMUP_N}" \
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

N="${RETAIL_N_TASKS}"
SUFFIX="n${N}"

echo "================================================================"
echo "τ-Bench RETAIL EchoKV-3C sweep — n=${N} (${RETAIL_WARMUP_N} warmup + $((N-RETAIL_WARMUP_N)) measure)"
echo "  CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "  log dir = ${LOG_DIR}"
echo "  out dir = ${OUT_DIR}"
echo "  start   = $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

run_echokv_cell ""                       "echokv_frozen_${SUFFIX}"
run_echokv_cell "--continuous-dynamic"   "echokv_dynamic_${SUFFIX}"

echo
echo "================================================================"
echo "Both retail EchoKV cells complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo
echo "Cross-cell summary on retail-only (then merged airline+retail "
echo "below) — EchoKV vs static baselines:"
"${PY}" - "${SUFFIX}" << 'EOF'
import json, math, sys
from pathlib import Path

retail_suffix = sys.argv[1]
base = Path("results/serving/session15_70b")
def load(p):
    if not p.exists(): return None
    b = json.load(open(p))
    if "measurement" in b:
        c = b["measurement"]
        c["bits"] = b["memory_projection"]["avg_bits"]
        c["tiers"] = b["memory_projection"]["tier_counts"]
    else:
        c = b["cells"][0]
    return c

def per_req(c):
    return {r["request_id"]: int(r["correct"]) for r in c["per_request"]}

def mcnemar(a, b):
    common = set(a) & set(b)
    bw = sum(1 for k in common if b[k] < a[k])
    cw = sum(1 for k in common if b[k] > a[k])
    n = bw + cw
    if n == 0: return 0, 0, 1.0, 0.0, len(common)
    k = max(bw, cw)
    p = sum(math.comb(n, i) * 0.5**n for i in range(k, n+1))
    return bw, cw, min(2*p, 1.0), (cw - bw) / max(len(common), 1), len(common)

# Cell map: name → (retail filename, airline filename)
specs = [
    ("FP16",       f"fp16_{retail_suffix}",           "fp16_n50"),
    ("INT4",       f"int4_{retail_suffix}",           "int4_n50"),
    ("INT8",       f"int8_{retail_suffix}",           "int8_n50"),
    ("NVFP4",      f"nvfp4_{retail_suffix}",          "nvfp4_n50"),
    ("EchoKV-frz", f"echokv_frozen_{retail_suffix}",  "echokv_frozen_n50"),
    ("EchoKV-dyn", f"echokv_dynamic_{retail_suffix}", "echokv_dynamic_n50"),
]
cells_retail, cells_merged = {}, {}
for label, retail_name, airline_name in specs:
    c_retail = load(base / f"taubench_retail_{retail_name}.json")
    c_airline = load(base / f"taubench_airline_{airline_name}.json")
    if c_retail is not None:
        cells_retail[label] = c_retail
    merged = {}
    if c_airline: merged.update(per_req(c_airline))
    if c_retail: merged.update(per_req(c_retail))
    if merged:
        cells_merged[label] = merged

print(f"  {'cell':<12} {'retail-only':>20} {'merged airline+retail':>26}")
for label, *_ in specs:
    rs = "—"
    if label in cells_retail:
        c = cells_retail[label]
        n_corr = sum(r["correct"] for r in c["per_request"])
        rs = f"{c['accuracy']:.3f} ({n_corr}/{len(c['per_request'])})"
    ms = "—"
    if label in cells_merged:
        m = cells_merged[label]
        n_corr = sum(m.values())
        ms = f"{n_corr/max(len(m),1):.3f} ({n_corr}/{len(m)})"
    print(f"  {label:<12} {rs:>20} {ms:>26}")

print()
print("Merged airline+retail paired McNemar (the headline numbers for the paper):")
fp = cells_merged.get("FP16", {})
for label in ["INT4", "INT8", "NVFP4", "EchoKV-frz", "EchoKV-dyn"]:
    other = cells_merged.get(label, {})
    if not fp or not other: continue
    bw, cw, p, d, n = mcnemar(fp, other)
    print(f"  {label:<11} vs FP16  Δ={d:+.4f}  (b/c={bw}/{cw}, p={p:.4f}, paired n={n})")

# Also EchoKV vs NVFP4 — the Pareto headline
print()
nv = cells_merged.get("NVFP4", {})
for label in ["EchoKV-frz", "EchoKV-dyn"]:
    other = cells_merged.get(label, {})
    if not nv or not other: continue
    bw, cw, p, d, n = mcnemar(nv, other)
    print(f"  {label:<11} vs NVFP4 Δ={d:+.4f}  (b/c={bw}/{cw}, p={p:.4f}, paired n={n})")
EOF
echo "================================================================"
