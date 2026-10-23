#!/usr/bin/env bash
# τ-Bench RETAIL static-baseline sweep — FP16 → INT4 → INT8 → NVFP4
#
# Companion to scripts/10_taubench_static_sweep.sh (airline). Does NOT
# re-run airline; only runs the retail env. The exit summary merges
# the new retail per-task records with the existing airline JSONs to
# print airline+retail paired McNemar.
#
# Default RETAIL_N_TASKS=50: gives 95 paired tasks total when merged
# with the existing 45 airline measurement tasks (~1.45× more data,
# SE ~0.69× of airline-only). 50 is the sweet spot — going to the
# full 115 only buys ~30% additional SE reduction at 2× the wall.
# Override with: RETAIL_N_TASKS=115 bash scripts/14_...
#
# Wall: ~5 h at default n=50 (4 cells × ~1.2 h each).
#
# Each cell needs a fresh vLLM process because the static-quant hooks
# are sticky in-process. ~1.5–3 h wall per cell at 115 tasks (FP16 is
# fastest, INT/NVFP4 cells take 2.5–3× longer per-turn due to BF16-
# emulated quantize/dequantize). Total: ~10–12 h.
#
# Run via:
#   tmux new -s tau-retail 'bash scripts/14_taubench_retail_static_sweep.sh'

set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
RETAIL_N_TASKS="${RETAIL_N_TASKS:-50}"   # override with env var if needed

STAMP=$(date -u +%Y%m%d_%H%M%S)
LOG_DIR="/tmp/taubench_retail_sweep_${STAMP}"
OUT_DIR="results/serving/session15_70b"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

PY="/opt/conda/envs/scalable/bin/python"

run_cell() {
    local cfg="$1"
    local out_name="$2"
    local out="${OUT_DIR}/taubench_retail_${out_name}.json"
    local log="${LOG_DIR}/${out_name}.log"
    echo "================================================================"
    echo "Cell: ${cfg}  →  ${out}"
    echo "  start: $(date -u +%H:%M:%S)"
    echo "================================================================"
    "${PY}" -m evaluation.serving.run_session15_70b_taubench \
        --tensor-parallel-size 8 \
        --max-model-len 16384 \
        --env retail \
        --n-tasks "${RETAIL_N_TASKS}" \
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

N="${RETAIL_N_TASKS}"
SUFFIX="n${N}"

echo "================================================================"
echo "τ-Bench RETAIL static-baseline sweep — n=${N}, max-turns=15, TP=8"
echo "  CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "  log dir = ${LOG_DIR}"
echo "  out dir = ${OUT_DIR}"
echo "  start   = $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

run_cell fp16        "fp16_${SUFFIX}"
run_cell static_int4 "int4_${SUFFIX}"
run_cell static_int8 "int8_${SUFFIX}"
run_cell nvfp4       "nvfp4_${SUFFIX}"

echo
echo "================================================================"
echo "All retail static cells complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo
echo "Aggregate summary (retail-only + airline+retail merged):"
"${PY}" - "${SUFFIX}" << 'EOF'
import json, math, sys
from pathlib import Path

retail_suffix = sys.argv[1]
base = Path("results/serving/session15_70b")

def load(p):
    if not p.exists(): return None
    b = json.load(open(p))
    if "measurement" in b: return b["measurement"]
    return b["cells"][0]

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

# (label, retail filename, airline filename)
names = [("FP16",  f"fp16_{retail_suffix}",  "fp16_n50"),
         ("INT4",  f"int4_{retail_suffix}",  "int4_n50"),
         ("INT8",  f"int8_{retail_suffix}",  "int8_n50"),
         ("NVFP4", f"nvfp4_{retail_suffix}", "nvfp4_n50")]

print(f"  {'cell':<6} {'retail':>20} {'airline':>20} {'merged':>20}")
merged = {}
for label, retail_name, airline_name in names:
    c_retail = load(base / f"taubench_retail_{retail_name}.json")
    c_airline = load(base / f"taubench_airline_{airline_name}.json")
    rs = f"{c_retail['accuracy']:.3f} ({sum(r['correct'] for r in c_retail['per_request'])}/{len(c_retail['per_request'])})" if c_retail else "(missing)"
    as_ = f"{c_airline['accuracy']:.3f} ({sum(r['correct'] for r in c_airline['per_request'])}/{len(c_airline['per_request'])})" if c_airline else "(missing)"
    m = {}
    if c_airline: m.update(per_req(c_airline))
    if c_retail: m.update(per_req(c_retail))
    merged[label] = m
    n_correct = sum(m.values())
    ms = f"{n_correct/max(len(m),1):.3f} ({n_correct}/{len(m)})" if m else "—"
    print(f"  {label:<6} {rs:>20} {as_:>20} {ms:>20}")

print()
print("Merged (airline+retail) paired McNemar vs FP16:")
fp = merged.get("FP16", {})
for label in ["INT4", "INT8", "NVFP4"]:
    other = merged.get(label, {})
    if not fp or not other: continue
    bw, cw, p, d, n = mcnemar(fp, other)
    print(f"  {label:<6} vs FP16  Δ={d:+.4f}  (b/c={bw}/{cw}, p={p:.4f}, paired n={n})")
EOF
echo "================================================================"
