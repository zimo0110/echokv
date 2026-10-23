#!/usr/bin/env bash
# HotpotQA 4-config sweep at n=500, max_num_seqs=8 — same protocol
# that gave us the GSM8K-Hard Pareto headline. The point of this run
# is multi-workload generalisation: if EchoKV-3C also matches FP16 on
# a second workload at meaningful memory savings, the system is
# general.
#
# Sequential 4 cells (FP16 → INT4 → INT8 → EchoKV-3C abs-canary). Each
# loads vLLM separately. Per-cell JSONs land independently so a
# mid-sweep failure leaves earlier cells usable.
#
# Run as: bash scripts/04_hotpotqa_n500_mns8.sh
# Or in tmux:
#   tmux new -s hotpot500 'bash scripts/04_hotpotqa_n500_mns8.sh'
#
# Expected per-cell wall on 70B+TP=4 (HotpotQA = ~1200-tok prompts,
# max_new_tokens=64):
#   FP16:        ~5 min
#   static_int4: ~15 min
#   static_int8: ~15 min
#   EchoKV-3C:   ~20 min
# Total: ~50-60 min.

set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

STAMP=$(date -u +%Y%m%d_%H%M%S)
LOG_DIR="/tmp/hotpotqa_n500_${STAMP}"
mkdir -p "${LOG_DIR}"
OUT_DIR="results/serving/session14_70b/hotpotqa_n500_mns8"
mkdir -p "${OUT_DIR}"

PY="/opt/conda/envs/scalable/bin/python"

run_baseline() {
    local cfg="$1"
    local name="$2"
    local out="${OUT_DIR}/${name}_n500_mns8.json"
    local log="${LOG_DIR}/${name}.log"
    echo "----------------------------------------------------------------"
    echo "Cell ${name}: $(date -u +%H:%M:%S) → ${out}"
    echo "----------------------------------------------------------------"
    "${PY}" -m evaluation.serving.run_session14_70b_hotpotqa \
        --n-prompts 500 \
        --configs "${cfg}" \
        --max-num-seqs 8 \
        --max-new-tokens 64 \
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
    local out="${OUT_DIR}/echokv3c_abscanary_tau022_n500_mns8.json"
    local log="${LOG_DIR}/echokv3c.log"
    echo "----------------------------------------------------------------"
    echo "Cell echokv3c (abs-canary, τ=0.022): $(date -u +%H:%M:%S) → ${out}"
    echo "----------------------------------------------------------------"
    "${PY}" -m evaluation.serving.run_session14_70b_hotpotqa_echokv \
        --n-prompts 500 --warmup-n 16 \
        --multi-precision --absolute-canary --kref-window-size 64 \
        --tau-budget 0.022 \
        --max-num-seqs 8 --max-new-tokens 64 --seed 0 \
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
echo "HotpotQA 4-config sweep — n=500, max_num_seqs=8"
echo "(matches the GSM8K-Hard headline protocol)"
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
echo "HotpotQA n=500 mns=8 paired Pareto table:"
"${PY}" << EOF
import json, math
from pathlib import Path

base = Path("${OUT_DIR}")
def load(name, key='cells[0]'):
    p = base / f'{name}_n500_mns8.json'
    if not p.exists():
        return None
    b = json.load(open(p))
    if 'measurement' in b:
        m = b['measurement']
    else:
        m = b['cells'][0]
    return {r['request_id']: r['correct'] for r in m['per_request']}, m['accuracy']

fp16, fp16_acc = load('fp16') or (None, None)
int4, int4_acc = load('int4') or (None, None)
int8, int8_acc = load('int8') or (None, None)
ech_path = base / 'echokv3c_abscanary_tau022_n500_mns8.json'
ech_blob = json.load(open(ech_path)) if ech_path.exists() else {}
ech_m = ech_blob.get('measurement', {})
proj = ech_blob.get('memory_projection', {})
ech = {r['request_id']: r['correct'] for r in ech_m.get('per_request', [])} if ech_m else None
ech_acc = ech_m.get('accuracy') if ech_m else None

def show(name, acc, bits, savings):
    if acc is None:
        print(f'  {name:<22} (missing)')
    else:
        print(f'  {name:<22} acc={acc:.3f}  bits={bits:<6.2f}  {savings}')

show('FP16',          fp16_acc,        16.00, '(baseline)')
show('static_int4',   int4_acc,         4.00, '75.0% savings vs FP16')
show('static_int8',   int8_acc,         8.00, '50.0% savings vs FP16')
show('EchoKV-3C',     ech_acc, proj.get('avg_bits', float('nan')),
     f'{proj.get("savings_vs_fp16_pct", float("nan")):.1f}% savings vs FP16  '
     f'{proj.get("tier_counts")}')

def mcnemar(a, b):
    common = set(a) & set(b)
    bw = sum(1 for k in common if b[k] < a[k])
    cw = sum(1 for k in common if b[k] > a[k])
    n = bw + cw
    if n == 0:
        return 0, 0, 1.0, 0.0
    k = max(bw, cw)
    p_one = sum(math.comb(n, i) * 0.5**n for i in range(k, n+1))
    p_two = min(2 * p_one, 1.0) if n > 1 else 1.0
    delta = (cw - bw) / len(common)
    return bw, cw, p_two, delta

print()
print('Paired McNemar (same prompts):')
for ref, ref_name, cand, cand_name in [
    (fp16, 'FP16', int4,  'INT4'),
    (fp16, 'FP16', int8,  'INT8'),
    (fp16, 'FP16', ech,   'EchoKV-3C'),
    (int4, 'INT4', ech,   'EchoKV-3C'),
]:
    if ref is None or cand is None:
        continue
    bw, cw, p, d = mcnemar(ref, cand)
    print(f'  {cand_name:<10} vs {ref_name:<6}  Δ={d:+.4f}  (b/c={bw}/{cw}, p={p:.4f})')
EOF
echo "================================================================"
