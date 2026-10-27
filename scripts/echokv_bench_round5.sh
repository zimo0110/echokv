#!/usr/bin/env bash
# Round 5 on a CUDA box (after scripts/echokv_gpu_validate.sh set up .venv-gpu).
# Measures the decode-kernel rework + autotuning, the leaner write path, and
# long contexts / tight memory budgets. Pick steps with STEPS (default: all):
#
#   MODEL=unsloth/Llama-3.1-8B-Instruct bash scripts/echokv_bench_round5.sh
#   STEPS="unit profile latency" bash scripts/echokv_bench_round5.sh     # ~35 min
#
#   unit      unit tests: kernels compile + match the oracle on this GPU   ~1 min
#   profile   stock vs EchoKV INT4, 64 x (1K in, 128 out)                  ~8 min
#   latency   serving, 8K prompts, 16 and 32 users, stock vs INT4          ~25 min
#   ctx64k    throughput, 8 x 64K prompts, 16 GiB KV, stock vs INT4        ~10 min
#   ctx128k   throughput, 4 x ~128K prompts, 24 GiB KV, stock vs INT4      ~15 min
#   budget    quality: live controller (tau 0.005) under 4- and 6-bit budgets ~10 min
#
# Results: results/echokv_gpu/<model>/round5_*.
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-unsloth/Llama-3.1-8B-Instruct}
TP=${TP:-1}
STEPS=${STEPS:-unit profile latency ctx64k ctx128k budget}
OUT=results/echokv_gpu/$(basename "$MODEL")
mkdir -p "$OUT"
[ -x .venv-gpu/bin/python ] || { echo "run scripts/echokv_gpu_validate.sh first (sets up .venv-gpu)"; exit 1; }
PY=.venv-gpu/bin/python
export PATH="$PWD/.venv-gpu/bin:$PATH" VLLM_ALLOW_INSECURE_SERIALIZATION=1
FAILED=()

want() { [[ " $STEPS " == *" $1 "* ]]; }
run() {  # run <name> <command...>
  local name=$1; shift
  echo "== $name ($(date +%H:%M))"
  if "$@" 2>&1 | tee "$OUT/round5_$name.txt"; then :; else FAILED+=("$name"); fi
}

want unit && run unit $PY -m pytest -q tests/echokv_vllm

want profile && run profile $PY evaluation/serving/echokv_profile.py --model "$MODEL" \
  --kv-cache-gib 8 --num-prompts 64 --input-len 1024 --output-len 128 \
  --out "$OUT/round5_profile.json"

want latency && run latency $PY evaluation/serving/echokv_latency.py --model "$MODEL" --tp "$TP" \
  --kv-cache-gib 8 --input-len 8192 --output-len 256 --concurrency 16 32 \
  --configs stock echokv-int4 --out "$OUT/round5_latency.json"

# 16 GiB holds 2 of these 64K requests at bf16 and ~6 at INT4.
want ctx64k && run ctx64k $PY evaluation/serving/echokv_throughput.py --model "$MODEL" --tp "$TP" \
  --kv-cache-gib 16 --num-prompts 8 --input-len 65536 --output-len 256 \
  --configs stock echokv-int4 --out "$OUT/round5_ctx64k.json"

# Llama-3.1 allows 131,072 positions; 130,048 + 256 out leaves headroom.
# 24 GiB holds 1 of these at bf16 and ~4 at INT4.
want ctx128k && run ctx128k $PY evaluation/serving/echokv_throughput.py --model "$MODEL" --tp "$TP" \
  --kv-cache-gib 24 --num-prompts 4 --input-len 130048 --output-len 256 \
  --configs stock echokv-int4 --out "$OUT/round5_ctx128k.json"

# The controller wants INT8 for some layers at tau 0.005; a 4-bit budget
# forces those blocks down (demotion / lower tiers), a 6-bit one partly.
want budget && run budget $PY evaluation/serving/echokv_quality_sweep.py --model "$MODEL" --tp "$TP" \
  --dtype bfloat16 --kv-cache-mib 8192 --passages 32 --passage-tokens 2048 \
  --static 4 --taus 0.005 --mem-bits 4 6 8 --quiet --out "$OUT/round5_budget.json"

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All done ($(date +%H:%M)). Results: $OUT/round5_*"
else
  echo "Failed: ${FAILED[*]} (logs: $OUT/round5_*.txt)"
  exit 1
fi
