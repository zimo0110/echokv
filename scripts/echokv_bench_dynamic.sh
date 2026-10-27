#!/usr/bin/env bash
# Dynamic EchoKV (live controller, no pinned tier) on a CUDA box, after
# scripts/echokv_gpu_validate.sh set up .venv-gpu. The controller picks each
# layer's tier from the canary, inside a 4-bit or 6-bit memory budget.
# Pinned INT4 runs alongside as a same-session reference; stock vLLM numbers
# come from round 5 on the same machine (STOCK=1 reruns them, +~25 min).
#
#   MODEL=unsloth/Llama-3.1-8B-Instruct bash scripts/echokv_bench_dynamic.sh      # ~45 min
#
#   unit      unit tests + real-vLLM e2e (new decode kernels compile)         ~3 min
#   profile   64 x (1K in, 128 out): pinned INT4 vs live, canary cost        ~5 min
#   short     256 x (1K in, 256 out), 8 GiB KV                               ~6 min
#   ctx32k    16 x (32K in, 256 out), 8 GiB KV                               ~15 min
#   latency   serving, 8K prompts, 16 and 32 users, 8 GiB KV                 ~16 min
#
# Results: results/echokv_gpu/<model>/dynamic_*.
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-unsloth/Llama-3.1-8B-Instruct}
STEPS=${STEPS:-unit profile short ctx32k latency}
OUT=results/echokv_gpu/$(basename "$MODEL")
mkdir -p "$OUT"
[ -x .venv-gpu/bin/python ] || { echo "run scripts/echokv_gpu_validate.sh first (sets up .venv-gpu)"; exit 1; }
PY=.venv-gpu/bin/python
export PATH="$PWD/.venv-gpu/bin:$PATH" VLLM_ALLOW_INSECURE_SERIALIZATION=1
FAILED=()
LIVE="echokv-live-mem4 echokv-live-mem6"
STOCK_CFG=$([ "${STOCK:-0}" = 1 ] && echo stock)

want() { [[ " $STEPS " == *" $1 "* ]]; }
run() {  # run <name> <command...>
  local name=$1; shift
  echo "== $name ($(date +%H:%M))"
  if "$@" 2>&1 | tee "$OUT/dynamic_$name.txt"; then :; else FAILED+=("$name"); fi
}

want unit && run unit bash -c "$PY -m pytest -q tests/echokv_vllm && $PY -m pytest -q -m slow tests/echokv_vllm/test_vllm_e2e.py"

want profile && run profile $PY evaluation/serving/echokv_profile.py --model "$MODEL" \
  --kv-cache-gib 8 --num-prompts 64 --input-len 1024 --output-len 128 \
  --configs echokv-int4 echokv-live-mem4 --out "$OUT/dynamic_profile.json"

want short && run short $PY evaluation/serving/echokv_throughput.py --model "$MODEL" \
  --kv-cache-gib 8 --num-prompts 256 --input-len 1024 --output-len 256 \
  --configs $STOCK_CFG echokv-int4 $LIVE --out "$OUT/dynamic_short.json"

want ctx32k && run ctx32k $PY evaluation/serving/echokv_throughput.py --model "$MODEL" \
  --kv-cache-gib 8 --num-prompts 16 --input-len 32768 --output-len 256 \
  --configs $STOCK_CFG echokv-int4 $LIVE --out "$OUT/dynamic_ctx32k.json"

want latency && run latency $PY evaluation/serving/echokv_latency.py --model "$MODEL" \
  --kv-cache-gib 8 --input-len 8192 --output-len 256 --concurrency 16 32 \
  --configs $STOCK_CFG $LIVE --out "$OUT/dynamic_latency.json"

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All done ($(date +%H:%M)). Results: $OUT/dynamic_*"
else
  echo "Failed: ${FAILED[*]} (logs: $OUT/dynamic_*.txt)"
  exit 1
fi
