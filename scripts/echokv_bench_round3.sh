#!/usr/bin/env bash
# Round 3 on a CUDA box (after scripts/echokv_gpu_validate.sh set up .venv-gpu):
#   1. profile: where per-step time goes, stock vs EchoKV INT4
#   2. 32K-token contexts: throughput at fixed KV memory (stock / INT8 / INT4)
#   3. serving latency: TTFT / TPOT / ITL at rising concurrency, 8K prompts
#
#   MODEL=unsloth/Llama-3.1-8B-Instruct bash scripts/echokv_bench_round3.sh
#
# Results: results/echokv_gpu/<model>/round3_*. ~1 hour on a DGX Spark.
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-unsloth/Llama-3.1-8B-Instruct}
TP=${TP:-1}
OUT=results/echokv_gpu/$(basename "$MODEL")
mkdir -p "$OUT"
[ -x .venv-gpu/bin/python ] || { echo "run scripts/echokv_gpu_validate.sh first (sets up .venv-gpu)"; exit 1; }
PY=.venv-gpu/bin/python
export PATH="$PWD/.venv-gpu/bin:$PATH" VLLM_ALLOW_INSECURE_SERIALIZATION=1
FAILED=()

run() {  # run <name> <command...>
  local name=$1; shift
  echo "== $name ($(date +%H:%M))"
  if "$@" 2>&1 | tee "$OUT/round3_$name.txt"; then :; else FAILED+=("$name"); fi
}

run unit_tests $PY -m pytest -q tests/echokv_vllm

run profile $PY evaluation/serving/echokv_profile.py --model "$MODEL" \
  --kv-cache-gib 8 --num-prompts 64 --input-len 1024 --output-len 128 \
  --out "$OUT/round3_profile.json"

# 8 GiB holds 2 of these 32K requests at bf16, ~3.5 at INT8, ~6 at INT4.
run throughput_32k $PY evaluation/serving/echokv_throughput.py --model "$MODEL" --tp "$TP" \
  --kv-cache-gib 8 --num-prompts 16 --input-len 32768 --output-len 256 \
  --configs stock echokv-int8 echokv-int4 --out "$OUT/round3_throughput_32k.json"

run latency_8k $PY evaluation/serving/echokv_latency.py --model "$MODEL" --tp "$TP" \
  --kv-cache-gib 8 --input-len 8192 --output-len 256 --concurrency 4 16 32 \
  --configs stock echokv-int4 --out "$OUT/round3_latency_8k.json"

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All done ($(date +%H:%M)). Results: $OUT/round3_*"
else
  echo "Failed: ${FAILED[*]} (logs: $OUT/round3_*.txt)"
  exit 1
fi
