#!/usr/bin/env bash
# Round 2 on a CUDA box, after scripts/echokv_gpu_validate.sh has set up .venv-gpu:
#   1. INT4 quality re-check on the current code (vs stock)
#   2. throughput for all configs on the current code (short prompts)
#   3. memory-bound throughput: long prompts, many users (stock vs INT4)
#
#   MODEL=unsloth/Llama-3.1-8B-Instruct bash scripts/echokv_bench_round2.sh
#
# Results: results/echokv_gpu/<model>/round2_*.{json,txt}. ~45 min on a DGX Spark.
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
  if "$@" 2>&1 | tee "$OUT/round2_$name.txt"; then :; else FAILED+=("$name"); fi
}

run quality_int4 $PY evaluation/serving/echokv_quality_sweep.py --model "$MODEL" --tp "$TP" \
  --dtype bfloat16 --kv-cache-mib 8192 --passages 32 --passage-tokens 2048 \
  --static 4 --taus --quiet --out "$OUT/round2_quality_int4.json"

run throughput_short $PY evaluation/serving/echokv_throughput.py --model "$MODEL" --tp "$TP" \
  --kv-cache-gib 8 --num-prompts 256 --input-len 1024 --output-len 256 \
  --out "$OUT/round2_throughput_short.json"

# 8 GiB holds ~7 of these 8K-token requests at bf16 and ~23 at INT4.
run throughput_long $PY evaluation/serving/echokv_throughput.py --model "$MODEL" --tp "$TP" \
  --kv-cache-gib 8 --num-prompts 64 --input-len 8192 --output-len 256 \
  --configs stock echokv-int4 --out "$OUT/round2_throughput_long.json"

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All done ($(date +%H:%M)). Results: $OUT/round2_*"
else
  echo "Failed: ${FAILED[*]} (logs: $OUT/round2_*.txt)"
  exit 1
fi
