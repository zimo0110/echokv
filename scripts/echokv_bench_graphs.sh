#!/usr/bin/env bash
# Decode steps as full CUDA graphs (round 7) on a CUDA box, after
# scripts/echokv_gpu_validate.sh set up .venv-gpu. The H100 profile showed
# EchoKV CPU-bound: the GPU idled ~2/3 of the time while Python launched
# per-layer work. Now the builder does all host work once per step and
# decode-only steps replay one CUDA graph. The question is whether wall
# time now tracks GPU-busy time.
#
#   MODEL=unsloth/Llama-3.1-8B-Instruct bash scripts/echokv_bench_graphs.sh      # ~35 min
#
#   unit      unit tests + real-vLLM e2e, eager and with CUDA graphs          ~5 min
#   profile   64 x (1K in, 128 out): stock vs INT4 vs live, wall vs GPU busy  ~6 min
#   short     256 x (1K in, 256 out), 8 GiB KV: stock, INT4, live            ~8 min
#   latency   serving, 8K prompts, 16 and 32 users, 8 GiB KV                 ~16 min
#
# Results: results/echokv_gpu/<model>/graphs_*.
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-unsloth/Llama-3.1-8B-Instruct}
STEPS=${STEPS:-unit profile short latency}
OUT=results/echokv_gpu/$(basename "$MODEL")
mkdir -p "$OUT"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv > "$OUT/gpus.txt" 2>/dev/null || true
# PY=... uses a Python that already has vLLM + this plugin (e.g. on Modal).
PY=${PY:-.venv-gpu/bin/python}
[ -x "$PY" ] || { echo "run scripts/echokv_gpu_validate.sh first (sets up .venv-gpu)"; exit 1; }
export PATH="$PWD/.venv-gpu/bin:$PATH" VLLM_ALLOW_INSECURE_SERIALIZATION=1
FAILED=()

want() { [[ " $STEPS " == *" $1 "* ]]; }
run() {  # run <name> <command...>
  local name=$1; shift
  echo "== $name ($(date +%H:%M))"
  if "$@" 2>&1 | tee "$OUT/graphs_$name.txt"; then :; else FAILED+=("$name"); fi
}

want unit && run unit bash -c "$PY -m pytest -q tests/echokv_vllm && $PY -m pytest -q -m slow tests/echokv_vllm/test_vllm_e2e.py"

want profile && run profile $PY evaluation/serving/echokv_profile.py --model "$MODEL" \
  --kv-cache-gib 8 --num-prompts 64 --input-len 1024 --output-len 128 \
  --configs stock echokv-int4 echokv-live-mem4 --out "$OUT/graphs_profile.json" \
  --trace-dir "$OUT/graphs_profile_traces"

want short && run short $PY evaluation/serving/echokv_throughput.py --model "$MODEL" \
  --kv-cache-gib 8 --num-prompts 256 --input-len 1024 --output-len 256 \
  --configs stock echokv-int4 echokv-live-mem4 --out "$OUT/graphs_short.json"

want latency && run latency $PY evaluation/serving/echokv_latency.py --model "$MODEL" \
  --kv-cache-gib 8 --input-len 8192 --output-len 256 --concurrency 16 32 \
  --configs stock echokv-int4 echokv-live-mem4 --out "$OUT/graphs_latency.json"

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All done ($(date +%H:%M)). Results: $OUT/graphs_*"
else
  echo "Failed: ${FAILED[*]} (logs: $OUT/graphs_*.txt)"
  exit 1
fi
