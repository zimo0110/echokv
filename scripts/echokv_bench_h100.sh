#!/usr/bin/env bash
# EchoKV on a 4x H100 (80 GB) machine: Llama-3.1-8B on one GPU, then
# Llama-3.3-70B with tensor parallelism 4. Ungated model mirrors, no HF token.
#
#   bash scripts/echokv_bench_h100.sh                    # everything, ~2.5 h
#   STEPS="setup unit e2e profile8b" bash scripts/echokv_bench_h100.sh   # ~15 min smoke
#
#   setup        install vLLM 0.32.0 + plugin into .venv-gpu; check driver/GPUs
#   unit         unit tests (Triton kernels compiled for sm_90, vs the oracle)
#   e2e          real vLLM boot; checks the host slot mapping on every step
#   profile8b    8B, stock vs INT4, 64 x (1K in, 128 out): is a fast GPU CPU-bound?
#   quality8b    8B WikiText ΔNLL: stock, INT8, INT4, live tau 0.005 / 0.01
#   tput8b       8B throughput, 256 x (1K in, 256 out), 8 GiB KV: stock/INT8/INT4
#   tput8b_32k   8B throughput, 16 x (32K in, 256 out), 8 GiB KV: stock/INT8/INT4/live
#   latency8b    8B serving, 8K prompts, 16/32/64 users, 8 GiB KV: stock/INT4/live
#   quality70b   70B TP=4 ΔNLL: stock, INT8, INT4, live tau 0.005 / 0.01
#   tput70b_8k   70B TP=4 throughput, 64 x (8K in, 256 out), 8 GiB KV per GPU
#   tput70b_32k  70B TP=4 throughput, 16 x (32K in, 256 out), 8 GiB KV per GPU, incl. live
#   latency70b   70B TP=4 serving, 8K prompts, 16/32 users, 8 GiB KV per GPU, incl. live
#
# "live" = dynamic EchoKV: the controller picks each layer's tier under a
# 4-bit budget (ECHOKV_MEM_BITS=4, default tau), nothing pinned.
#
# The 70B weights (~140 GB) download in the background from the start, so
# the 8B steps overlap with it. Results: results/echokv_gpu/h100_<model>/.
# A failed step is recorded and the rest still run; send the whole folder.
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL8=${MODEL8:-unsloth/Llama-3.1-8B-Instruct}
MODEL70=${MODEL70:-unsloth/Llama-3.3-70B-Instruct}
STEPS=${STEPS:-setup unit e2e profile8b quality8b tput8b tput8b_32k latency8b quality70b tput70b_8k tput70b_32k latency70b}
# H100s have dedicated memory (unlike the DGX Spark), so vLLM's free-memory
# check can use most of it; the KV size is set explicitly per step anyway.
export ECHOKV_GPU_UTIL=${ECHOKV_GPU_UTIL:-0.9}
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
OUT8=results/echokv_gpu/h100_$(basename "$MODEL8")
OUT70=results/echokv_gpu/h100_$(basename "$MODEL70")
mkdir -p "$OUT8" "$OUT70"
FAILED=()

want() { [[ " $STEPS " == *" $1 "* ]]; }
run() {  # run <out dir> <name> <command...>
  local out=$1 name=$2; shift 2
  echo "== $name ($(date +%H:%M))"
  if "$@" 2>&1 | tee "$out/$name.txt"; then :; else FAILED+=("$name"); fi
}

if want setup || [ ! -x .venv-gpu/bin/python ]; then
  SETUP_ONLY=1 bash scripts/echokv_gpu_validate.sh || exit 1
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv | tee "$OUT8/gpus.txt" "$OUT70/gpus.txt"
fi
PY=.venv-gpu/bin/python
export PATH="$PWD/.venv-gpu/bin:$HOME/.local/bin:$PATH"
NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# -- 70B download, in the background --------------------------------------
DL_PID=
if [[ "$STEPS" == *70b* ]]; then
  [ "$NGPU" -ge 4 ] || { echo "70B steps need 4 GPUs, found $NGPU"; exit 1; }
  FREE_GB=$(df -Pk "${HF_HOME:-$HOME/.cache/huggingface}" 2>/dev/null | awk 'NR==2 {print int($4/1048576)}')
  [ -z "$FREE_GB" ] && FREE_GB=$(df -Pk "$HOME" | awk 'NR==2 {print int($4/1048576)}')
  [ "$FREE_GB" -ge 170 ] || echo "WARNING: ~${FREE_GB} GB free for the HF cache; $MODEL70 needs ~140 GB (set HF_HOME to a bigger disk)"
  echo "== downloading $MODEL70 in the background (log $OUT70/download.txt)"
  $PY -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], allow_patterns=['*.json', '*.safetensors', 'tokenizer*'])" \
    "$MODEL70" >"$OUT70/download.txt" 2>&1 &
  DL_PID=$!
fi

# -- Llama-3.1-8B, one GPU --------------------------------------------------
export CUDA_VISIBLE_DEVICES=0
want unit && run "$OUT8" unit $PY -m pytest -q tests/echokv_vllm
want e2e && run "$OUT8" e2e $PY -m pytest -q -m slow tests/echokv_vllm/test_vllm_e2e.py

want profile8b && run "$OUT8" profile8b $PY evaluation/serving/echokv_profile.py --model "$MODEL8" \
  --kv-cache-gib 8 --num-prompts 64 --input-len 1024 --output-len 128 \
  --out "$OUT8/profile8b.json" --trace-dir "$OUT8/profile8b_traces"

want quality8b && run "$OUT8" quality8b $PY evaluation/serving/echokv_quality_sweep.py --model "$MODEL8" \
  --dtype bfloat16 --kv-cache-mib 8192 --passages 32 --passage-tokens 2048 \
  --static 8 4 --taus 0.005 0.01 --quiet --out "$OUT8/quality8b.json"

want tput8b && run "$OUT8" tput8b $PY evaluation/serving/echokv_throughput.py --model "$MODEL8" \
  --kv-cache-gib 8 --num-prompts 256 --input-len 1024 --output-len 256 \
  --configs stock echokv-int8 echokv-int4 --out "$OUT8/tput8b.json"

# 8 GiB holds 2 of these at bf16, ~3.5 at INT8, ~6 at INT4.
want tput8b_32k && run "$OUT8" tput8b_32k $PY evaluation/serving/echokv_throughput.py --model "$MODEL8" \
  --kv-cache-gib 8 --num-prompts 16 --input-len 32768 --output-len 256 \
  --configs stock echokv-int8 echokv-int4 echokv-live-mem4 --out "$OUT8/tput8b_32k.json"

want latency8b && run "$OUT8" latency8b $PY evaluation/serving/echokv_latency.py --model "$MODEL8" \
  --kv-cache-gib 8 --input-len 8192 --output-len 256 --concurrency 16 32 64 \
  --configs stock echokv-int4 echokv-live-mem4 --out "$OUT8/latency8b.json"

# -- Llama-3.3-70B, TP=4 ----------------------------------------------------
unset CUDA_VISIBLE_DEVICES
if [ -n "$DL_PID" ]; then
  echo "== waiting for the $MODEL70 download ($(date +%H:%M))"
  wait "$DL_PID" || { echo "download failed, see $OUT70/download.txt"; FAILED+=(download70b); }
fi

# Per GPU, with 2 of the 8 KV heads each: 80 KB/token at bf16, so 8 GiB holds
# ~105K tokens (12 x 8K or 3 x 32K requests); INT4 holds ~3.2x that.
want quality70b && run "$OUT70" quality70b $PY evaluation/serving/echokv_quality_sweep.py --model "$MODEL70" --tp 4 \
  --dtype bfloat16 --kv-cache-mib 8192 --passages 32 --passage-tokens 2048 \
  --static 8 4 --taus 0.005 0.01 --quiet --out "$OUT70/quality70b.json"

want tput70b_8k && run "$OUT70" tput70b_8k $PY evaluation/serving/echokv_throughput.py --model "$MODEL70" --tp 4 \
  --kv-cache-gib 8 --num-prompts 64 --input-len 8192 --output-len 256 \
  --configs stock echokv-int4 --out "$OUT70/tput70b_8k.json"

want tput70b_32k && run "$OUT70" tput70b_32k $PY evaluation/serving/echokv_throughput.py --model "$MODEL70" --tp 4 \
  --kv-cache-gib 8 --num-prompts 16 --input-len 32768 --output-len 256 \
  --configs stock echokv-int8 echokv-int4 echokv-live-mem4 --out "$OUT70/tput70b_32k.json"

want latency70b && run "$OUT70" latency70b $PY evaluation/serving/echokv_latency.py --model "$MODEL70" --tp 4 \
  --kv-cache-gib 8 --input-len 8192 --output-len 256 --concurrency 16 32 \
  --configs stock echokv-int4 echokv-live-mem4 --out "$OUT70/latency70b.json"

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All done ($(date +%H:%M)). Results: $OUT8/ and $OUT70/"
else
  echo "Failed: ${FAILED[*]} (logs in $OUT8/ and $OUT70/)"
  exit 1
fi
