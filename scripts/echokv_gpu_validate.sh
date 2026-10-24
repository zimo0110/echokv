#!/usr/bin/env bash
# Validate the EchoKV vLLM plugin on a CUDA machine, end to end.
#
#   bash scripts/echokv_gpu_validate.sh                # Llama-3.1-8B, 1 GPU
#   MODEL=meta-llama/Llama-3.1-70B-Instruct TP=4 bash scripts/echokv_gpu_validate.sh
#
# Steps: install vLLM 0.32.0 + this plugin into ./.venv-gpu, run the unit
# tests (the Triton kernels compile and run natively and are checked against
# the PyTorch oracle), then the quality sweep and the throughput/capacity
# benchmark. Results land in results/echokv_gpu/.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
TP=${TP:-1}
KV_GIB=${KV_GIB:-8}
OUT=results/echokv_gpu/$(basename "$MODEL")
mkdir -p "$OUT"

if [ ! -d .venv-gpu ]; then
  python3 -m venv .venv-gpu
  .venv-gpu/bin/pip install -q --upgrade pip
  .venv-gpu/bin/pip install -q "vllm==0.32.0" datasets pytest
  .venv-gpu/bin/pip install -q -e .
fi
PY=.venv-gpu/bin/python
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

echo "== unit tests (Triton kernels on GPU)"
$PY -m pytest -q tests/echokv_vllm 2>&1 | tee "$OUT/pytest.txt"

echo "== end-to-end smoke (live controller, mixed tiers)"
$PY -m pytest -q -m slow tests/echokv_vllm/test_vllm_e2e.py 2>&1 | tee -a "$OUT/pytest.txt"

echo "== quality sweep (WikiText NLL vs stock)"
$PY evaluation/serving/echokv_quality_sweep.py --model "$MODEL" --tp "$TP" --dtype bfloat16 \
  --kv-cache-mib 8192 --passages 32 --passage-tokens 2048 --quiet --out "$OUT/quality.json" \
  | tee "$OUT/quality.txt"

echo "== throughput and KV capacity (same memory budget)"
$PY evaluation/serving/echokv_throughput.py --model "$MODEL" --tp "$TP" --kv-cache-gib "$KV_GIB" \
  --num-prompts 256 --input-len 1024 --output-len 256 --out "$OUT/throughput.json" \
  | tee "$OUT/throughput.txt"
