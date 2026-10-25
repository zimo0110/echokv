#!/usr/bin/env bash
# Validate the EchoKV vLLM plugin on a CUDA machine, end to end.
#
#   export HF_TOKEN=hf_...                             # access to the gated Llama repos
#   bash scripts/echokv_gpu_validate.sh                # Llama-3.1-8B, 1 GPU
#   MODEL=meta-llama/Llama-3.1-70B-Instruct TP=4 bash scripts/echokv_gpu_validate.sh
#
# Steps: install vLLM 0.32.0 + this plugin into ./.venv-gpu (Python 3.12 via
# uv; vLLM 0.32 needs >= 3.11), run the unit tests (the Triton kernels
# compile and run natively and are checked against the PyTorch oracle), the
# end-to-end vLLM test, the quality sweep and the throughput/capacity
# benchmark. Everything is logged to results/echokv_gpu/<model>/; a failing
# step is recorded and the remaining steps still run.
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
TP=${TP:-1}
KV_GIB=${KV_GIB:-8}
OUT=results/echokv_gpu/$(basename "$MODEL")
mkdir -p "$OUT"
FAILED=()

die() { echo "ERROR: $*" >&2; exit 1; }

# -- preflight ---------------------------------------------------------------
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: this needs a CUDA GPU machine"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "$OUT/gpus.txt"
NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
[ "$NGPU" -ge "$TP" ] || die "TP=$TP but only $NGPU GPU(s) visible"

if [ ! -x .venv-gpu/bin/python ]; then
  echo "== installing (first run only, a few minutes)"
  command -v uv >/dev/null || python3 -m pip install -q --user uv || die "could not install uv"
  export PATH="$HOME/.local/bin:$PATH"
  uv venv --python 3.12 .venv-gpu || die "uv venv failed"
  uv pip install --python .venv-gpu/bin/python "vllm==0.32.0" datasets pytest "huggingface_hub[cli]" \
    || die "installing vLLM 0.32.0 failed"
  uv pip install --python .venv-gpu/bin/python -e . || die "installing echokv_vllm failed"
fi
PY=.venv-gpu/bin/python
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

$PY - "$MODEL" <<'EOF' || die "cannot read $MODEL from Hugging Face: set HF_TOKEN (or run 'huggingface-cli login') with access to that repo"
import sys
from huggingface_hub import hf_hub_download
hf_hub_download(sys.argv[1], "config.json")
EOF

run() {  # run <name> <command...>: log to $OUT/<name>.txt, keep going on failure
  local name=$1; shift
  echo "== $name"
  if "$@" 2>&1 | tee "$OUT/$name.txt"; then :; else FAILED+=("$name"); fi
}

# -- steps -------------------------------------------------------------------
run unit_tests $PY -m pytest -q tests/echokv_vllm
run e2e_test $PY -m pytest -q -m slow tests/echokv_vllm/test_vllm_e2e.py
run quality $PY evaluation/serving/echokv_quality_sweep.py --model "$MODEL" --tp "$TP" \
  --dtype bfloat16 --kv-cache-mib 8192 --passages 32 --passage-tokens 2048 --quiet \
  --out "$OUT/quality.json"
run throughput $PY evaluation/serving/echokv_throughput.py --model "$MODEL" --tp "$TP" \
  --kv-cache-gib "$KV_GIB" --num-prompts 256 --input-len 1024 --output-len 256 \
  --out "$OUT/throughput.json"

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All steps passed. Results: $OUT/"
else
  echo "Failed steps: ${FAILED[*]} (logs in $OUT/). Send the whole $OUT/ folder."
  exit 1
fi
