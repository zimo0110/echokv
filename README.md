# EchoKV

Closed-loop, per-layer mixed-precision KV cache (INT8 / INT4 / INT2) for
vLLM, shipped as a vLLM plugin (`echokv_vllm/`, vLLM 0.32.0, no fork).
Results: [RESULTS.md](RESULTS.md). Design and limitations: [docs/ECHOKV_VLLM_PLUGIN.md](docs/ECHOKV_VLLM_PLUGIN.md).

## Run the GPU validation

On a Linux machine with an NVIDIA GPU (H100/A100 80GB recommended):

```bash
git clone git@github.com:zimo0110/echokv.git && cd echokv
export HF_TOKEN=hf_...     # a token whose account accepted the Llama 3.1 license
bash scripts/echokv_gpu_validate.sh                                  # Llama-3.1-8B, 1 GPU
MODEL=meta-llama/Llama-3.1-70B-Instruct TP=4 bash scripts/echokv_gpu_validate.sh   # 4 GPUs
```

The script installs everything itself (uv, Python 3.12, vLLM 0.32.0, this
plugin) into `.venv-gpu/`, then runs the unit tests (Triton kernels on the
GPU), an end-to-end vLLM test, a WikiText quality sweep and a
throughput/KV-capacity benchmark against stock vLLM. All output goes to
`results/echokv_gpu/<model>/`; send that folder back. A failing step is
logged and the rest still run.

Needs: ~100 GB free disk for 8B (~300 GB for 70B), internet access for
pip and Hugging Face.

## Use it

```bash
pip install vllm==0.32.0 && pip install -e .
ECHOKV_ENABLE=1 python -c '
from vllm import LLM
llm = LLM("meta-llama/Llama-3.1-8B-Instruct", attention_config={"backend": "CUSTOM"})
print(llm.generate("The capital of Japan is")[0].outputs[0].text)'
```

Settings are `ECHOKV_*` environment variables (see `echokv_vllm/config.py`),
e.g. `ECHOKV_MEM_BITS` for the memory budget (default 4 bits: INT4, 3.2× the
KV capacity for +0.7% perplexity at 8B; 8 for INT8), `ECHOKV_TAU0` for the
quality/memory trade-off.

## Repo layout

- `echokv_vllm/`: the plugin (store, Triton kernels, canary, controller, vLLM backend)
- `tests/echokv_vllm/`: its tests (`pytest tests/echokv_vllm`; `-m slow` boots vLLM)
- `evaluation/serving/echokv_*.py`: quality sweep, throughput benchmark, smoke driver
- `docker/triton-interp/`: run the Triton kernels without a GPU (CPU interpreter)
- `vllm_patches/`: one fix needed only to run vLLM 0.32 on macOS CPU
- everything else: the original research code, paper drafts and results
