# EchoKV results

Current results for the EchoKV vLLM plugin (`echokv_vllm/`, vLLM 0.32.0, no fork).
Design and methodology: [docs/ECHOKV_VLLM_PLUGIN.md](docs/ECHOKV_VLLM_PLUGIN.md).
Last updated: 2026-10-25.

## TL;DR

On a real GPU (NVIDIA DGX Spark, Llama-3.1-8B-Instruct):

- **It works end to end in stock vLLM.**
  - All 52 unit tests pass, including the Triton kernels compiled for the GPU.
  - The live controller changes tiers mid-request, and blocks at different
    precisions coexist.
- **INT4 is nearly lossless: +0.7% perplexity for 3.19× KV capacity** in the
  same memory.
- **INT8 is lossless: 1.77× capacity.**
- **EchoKV INT4 is 1.35× faster than stock vLLM** in the same memory
  (462.5 vs 342.5 output tok/s). The extra KV capacity lets vLLM run about
  3× more requests at once. Before the speed fixes it was 0.49× stock.

## GPU: DGX Spark (GB10), Llama-3.1-8B-Instruct, bf16

Raw files: [`results/echokv_gpu/dgx_spark_Llama-3.1-8B-Instruct/`](results/echokv_gpu/dgx_spark_Llama-3.1-8B-Instruct/).

### Quality

WikiText-103 test set, 32 passages × 2,048 tokens. ΔNLL is the change in
prompt NLL against stock vLLM (NLL 1.886). Lower is better.

| config | ΔNLL | perplexity | stored bits | tiers chosen |
|---|---:|---:|---:|---|
| EchoKV INT8 | −0.0006 | ≈ 0% | 8.0 | all INT8 |
| **EchoKV INT4** | **+0.0066** | **+0.7%** | **4.0** | all INT4 |
| EchoKV INT2 | +0.196 | +22% | 2.0 | all INT2 |
| live, τ=0.005 | +0.0065 | +0.7% | 4.25 | INT4, layers 13–14 INT8 |
| live, τ=0.01 (default) | +0.0066 | +0.7% | 4.0 | all INT4 |
| live, τ=0.02 | +0.0066 | +0.7% | 4.0 | all INT4 |
| live, τ=0.05 | +0.0152 | +1.5% | 3.94 | layer 0 INT2, rest INT4 |

INT4 differences below ~0.003 nats are noise: they come from rounding flips
(measured, see the docs).

### Capacity and throughput

8 GiB KV-cache budget for every config. 256 requests × (1,024 tokens in,
256 out). CUDA graphs on (piecewise).

**Current code** (commit `c6b4743`), from `throughput_v2.json`:

| config | KV tokens | capacity | output tok/s | speed |
|---|---:|---:|---:|---:|
| stock vLLM (bf16) | 65,536 | 1.00× | 342.5 | 1.00× |
| **EchoKV INT4** | 209,024 | **3.19×** | **462.5** | **1.35×** |

Wall time for the 256 requests: 142 s for EchoKV vs 191 s for stock.

**Before the speed fixes** (from `throughput.json`). The fixes in commit
`7372522` were:
- a per-step batch layout built once instead of per layer
- fused bf16 prefill attention
- fp16 tensor-core decode dots

| config | KV tokens | capacity | output tok/s | speed |
|---|---:|---:|---:|---:|
| stock vLLM (bf16) | 65,536 | 1.00× | 347.0 | 1.00× |
| EchoKV INT8 | 116,288 | 1.77× | 158.4 | 0.46× |
| EchoKV INT4 | 209,024 | 3.19× | 170.6 | 0.49× |
| live, 6-bit budget | 149,440 | 2.28× | 156.3 | 0.45× |
| live, 4-bit budget | 209,024 | 3.19× | 168.9 | 0.49× |

The quality table above was measured before the speed fixes. A
re-check of INT4 on the current code is pending.

The DGX Spark has ~273 GB/s memory bandwidth (an H100 has ~3.3 TB/s), so
treat absolute tok/s as Spark-specific. The ratios against stock are the
meaningful numbers.

### What this means

- **The quantizer is the big win.** Per-channel K quantization with fp16
  min makes INT4 nearly free at 8B.
- **The controller behaves correctly, but there's little to adapt here.**
  INT4 is already safe in every layer, so the live controller picks INT4
  almost everywhere.
- **INT2 costs more than it saves.** Moving one layer to INT2 at τ=0.05 cost
  +0.009 nats for 0.06 bits saved, so INT2 should be a last resort.

## Choosing the K quantization scheme

Simulated in HF transformers on post-RoPE K/V, WikiText-103. ΔNLL against
full precision; K uses the listed scheme, V is per-token.

| scheme | Qwen2.5-0.5B INT4 | INT2 | Llama-3.2-1B INT4 | INT2 |
|---|---:|---:|---:|---:|
| per-token K (paper's scheme) | +1.67 | +5.50 | +0.041 | +3.78 |
| center + Hadamard | +0.28 | +3.49 | +0.044 | +2.30 |
| **per-channel K (chosen)** | **+0.011** | **+0.71** | **+0.018** | **+0.69** |

This study also turned up a bug in the repo's reference quantizer
(`evaluation/motivation/kivi.py`). Its clamped integer zero-point can't
represent groups that don't span zero: values in [10, 11] come back about
10 off at any bit width. The paper's emulation numbers used it, so treat
them with care.

## CPU: macOS, fp32, vLLM 0.32.0 (small models)

Quality vs memory, warm controller, WikiText prompt NLL:

| config | Llama-3.2-1B ΔNLL | bits | Qwen2.5-0.5B ΔNLL | bits |
|---|---:|---:|---:|---:|
| INT8 | +0.0001 | 8.0 | +0.0001 | 8.0 |
| INT4 | +0.0209 | 4.0 | +0.0116 | 4.0 |
| INT2 | +0.746 | 2.0 | +0.749 | 2.0 |
| live τ=0.01 | +0.0209 | 4.0 | +0.0030 | 6.67 |

Correctness checks:
- **Attention path:** matches stock vLLM to 1e-5.
- **Store vs oracle:** bit-identical.
- **TP=2:** both ranks reach identical tier decisions.
- **Repeated prompts:** identical greedy outputs, like stock vLLM.
- **Stress test:** no errors with a 19-block cache serving 48 requests.

## Known limitations

- **Preempted requests can produce different outputs.** A request vLLM
  preempts and recomputes can give a different (equally valid) greedy
  output.
- **The memory budget is per layer.** Layers can't borrow chunks from each
  other.
- **Unsupported attention variants:** sliding window, ALiBi, soft-cap,
  sinks and MLA.

## Reproduce

```bash
bash scripts/echokv_gpu_validate.sh                                   # tests + quality + throughput
MODEL=unsloth/Llama-3.1-8B-Instruct bash scripts/echokv_gpu_validate.sh  # ungated Llama mirror
```

## Next

1. Re-check INT4 quality on the current code, then re-run the other
   configs (INT8, live) for throughput.
2. Profile on GPU, then work on a fused prefill kernel, a fused write
   kernel and CUDA graphs for decode.
3. Run memory-bound benchmarks (long contexts, many users), where the
   capacity gain should matter most.
4. Run 70B on 4× H100 (TP=4).
5. Try long-context workloads (e.g. RULER), where per-layer adaptation has
   more room to help.
