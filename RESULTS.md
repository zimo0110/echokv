# EchoKV results

Current results for the EchoKV vLLM plugin (`echokv_vllm/`, vLLM 0.32.0, no fork).
Design and methodology: [docs/ECHOKV_VLLM_PLUGIN.md](docs/ECHOKV_VLLM_PLUGIN.md).
Last updated: 2026-10-26.

## TL;DR

On a real GPU (NVIDIA DGX Spark, Llama-3.1-8B-Instruct):

- **It works end to end in stock vLLM.**
  - All 77 unit tests pass, including the Triton kernels and flash-attention
    compiled for the GPU.
  - The live controller changes tiers mid-request, and blocks at different
    precisions coexist.
- **INT4 is nearly lossless: +0.6–0.7% perplexity for 3.19× KV capacity**
  in the same memory. INT8 is lossless at 1.77×.
- **Faster than stock vLLM wherever memory is the limit**, all in the same
  8 GiB KV budget:
  - **32K-token contexts: 3.86× throughput (INT4), 3.05× (INT8).**
  - Serving with 8K prompts at 16 concurrent users: **12× lower median
    time-to-first-token** (3.7 s vs 44.8 s) and 1.5× throughput.
  - At 32 users: 6× lower TTFT and 1.64× throughput.
  - Short prompts (1K): INT4 1.35×, INT8 1.15×.
- **Very long contexts:** 64K-token prompts 1.51× and 128K-token prompts
  1.28× (16 and 24 GiB KV). Stock fits one such request at a time,
  INT4 fits 4–6.
- **No penalty when memory isn't the limit.** At 4 concurrent users,
  throughput, TTFT and time-per-token are identical to stock.
- **The trade-off:** under load, each request decodes alongside more others,
  so time per output token is higher (209 vs 136 ms at 16 users). That's
  set by how many requests run at once, not by kernel speed (see round 5).
  Total waiting is much lower: median end-to-end 57 s vs 80 s.

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

**Current code**, short prompts: 256 × (1,024 in, 256 out). From
`round2_throughput_short.json`.

| config | KV tokens | capacity | output tok/s | speed |
|---|---:|---:|---:|---:|
| stock vLLM (bf16) | 65,536 | 1.00× | 341.4 | 1.00× |
| EchoKV INT8 | 116,288 | 1.77× | 394.1 | **1.15×** |
| EchoKV INT4 | 209,024 | 3.19× | 461.0 | **1.35×** |
| live, 6-bit budget | 149,440 | 2.28× | 404.5 | **1.18×** |
| live, 4-bit budget | 209,024 | 3.19× | 456.9 | **1.34×** |

**Current code**, long prompts (memory-bound): 64 × (8,192 in, 256 out).
At this length 8 GiB holds about 7 requests in bf16 and about 23 in INT4.
From `round2_throughput_long.json`.

| config | KV tokens | output tok/s | total tok/s | wall time | speed |
|---|---:|---:|---:|---:|---:|
| stock vLLM (bf16) | 65,536 | 46.5 | 1,534 | 352 s | 1.00× |
| **EchoKV INT4** | 209,024 | **65.1** | **2,149** | **252 s** | **1.40×** |

Speed tracks capacity: the more KV tokens fit, the bigger the batches vLLM
can run.

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

The quality table above was measured before the speed fixes. INT4
re-checked on the current code: ΔNLL +0.0055 (`round2_quality_int4.txt`).

The DGX Spark has ~273 GB/s memory bandwidth (an H100 has ~3.3 TB/s), so
treat absolute tok/s as Spark-specific. The ratios against stock are the
meaningful numbers.

### Long contexts and serving latency (current code, commit `416aa80`)

**32K-token contexts:** 16 × (32,768 in, 256 out), 8 GiB KV. At this
length, stock fits 2 requests in memory, INT8 about 3.5 and INT4 about 6.
From `round3_throughput_32k.json`.

| config | KV tokens | output tok/s | total tok/s | wall time | speed |
|---|---:|---:|---:|---:|---:|
| stock vLLM (bf16) | 65,536 | 4.2 | 543 | 974 s | 1.00× |
| EchoKV INT8 | 116,288 | 12.8 | 1,655 | 319 s | **3.05×** |
| EchoKV INT4 | 209,024 | 16.2 | 2,094 | 252 s | **3.86×** |

**Serving latency:** `vllm serve` driven by `vllm bench serve`. 8,192-token
prompts, 256 out, closed loop at fixed concurrency, 8 GiB KV. Times in ms.
From `round3_latency_8k.json`.

| config | users | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | E2E p50 |
|---|---:|---:|---:|---:|---:|---:|
| stock | 4 | 35 | 3,040 | 7,483 | 101 | 28,759 |
| EchoKV INT4 | 4 | 35 | 3,010 | 7,776 | 101 | 28,817 |
| stock | 16 | 45 | 44,847 | 78,728 | 136 | 79,569 |
| EchoKV INT4 | 16 | **68** | **3,734** | 32,333 | 209 | **57,168** |
| stock | 32 | 45 | 124,797 | 158,269 | 136 | 159,507 |
| EchoKV INT4 | 32 | **74** | **21,162** | 93,518 | 294 | **97,297** |

**Profile:** 64 × (1,024 in, 128 out). From
`round3_profile_summary.txt`. The Spark ran ~2× slower overall in this
session for both configs, so only the comparison between them is
meaningful.
- **Wall time:** EchoKV 66.7 s vs stock 80.5 s.
- **Prefill attention:** 1.3 s GPU (was 5.3 s with the fused kernel). It
  now dequantizes once, then runs flash-attention.
- **Decode kernel:** 6.6 s.
- **Remaining cost:** ~10 s of GPU idle time, caused by a blocking
  host-to-device copy in the write path. This is fixed in round 4 below.

**Round 4 profile** (current code, commit `1546036`): pinned non-blocking
host-to-device copies, and the canary no longer waits on the GPU. Same
workload; from `round4_profile.json`. GEMM times match the first round-3
session to within 1%, so the machine ran at full speed.

| | fused prefill (round 3, first session) | **current** | stock (same-speed session) |
|---|---:|---:|---:|
| wall time | 31.5 s | **26.8 s** | 35.8 s |
| GPU idle | 1.0 s | **0.9 s** | 0.1 s |
| prefill attention (GPU) | 5.3 s | **0.73 s** | — |

EchoKV now finishes this short-prompt workload **1.34× faster than stock**.
At 16 users with 8K prompts, round-4 latency matches round 3 (68 tok/s,
TTFT p50 3.7 s, TPOT 208 ms; `round4_latency_c16.json`). That load is
decode-bound on the GPU, so removing CPU stalls doesn't change it.

### Round 5 (commit `e389602`)

Changes:
- A shared tile loader for the decode, fused-prefill and dequantize
  kernels. It reads K's scale/min once per tile instead of once per token.
- Decode-kernel autotuning per GPU.
- Per-step write bookkeeping done once per step on the host.

All 77 unit tests pass on the GPU (`round5_unit.txt`).

**Profile:** same workload as round 4, 64 × (1,024 in, 128 out).
From `round5_profile.json`.

| | round 4 | round 5 | stock |
|---|---:|---:|---:|
| wall time | 26.8 s | **26.5 s** | 35.6 s |
| GPU idle | 0.9 s | 0.8 s | 0.1 s |
| decode kernel (GPU) | 3.43 s | 3.37 s | — |
| KV writes (GPU) | 0.86 s | **0.56 s** | — |
| prefill attention (GPU) | 0.73 s | 0.73 s | — |

- **Writes are 35% cheaper on the GPU.**
- **Decode barely moved (−2%).** The per-token stats loads weren't what
  limited it.
- **Decode is 13% of GPU time.** Most of the rest is the model's matrix
  multiplies, the same work stock does.

**CPU waiting on the GPU.** The trace shows the host blocked on the GPU
for ~23 s of the 26.5 s wall time:

| where | cause | calls | CPU blocked |
|---|---|---:|---:|
| block open | `sealed[idx] = 0`: the scalar becomes a CPU tensor, and copying it blocks | 608 | 12.7 s |
| metadata build | copying `slot_mapping` and `seq_lens` back from the GPU | 715 | 8.5 s |
| write | `block_res[idx] = -1` (same scalar issue) | 1,120 | 1.6 s |

On the Spark this barely costs anything: the GPU is the bottleneck, so
the CPU would be waiting somewhere anyway. On a faster GPU, the CPU would
sit in these waits instead of preparing the next step.

The fix, after `e389602`:
- `index_fill_` replaces the scalar assignments.
- The builder computes the slot mapping on the host instead of copying it
  back. It reads vLLM's CPU block table (V1 model runner) or a host mirror
  of the block-table writes (V2 runner, the default on CUDA). It falls back to the copy under
  speculative decoding or context parallelism.
- `ECHOKV_CHECK_HOST_SLOTS=1` checks every step against vLLM's own
  mapping. The end-to-end test runs with it on.

This fix still needs a GPU run.

**Serving latency:** 8,192-token prompts, 256 out, 8 GiB KV. Matches
rounds 3 and 4 to within ~1%. From `round5_latency.json`.

| config | users | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | E2E p50 |
|---|---:|---:|---:|---:|---:|---:|
| stock | 16 | 46 | 44,257 | 77,586 | 135 | 78,583 |
| EchoKV INT4 | 16 | **68** | **3,766** | 32,595 | 210 | **57,416** |
| stock | 32 | 46 | 123,440 | 156,508 | 135 | 157,880 |
| EchoKV INT4 | 32 | **74** | **21,423** | 93,916 | 295 | **97,777** |

**Why time per token is higher.** TPOT ≈ (requests decoding at once) ÷
(output throughput):

| | requests in memory | ÷ out tok/s | predicted TPOT | measured TPOT |
|---|---:|---:|---:|---:|
| stock | 7 | 46 | 152 ms | 135 ms |
| EchoKV, 16 users | 16 | 68 | 235 ms | 210 ms |
| EchoKV, 32 users | 24 | 74 | 324 ms | 295 ms |

Stock holds 7 of these requests in 8 GiB; INT4 holds 24.
- **Throughput is mostly prefill.** Each request prefills 8,192 tokens
  for 256 output tokens, and prefill is the same work in both configs.
- **More requests share each step.** Running more requests at once raises
  each one's TPOT, while the queueing wait (TTFT) falls.
- **A faster decode kernel can't change this much.** A deployment that
  needs stock-level TPOT can cap concurrency with `--max-num-seqs`, at the
  cost of TTFT.

**Very long contexts:** stock vs INT4. From `round5_ctx64k.json` and
`round5_ctx128k.json`.

| workload | KV | stock fits | INT4 fits | stock wall | INT4 wall | speed |
|---|---:|---:|---:|---:|---:|---:|
| 8 × (65,536 in, 256 out) | 16 GiB | 1 | 6 | 429 s | 284 s | **1.51×** |
| 4 × (130,048 in, 256 out) | 24 GiB | 1 | 4 | 471 s | 367 s | **1.28×** |

- **Both configs are prefill-bound** at these lengths (~1,400–1,850
  prompt tok/s).
- **The gain is decoding together.** Stock decodes its requests one at a
  time; INT4 decodes them together.
- **Longer outputs would widen the gap.**
- **128K needs EchoKV in a smaller budget.** Stock can't hold one 128K
  request in 8 GiB; INT4 holds 1.6.

**Quality under a memory budget:** live controller at τ=0.005 with
`ECHOKV_MEM_BITS` 4, 6 and 8. Same quality setup as above. From
`round5_budget.json`.

| config | ΔNLL | stored bits | tiers chosen |
|---|---:|---:|---|
| static INT4 | +0.0065 | 4.0 | all INT4 |
| live, 4-bit budget | +0.0065 | 4.0 | all INT4 (forced) |
| live, 6-bit budget | +0.0053 | 4.12 | INT4, layer 14 INT8 |
| live, 8-bit budget | +0.0061 | 4.25 | INT4, layers 13–14 INT8 |

- **The budget caps correctly.** The controller wants INT8 for layers
  13–14 at τ=0.005. An 8-bit budget lets both have it, a 6-bit budget
  one, and a 4-bit budget none.
- **Quality stays within INT4's noise (±0.003) in all three.**

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

1. Confirm the GPU-sync fix on the GPU. On the Spark, expect the CPU
   wait to drop and wall time to barely change.
2. Run on a fast GPU (H100). Per-layer Python cost is ~1 ms per layer per
   step, so a fast GPU may become CPU-bound. If it does, the next step is
   a device-side write path that CUDA graphs can capture.
3. Make INT4 the default `ECHOKV_MEM_BITS`.
4. Run 70B on 4× H100 (TP=4).
5. Try long-context workloads (e.g. RULER), where per-layer adaptation has
   more room to help.
