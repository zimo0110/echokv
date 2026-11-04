# EchoKV results

Current results for the EchoKV vLLM plugin (`echokv_vllm/`, vLLM 0.32.0, no fork).
Design and methodology: [docs/ECHOKV_VLLM_PLUGIN.md](docs/ECHOKV_VLLM_PLUGIN.md).
Last updated: 2026-11-03.

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
  - Short prompts (1K): INT4 1.51× (round 7, CUDA graphs), INT8 1.15×.
- **Very long contexts:** 64K-token prompts 1.51× and 128K-token prompts
  1.28× (16 and 24 GiB KV). Stock fits one such request at a time,
  INT4 fits 4–6.
- **No penalty when memory isn't the limit.** At 4 concurrent users,
  throughput, TTFT and time-per-token are identical to stock.
- **The trade-off:** under load, each request decodes alongside more others,
  so time per output token is higher (209 vs 136 ms at 16 users). That's
  set by how many requests run at once, not by kernel speed (see round 5).
  Total waiting is much lower: median end-to-end 57 s vs 80 s.

On Modal (round 6): Llama-3.3-70B with TP=4 on 4× H100, and Llama-3.1-8B
on one H200 (Modal substituted H200s for that run; see
[`MODAL_RUNS.md`](results/echokv_gpu/MODAL_RUNS.md)):

- **Quality holds at 70B:** INT4 costs +0.006 NLL (+0.6% perplexity) for
  3.16× KV capacity. INT8 is lossless.
- **Speed doesn't yet:** 0.6× stock on 8B with short prompts (H200),
  0.93–0.98× on 70B (H100). The profile shows the cause: EchoKV is
  CPU-bound. The GPU does the same work as stock but sits idle about
  two thirds of the time (8.1 s wall vs 3.5 s GPU busy on one H100) while
  Python launches per-layer work. A slow GPU (the Spark) hid this.
- **Round 7 removes it:** decode steps now replay as full CUDA graphs.
  On one H100, EchoKV's wall time fell from 8.1 s to 3.8 s, the same as
  stock, and short-prompt throughput is 0.90× stock (8B). At 8K prompts
  it matches stock's throughput with 5× lower time to first token.

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

All unit tests pass on the GPU (`round5_unit.txt`, `round5b_unit.txt`).

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

**Confirmed on the GPU** (commit `7957e9d`, `round5b_profile.json`):
- The end-to-end test passes with `ECHOKV_CHECK_HOST_SLOTS=1`, so the
  host mapping matched vLLM's on every step.
- CPU time in `echokv.build` fell from 8.5 s to 0.14 s.
- CPU time in `write.open` fell from 12.8 s to 0.3 s.
- Wall time is 26.3 s (vs 26.5 s); the Spark was GPU-bound either way.

Where the CPU waits now:
- **Kernel launches (10.8 s):** the CPU runs far enough ahead to fill the
  GPU's launch queue.
- **vLLM's own once-per-step sync:** stock vLLM has it too.

The remaining per-layer cost on a faster GPU is launches. Each decode call
merges its split results with ~13 small PyTorch kernels, which could be
folded into the Triton kernel.

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

### Round 7: decode steps as CUDA graphs (commit `8dc2792`)

The builder does all host work once per step, and vLLM replays
decode-only steps as one CUDA graph (see Next). The first Spark run
crashed in the canary: vLLM builds layers under bf16 as the default
dtype, so the K_ref buffer came out bf16. Commit `8dc2792` fixes it, and
the GPU e2e test now runs in bf16. All tests pass on the GPU, including
the e2e with graphs (`graphs_unit.txt`).

**Profile:** 64 × (1,024 in, 128 out), from `graphs_profile.json`.

| | round 5 | round 7 | stock |
|---|---:|---:|---:|
| wall time | 26.5 s | **24.6 s** | 35.2 s |
| GPU busy | 25.7 s | 24.5 s | 35.1 s |
| decode kernel (GPU) | 3.37 s | 3.14 s | — |
| KV writes (GPU) | 0.56 s | **0.16 s** | — |

- **No GPU idle time left.** It was already small on the Spark, whose GPU
  is slow enough to hide Python. The H100 will show what this is worth.
- **KV writes cost about a third of what they did,** now two kernels per
  layer: 0.56 s of GPU time in round 5, 0.16 s of write kernels in
  round 7. (The round-7 `echokv.write` scope reads 0.12 s, but it only
  sees the 288 eager calls: graph replays bypass the profiler scopes.)
- **The decode kernel is the largest EchoKV cost:** 13% of GPU time.

**Throughput,** 256 × (1,024 in, 256 out), 8 GiB KV. From
`graphs_short.json`.

| config | out tok/s | speed | round 5 |
|---|---:|---:|---:|
| stock vLLM (bf16) | 349.1 | 1.00× | 1.00× |
| EchoKV INT4 | 528.4 | **1.51×** | 1.35× |
| live, 4-bit budget | 525.9 | **1.51×** | 1.34× |

**Serving latency,** 8,192-token prompts, 256 out, 8 GiB KV. From
`graphs_latency.json`.

| config | users | out tok/s | TTFT p50 | TPOT p50 | E2E p50 |
|---|---:|---:|---:|---:|---:|
| stock | 16 | 46 | 44,263 | 134 | 78,379 |
| EchoKV INT4 | 16 | **70** | **3,740** | 204 | **55,833** |
| live, 4-bit | 16 | 68 | 3,920 | 209 | 57,448 |
| stock | 32 | 46 | 123,328 | 134 | 157,567 |
| EchoKV INT4 | 32 | **76** | **21,189** | 288 | **95,689** |
| live, 4-bit | 32 | 73 | 22,070 | 298 | 98,615 |

These are 2–3% better than round 5. Prefill dominates this workload, and
the TPOT gap is still set by concurrency, not kernel speed.

### Round 7 on one H100 (commit `ac13e59`)

Same script and workloads as the Spark run above, Llama-3.1-8B-Instruct,
on one Modal H100. Raw files:
`results/echokv_gpu/20261103_143942/Llama-3.1-8B-Instruct/`. All 93 unit
tests and the three GPU e2e tests pass, including the check that greedy
text with CUDA graphs matches eager.

**The CPU bottleneck is gone.** 64 × (1,024 in, 128 out), from
`graphs_profile.json`:

| config | round 6 wall | round 6 GPU busy | round 7 wall | round 7 GPU busy |
|---|---:|---:|---:|---:|
| stock | 3.7 s | 3.6 s | 3.8 s | 3.6 s |
| EchoKV INT4 | 8.1 s | 3.5 s | **3.8 s** | 3.5 s |

The largest EchoKV cost is now the decode kernel: 1.08 s of GPU time,
against 0.53 s for stock's flash-attention (prefill and decode together).
KV writes are 0.07 s.

**Throughput,** 256 × (1,024 in, 256 out), 8 GiB KV, from
`graphs_short.json`. Memory isn't the limit here, so this measures overhead.

| config | KV tokens | out tok/s | speed |
|---|---:|---:|---:|
| stock vLLM (bf16) | 65,536 | 3,477 | 1.00× |
| EchoKV INT4 | 209,024 (3.19×) | 3,135 | **0.90×** |
| live, 4-bit budget | 209,024 (3.19×) | 3,143 | **0.90×** |

Round 6 measured 0.61× on this workload, but on an H200, so it is not a
like-for-like comparison.

**Serving latency,** 8,192-token prompts, 256 out, 8 GiB KV, medians in ms,
from `graphs_latency.json`:

| config | users | out tok/s | TTFT p50 | TPOT p50 | E2E p50 |
|---|---:|---:|---:|---:|---:|
| stock | 16 | 446 | 4,747 | 13 | 7,946 |
| EchoKV INT4 | 16 | 427 | **954** | 33 | 9,454 |
| live, 4-bit | 16 | 439 | **951** | 33 | 9,198 |
| stock | 32 | 447 | 12,867 | 13 | 16,077 |
| EchoKV INT4 | 32 | 442 | **2,689** | 51 | 15,982 |
| live, 4-bit | 32 | 443 | **2,688** | 51 | 15,797 |

- **Throughput matches stock** (0.96–0.99×) **with 4.8–5× lower time to
  first token.**
- **Time per output token is 2.5–4× stock's.** EchoKV admits more
  requests at once, so each decodes alongside more others, and its decode
  kernel is about 2× flash-attention's GPU time.

## GPU: Modal H100 / H200, round 6 (commit `b95f4b8`)

Run with `scripts/echokv_bench_h100.sh`. **The 8B numbers below are from
an H200** (141 GB, ~1.4× an H100's memory bandwidth): the run asked for
H100s and Modal filled it with H200s (`gpus.txt`). The 70B numbers and
the 8B smoke-run profile are from H100 80GB. Raw files:
- `results/echokv_gpu/20261101_183932/h100_Llama-3.1-8B-Instruct/` (8B, one H200, despite the folder name)
- `results/echokv_gpu/20261101_213936/h100_Llama-3.3-70B-Instruct/` (70B, 4× H100, TP=4)
- `results/echokv_gpu/20261101_180614/h100_Llama-3.1-8B-Instruct/` (8B smoke run, one H100)

All 85 unit tests and the end-to-end test pass on sm_90.

### Quality

WikiText-103, 32 passages × 2,048 tokens. ΔNLL is against stock vLLM.

| config | 8B ΔNLL | 70B ΔNLL |
|---|---:|---:|
| stock NLL | 1.8860 | 1.2312 |
| INT8 | −0.0006 | −0.0003 |
| INT4 | +0.0050 | +0.0057 |
| live τ=0.005 | +0.0051 (one layer INT8) | +0.0044 |
| live τ=0.01 | +0.0050 | +0.0068 (one layer INT2) |

### Throughput

8 GiB of KV per GPU. EchoKV holds 3.19× the tokens (8B) or 3.16× (70B).
Speed is relative to stock.

| workload | stock tok/s | INT8 | INT4 | live (4-bit budget) |
|---|---:|---:|---:|---:|
| 8B, 256 × (1K in, 256 out) | 4,025 | 0.57× | 0.61× | — |
| 8B, 16 × (32K in, 256 out) | 93.3 | 0.72× | 1.05× | 0.93× |
| 70B, 64 × (8K in, 256 out) | 253.9 | — | 0.93× | — |
| 70B, 16 × (32K in, 256 out) | 57.1 | 0.71× | 0.98× | 0.98× |

### Serving latency

8K-token prompts, 256 output tokens. TTFT is time to first token, TPOT is
time per output token, E2E is end-to-end request time; all are medians.

| model, config | users | out tok/s | TTFT | TPOT | E2E |
|---|---:|---:|---:|---:|---:|
| 8B stock | 16 / 32 / 64 | 522 / 522 / 532 | 4.2 / 11.2 / 27.2 s | 10 / 10 / 10 ms | 6.9 / 13.9 / 29.8 s |
| 8B INT4 | 16 / 32 / 64 | 359 / 447 / 441 | 2.0 / 2.7 / 18.3 s | 32 / 51 / 51 ms | 11.3 / 15.6 / 31.3 s |
| 8B live | 16 / 32 / 64 | 382 / 390 / 387 | 1.2 / 2.7 / 20.4 s | 37 / 58 / 59 ms | 10.6 / 17.6 / 35.3 s |
| 70B stock | 16 / 32 | 252 / 252 | 4.1 / 18.1 s | 39 / 39 ms | 14.0 / 27.8 s |
| 70B INT4 | 16 / 32 | 147 / 223 | 3.1 / 3.1 s | 86 / 129 ms | 27.3 / 35.9 s |
| 70B live | 16 / 32 | 168 / 229 | 3.0 / 2.4 s | 82 / 128 ms | 23.9 / 35.0 s |

### Why it's slower: the CPU, not the GPU

The 8B profile (64 × (1K in, 128 out)) settles it:

| config | H200 wall | H200 GPU busy | H100 wall | H100 GPU busy |
|---|---:|---:|---:|---:|
| stock | 3.3 s | 3.1 s | 3.7 s | 3.6 s |
| EchoKV INT4 | 9.5 s | 3.3 s | 8.1 s | 3.5 s |

- **EchoKV's GPU work is within 10% of stock, but its GPU idles about two
  thirds of the time.**
- **Where the CPU time goes:** each layer spends about 0.94 ms of Python
  per step on writes and 0.62 ms on decode (H200 profile: 3,988 ms over
  4,224 write calls, 2,589 ms over 4,192 decode calls). Each Triton
  launch alone costs about 0.1 ms of Python.
- **Per step that adds up:** about 50 ms of Python against about 10 ms for
  stock's whole step. Stock replays decode as one CUDA graph; EchoKV's
  attention ran eagerly, layer by layer.
- **70B looks better,** probably because each step has more GPU work
  per Python launch to hide behind (70B was not profiled).
- **The Spark hid all of this:** its GPU is slow enough that the Python
  overlapped.
- **The next limit, once the CPU is fixed:** the decode kernel's GPU time
  is 2.5× flash-attention's (1.06 s against 0.43 s in this profile).

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
bash scripts/echokv_bench_h100.sh                                      # 4x H100: 8B, then 70B TP=4
bash scripts/echokv_bench_graphs.sh                                    # round 7: CUDA graphs, ~35 min
```

## Next

**Round 6 (branch `round6`) has run on 4× H100; results are above.** Its
changes:
- **Decode split-merge in Triton:** a merge kernel replaces ~13 small
  PyTorch kernels per layer per step.
- **INT4 is the default:** `ECHOKV_MEM_BITS=4`, with INT4 as the starting
  tier.

**Round 7 (branch `round7`): decode steps run as full CUDA graphs.**
Validated on the Spark and one H100 (results above). The canary's measurements and
tier decisions are unchanged: a CPU run is identical to round 6.
- **The builder does all host-side work, once per step, for all layers:**
  - write planning
  - block opens at each layer's tier, written as page headers
  - evictions
  - K_ref ring positions
  - one host-to-device upload of the step's metadata
  - the canary, batched over layers, with one all-reduce under TP
- **Each layer only launches kernels:**
  - KV update: K-finalize, plus one fused token kernel (V quantize,
    residual K, sealed K and K_ref).
  - Decode: split attention and merge.
  - A query-row copy for the canary.
- **The builder now reports `UNIFORM_SINGLE_TOKEN_DECODE`,** so vLLM
  replays decode-only steps as one graph. The metadata lives at fixed
  addresses, kernels mask padding with device-side counts, and decode uses
  a fixed split grid, with each sequence's split size taken from its own
  length (still batch-invariant).

1. **The decode kernel is the next limit:** about 2× flash-attention's
   GPU time on the H100 now that the CPU overhead is gone (round 7 above).
2. **Rerun 70B (TP=4) on `round7`** with `scripts/echokv_bench_h100.sh`.
3. **Try long-context workloads** (e.g. RULER), where per-layer adaptation
   has more room to help.
