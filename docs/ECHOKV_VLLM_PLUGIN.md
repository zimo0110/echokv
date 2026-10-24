# EchoKV for vLLM

`echokv_vllm/` adds a closed-loop, mixed-precision KV cache to stock vLLM
0.32 as an attention-backend plugin. There is no vLLM fork. Each layer's
KV precision (INT8, INT4 or INT2) is chosen live by a canary-driven
controller, every cache block keeps the precision it was written at, and
the savings are real: vLLM's own allocator fits more tokens in the same
memory.

## Quick start

```bash
pip install vllm==0.32.0
pip install -e .                       # registers the vllm.general_plugins entry point
export ECHOKV_ENABLE=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1
```

```python
from vllm import LLM
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct",
          attention_config={"backend": "CUSTOM"},   # CUDA; on CPU the plugin takes the CPU_ATTN slot
          block_size=64)                            # any multiple of 32
```

All settings are `ECHOKV_*` environment variables (`echokv_vllm/config.py`):

| Variable | Default | Meaning |
|---|---|---|
| `ECHOKV_MEM_BITS` | 8 | Memory budget in bits/element (2, 4, 6, 8). vLLM sizes the cache for it. |
| `ECHOKV_TAU0` | 0.01 | Per-layer canary KL budget: the quality/memory knob. |
| `ECHOKV_FIXED_BITS` | unset | Pin every layer to one tier, with the controller off (static baselines). |
| `ECHOKV_INIT_BITS` | 8 | Starting tier. |
| `ECHOKV_CANARY_INTERVAL` | 64 | Canary fires every N forward steps per layer. |
| `ECHOKV_TAU_SCALING` | none | `sqrt`: τ0·√(max(L, L0)/L0), loosening only above L0. |

Runtime control (read state, freeze, set tiers or τ) goes through
`llm.apply_model` with the functions in `echokv_vllm/control.py`.

## Design

### Quantization (decided by measurement)

* **K: per channel, over groups of 32 tokens (KIVI).** The open group of
  each block is held in a full-precision residual buffer until its 32nd
  token arrives.
* **V: per token, in groups of 32 along head_dim.**
* Both store an **fp16 scale and fp16 min** per group.

Two findings drove this:

1. **Bug in the repo's reference quantizer** (`evaluation/motivation/kivi.py`).
   It used an integer zero-point clamped to `[0, 2^b−1]`, which cannot
   represent groups that don't span zero. Values in [10, 11] reconstruct
   about 10 off, at INT8 and INT4 alike. K outlier channels are exactly
   such groups. The paper's emulation numbers were produced with this
   quantizer.
2. **Per-channel K wins by a wide margin** (WikiText-103, ΔNLL vs FP,
   simulated with post-RoPE K/V):

   | ΔNLL (nats) | Qwen2.5-0.5B INT4 | INT2 | Llama-3.2-1B INT4 | INT2 |
   |---|---|---|---|---|
   | per-token K (old) | +1.67 | +5.50 | +0.041 | +3.78 |
   | center + Hadamard | +0.28 | +3.49 | +0.044 | +2.30 |
   | **per-channel K** | **+0.011** | **+0.71** | **+0.018** | **+0.69** |

   K group 16 is slightly better at INT2, but its metadata costs 2
   bits/element. 32 costs 1 bit/element.

### Storage (`tiered_cache.py`)

Each vLLM page (one block of `bs` tokens, one layer) is laid out as:

```
[ header | V scale/min [bs,H,D/32] | K scale/min [bs/32,H,D] | k_c data chunks ]
```

* A chunk holds one block's K+V codes at INT2, and `k_c = mem_bits/2`.
* An INT2 block lives in its own chunk 0. INT4 takes 1 extra chunk and
  INT8 takes 3, from a per-layer pool made of the other chunk slots.
* The backend's `customize_spec` declares exactly these bytes, so vLLM's
  profiler, allocator and scheduler budget them.
* **Forward-only tiers:** a block's tier is set when its first token is
  written, and changing a layer's tier never rewrites old blocks.
  Attention reads every block at its own tier.
* **Memory pressure:** when the pool can't give a new block its target
  tier, the least-recently-written full blocks are requantized to INT2 in
  place, freeing their chunks. Freed and prefix-cached blocks would
  otherwise hold chunks forever. Only if no block can be demoted does the
  new block open lower.
* **Residual eviction:** if residual slots run out, the least-recently
  written open group is sealed. Its tokens are quantized now; any later
  tokens in that group reuse its stats, clamped. That is lossy, never
  corrupt.

### Write path

* Once per step, the metadata builder runs a `WritePlanner` on the host.
  From the slot mapping alone it works out which blocks open, which K
  groups complete, residual slot assignment and evictions. It needs one
  small device-to-host copy.
* Every layer applies the same plan with no host syncs:
  * host-side chunk allocator with tiny tag/table uploads
  * Triton V-write and K-finalize kernels; the tier is a runtime value,
    so one launch covers all tiers
  * residual scatter, with masked lanes sent to a trash row

### Read path

* **Decode:** a Triton paged-attention kernel (flash-decoding with split
  KV) reads each block at its tier and takes the open group's K from the
  residual.
  * **Batch-invariant:** fixed split boundaries per sequence, so a
    request's output doesn't depend on its batch neighbours.
* **Prefill chunks:** a PyTorch gather, dequantize and SDPA.

### Controller (`canary.py`, `controller.py`)

* **Canary:** an FP16 ring of the last 64 keys per layer, filled from the
  write path. Every `ECHOKV_CANARY_INTERVAL` steps it computes
  δ̂_b = KL(softmax(Q·K̃_b) ‖ softmax(Q·K_ref)) for b ∈ {2, 4, 8}.
  * K̃_b is the cache's exact K format, so the canary measures what the
    cache does.
  * Under TP, δ̂ is averaged across ranks, so every rank makes the same
    decision.
* **Controller:** keeps an EMA of δ̂ per tier and picks the lowest tier
  under τ.

### CUDA graphs

* The builder reports `AttentionCGSupport.NEVER`.
* In vLLM's default piecewise mode, attention and the KV update run
  eagerly and everything between them is graphed, so serving works with
  CUDA graphs on.

## Results (macOS CPU, fp32, vLLM 0.32.0)

**Quality vs memory.** Prompt NLL on WikiText-103 against stock vLLM. Live
rows warm up on generation first.

| | Llama-3.2-1B ΔNLL | bits | Qwen2.5-0.5B ΔNLL | bits |
|---|---|---|---|---|
| static INT8 | +0.0001 | 8.0 | +0.0001 | 8.0 |
| static INT4 | +0.0209 | 4.0 | +0.0116 | 4.0 |
| static INT2 | +0.746 | 2.0 | +0.749 | 2.0 |
| live τ=0.005 | +0.0015 | 7.75 | +0.0001 | 8.0 |
| live τ=0.01 | +0.0209 | 4.0 | +0.003–0.0045 | 6.7–7.0 |
| live τ=0.02 | +0.0209 | 4.0 | +0.0114 | 4.5 |

τ is a real quality/memory knob, and the controller avoids INT2. On these
small models, per-layer adaptivity is roughly neutral against a uniform
INT8/INT4 mix at equal bits. The big gain came from the quantizer.

**KV capacity in vLLM, same budget.** Llama-3.2-1B, 256 MiB, measured from
vLLM's allocator. Per-token byte counts match the layout's prediction
exactly.

| | vs fp16 |
|---|---|
| INT8 | 1.8× |
| INT4 | 3.2× |
| INT2 | 5.6× |

These include scale/min overhead.

**Correctness checks**
* Attention path vs stock vLLM with FP K/V: mean |Δlogprob| 1e-5.
* Store vs oracle: bit-identical.
* TP=2: identical EMAs and tiers on both ranks.
* Greedy outputs for repeated prompts: 36/36 identical without memory
  pressure, like stock vLLM.
* Stress test: 48 requests in a 19-block cache with prefix caching. No
  errors; 87 demotions; output stays coherent.

## Testing

```bash
pytest tests/echokv_vllm                     # 40 tests; Triton cases skip without CUDA
pytest -m slow tests/echokv_vllm             # + boots real vLLM
docker build -t echokv-triton-interp docker/triton-interp
docker run --rm -v "$PWD":/work echokv-triton-interp python -m pytest -q -p no:cacheprovider tests/echokv_vllm
bash scripts/echokv_gpu_validate.sh          # on a CUDA box: tests, quality sweep, throughput
```

The interpreter run puts every store write and decode in the tests
through the Triton kernels and checks them against the PyTorch oracle.
That verifies kernel logic without a GPU, but **the kernels have not yet
been compiled or timed on a real GPU**. `scripts/echokv_gpu_validate.sh`
does that.

macOS needs one vLLM 0.32 fix (`vllm_patches/0001-*.patch`, CPU
slot-mapping shim argument names).

## Limitations

* **Preempted or re-chunked requests can produce different (equally
  valid) greedy outputs.** During decode a query sees its own K group in
  full precision (from the residual); when vLLM recomputes a preempted
  request as a prefill, those tokens are read back quantized. The fix is
  prefill attention with "own group exact" semantics, merging two partial
  attentions by log-sum-exp.
* **The memory budget is per layer.** A sensitive layer cannot borrow
  chunks from an insensitive one, because pools live inside each layer's
  pages.
* **Prefill attention is PyTorch** (gather, dequantize, SDPA). A fused
  Triton prefill kernel is the next performance step after measuring on
  GPU.
* **Unsupported:** sliding window, ALiBi, soft-cap, sinks, MLA,
  encoder-decoder. Head size must be divisible by 32 (powers of two for
  the Triton path).
