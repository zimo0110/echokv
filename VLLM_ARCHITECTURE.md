# EchoKV vLLM Fork — Architecture

This document covers the design of the EchoKV-native KV cache integration
in the vLLM fork at [/mnt/nvme_tmp10/nazar_vllm_fork](/mnt/nvme_tmp10/nazar_vllm_fork)
(branch `feature/echokv-int4-cache`).

The companion document — [VLLM_CHANGELOG.md](VLLM_CHANGELOG.md) — is the
chronological record of how we got here, phase by phase.

## 1. Goal

Make vLLM v1 serve a model with a **KIVI-style group-quantized KV cache**
where the bit-width is **chosen per attention layer at engine init** and
can be updated per layer by the EchoKV controller at runtime.

KIVI quantization (asymmetric, per-group, INTb) stores three tensors per
(token, kv-head) pair:

  - `data`  — packed sub-byte codes, `(head_size * bits) / 8` bytes
  - `scale` — fp16, one per group, `head_size / group_size` per head
  - `zp`    — uint8, one per group, `head_size / group_size` per head

Reading a slot is `(code - zp) * scale` per group, dequantised inline
before the Q·K^T multiply.

## 2. End-to-end data flow

```
                 ┌──────────────────────────────────────────────┐
   user prompt → │  vLLM engine init                            │
                 │   ↓                                          │
                 │  Attention.get_kv_cache_spec  ← per-layer    │
                 │   ↓ (one spec per layer)        bits dict    │
                 │  cache manager: groups specs by bits/size    │
                 │   ↓ per group                                │
                 │  paged data buffer (cache manager-allocated) │
                 └──────────────────────────────────────────────┘
                                       │
                 ┌─────────────────────┴───────────────┐
                 │  forward path (per layer, per step) │
                 │                                     │
   K/V (BF16) ──→│ do_kv_cache_update                  │
                 │   → triton_reshape_and_cache_..._kivi│
                 │      writes K_data, V_data into     │
                 │      paged buffer + K/V scale/zp    │
                 │      into per-layer side buffer     │
                 │                                     │
   Q (BF16)  ───→│ _forward_kivi                       │
                 │   → unified_attention_2d_kivi       │
                 │      gathers data + scale + zp,     │
                 │      inline dequant, Q·K^T, softmax │
                 │      ── output (BF16)               │
                 └─────────────────────────────────────┘
```

## 3. Cache layout (Phase F.2 — current)

Each KIVI layer's storage is split across **two byte buffers**:

### 3.1 Paged data buffer

Managed by vLLM's cache manager. Holds only the K/V data sections,
section-major:

```
[ K_data (all blocks) | V_data (all blocks) ]
```

Page size:

```
page_size_bytes = block_size · num_kv_heads · 2 · (head_size · bits / 8)
```

Because the page size scales linearly with `bits`, vLLM's existing
`unify_kv_cache_spec_page_size` (which doubles `block_size` on layers
with smaller pages) cleanly balances heterogeneous-bits groups:

| Layer bits | Spec block_size after unify | Per-block bytes (head=128, kv_heads=8) |
|-----------:|----------------------------:|----------------------------------------:|
|       INT8 |                          16 |                                  32 768 |
|       INT4 |                          32 |                                  32 768 |
|       INT2 |                          64 |                                  32 768 |

### 3.2 Per-layer scale/zp buffer

Allocated outside the cache manager, lazily on a layer's first forward
when `num_blocks` becomes known from the bound paged cache. Holds the
four metadata sections, section-major:

```
[ K_scale (all blocks) | K_zp (all blocks) | V_scale (all blocks) | V_zp (all blocks) ]
```

Per-block bytes:

```
scalezp_per_block = block_size · num_kv_heads · 2 · (n_groups · 3)
```

where `n_groups = head_size / group_size` and the `· 3` is `2 (fp16
scale) + 1 (uint8 zp)`. Independent of `bits`.

Total side-buffer size per layer:
`num_blocks · scalezp_per_block`.

### 3.3 Why split

KIVI's per-token storage is `data + scale + zp`. The scale/zp footprint
is constant in `bits` while the data section scales with `bits`. With
both in one buffer, `block_size` adjustment cannot make pages match
across heterogeneous-bits layers (the constant term doesn't scale).
Externalizing scale/zp removes the constant term from the paged budget;
the residual data section then scales cleanly so vLLM's existing
unification logic just works.

The naïve alternative (pad smaller pages to the largest) wastes a lot
of memory at INT2; the split-buffer design wastes none.

### 3.4 Section-major (not block-major)

Inside each buffer, all blocks of one section are contiguous before the
next section starts. Per-section storage is C-contiguous over
`(num_blocks, block_size, num_kv_heads, last_dim)`, so `.view(...)`
returns true zero-copy contiguous views. A naïve block-major layout
(`[K_data block 0 | K_scale block 0 | ... | V_zp block 0 | K_data block 1 | ...]`)
produces non-contiguous slices with surprising strides that break
Triton's pointer arithmetic in the attention kernel. (We hit and fixed
this in Phase D.4 — it surfaced as ~5% relative output error.)

### 3.5 Profile-aware accounting (Phase F.2.1)

The split-buffer layout puts the scale/zp tensors outside vLLM's paged
KV-cache budget. To prevent the side buffers from OOMing the engine
during the first forward, the worker's profiling pass and cache-init
pass each handle one half of the accounting:

  * **`determine_available_memory`** subtracts the side-buffer overhead
    upfront. The helper walks each KIVI layer's spec, computes its
    post-unify `block_size` (smaller bits get larger block_size after
    `unify_kv_cache_spec_page_size`), totals the data and side
    per-block bytes, and reduces `available_kv_cache_memory_bytes` by
    the side fraction. The fraction is computed against
    *per-group* data because vLLM allocates one buffer per
    cache-group; for heterogeneous specs this matches what the cache
    manager actually allocates.

  * **`initialize_from_config`** eagerly allocates each layer's side
    buffer immediately after vLLM's `bind_kv_cache`, so the subsequent
    warmup forward pass sees the real memory footprint and doesn't
    OOM trying to fault them in alongside peak activation memory.

### 3.6 The `KiviCacheView`

Both buffers are sliced into a single `KiviCacheView` with six tensor
fields (`k_data`, `k_scale`, `k_zp`, `v_data`, `v_scale`, `v_zp`). The
kernels take the view and don't know whether the six tensors come from
one buffer (legacy single-buffer mode, used in kernel unit tests) or
two (the production split-buffer mode).

Source: [`vllm/v1/attention/ops/kivi_cache_layout.py`](/mnt/nvme_tmp10/nazar_vllm_fork/vllm/v1/attention/ops/kivi_cache_layout.py).

## 4. Configuration surface

### 4.1 `kv_cache_dtype` (engine-global)

A single string on `CacheConfig.cache_dtype`, accepted via `LLM(kv_cache_dtype=...)`:

  - `int2_kivi` / `int4_kivi` / `int8_kivi` — KIVI INTb at the indicated bit-width
  - `nvfp4_e2m1` — NVFP4 (E2M1) layout (currently shares the INT4 storage shape)
  - other fp8/auto values — unchanged from upstream vLLM

### 4.2 `kivi_bits_per_layer` (per-layer override)

A `dict[str, int] | None` on `CacheConfig.kivi_bits_per_layer`, accepted
via `LLM(kivi_bits_per_layer=...)`. Maps `Attention.layer_name` (e.g.
`model.layers.0.self_attn.attn`) to the bit-width that layer's spec
should use. Layers not present in the dict fall back to the bit-width
encoded in `cache_dtype`.

Validator (in `CacheConfig`):

  - keys are non-empty strings
  - values are in `{2, 4, 8}`

Heterogeneous values across layers are accepted (Phase F.2). The earlier
F.1 guard that rejected mixed bits is gone.

### 4.3 `apply_per_layer_kivi_precision` (engine-init hook)

`TritonAttentionImpl.apply_per_layer_kivi_precision(bits, group_size=32)`
sets a layer's bit-width *before* the data buffer is allocated. Called
by `Attention.__init__` to apply the per-layer override from
`CacheConfig.kivi_bits_per_layer` (Phase F.1). Sets both the active
`kivi_bits` and the buffer's `kivi_buffer_bits` to the same value —
the spec then sizes the data buffer accordingly.

For runtime swaps after the buffer is allocated, use
[`swap_kivi_bits`](#44-swap_kivi_bits-runtime-hook) instead.

### 4.4 `swap_kivi_bits` (runtime hook, Phase F.3.1)

`TritonAttentionImpl.swap_kivi_bits(new_bits)` mutates only the active
`kivi_bits`, leaving `kivi_buffer_bits` pinned at the engine-init
value. Called by the EchoKV controller between request batches.

Constraints:

  * `new_bits ∈ {2, 4, 8}`
  * `new_bits ≤ kivi_buffer_bits` (the buffer was sized for the maximum;
    upsizing past it is rejected)

Caller contract (drain-and-swap): no in-flight requests may reference
the cache when `swap_kivi_bits` is called. Existing cache blocks were
written at the previous active bits; they're uninterpretable at the
new active bits. Mid-request swaps need per-block bit tagging
(Phase F.3.2).

The host-side helper
[`swap_tier_vector_to_model`](evaluation/serving/fork_tier.py) walks
every Attention's impl and calls `swap_kivi_bits` in lockstep across
TP ranks via `llm.apply_model`.

## 5. Code layout

### 5.1 In the vLLM fork

| File | Role |
|------|------|
| [`vllm/config/cache.py`](/mnt/nvme_tmp10/nazar_vllm_fork/vllm/config/cache.py) | `CacheConfig` + `kivi_bits_per_layer` validator |
| [`vllm/engine/arg_utils.py`](/mnt/nvme_tmp10/nazar_vllm_fork/vllm/engine/arg_utils.py) | Plumbs `kivi_bits_per_layer` through `EngineArgs` |
| [`vllm/v1/kv_cache_interface.py`](/mnt/nvme_tmp10/nazar_vllm_fork/vllm/v1/kv_cache_interface.py) | `GroupQuantizedAttentionSpec` (data-only `real_page_size_bytes`); `UniformTypeKVCacheSpecs` knows `GroupQuantizedAttentionSpec` |
| [`vllm/model_executor/layers/attention/attention.py`](/mnt/nvme_tmp10/nazar_vllm_fork/vllm/model_executor/layers/attention/attention.py) | `Attention.get_kv_cache_spec` consults `kivi_bits_per_layer`; `Attention.__init__` aligns `impl.kivi_bits` post-construction |
| [`vllm/v1/attention/backends/triton_attn.py`](/mnt/nvme_tmp10/nazar_vllm_fork/vllm/v1/attention/backends/triton_attn.py) | `TritonAttentionBackend.get_kv_cache_shape` (data-only); `TritonAttentionImpl` (`_forward_kivi`, `do_kv_cache_update`, `apply_per_layer_kivi_precision`, `_ensure_kivi_scalezp_buffer`) |
| [`vllm/v1/attention/ops/kivi_cache_layout.py`](/mnt/nvme_tmp10/nazar_vllm_fork/vllm/v1/attention/ops/kivi_cache_layout.py) | Page-size helpers, `slice_kivi_cache`, `KiviCacheView` |
| [`vllm/v1/attention/ops/triton_reshape_and_cache_flash_kivi.py`](/mnt/nvme_tmp10/nazar_vllm_fork/vllm/v1/attention/ops/triton_reshape_and_cache_flash_kivi.py) | KIVI write kernel — quantises BF16/FP16 K/V, packs into the cache view |
| [`vllm/v1/attention/ops/triton_unified_attention_kivi.py`](/mnt/nvme_tmp10/nazar_vllm_fork/vllm/v1/attention/ops/triton_unified_attention_kivi.py) | KIVI read kernel — gathers + dequantises inline before Q·K^T |
| [`vllm/v1/core/single_type_kv_cache_manager.py`](/mnt/nvme_tmp10/nazar_vllm_fork/vllm/v1/core/single_type_kv_cache_manager.py) | `GroupQuantizedAttentionSpec → FullAttentionManager` registration |
| [`vllm/v1/worker/gpu_model_runner.py`](/mnt/nvme_tmp10/nazar_vllm_fork/vllm/v1/worker/gpu_model_runner.py) | `_reshape_kv_cache_tensors` synthesises per-layer `cache_dtype_str` for heterogeneous specs |

### 5.2 In the host repo

| File | Role |
|------|------|
| [`evaluation/serving/fork_tier.py`](evaluation/serving/fork_tier.py) | `build_uniform_tier_vector`, `apply_tier_vector_to_model`, `read_tier_vector_from_model` — TP-shippable host-side helpers |
| [`tests/test_phaseF1_tier_vector.py`](tests/test_phaseF1_tier_vector.py) | 13 unit tests (CPU-only, stub Attention model) for the helpers + the `CacheConfig.kivi_bits_per_layer` validator |
| [`tests/test_phaseC_kv_cache_spec.py`](tests/test_phaseC_kv_cache_spec.py), [`tests/test_phaseD_attn.py`](tests/test_phaseD_attn.py), [`tests/test_phaseD2_unified_attn_kivi.py`](tests/test_phaseD2_unified_attn_kivi.py), [`tests/test_phaseD3_features.py`](tests/test_phaseD3_features.py), [`tests/test_phaseD4_dispatch.py`](tests/test_phaseD4_dispatch.py), [`tests/test_phaseD5_write_path.py`](tests/test_phaseD5_write_path.py), [`tests/test_phaseE1_mixed_precision.py`](tests/test_phaseE1_mixed_precision.py), [`tests/test_phaseE3_int2_int8.py`](tests/test_phaseE3_int2_int8.py), [`tests/test_phaseF1_tier_vector.py`](tests/test_phaseF1_tier_vector.py) | 122 tests covering kernels, cache layout, dispatch, mixed-precision, write path, INT2/INT8 parametric kernels, and per-layer config plumbing |

## 6. Invariants

The fork integration upholds the same invariants as the host EchoKV
core (see [ARCHITECTURE.md](ARCHITECTURE.md) §9 for the full set):

  - **INV-1** (no per-block aux state): per-impl state is per-process,
    per-layer; no per-block aux state.
  - **INV-2** (no rewrite of existing blocks): runtime
    `apply_per_layer_kivi_precision` only affects *new* allocations.
  - **INV-3** (TP broadcast synchrony): each TP rank runs its own
    controller and converges independently to the same tier vector
    (deterministic seeding); the harness applies tier vectors to all
    ranks in lockstep via `llm.apply_model`.
  - **INV-4** (bit-exact non-canary path): the FP16 path is unchanged;
    the KIVI path activates only when `kv_cache_dtype` is one of the
    `int*_kivi` strings.

## 7. Dispatch logic

### 7.1 Engine-init dispatch

`Attention.get_kv_cache_spec` ([attention.py:516](/mnt/nvme_tmp10/nazar_vllm_fork/vllm/model_executor/layers/attention/attention.py#L516)) decides per layer:

```python
if is_kivi_dtype(cache_dtype_str) and self.sliding_window is None:
    spec_bits = kivi_bits(cache_dtype_str)         # global default
    if cache_config.kivi_bits_per_layer is not None:
        spec_bits = override.get(self.layer_name, spec_bits)
    return GroupQuantizedAttentionSpec(..., bits=spec_bits, ...)
elif self.sliding_window is not None:
    return SlidingWindowSpec(...)
else:
    return FullAttentionSpec(...)
```

KIVI + sliding window is currently rejected; the existing FP16 sliding
window is preserved. KIVI + MLA is rejected (asserted).

### 7.2 Forward dispatch

`TritonAttentionImpl.forward` ([triton_attn.py](/mnt/nvme_tmp10/nazar_vllm_fork/vllm/v1/attention/backends/triton_attn.py)) routes:

```python
if is_kivi_dtype(self.kv_cache_dtype):
    return self._forward_kivi(query, kv_cache, attn_metadata, output)
# ... existing FP16 / FP8 / MLA paths ...
```

`do_kv_cache_update` mirrors the same branch for the write path.

### 7.3 Cache grouping

Heterogeneous `GroupQuantizedAttentionSpec` layers go through:

  1. `is_kv_cache_spec_uniform(specs)` → False (different `bits` →
     `merge` raises `AssertionError`).
  2. `UniformTypeKVCacheSpecs.from_specs(specs)` — checks
     `is_uniform_type` which now has a `GroupQuantizedAttentionSpec`
     branch requiring matching `bits / group_size / quantize_v`.
     Returns None for mixed bits → falls through.
  3. `unify_kv_cache_spec_page_size(specs)` — adjusts `block_size` on
     smaller-page layers so all pages match the max. **This is the load-
     bearing step**: with the data-only page-size from F.2, the math
     closes (page = `block_size · K · 2 · (D · bits / 8)` is invariant
     under doubling block_size while halving bits).
  4. `_get_kv_cache_groups_uniform_page_size(specs)` — groups layers by
     spec equality. Different-bits specs end up in different groups,
     each with its own block table; `merge` within a group sees only
     uniform-bits specs and succeeds.

## 8. Known limitations / follow-ups

### 8.1 ~~Profiler doesn't account for the per-layer scale/zp buffer~~ — fixed in Phase F.2.1

`gpu_worker.py:determine_available_memory` now subtracts the per-layer
scale/zp side-buffer overhead from `available_kv_cache_memory_bytes`
before returning, and `gpu_worker.py:initialize_from_config` eagerly
allocates each layer's side buffer immediately after the data buffer
is bound (so vLLM's subsequent warmup forward sees the true memory
footprint rather than triggering lazy allocation alongside peak
activation memory).

Callers can use `gpu_memory_utilization=0.85` and `max_model_len=4096`
(upstream defaults) at uniform or heterogeneous bits without OOM —
all the F.1, F.2, and F.3.0 demos run at those settings post-F.2.1.

### 8.2 Sliding window + KIVI not yet supported

`Attention.get_kv_cache_spec` falls through to `SlidingWindowSpec` when
`self.sliding_window is not None`. Wiring the KIVI path through the
sliding-window backend is a follow-up; the kernel already supports
sliding-window via `SLIDING_WINDOW` (Phase D.3), but the
spec/backend/cache-manager wiring isn't in place.

### 8.3 NVFP4 is layout-only, not a real kernel

`nvfp4_e2m1` is recognised as a KIVI dtype and uses the same
`(uint8 data + fp16 scale + uint8 zp)` storage shape, but the actual
quant codes for E2M1 floats and the zero-point semantics aren't
implemented. The paper's NVFP4 baseline plan (PAPER_DRAFT.md §7.1) calls
for this kernel.

### 8.4 ~~F.3.1 — runtime tier swapping within one engine~~ — fixed in Phase F.3.1

Phase F.3.1 implemented the **allocate-at-max-bits + tolerant slice**
design. ``slice_kivi_cache`` accepts a ``buffer_bits`` parameter
distinct from the active ``bits``; when ``buffer_bits > bits`` the K
and V views are sized at the active bits (so kernel pointer arithmetic
is correct) while V's offset within the buffer uses ``buffer_bits``
(so V starts in its allocated region, not the unused tail of the K
section). ``TritonAttentionImpl.swap_kivi_bits(new_bits)`` is the
runtime hook; ``swap_tier_vector_to_model`` is the host-side
TP-shippable counterpart.

The contract is **drain-and-swap**: the caller must ensure no
in-flight requests reference the cache before swapping (existing
blocks were written at the previous bit-width and are uninterpretable
at the new bit-width). Mid-request swaps (per-block bit tagging) are
deferred to Phase F.3.2.

For the warmup-then-freeze pattern Phase F.3.0 still works exactly as
documented — F.3.1 is additive. Common usage:

  * **Boot at the maximum bit-width** any layer might ever use (e.g.
    ``kv_cache_dtype="int8_kivi"`` with no ``kivi_bits_per_layer``).
    Each layer's data buffer is sized for INT8.
  * **Initial tier**: optionally pin a starting tier via
    ``apply_per_layer_kivi_precision`` post-init or
    ``kivi_bits_per_layer`` at init.
  * **Swap at runtime** via ``swap_tier_vector_to_model`` between
    request batches — controller decisions take effect immediately on
    the next batch without engine restart.

## 9. Performance characteristics (current snapshot, no perf tuning)

Llama-3.1-8B-Instruct, TP=1, H100 80GB, `block_size=16`,
`enforce_eager=True`, `gpu_memory_utilization=0.85`, `max_model_len=4096`
(upstream serving defaults; F.2.1's profile-aware accounting + eager
side-buffer allocation lets all configurations run at these settings):

  - Uniform INT4 (`kv_cache_dtype="int4_kivi"`, no per-layer dict):
    ~107 toks/s input, ~146 toks/s output on the 3-prompt smoke.
  - Heterogeneous 16 INT4 + 16 INT8: same correctness, ~107 toks/s
    input, ~153 toks/s output.
  - Heterogeneous 10 INT8 / 10 INT4 / 12 INT2 ladder: ~105 toks/s
    input, ~151 toks/s output.

Headline accuracy parity (matching FP16) on the 3-prompt smoke is
maintained for all three configurations.

## 10. How to use

### 10.1 Boot at uniform INT4

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    kv_cache_dtype="int4_kivi",
    tensor_parallel_size=1,
    dtype="bfloat16",
    block_size=16,
    enforce_eager=True,
    gpu_memory_utilization=0.85,
)
out = llm.generate(["Hello"], SamplingParams(temperature=0.0, max_tokens=24))
```

### 10.2 Boot at heterogeneous mix

```python
from vllm import LLM, SamplingParams
from evaluation.serving.fork_tier import (
    apply_tier_vector_to_model,
    read_tier_vector_from_model,
)

NUM_LAYERS = 32  # Llama-3.1-8B
tier = {
    f"model.layers.{i}.self_attn.attn": (8 if i % 2 == 0 else 4)
    for i in range(NUM_LAYERS)
}

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    kv_cache_dtype="int8_kivi",      # max bits in the dict
    kivi_bits_per_layer=tier,
    tensor_parallel_size=1,
    dtype="bfloat16",
    block_size=16,
    enforce_eager=True,
    gpu_memory_utilization=0.85,     # F.2.1 reserves scale/zp upfront
)

# Verify the per-layer dispatch.
rank_vectors = llm.apply_model(read_tier_vector_from_model)
assert rank_vectors[0] == [tier[f"model.layers.{i}.self_attn.attn"]
                           for i in range(NUM_LAYERS)]
```

### 10.3 Update tier vector at runtime (controller hook)

```python
# Same dict shape; values that match the spec's allocated bits work,
# others raise ValueError on the next forward.
def _push(model):
    return apply_tier_vector_to_model(model, new_tier_vector)

llm.apply_model(_push)   # broadcast across TP ranks
```

### 10.4 Warmup-then-freeze with the controller (Phase F.3.0)

Wire the EchoKV ``CanaryHarness`` against the fork to converge a tier
vector, then boot a fresh engine with that vector pinned via
``kivi_bits_per_layer``. End-to-end the loop is: canary observes Q/K
via forward-pre-hooks → controller ticks per-layer δ̂ EMAs → resulting
tier vector seeds ``LLM(kivi_bits_per_layer=...)`` for measurement.

```bash
# Single-pass canary observation (warmup only):
CUDA_VISIBLE_DEVICES=4 \
VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
python evaluation/serving/run_phaseF3_canary_fork.py --tp 1

# Full warmup-then-freeze across two engine boots:
CUDA_VISIBLE_DEVICES=4 \
python evaluation/serving/run_phaseF3_two_phase.py --tp 1
```

The two-phase driver runs each engine in its own subprocess so the
second engine starts with a clean GPU state (NCCL groups, vLLM's pool,
and per-layer scale/zp side buffers from the first engine all linger
past ``del llm``). Use observation-only mode on the fork
(``CanaryHarness(tier_application=False)``) — the cache is already
quantised at write time, so shadow-quantizing K/V before attention
would double-apply noise.

### 10.5 Runtime tier swap inside one engine (Phase F.3.1)

```python
from vllm import LLM, SamplingParams
from evaluation.serving.fork_tier import (
    build_uniform_tier_vector,
    swap_tier_vector_to_model,
)

# Boot at the maximum bit-width. With no kivi_bits_per_layer the buffer
# is sized for the engine-global cache_dtype's bits.
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    kv_cache_dtype="int8_kivi",
    tensor_parallel_size=1,
    dtype="bfloat16",
    block_size=16,
    enforce_eager=True,
    gpu_memory_utilization=0.85,
)

# Batch 1 at active=INT8.
sp = SamplingParams(temperature=0.0, max_tokens=32)
llm.generate(["..."], sp)

# Drain (handled implicitly by `generate` returning), then swap to INT4.
def _swap(model):
    return swap_tier_vector_to_model(model, build_uniform_tier_vector(32, 4))

llm.apply_model(_swap)   # broadcast across TP ranks

# Batch 2 at active=INT4 in the SAME engine — no restart.
llm.generate(["..."], sp)
```

`swap_tier_vector_to_model` keeps each layer's `kivi_buffer_bits`
pinned at engine-init bits and only mutates `kivi_bits` (the active
bit-width the kernel reads). It's a synchronous, idempotent call;
tied to `llm.apply_model` so all TP ranks see the same change before
the next forward.

## 11. Testing

122 tests across the host repo's `tests/` directory cover every phase
of the integration:

  - Phase A: `test_native_int4_kv.py` — kernel parity with the Phase B
    pack/unpack reference.
  - Phase C: `test_phaseC_kv_cache_spec.py` — `GroupQuantizedAttentionSpec`
    page-size accounting and merge semantics.
  - Phase D: `test_phaseD_attn.py`, `test_phaseD2_unified_attn_kivi.py`,
    `test_phaseD3_features.py`, `test_phaseD4_dispatch.py`,
    `test_phaseD5_write_path.py` — read kernel, full feature parity,
    dispatch wiring, write kernel, end-to-end pack→write→read.
  - Phase E.1: `test_phaseE1_mixed_precision.py` — per-layer config
    state and the runtime hook.
  - Phase E.3: `test_phaseE3_int2_int8.py` — INT2 and INT8 parametric
    kernel paths.
  - Phase F.1 / F.3.1: `test_phaseF1_tier_vector.py` — host-side
    helpers (init-time + runtime swap),
    `CacheConfig.kivi_bits_per_layer` validator, dict ↔ list forms,
    error paths, `swap_tier_vector_to_model` chained / upsize
    rejection / dict form.

End-to-end accuracy parity drivers under `evaluation/serving/`:

  - `run_phaseG_eval_fork.py` — RULER and HotpotQA at FP16 vs
    INT4_kivi on Llama-3.1-8B. Two-engine subprocess pattern; emits
    a per-prompt + summary record per cache dtype with a verdict on
    statistical parity. Verified: RULER 4K/8K/16K all 100%/100%,
    HotpotQA n=30 hard 0.467/0.500 (Δ=+0.033, within noise).

End-to-end smoke drivers under `/tmp/`:

  - `phaseE2_boot.py` — boot Llama-8B with `int4_kivi`, compare against FP16.
  - `phaseF_smoke.py` — Phase F.1 plumbing with a uniform per-layer dict.
  - `phaseF2_heterogeneous_smoke.py` — Phase F.2 heterogeneous patterns
    (`alternating-4-8`, `first-half-int8`, `ladder`).
  - `phaseF31_runtime_swap_smoke.py` — Phase F.3.1 runtime tier swap.
    Boots at INT8 buffer, runs three batches with active = 8 → 4 → 2,
    each swap inside one engine via `swap_tier_vector_to_model`.

Run the full unit suite:

```bash
PYTHONPATH=/mnt/nvme_tmp10/nazar_vllm_fork:/home/nazar/echokv \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
python -m pytest tests/test_native_int4_kv.py \
                 tests/test_phaseC_kv_cache_spec.py \
                 tests/test_phaseD_attn.py \
                 tests/test_phaseD2_unified_attn_kivi.py \
                 tests/test_phaseD3_features.py \
                 tests/test_phaseD4_dispatch.py \
                 tests/test_phaseD5_write_path.py \
                 tests/test_phaseE1_mixed_precision.py \
                 tests/test_phaseE3_int2_int8.py \
                 tests/test_phaseF1_tier_vector.py
```
