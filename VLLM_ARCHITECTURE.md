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

### 3.5 The `KiviCacheView`

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

### 4.3 `apply_per_layer_kivi_precision` (runtime hook)

`TritonAttentionImpl.apply_per_layer_kivi_precision(bits, group_size=32)`
mutates the impl's `kivi_bits` and `kivi_group_size`. Called by the
EchoKV controller at tier-update ticks. Per **INV-2**, only *new* block
allocations pick up the new precision; in-flight blocks continue at
their original bit-width.

A layer's runtime bits must match the bits its paged buffer was sized
for at engine init. The host-side helper
[`apply_tier_vector_to_model`](evaluation/serving/fork_tier.py) walks
every Attention's impl and calls the hook in lockstep, so the controller
keeps the impl's view consistent with what the spec already chose.

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

### 8.1 Profiler doesn't account for the per-layer scale/zp buffer

vLLM's KV-cache profiler computes `num_blocks` from
`gpu_memory_utilization` and the spec's `page_size_bytes` (data-only).
The per-layer scale/zp side buffer is allocated *on top of* this budget.
For Llama-3.1-8B at 32 layers and `block_size=16`, the side-buffer cost
is roughly 15% of the data-buffer cost.

**Workaround**: callers leave headroom (e.g. `gpu_memory_utilization=0.65`
in the heterogeneous smoke for full TP=1 Llama-8B).

**Proper fix** (small): have `CacheConfig` subtract the side-buffer
overhead from the cache budget upfront — compute
`scalezp_per_token_bytes` at config time and reduce the available cache
memory by that fraction before profiling.

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

### 8.4 F.3.1 — runtime tier swapping within one engine

Phase F.3.0 demonstrates the warmup-then-freeze pattern across two
engine boots:

  * Phase 1 — install ``CanaryHarness`` against the fork (observation
    only — the cache is already KIVI INTb downstream of the hook,
    shadow-quantizing K/V would double-apply noise). Run warmup
    prompts; the controller ticks per-layer δ̂ EMAs and converges to
    a tier vector. (See ``run_phaseF3_canary_fork.py``.)

  * Phase 2 — boot a fresh engine with
    ``LLM(kivi_bits_per_layer=<warmup tier>)`` so each layer's KV
    cache is allocated at the controller-chosen bits from the very
    first measurement token. (See ``run_phaseF3_two_phase.py``.)

True per-layer dynamic tier swapping inside one engine still requires
F.3.1 work: each layer's paged data buffer is sized for one
bit-width at engine init via the spec; ``apply_per_layer_kivi_precision``
mutates ``impl.kivi_bits`` in place but the next
``slice_kivi_cache`` asserts the buffer matches
``page_bytes(active_bits)`` exactly. Resolving this needs either
(a) allocate-at-max-bits + a tolerant slice that accepts
``buffer_size >= num_blocks * page_bytes(active_bits)``, or
(b) per-tier dedicated buffers with the impl swapping which buffer
is active.

## 9. Performance characteristics (current snapshot, no perf tuning)

Llama-3.1-8B-Instruct, TP=1, H100 80GB, `block_size=16`,
`enforce_eager=True`, `gpu_memory_utilization=0.65–0.85`, `max_model_len=4096`:

  - Uniform INT4 (`kv_cache_dtype="int4_kivi"`, no per-layer dict):
    ~110 toks/s input, ~155 toks/s output on the 3-prompt smoke.
  - Heterogeneous (16 INT4 + 16 INT8): same correctness, ~14–25
    toks/s output (decode-bound; the kernel hasn't been tuned for
    heterogeneous launch patterns yet).
  - Heterogeneous (10 INT8 / 10 INT4 / 12 INT2 ladder): ~24 toks/s
    output.

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
    gpu_memory_utilization=0.65,     # leave headroom for scale/zp
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
  - Phase F.1: `test_phaseF1_tier_vector.py` — host-side helpers,
    `CacheConfig.kivi_bits_per_layer` validator, dict ↔ list forms,
    error paths.

End-to-end smoke drivers under `/tmp/`:

  - `phaseE2_boot.py` — boot Llama-8B with `int4_kivi`, compare against FP16.
  - `phaseF_smoke.py` — Phase F.1 plumbing with a uniform per-layer dict.
  - `phaseF2_heterogeneous_smoke.py` — Phase F.2 heterogeneous patterns
    (`alternating-4-8`, `first-half-int8`, `ladder`).

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
