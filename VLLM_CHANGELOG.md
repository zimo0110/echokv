# EchoKV vLLM Fork — Changelog

Chronological record of every phase of the EchoKV-native KV cache
integration into the vLLM fork
([/mnt/nvme_tmp10/nazar_vllm_fork](/mnt/nvme_tmp10/nazar_vllm_fork),
branch `feature/echokv-int4-cache`).

For the design document, see
[VLLM_ARCHITECTURE.md](VLLM_ARCHITECTURE.md). For host-side test files
and the controller hook, see [evaluation/serving/fork_tier.py](evaluation/serving/fork_tier.py)
and [tests/test_phaseF1_tier_vector.py](tests/test_phaseF1_tier_vector.py).

Commit hashes refer to the fork branch unless prefixed `host:`.

---

## Phase F.2 — heterogeneous bits (split-buffer layout)

**Fork:** `780cbcc` echokv phase F.2: real heterogeneous KIVI bits via split-buffer layout
**Host:** `e1416c7` phase F.2 host: heterogeneous tier vector test now expects acceptance

### What changed

The blocker for real per-layer mixed-precision was that vLLM's cache
manager requires uniform `page_size_bytes` per block across groups. KIVI's
per-token storage is `data + scale + zp` per group; the data section
scales with `bits` but the scale/zp overhead is constant, so
`unify_kv_cache_spec_page_size`'s block_size adjustment couldn't close
the gap as long as both lived in one buffer.

F.2 splits each KIVI layer's storage into two byte buffers:

  - **paged data buffer** (cache-manager-allocated): `[K_data | V_data]`,
    page size `block_size · num_kv_heads · 2 · (head_size · bits / 8)`
    — scales cleanly with `bits` so block_size adjustment unifies pages
    (INT4 → block_size=32, INT8 → block_size=16, both 32 768 B/block at
    head=128, kv_heads=8).
  - **per-layer scale/zp buffer** (allocated outside the cache manager,
    lazily on first forward): `[K_scale | K_zp | V_scale | V_zp]`,
    independent of `bits`.

Files:
  - `kivi_cache_layout.py` — new `kivi_data_page_size_bytes`,
    `kivi_scalezp_per_block_bytes`, `kivi_scalezp_buffer_bytes`;
    `slice_kivi_cache` accepts both buffers (legacy single-buffer mode
    kept for kernel unit tests).
  - `GroupQuantizedAttentionSpec.real_page_size_bytes` is data-only.
  - `TritonAttentionImpl._kivi_scalezp_buffer` lazily allocated by
    `_ensure_kivi_scalezp_buffer`. `_resolve_kivi_block_size` uses
    `kivi_data_page_size_bytes`. Both `_forward_kivi` and
    `do_kv_cache_update` pass both buffers to `slice_kivi_cache`.
  - `TritonAttentionBackend.get_kv_cache_shape` is data-only.
  - `gpu_model_runner._reshape_kv_cache_tensors` synthesises a per-layer
    `cache_dtype_str` from the spec's `bits` so `get_kv_cache_shape`
    computes the correct page size for heterogeneous specs.
  - `CacheConfig.kivi_bits_per_layer` validator no longer rejects
    heterogeneous values (Phase F.1's "Phase F.2 pending" guard removed).

### Verification

  - 122 unit tests pass under the new layout.
  - Uniform-bits smoke (`/tmp/phaseF_smoke.py`): same outputs as before
    the refactor — no behavioral regression.
  - Heterogeneous smoke (`/tmp/phaseF2_heterogeneous_smoke.py`):
    - `alternating-4-8` (16 INT4 + 16 INT8): boots, all 32 layers report
      requested bits, generation produces "5", "3", "Paris …".
    - `ladder` (10 INT8 / 10 INT4 / 12 INT2): boots, counts match,
      generation produces "5", "3", "Paris, the City of Light …".

### Caveat

vLLM's KV-cache profiler doesn't yet account for the per-layer scale/zp
side buffer. Callers must leave headroom (e.g.
`gpu_memory_utilization=0.65` for 32-layer Llama-8B). Subtracting
side-buffer cost upfront in CacheConfig profiling is a small follow-up.

---

## Phase F.1 — per-layer KIVI bits at engine init

**Fork:** `6c7d188` echokv phase F.1: per-layer KIVI bits at engine init via CacheConfig
**Host:** `81f6c17` phase F.1: host-side per-layer tier-vector helpers + tests

### What changed

Adds the controller's hook for seeding an initial heterogeneous tier
vector at engine init, and clears the integration debt between Phase
E.1's runtime `apply_per_layer_kivi_precision` setter and the spec-time
per-layer page-size decision.

  - `CacheConfig.kivi_bits_per_layer: dict[str, int] | None` — maps
    `Attention.layer_name` to `{2, 4, 8}`. Plumbed through `EngineArgs`
    so callers can pass `LLM(kivi_bits_per_layer={...})`.
  - Pydantic validator rejects invalid bit-widths and (in F.1)
    heterogeneous values with a message pointing at F.2.
  - `Attention.get_kv_cache_spec` consults the dict and emits the
    `GroupQuantizedAttentionSpec` with the per-layer bit-width.
  - `Attention.__init__` mirrors the lookup post-impl-construction and
    pushes the bits into `impl.kivi_bits` via
    `apply_per_layer_kivi_precision`.
  - `GroupQuantizedAttentionSpec.merge` raises `AssertionError` (not
    `ValueError`) when bits differ, so vLLM's `is_kv_cache_spec_uniform`
    correctly detects heterogeneity.
  - `UniformTypeKVCacheSpecs.is_uniform_type` learns about
    `GroupQuantizedAttentionSpec`.

Host-side helpers in
[`evaluation/serving/fork_tier.py`](evaluation/serving/fork_tier.py):
  - `build_uniform_tier_vector(num_layers, bits)` — builds the dict shape
    `CacheConfig.kivi_bits_per_layer` expects (Llama-style layer names).
  - `apply_tier_vector_to_model(model, bits_per_layer)` — walks every
    Attention's `TritonAttentionImpl` and calls
    `apply_per_layer_kivi_precision`.
  - `read_tier_vector_from_model(model)` — diagnostic round-trip.

### Verification

  - Llama-3.1-8B-Instruct boots with
    `kv_cache_dtype="int4_kivi"` + `kivi_bits_per_layer=build_uniform_tier_vector(32, 4)`
    (uniform 32-layer dict), `llm.apply_model(read_tier_vector_from_model)`
    returns `[4]*32` per rank, generation matches the no-dict path
    ("5", "3", "Paris").
  - 13 new unit tests in `tests/test_phaseF1_tier_vector.py`
    (CPU-only, stub Attention model) cover helpers and the validator.

---

## Phase E.4 — fix BLOCK_M / BLOCK_Q grid mismatch

**Fork:** `dcf5286` echokv phase E.4: fix BLOCK_M/BLOCK_Q grid mismatch in KIVI attn

### Bug

Llama-3.1-8B with `kv_cache_dtype="int4_kivi"` produced gibberish output
("Saudi Arabia" instead of "Paris" for the capital of France) despite
all 109 unit tests passing. Methodical diagnostic eliminated GQA
parameters, prefill regime, and TP=1 vs TP=2 as suspects.

Root cause was in `TritonAttentionImpl._forward_kivi`: the call to
`unified_attention_2d_kivi` used the wrapper's defaults
`block_q=16, block_m=16`. The kernel invariant is
`BLOCK_M == num_queries_per_kv * BLOCK_Q`. For Llama-3.1
(`queries_per_kv=4`), `BLOCK_M=16` only allowed `BLOCK_Q=4` query
positions per program — but the launch grid was sized assuming
`BLOCK_Q=16`. Result: prefill silently dropped everything past the
first 4 query tokens.

### Fix

Pass `block_q=16, block_m=self.num_queries_per_kv * block_q` explicitly
in `_forward_kivi`.

### Verification

  - Llama-3.1-8B with `int4_kivi` now produces correct output:
    "Q: What is 2 + 3? A:" → "5", "What is 5 - 2? A:" → "3",
    "The capital of France is" → "a city of romance ... Paris ...".
  - 109/109 unit tests still pass.

---

## Phase E.3 — INT2 and INT8 kernel paths

**Fork:** `c93ea2c` echokv phase E.3: parametric BITS — INT2 and INT8 kernel support

### What changed

The Phase D kernels hard-coded `LEVELS_F=15` (INT4-specific). E.3 makes
the bit-width a parametric `BITS` constexpr in both kernels:

  - Quant ladder: `LEVELS = (1 << BITS) - 1` — INT2 → 3, INT4 → 15,
    INT8 → 255.
  - Pack/unpack math:
    - INT2: 4 codes per byte (`code = (byte >> (2 * parity)) & 3`).
    - INT4: 2 codes per byte (existing path).
    - INT8: 1 code per byte (`code = byte`).
  - `BYTES_PER_GROUP = (group_size · BITS) / 8` parameterised
    accordingly.

### Verification

`tests/test_phaseE3_int2_int8.py` adds parity + end-to-end tests for
INT2 and INT8 (write → read matches unpack-then-eager-SDPA within the
expected bit-width-dependent error band).

---

## Phase E.2 — real-model boot

**Fork:** `5914efb` echokv phase E.2: real-model boot — Llama-3.1-8B with int4_kivi
**Fork (partial):** `ecd6c5e` echokv phase E.2 (partial): unblock engine init for KIVI dtypes

### What changed

Three integration fixes to unblock end-to-end boot of Llama-3.1-8B-Instruct
with `kv_cache_dtype="int4_kivi"`:

  1. `Attention.get_kv_cache_spec` emits `GroupQuantizedAttentionSpec`
     with `dtype=torch.uint8` (storage dtype). Branches on
     `is_kivi_dtype(cache_config.cache_dtype)` ahead of the
     sliding-window / `FullAttentionSpec` branches.
  2. `single_type_kv_cache_manager.spec_manager_map` registers
     `GroupQuantizedAttentionSpec → FullAttentionManager`. Block-allocation
     semantics are identical to plain full attention; only the in-cache
     storage layout differs.
  3. `FullAttentionManager.find_longest_cache_hit` extends the
     `isinstance` check to include `GroupQuantizedAttentionSpec`.

Plus the prerequisites in `ecd6c5e`:
  - `STR_DTYPE_TO_TORCH_DTYPE` registers the four KIVI dtypes as
    `torch.uint8`.
  - `TritonAttentionBackend.get_kv_cache_stride_order` accepts an
    optional `cache_dtype_str` kwarg and returns a 2-element identity
    stride order for KIVI dtypes (matches the
    `(num_blocks, page_bytes)` shape).
  - `GPUModelRunner._reshape_kv_cache_tensors` forwards
    `cache_config.cache_dtype` to `get_kv_cache_stride_order` (with a
    `TypeError` fallback for backwards compatibility).

### Status at end of E.2

End-to-end data flow wired (engine init → cache allocator → per-layer
spec → 32-layer `_forward_kivi` + `do_kv_cache_update` → output). Output
text was *gibberish* ("Saudi Arabia" for "capital of France") — fixed in
Phase E.4.

109/109 unit tests pass.

---

## Phase E.1 — per-layer KIVI config + controller hook

**Fork:** `253e33d` echokv phase E.1: per-layer KIVI config + controller hook

### What changed

Replaces the Phase D.4/D.6 inline reads (`kivi_bits = kivi_bits(self.kv_cache_dtype)`)
and the inferred-from-`page_bytes` block_size kludge with explicit
per-impl state:

  - `self.kivi_bits` — INT{2,4,8}; initialised from `kv_cache_dtype`.
  - `self.kivi_group_size` — default 32 (KIVI default).
  - `self.kivi_block_size` — lazily resolved on first call, cached in
    `_resolve_kivi_block_size`.

The new `apply_per_layer_kivi_precision(bits, group_size=32)` method is
the EchoKV controller's hook for swapping a layer's precision at a
tier-update tick. Validates `bits ∈ {2, 4, 8}`, `group_size` even and
positive, and `head_size` divisibility. Per **INV-2**, only *new* block
allocations pick up the new precision; in-flight blocks continue at
their original bit-width.

### Verification

`tests/test_phaseE1_mixed_precision.py` (4 tests):
  - Two simultaneous KIVI layers at the same bit-width: kernel launches
    don't share constant pollution.
  - Two layers with different `group_size`: independent
    `page_size_bytes` / cache buffers, no cross-contamination.
  - `apply_per_layer_kivi_precision` validation paths.
  - `_resolve_kivi_block_size` lookup, caching, mismatch error.

---

## Phase D.6 — wire KIVI write path into `TritonAttentionImpl`

**Fork:** `61921bc` echokv phase D.6: wire KIVI write path into TritonAttentionImpl

### What changed

`do_kv_cache_update` branches on `is_kivi_dtype(self.kv_cache_dtype)`;
for KIVI it slices the flat byte buffer via `slice_kivi_cache` and
dispatches the write kernel from D.5. The fp8 / fp16 paths are
unchanged.

### Status

Runtime path is now end-to-end through both kernels:

```
model produces K/V (BF16)
   → do_kv_cache_update routes to KIVI write kernel
      → per (token, kv-head, group) quantise-and-pack into the
        section-major cache buffer
   → forward() routes to KIVI attention kernel
      → per (HEAD_SIZE, TILE_SIZE) tile, gather bytes + scale + zp,
        extract nibbles, dequantise inline before Q·K^T
```

99/99 tests passing across phases A–D on H100.

---

## Phase D.5 — write-path kernel

**Fork:** `b35f49b` echokv phase D.5: write-path kernel (BF16 K/V → packed KIVI cache)

### What changed

New kernel `reshape_and_cache_kernel_kivi` quantises BF16/FP16 K and V
on the fly per `(token, kv_head, group)` and writes the packed bytes +
fp16 scale + uint8 zp into the section-major `KiviCacheView` at the slot
indicated by `slot_mapping[token]`. Negative slots are skipped (matches
the FP16 `reshape_and_cache` contract).

Numerical contract: bit-for-bit equivalent to running
`pack_int4_per_group` on the BF16 K/V and writing each slot's output
into the cache view in Python — verified by `torch.equal` across all 6
cache sub-tensors (data, scale, zp × {K, V}).

Public API:

```python
triton_reshape_and_cache_flash_kivi(
    key, value, cache_view, slot_mapping, group_size=32, bits=4,
)
```

### Verification

`tests/test_phaseD5_write_path.py` (6 tests):
  - bf16/fp16 × group_size {16, 32}: bit-for-bit equivalence to
    `pack_int4_per_group`.
  - Padding-token semantics (`slot_mapping == -1` no-op).
  - End-to-end: write via D.5 → read via D.3 attention kernel matches
    unpack-then-eager-SDPA within 5% relative error.

---

## Phase D.4 — dispatcher wiring + section-major cache layout

**Fork:** `e9bc069` echokv phase D.4: dispatcher wiring + section-major cache layout

### What changed

Wires the KIVI attention kernel into `TritonAttentionBackend` so that
`kv_cache_dtype ∈ {int2_kivi, int4_kivi, int8_kivi, nvfp4_e2m1}` routes
through the new path:

  - `TritonAttentionBackend.supported_kv_cache_dtypes` — adds the four KIVI dtypes.
  - `get_kv_cache_shape` returns `(num_blocks, page_size_bytes)` flat
    uint8 for KIVI; defers to `kivi_cache_layout.kivi_page_size_bytes`
    so the spec class and the layout helper agree on a single source of
    truth.
  - `TritonAttentionImpl._forward_kivi` slices the cache via
    `slice_kivi_cache` and dispatches to `unified_attention_2d_kivi`.

### The section-major layout decision

New module `vllm/v1/attention/ops/kivi_cache_layout.py` chose
section-major over block-major, with the latter producing non-contiguous
slices (surprising strides) that break Triton's pointer arithmetic. The
naïve block-major variant surfaced as ~5% relative output error on the
end-to-end test; section-major eliminates it.

### Verification

`tests/test_phaseD4_dispatch.py` (13 tests): page-size accounting,
shape contracts, write-through (zero-copy), C-contiguity, end-to-end
pack→slice→attention vs unpack→eager-SDPA.

---

## Phase D.3 — full feature parity for KIVI unified attention

**Fork:** `c2018b3` echokv phase D.3: full feature parity for KIVI unified attention

### What changed

Layers back the optional features stripped from the D.2
proof-of-concept so the KIVI kernel becomes a drop-in replacement for
`kernel_unified_attention_2d`:

  - `USE_SINKS` (Streaming-LLM attention sinks)
  - `USE_ALIBI_SLOPES` + `USE_ALIBI_SQRT`
  - `USE_QQ_BIAS`
  - `USE_SOFTCAP` (Gemma-style)
  - `SLIDING_WINDOW` (Mistral-style; tile pruning + per-tile mask + V
    zeroing)
  - `USE_MM_PREFIX` + `MAX_MM_RANGES` (multimodal bidirectional ranges)
  - `USE_FP8` output (out_scale + clamp at epilogue)

Structure matches `triton_unified_attention.py` line-for-line; the only
diff is the K/V load blocks, which still use the inline INT4 dequant
pattern from D.2.

### Verification

`tests/test_phaseD3_features.py` (6 tests): softcap, sliding window
{4, 8, 16}, alibi (linear), softcap+sliding combined.

---

## Phase D.2 — unified attention kernel with inline INT4 dequant

**Fork:** `fbb203e` echokv phase D.2: unified attention kernel with inline INT4 dequant

### What changed

New kernel `kernel_unified_attention_2d_kivi` mirrors the
`BLOCK_M × TILE_SIZE × HEAD_SIZE_PADDED` tiling structure of the
production `kernel_unified_attention_2d` but loads K/V from the
EchoKV-native packed cache layout (`uint8 data + fp16 scale + uint8 zp`
per `GroupQuantizedAttentionSpec`) and dequantises inline before Q·K^T.

Dequant pattern: for each `(HEAD_SIZE_PADDED, TILE_SIZE)` tile,

  1. Gather packed bytes at `byte_idx = offs_d // 2`.
  2. Gather per-group fp16 scale and uint8 zp at
     `group_idx = offs_d // GROUP_SIZE`.
  3. Extract nibble via `parity = offs_d % 2` (low/high).
  4. Compute `(q − zp) · scale` in fp32, cast to Q's dtype for the
     dot-product.

Same pattern works for V with d on the inner dim.

### Verification

`tests/test_phaseD2_unified_attn_kivi.py`: builds a real vLLM paged
cache via Phase B `pack_int4_per_group`, runs the new kernel against an
unpack-then-eager-SDPA reference, passes within 5% relative error at
`group_size ∈ {16, 32}` (intrinsic fp16-scale precision propagated
through softmax).

---

## Phase C — `GroupQuantizedAttentionSpec` + `kv_cache_dtype` literals

**Fork:** `b25bf9f` echokv phase C: GroupQuantizedAttentionSpec + kv_cache_dtype literals

### What changed

Cache-side scaffolding for KIVI-style asymmetric per-group INTb KV cache
(foundation for the EchoKV adaptive per-layer precision controller):

  - `CacheDType` literal extended with `int2_kivi`, `int4_kivi`,
    `int8_kivi`, `nvfp4_e2m1`.
  - New `GroupQuantizedAttentionSpec(AttentionSpec)`:
    - `real_page_size_bytes` accounts for packed uint8 data + fp16 scale
      + uint8 zp.
    - `__post_init__` validates `bits ∈ {2,4,8}`, even `group_size`
      dividing `head_size`.
    - `merge()` refuses different-bits/group_size/quantize_v layers,
      forcing per-precision routing into separate KV-cache groups
      (load-bearing for EchoKV's per-layer mixed precision; tightened
      to `AssertionError` in F.1).
    - `quantize_v=False` supports asymmetric K-only quantization.
  - `attention/backend.py` recognises the four new dtypes via
    `is_quantized_kv_cache` and a new `is_kivi_quant_kv_cache` helper.

### Verification

`tests/test_phaseC_kv_cache_spec.py` (32 tests).

---

## Earlier phases (host-only)

Phases A and B preceded the fork integration and live entirely in the
host repo. They built the in-process kernel parity that the fork's
write/read kernels were modelled on:

  - **Phase A** — `kernels/native_int4_kv.py` and
    `tests/test_native_int4_kv.py`. Triton kernel `round_trip_int4_per_group`
    matches `evaluation.motivation.kivi.simulate_quantization` bit-for-bit
    across float32 / fp16 / bf16 × group_size {16, 32, 64} × heavy-tailed
    inputs. Smoke runner: `scripts/19_phaseA_smoke.sh`.
  - **Phase B** — `pack_int4_per_group` / `unpack_int4_per_group` host-side
    pack/unpack reference, used as ground truth for D.4 (slice) and D.5
    (write) parity tests.

These weren't separately commit-tagged in the fork branch since they
don't touch vLLM source.

---

## Test inventory (current)

122 unit tests pass on H100 against the current fork state:

| Phase | Test file | Tests | Coverage |
|------:|-----------|------:|----------|
|     A | test_native_int4_kv.py            | 12+ | Kernel parity |
|     C | test_phaseC_kv_cache_spec.py      | 32 | Spec page-size, merge, kv_cache_dtype literals |
|     D | test_phaseD_attn.py               | varies | Read kernel correctness |
|   D.2 | test_phaseD2_unified_attn_kivi.py | 4 | Inline dequant kernel |
|   D.3 | test_phaseD3_features.py          | 6 | Sinks, alibi, softcap, sliding window, multimodal |
|   D.4 | test_phaseD4_dispatch.py          | 13 | Page size, slice, contiguity, end-to-end |
|   D.5 | test_phaseD5_write_path.py        | 6 | Bit-for-bit write parity, padding, end-to-end |
|   E.1 | test_phaseE1_mixed_precision.py   | 4 | Per-layer state, runtime hook |
|   E.3 | test_phaseE3_int2_int8.py         | varies | INT2 + INT8 parametric kernels |
|   F.1 | test_phaseF1_tier_vector.py       | 13 | Host helpers + CacheConfig validator |

Ten end-to-end smoke drivers under `/tmp/`:

  - `phaseE2_boot.py` — Llama-8B boot at uniform INT4 vs FP16
  - `phaseF_smoke.py` — F.1 plumbing with uniform per-layer dict
  - `phaseF2_heterogeneous_smoke.py` — F.2 mixed-precision patterns

---

## Outstanding follow-ups

1. **Profiler accounting for the per-layer scale/zp side buffer.**
   `gpu_memory_utilization` should subtract side-buffer cost upfront so
   `0.85` works at heterogeneous bits without OOM. Currently the
   workaround is `0.65`. Small fix in `CacheConfig` profiling.

2. **Sliding window + KIVI.** `Attention.get_kv_cache_spec` falls
   through to `SlidingWindowSpec` when `self.sliding_window` is set.
   The kernel already supports `SLIDING_WINDOW` (Phase D.3); only the
   spec/backend/cache-manager wiring is missing.

3. **NVFP4 actual kernel.** `nvfp4_e2m1` is layout-only today
   (shares INT4 storage shape). The paper's NVFP4 baseline plan
   (PAPER_DRAFT.md §7.1) calls for E2M1 quant codes + matching
   pack/unpack and dequant paths.

4. **Closed-loop controller against the fork.** The
   `CanaryHarness` runs in shadow mode against upstream vLLM today.
   Wiring it to drive `apply_tier_vector_to_model` against the fork's
   real KIVI cache is the natural Phase F.3.
