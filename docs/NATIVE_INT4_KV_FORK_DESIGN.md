# Native INT4 KV cache — vLLM fork design

> Living spec for the Phase-11 work (PAPER_DRAFT §7). Branch:
> `feature/native-int4-kv-cache`.

## Goal

Add per-layer mixed-precision INT4 / NVFP4 KV cache to vLLM 0.16,
with KIVI-equivalent (per-group, group-32) quantization, paged
storage, and inline dequant in the attention kernel. Per-layer
dispatch lets EchoKV's controller select bit-widths.

End-state (target): wall direction inverts vs FP16 (quantized faster,
not slower). Accuracy parity with our existing emulated cells.

## Storage contract

For each transformer layer's KV cache:

```
data_K:   uint8,  shape (num_blocks, block_size, num_heads, head_dim // 2)
data_V:   uint8,  shape (num_blocks, block_size, num_heads, head_dim // 2)
scale_K:  float16, shape (num_blocks, block_size, num_heads, head_dim // group_size)
scale_V:  float16, shape (num_blocks, block_size, num_heads, head_dim // group_size)
zp_K:     uint8,   shape (num_blocks, block_size, num_heads, head_dim // group_size)
zp_V:     uint8,   shape (num_blocks, block_size, num_heads, head_dim // group_size)
```

Notes:
- `head_dim` must be divisible by `group_size`. For Llama-3.1-70B
  with `head_dim=128` and `group_size=32`, that's 4 groups per (token, head).
- Two INT4 codes per byte: low nibble = even index, high nibble = odd
  index along `head_dim`.
- Scale stored as fp16 to stay aligned with vLLM's existing fp8
  scale tensors. ZP stored as uint8 (range 0..15).
- Scale + ZP layout matches the data layout's first three dims so a
  given (block, slot, head) loads its data + scale + zp from the same
  page address with a small offset.
- Memory cost vs BF16 baseline:
  - Data: 4 bits/elem instead of 16 → 4× saving.
  - Scale: 16 bits per `group_size` elements → +16/32 = 0.5 bits/elem.
  - ZP: 8 bits per `group_size` elements → +8/32 = 0.25 bits/elem.
  - **Net: 4.75 bits/elem ≈ 70% saving** vs BF16 (KIVI claims 75%
    because they don't carry zero-point as a separate tensor; we
    do for symmetry with our existing `kivi.py` reference).

## Per-layer mixed precision

Each layer's KV cache is independently allocated. `KVCacheSpec` (in
`vllm/v1/kv_cache_interface.py`) is already keyed per-layer; we add
a new spec subclass:

```python
@dataclass(frozen=True, kw_only=True)
class GroupQuantizedAttentionSpec(AttentionSpec):
    bits: int                    # 2, 4, or 8
    group_size: int = 32
    # The base layout is the same as FullAttentionSpec; only the
    # storage dtype + extra scale/zp tensors differ.
```

Layers can advertise different `bits` values. `EchoKV-3C` reuses
its existing controller; on each tier-update tick the controller
calls a new `apply_per_layer_bit_widths(layer_idx -> bits)` method
on the cache manager that swaps the layer's cache spec at the next
new-block allocation (INV-2: existing blocks never change).

## Patch points

| file | change |
|---|---|
| `vllm/config/cache.py` | add `kv_cache_dtype` choices: `int4_kivi`, `int8_kivi`, `nvfp4_e2m1` |
| `vllm/v1/kv_cache_interface.py` | new `GroupQuantizedAttentionSpec` subclass; `page_size_bytes` includes scale + zp storage |
| `vllm/v1/attention/backend.py` | extend `is_quantized_kv_cache` to recognise the new dtypes; add to `supported_kv_cache_dtypes` of the chosen backend |
| `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py` | new kernel `reshape_and_cache_kernel_int4_kivi` (write path) — quantizes K/V per-group and writes packed uint8 + scale + zp |
| `vllm/v1/attention/ops/triton_unified_attention.py` | new kernel variant that loads packed uint8 + scale + zp and dequantizes inline before the Q·K^T multiply |
| `vllm/v1/attention/backends/triton_attn.py` | dispatch to the new kernel when `kv_cache_dtype.startswith("int4")` |

The change is contained — we don't touch the block allocator at the
page level (page_size_bytes already accounts for everything we need),
and we don't change the public LLM API.

## Phasing

* **Phase A (foundational kernel)**: Triton round-trip
  quant+dequant, validated bit-for-bit against
  `evaluation.motivation.kivi.simulate_quantization`. Done; lives in
  `kernels/native_int4_kv.py`. Tests in `tests/test_native_int4_kv.py`
  (waiting on GPU availability to actually run; static + import check
  clean).

* **Phase B (write path)**: A separate Triton kernel that takes
  BF16 K and V and writes packed uint8 + scale + zp into a
  pre-allocated cache buffer. Bit-for-bit parity with Phase A's
  round-trip output. Replaces `simulate_quantization` calls in
  the existing forward-pre-hook path so we can A/B-compare emulated
  vs Triton-fused on the same in-process driver before touching vLLM.
  Estimated 1–2 days.

* **Phase C (cache-spec + allocator)**: Extend `AttentionSpec` with
  the new subclass; teach the allocator to reserve scale + zp storage
  alongside the main cache; add `kv_cache_dtype` choice. Estimated
  2–3 days.

* **Phase D (read path / attention kernel)**: New Triton attention
  kernel variant in `triton_unified_attention.py` (or a sibling file)
  that loads packed uint8 + scale + zp inline. Estimated 4–5 days.

* **Phase E (per-layer mixed precision)**: Allow different layers
  to use different `bits`. Wire into the model runner's per-layer
  cache lookup. Estimated 2–3 days. **This is the novel piece.**

* **Phase F (NVFP4 variant)**: Same shape as INT4 but packing E2M1
  codes instead of asymmetric INTs. Lighter than INT4 because no
  zero-point. Estimated 2 days.

* **Phase G (EchoKV controller integration)**: Controller's
  `tier_application` calls `apply_per_layer_bit_widths` on the
  cache manager at each tick. Estimated 1 day.

* **Phase H (validation + benchmarks)**: τ-Bench retail rerun on the
  fork; replace emulated wall numbers in the paper. Estimated 2 days.

Total: **15–20 days** of focused work. Within the 2-week sprint we
can realistically deliver Phases A–D + first uniform-INT4 cell; E–H
are stretch goals.

## Risk register

1. **Triton attention kernel rewrite** is the biggest risk. vLLM's
   `triton_unified_attention.py` is ~800 LOC of optimised
   flash-attention-style code with autotuning. Adding an INT4 dequant
   path inline without breaking the fused softmax + matmul is
   non-trivial. Mitigation: start from a copy of the kernel, add the
   load+dequant prelude, validate that fp16 reconstruction matches
   the original output before optimising.

2. **Mixed precision per-layer** has no precedent in vLLM. The cache
   manager assumes uniform `KVCacheSpec.merge` semantics within a
   group. We may need to disable the merge step for our spec subclass.

3. **Wall-clock validation** depends on the kernel being properly
   memory-bandwidth-bound. If our naïve Triton implementation has
   bad shared-memory access patterns we'd regress vs FP16. Mitigation:
   profile early with Nsight Compute; we can reuse vLLM's existing
   block_size and tile sizes.

4. **Time overrun**. The 2-week sprint is tight. If Phase D (attention
   kernel) takes longer than 5 days we deliver Phase A–C + a paper
   note saying "kernel work is in progress". Bail-out point is end of
   week 1.

## Today's deliverables (completed)

- ✅ Branch `feature/native-int4-kv-cache` created off `master`.
- ✅ Phase A kernel `kernels/native_int4_kv.py` — fused round-trip;
  passes import + AST check.
- ✅ `tests/test_native_int4_kv.py` — 4 parametrized parity tests
  + 4 contract tests. Waiting on GPU to run.
- ✅ This design doc.

## Next (after τ-Bench retail sweep finishes)

- Run the kernel parity tests on real GPU (5-min run on a single H100).
- If parity holds, swap `simulate_quantization` calls for
  `round_trip_int4_per_group` in `nvfp4_tp.py` / `static_int4_tp.py`
  (Phase B), re-measure τ-Bench wall numbers — should narrow the
  3× gap to ~1.3×.
- Begin Phase C: cache-spec + allocator changes.
