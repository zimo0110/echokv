"""Phase C: GroupQuantizedAttentionSpec correctness tests.

Validates the new vLLM cache-spec class added to
``vllm/v1/kv_cache_interface.py`` in our fork at
``/mnt/nvme_tmp10/nazar_vllm_fork``. Tests cover:

  * page-size accounting at INT2 / INT4 / INT8 with K and V both quantized
  * page-size accounting with V unquantized (asymmetric K/V)
  * input validation (bits, group_size, head_size divisibility)
  * merge semantics (same-bits OK; different-bits rejected)
  * is_quantized_kv_cache + is_kivi_quant_kv_cache helpers
  * Literal CacheDType accepts the new strings

These tests do NOT instantiate an LLM or load a model. They exercise the
spec class and dtype guards in isolation. For the runtime read/write path
see Phase D (attention kernel) and Phase E (per-layer dispatch).

Run from the echokv repo root:
    PYTHONPATH=/mnt/nvme_tmp10/nazar_vllm_fork:$PYTHONPATH \\
        python -m pytest tests/test_phaseC_kv_cache_spec.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Prepend the vLLM fork to sys.path so ``import vllm.v1.kv_cache_interface``
# loads our patched copy. This must happen before any vllm import.
_VLLM_FORK = Path("/mnt/nvme_tmp10/nazar_vllm_fork")
if not _VLLM_FORK.is_dir():
    pytest.skip(f"vLLM fork not found at {_VLLM_FORK}",
                allow_module_level=True)
sys.path.insert(0, str(_VLLM_FORK))

# Sanity: confirm the import resolves to the fork. If torch/triton isn't
# available we still want the structural tests to run (the spec class is
# pure Python with no CUDA dependency).
import importlib.util
spec_module = importlib.util.find_spec("vllm.v1.kv_cache_interface")
if spec_module is None or _VLLM_FORK.as_posix() not in (spec_module.origin or ""):
    pytest.skip("vllm.v1.kv_cache_interface not loadable from the fork "
                f"(resolved to {spec_module.origin if spec_module else 'None'})",
                allow_module_level=True)

import torch  # used only as a dtype source

from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    GroupQuantizedAttentionSpec,
)
from vllm.v1.attention.backend import (
    is_kivi_quant_kv_cache,
    is_quantized_kv_cache,
)
from vllm.config.cache import CacheDType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spec(*, bits=4, group_size=32, head_size=128,
               num_kv_heads=8, block_size=16, dtype=torch.bfloat16,
               quantize_v=True):
    return GroupQuantizedAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        bits=bits,
        group_size=group_size,
        quantize_v=quantize_v,
    )


# ---------------------------------------------------------------------------
# Page-size accounting (the math from the design doc)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bits,expected_per_token_per_head", [
    # Phase F.2 split layout: ``real_page_size_bytes`` counts only the
    # paged data section (scale/zp moved to a per-layer side buffer).
    # head_size=128, group_size=32, n_groups_per_head=4 (still relevant
    # for the side-buffer math, just not for the paged page size).
    # K data bytes:
    #   bits=2: 128*2/8 = 32
    #   bits=4: 128*4/8 = 64
    #   bits=8: 128*8/8 = 128
    (2, 32),
    (4, 64),
    (8, 128),
])
def test_page_size_quantize_kv(bits, expected_per_token_per_head):
    """Per (token, kv-head) bytes in the paged data buffer = K + V (data only)."""
    block_size, num_kv_heads = 16, 8
    spec = _make_spec(bits=bits, block_size=block_size,
                      num_kv_heads=num_kv_heads, quantize_v=True)
    # K + V data sections, each ``expected_per_token_per_head`` bytes.
    expected = block_size * num_kv_heads * 2 * expected_per_token_per_head
    assert spec.real_page_size_bytes == expected, (
        f"got {spec.real_page_size_bytes}, want {expected} "
        f"(bits={bits}, data-only-per-head={expected_per_token_per_head})"
    )


def test_page_size_quantize_k_only():
    """When V is unquantized, V uses dtype-sized bytes (BF16 → 2 B/elem).
    K's data-only count under Phase F.2 is 64 bytes/head at INT4."""
    spec = _make_spec(bits=4, head_size=128, group_size=32,
                      num_kv_heads=8, block_size=16, dtype=torch.bfloat16,
                      quantize_v=False)
    # K data: 64 B/(token, head); V dense: 128 * 2 = 256 B/(token, head)
    block_size, num_kv_heads = 16, 8
    expected = block_size * num_kv_heads * (64 + 256)
    assert spec.real_page_size_bytes == expected


def test_page_size_smaller_than_fp16_baseline():
    """At INT4, the data-only paged page is exactly 1/4 of the FP16 page
    (4× fewer bits). Phase F.2 layout makes this ratio clean — earlier
    layouts included scale/zp metadata which dragged the ratio toward
    ~0.30."""
    common = dict(block_size=16, num_kv_heads=8, head_size=128,
                  dtype=torch.bfloat16)
    int4_spec = GroupQuantizedAttentionSpec(
        **common, bits=4, group_size=32, quantize_v=True,
    )
    fp16_spec = FullAttentionSpec(**common)
    ratio = int4_spec.real_page_size_bytes / fp16_spec.real_page_size_bytes
    assert 0.24 <= ratio <= 0.26, (
        f"INT4 / FP16 page-size ratio {ratio:.3f} not in expected 0.25 band"
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_bits", [1, 3, 5, 6, 7, 16])
def test_unsupported_bits_raises(bad_bits):
    with pytest.raises(ValueError, match="bits must be 2, 4, or 8"):
        _make_spec(bits=bad_bits)


@pytest.mark.parametrize("bad_gs", [-1, 0, 7, 31, 33])
def test_bad_group_size_raises(bad_gs):
    with pytest.raises(ValueError):
        _make_spec(group_size=bad_gs)


def test_head_size_indivisible_raises():
    with pytest.raises(ValueError, match="not divisible"):
        _make_spec(head_size=130, group_size=32)


# ---------------------------------------------------------------------------
# merge() semantics
# ---------------------------------------------------------------------------

def test_merge_same_bits_succeeds():
    """Layers with identical bits/group_size/quantize_v merge into one group."""
    a = _make_spec(bits=4)
    b = _make_spec(bits=4)
    merged = GroupQuantizedAttentionSpec.merge([a, b])
    assert merged == a


def test_merge_different_bits_rejected():
    """Mixed-precision layers must live in separate KV-cache groups; merge
    refuses to fold them together (per design-doc INV-2 routing).

    Phase F.1 update: raises AssertionError (not ValueError) so vLLM's
    ``is_kv_cache_spec_uniform`` correctly detects the heterogeneity and
    routes to the multi-group path. ValueError would propagate as an
    unhandled exception during init."""
    a = _make_spec(bits=4)
    b = _make_spec(bits=8)
    with pytest.raises(AssertionError, match="different bits"):
        GroupQuantizedAttentionSpec.merge([a, b])


def test_merge_different_group_size_rejected():
    a = _make_spec(group_size=16)
    b = _make_spec(group_size=32)
    with pytest.raises(AssertionError, match="same group_size"):
        GroupQuantizedAttentionSpec.merge([a, b])


def test_merge_different_quantize_v_rejected():
    a = _make_spec(quantize_v=True)
    b = _make_spec(quantize_v=False)
    with pytest.raises(AssertionError, match="quantize_v"):
        GroupQuantizedAttentionSpec.merge([a, b])


# ---------------------------------------------------------------------------
# kv_cache_dtype guards
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [
    "int2_kivi", "int4_kivi", "int8_kivi", "nvfp4_e2m1",
])
def test_kivi_dtype_recognized_as_quantized(dtype):
    assert is_quantized_kv_cache(dtype)
    assert is_kivi_quant_kv_cache(dtype)


@pytest.mark.parametrize("dtype", ["fp8", "fp8_e4m3", "fp8_e5m2"])
def test_fp8_still_recognized_as_quantized(dtype):
    """Pre-existing fp8 path must keep working."""
    assert is_quantized_kv_cache(dtype)
    assert not is_kivi_quant_kv_cache(dtype)


@pytest.mark.parametrize("dtype", ["auto", "bfloat16"])
def test_unquantized_dtype_not_flagged(dtype):
    assert not is_quantized_kv_cache(dtype)
    assert not is_kivi_quant_kv_cache(dtype)


# ---------------------------------------------------------------------------
# CacheDType Literal
# ---------------------------------------------------------------------------

def test_cache_dtype_literal_includes_new_options():
    """Confirm the Literal accepts our new strings (sanity-checking the
    typing.Literal extension)."""
    import typing
    args = typing.get_args(CacheDType)
    for new in ("int2_kivi", "int4_kivi", "int8_kivi", "nvfp4_e2m1"):
        assert new in args, f"{new} missing from CacheDType Literal"
    # Existing options preserved
    for old in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
        assert old in args, f"existing CacheDType {old} disappeared"


# ---------------------------------------------------------------------------
# Type identity (subclass relationship)
# ---------------------------------------------------------------------------

def test_subclass_of_attention_spec():
    """GroupQuantizedAttentionSpec must be an AttentionSpec subclass so
    callers that branch on isinstance(_, AttentionSpec) still see it."""
    spec = _make_spec()
    assert isinstance(spec, AttentionSpec)
