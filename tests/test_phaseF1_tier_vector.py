"""Phase F.1: per-layer tier-vector plumbing tests.

Covers the host-side helpers in ``evaluation.serving.fork_tier`` and the
``CacheConfig.kivi_bits_per_layer`` validator. The end-to-end engine-init
roundtrip is in ``/tmp/phaseF_smoke.py`` (requires GPU + Llama weights).

These tests use a stub model with mock Attention impls — no GPU needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_VLLM_FORK = Path("/mnt/nvme_tmp10/nazar_vllm_fork")
if not _VLLM_FORK.is_dir():
    pytest.skip(
        f"vLLM fork not found at {_VLLM_FORK}", allow_module_level=True,
    )
sys.path.insert(0, str(_VLLM_FORK))

from evaluation.serving.fork_tier import (              # noqa: E402
    apply_tier_vector_to_model,
    build_uniform_tier_vector,
    read_tier_vector_from_model,
)


class _StubImpl:
    """Mocks the surface ``apply_tier_vector_to_model`` calls on
    ``TritonAttentionImpl``."""

    def __init__(self, bits: int = 4, group_size: int = 32) -> None:
        self.kivi_bits = int(bits)
        self.kivi_group_size = int(group_size)

    def apply_per_layer_kivi_precision(
        self, bits: int, group_size: int = 32,
    ) -> None:
        if bits not in (2, 4, 8):
            raise ValueError(f"bits must be 2, 4, or 8 (got {bits})")
        self.kivi_bits = int(bits)
        self.kivi_group_size = int(group_size)


class _StubAttention(torch.nn.Module):
    """Module whose class name is ``Attention`` (matches the helper's
    selector) and which exposes an ``impl`` with the KIVI hook."""

    def __init__(self, bits: int = 4) -> None:
        super().__init__()
        self.impl = _StubImpl(bits=bits)


# Force the class name to "Attention" without adopting vLLM's full type.
_StubAttention.__name__ = "Attention"


class _StubModel(torch.nn.Module):
    def __init__(self, num_layers: int = 4, bits: int = 4) -> None:
        super().__init__()
        layers = torch.nn.ModuleList(
            [_StubAttention(bits=bits) for _ in range(num_layers)]
        )
        # Mimic Llama-style naming so the layer-id regex matches:
        # ``model.layers.{i}.self_attn.attn``.
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            block = torch.nn.Module()
            block.self_attn = torch.nn.Module()
            block.self_attn.attn = layers[i]
            self.model.layers.append(block)


def test_read_uniform_tier_vector():
    model = _StubModel(num_layers=4, bits=4)
    assert read_tier_vector_from_model(model) == [4, 4, 4, 4]


def test_apply_list_form_updates_each_layer():
    model = _StubModel(num_layers=4, bits=4)
    n = apply_tier_vector_to_model(model, [2, 4, 8, 4])
    assert n == 4
    assert read_tier_vector_from_model(model) == [2, 4, 8, 4]


def test_apply_dict_form_matches_layer_names():
    model = _StubModel(num_layers=3, bits=4)
    vec = build_uniform_tier_vector(num_layers=3, bits=8)
    n = apply_tier_vector_to_model(model, vec)
    assert n == 3
    assert read_tier_vector_from_model(model) == [8, 8, 8]


def test_apply_dict_missing_layer_raises():
    model = _StubModel(num_layers=3, bits=4)
    bad = {
        "model.layers.0.self_attn.attn": 4,
        "model.layers.1.self_attn.attn": 4,
        "model.layers.99.self_attn.attn": 4,  # wrong index
    }
    with pytest.raises(KeyError, match="missing"):
        apply_tier_vector_to_model(model, bad)


def test_apply_list_length_mismatch_raises():
    model = _StubModel(num_layers=3, bits=4)
    with pytest.raises(ValueError, match=r"4 entries .* 3 attention"):
        apply_tier_vector_to_model(model, [4, 4, 4, 4])


def test_apply_propagates_invalid_bits():
    """If a value is not in {2,4,8}, the impl's setter raises and we
    surface it (the helper does no extra validation)."""
    model = _StubModel(num_layers=2, bits=4)
    with pytest.raises(ValueError, match="bits must be 2, 4, or 8"):
        apply_tier_vector_to_model(model, [4, 3])


def test_no_kivi_impls_raises():
    """Plain torch model without KIVI impls => helper refuses cleanly."""
    plain = torch.nn.Linear(4, 4)
    with pytest.raises(RuntimeError, match="apply_per_layer_kivi_precision"):
        apply_tier_vector_to_model(plain, [4])


def test_build_uniform_tier_vector_shape():
    vec = build_uniform_tier_vector(num_layers=5, bits=4)
    assert len(vec) == 5
    assert all(b == 4 for b in vec.values())
    expected_keys = {f"model.layers.{i}.self_attn.attn" for i in range(5)}
    assert set(vec.keys()) == expected_keys


def test_build_uniform_tier_vector_rejects_bad_bits():
    with pytest.raises(ValueError, match="bits must be 2, 4, or 8"):
        build_uniform_tier_vector(num_layers=2, bits=5)


# ---------------------------------------------------------------------------
# CacheConfig validator (Pydantic field_validator)
# ---------------------------------------------------------------------------

def test_cache_config_uniform_dict_accepted():
    from vllm.config.cache import CacheConfig
    cfg = CacheConfig(
        block_size=16,
        cache_dtype="int4_kivi",
        kivi_bits_per_layer={
            "model.layers.0.self_attn.attn": 4,
            "model.layers.1.self_attn.attn": 4,
        },
    )
    assert cfg.kivi_bits_per_layer is not None
    assert set(cfg.kivi_bits_per_layer.values()) == {4}


def test_cache_config_heterogeneous_dict_accepted_post_f2():
    """Phase F.2 unblocked heterogeneous bits at allocation time
    (split-buffer layout: scale/zp moved off the paged budget so
    block_size adjustment unifies pages across bit-widths). The earlier
    Phase F.1 validator that rejected mixed values is removed."""
    from vllm.config.cache import CacheConfig
    cfg = CacheConfig(
        block_size=16,
        cache_dtype="int4_kivi",
        kivi_bits_per_layer={
            "model.layers.0.self_attn.attn": 4,
            "model.layers.1.self_attn.attn": 8,
        },
    )
    assert cfg.kivi_bits_per_layer is not None
    assert sorted(cfg.kivi_bits_per_layer.values()) == [4, 8]


def test_cache_config_invalid_bits_rejected():
    from pydantic import ValidationError

    from vllm.config.cache import CacheConfig
    with pytest.raises(ValidationError, match=r"\{2, 4, 8\}"):
        CacheConfig(
            block_size=16,
            cache_dtype="int4_kivi",
            kivi_bits_per_layer={"model.layers.0.self_attn.attn": 3},
        )


def test_cache_config_none_passes_through():
    from vllm.config.cache import CacheConfig
    cfg = CacheConfig(
        block_size=16,
        cache_dtype="int4_kivi",
        kivi_bits_per_layer=None,
    )
    assert cfg.kivi_bits_per_layer is None


# ---------------------------------------------------------------------------
# Phase F.3.1: runtime swap helpers
# ---------------------------------------------------------------------------


class _StubImplF31(_StubImpl):
    """Adds Phase F.3.1 fields/methods to the stub: ``kivi_buffer_bits``
    + ``swap_kivi_bits``."""

    def __init__(self, bits: int = 8) -> None:
        super().__init__(bits=bits)
        self.kivi_buffer_bits = int(bits)

    def swap_kivi_bits(self, new_bits: int) -> None:
        if new_bits not in (2, 4, 8):
            raise ValueError(f"bits must be 2, 4, or 8 (got {new_bits})")
        if new_bits > self.kivi_buffer_bits:
            raise ValueError(
                f"swap_kivi_bits cannot upsize past buffer bits "
                f"({self.kivi_buffer_bits}); requested {new_bits}"
            )
        self.kivi_bits = int(new_bits)


class _F31Attention(torch.nn.Module):
    def __init__(self, bits: int = 8) -> None:
        super().__init__()
        self.impl = _StubImplF31(bits=bits)


_F31Attention.__name__ = "Attention"


class _F31Model(torch.nn.Module):
    def __init__(self, num_layers: int = 4, bits: int = 8) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            block = torch.nn.Module()
            block.self_attn = torch.nn.Module()
            block.self_attn.attn = _F31Attention(bits=bits)
            self.model.layers.append(block)


def test_read_buffer_bits_distinguishes_active_from_buffer():
    from evaluation.serving.fork_tier import (
        read_buffer_bits_from_model, read_tier_vector_from_model,
        swap_tier_vector_to_model,
    )
    model = _F31Model(num_layers=4, bits=8)
    assert read_tier_vector_from_model(model) == [8, 8, 8, 8]
    assert read_buffer_bits_from_model(model) == [8, 8, 8, 8]
    swap_tier_vector_to_model(model, [4, 4, 4, 4])
    # Active bits drop, buffer bits do NOT.
    assert read_tier_vector_from_model(model) == [4, 4, 4, 4]
    assert read_buffer_bits_from_model(model) == [8, 8, 8, 8]


def test_swap_tier_vector_rejects_upsize_past_buffer():
    from evaluation.serving.fork_tier import swap_tier_vector_to_model

    model = _F31Model(num_layers=2, bits=4)  # buffers at INT4
    with pytest.raises(ValueError, match="upsize past buffer"):
        swap_tier_vector_to_model(model, [8, 8])


def test_swap_tier_vector_chains():
    """Applying multiple swaps in sequence keeps buffer_bits pinned."""
    from evaluation.serving.fork_tier import (
        read_buffer_bits_from_model, read_tier_vector_from_model,
        swap_tier_vector_to_model,
    )
    model = _F31Model(num_layers=4, bits=8)
    swap_tier_vector_to_model(model, [4, 4, 4, 4])
    swap_tier_vector_to_model(model, [2, 2, 2, 2])
    assert read_tier_vector_from_model(model) == [2, 2, 2, 2]
    assert read_buffer_bits_from_model(model) == [8, 8, 8, 8]
    swap_tier_vector_to_model(model, [4, 4, 4, 4])  # back up within buffer
    assert read_tier_vector_from_model(model) == [4, 4, 4, 4]


def test_swap_tier_vector_dict_form():
    from evaluation.serving.fork_tier import (
        build_uniform_tier_vector, read_tier_vector_from_model,
        swap_tier_vector_to_model,
    )
    model = _F31Model(num_layers=3, bits=8)
    swap_tier_vector_to_model(model, build_uniform_tier_vector(3, 4))
    assert read_tier_vector_from_model(model) == [4, 4, 4]


def test_swap_tier_vector_no_kivi_impls_raises():
    from evaluation.serving.fork_tier import swap_tier_vector_to_model
    plain = torch.nn.Linear(4, 4)
    with pytest.raises(RuntimeError, match="apply_per_layer_kivi_precision"):
        swap_tier_vector_to_model(plain, [4])
