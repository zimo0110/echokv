"""Phase F.1: per-layer tier-vector helpers for the EchoKV vLLM fork.

The fork's ``TritonAttentionImpl`` exposes ``apply_per_layer_kivi_precision``
(Phase E.1) for runtime per-layer bit-width changes; ``CacheConfig.kivi_bits_per_layer``
(Phase F.1) lets the same vector be supplied at engine init so the spec
allocator picks up the per-layer override.

This module provides three helpers:

  * ``build_uniform_tier_vector(num_layers, bits, prefix)`` — convenience
    constructor for the dict shape ``CacheConfig.kivi_bits_per_layer`` expects.

  * ``apply_tier_vector_to_model(model, bits_per_layer)`` — walks ``model``,
    finds each Attention's ``TritonAttentionImpl``, and pushes the per-layer
    bit-width via ``apply_per_layer_kivi_precision``. Used to keep the impl
    state coherent if the controller swaps tiers post-init.

  * ``read_tier_vector_from_model(model)`` — returns the current per-layer
    ``kivi_bits`` for diagnostics; the inverse of ``apply_tier_vector_to_model``.

Phase F.1 limitation: the underlying KV-cache buffer is allocated once at
engine init from the spec, so ``apply_tier_vector_to_model`` is effectively
a no-op for any layer whose target bit-width differs from the bit-width its
buffer was allocated with. Heterogeneous bits at allocation time is gated
on Phase F.2 (cache-manager support for per-layer page sizes); the
``CacheConfig.kivi_bits_per_layer`` validator currently rejects mixed values.

For TP > 1 the helpers must be shipped via ``llm.apply_model`` so each TP
worker walks its local model slice. They live in a regular module (and
not the driver script) for the same picklability reason as ``canary_tp.py``.
"""

from __future__ import annotations

import re
from typing import Any

_LAYER_IDX_RE = re.compile(r"layers\.(\d+)")


def _collect_kivi_impls(model: Any) -> list[tuple[int, str, Any]]:
    """Return ``[(layer_id, layer_name, impl), ...]`` sorted by layer_id for
    every Attention in ``model`` whose impl exposes
    ``apply_per_layer_kivi_precision``."""
    out: list[tuple[int, str, Any]] = []
    for name, module in model.named_modules():
        if type(module).__name__ != "Attention":
            continue
        m = _LAYER_IDX_RE.search(name)
        if m is None:
            continue
        impl = getattr(module, "impl", None)
        if impl is None or not hasattr(impl, "apply_per_layer_kivi_precision"):
            continue
        out.append((int(m.group(1)), name, impl))
    out.sort(key=lambda t: t[0])
    ids = [t[0] for t in out]
    if ids != list(range(len(ids))):
        raise RuntimeError(f"non-contiguous attention layer ids: {ids[:8]} ...")
    return out


def build_uniform_tier_vector(
    num_layers: int,
    bits: int,
    *,
    prefix: str = "model.layers",
    suffix: str = "self_attn.attn",
) -> dict[str, int]:
    """Construct a uniform ``kivi_bits_per_layer`` dict of length ``num_layers``.

    The keys must match the ``layer_name`` (a.k.a. ``prefix``) that each
    ``Attention`` module is registered with, e.g.
    ``model.layers.0.self_attn.attn`` for Llama-style models.
    """
    if bits not in (2, 4, 8):
        raise ValueError(f"bits must be 2, 4, or 8 (got {bits})")
    return {f"{prefix}.{i}.{suffix}": int(bits) for i in range(int(num_layers))}


def apply_tier_vector_to_model(
    model: Any,
    bits_per_layer: list[int] | dict[str, int],
    group_size: int = 32,
) -> int:
    """Push ``bits_per_layer`` to every fork-side ``TritonAttentionImpl``
    inside ``model``. Returns the number of layers updated.

    ``bits_per_layer`` may be a list (one entry per Attention layer, in
    ascending layer-id order) or a dict keyed by ``layer_name``.

    Raises if the layer count doesn't match (list form) or if any impl
    lacks the EchoKV per-layer hook (i.e. the fork branch isn't loaded).
    """
    impls = _collect_kivi_impls(model)
    if not impls:
        raise RuntimeError(
            "no Attention.impl exposes apply_per_layer_kivi_precision; "
            "is this running against the EchoKV fork "
            "(feature/echokv-int4-cache)?"
        )

    if isinstance(bits_per_layer, dict):
        if len(bits_per_layer) != len(impls):
            raise ValueError(
                f"tier vector dict has {len(bits_per_layer)} entries "
                f"but model has {len(impls)} attention layers"
            )
        for _, name, impl in impls:
            if name not in bits_per_layer:
                raise KeyError(
                    f"layer {name!r} missing from tier vector dict"
                )
            impl.apply_per_layer_kivi_precision(
                int(bits_per_layer[name]), int(group_size)
            )
    else:
        if len(bits_per_layer) != len(impls):
            raise ValueError(
                f"tier vector has {len(bits_per_layer)} entries but model "
                f"has {len(impls)} attention layers"
            )
        for (_, _, impl), bits in zip(impls, bits_per_layer):
            impl.apply_per_layer_kivi_precision(int(bits), int(group_size))

    return len(impls)


def read_tier_vector_from_model(model: Any) -> list[int]:
    """Read each layer's current ``kivi_bits`` in ascending layer-id order.
    Returns an empty list if the fork's KIVI hook is not present on any
    Attention impl."""
    impls = _collect_kivi_impls(model)
    return [int(impl.kivi_bits) for _, _, impl in impls]


def read_buffer_bits_from_model(model: Any) -> list[int]:
    """Read each layer's *buffer* bit-width — the value the data buffer
    was allocated for at engine init. Independent of the active
    ``kivi_bits`` after a Phase F.3.1 runtime swap."""
    impls = _collect_kivi_impls(model)
    return [int(getattr(impl, "kivi_buffer_bits", impl.kivi_bits))
            for _, _, impl in impls]


def swap_tier_vector_to_model(
    model: Any,
    bits_per_layer: list[int] | dict[str, int],
) -> int:
    """Phase F.3.1: change every layer's *active* ``kivi_bits`` to the
    new value in ``bits_per_layer`` without reallocating the data
    buffer. Returns the number of layers updated.

    Caller responsibility: ensure no in-flight requests reference the
    KV cache before calling. Existing cache blocks were written at
    the previous bit-width and become uninterpretable at the new
    bit-width; per-block bit tagging is a future Phase F.3.2.

    Each new value must be ≤ the layer's ``kivi_buffer_bits`` (the
    bit-width its data buffer was sized for at engine init); upsizes
    raise ``ValueError`` per the impl's ``swap_kivi_bits``.
    """
    impls = _collect_kivi_impls(model)
    if not impls:
        raise RuntimeError(
            "no Attention.impl exposes apply_per_layer_kivi_precision; "
            "is this running against the EchoKV fork "
            "(feature/echokv-int4-cache)?"
        )

    if isinstance(bits_per_layer, dict):
        if len(bits_per_layer) != len(impls):
            raise ValueError(
                f"tier vector dict has {len(bits_per_layer)} entries "
                f"but model has {len(impls)} attention layers"
            )
        for _, name, impl in impls:
            if name not in bits_per_layer:
                raise KeyError(f"layer {name!r} missing from tier vector dict")
            impl.swap_kivi_bits(int(bits_per_layer[name]))
    else:
        if len(bits_per_layer) != len(impls):
            raise ValueError(
                f"tier vector has {len(bits_per_layer)} entries but "
                f"model has {len(impls)} attention layers"
            )
        for (_, _, impl), bits in zip(impls, bits_per_layer):
            impl.swap_kivi_bits(int(bits))
    return len(impls)
