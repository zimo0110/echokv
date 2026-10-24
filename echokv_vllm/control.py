"""Inspect and steer EchoKV layers inside a running engine.

Every function takes the model as its first argument, so it can be shipped
to all workers with ``llm.apply_model(fn)`` (use ``functools.partial`` for
the extra arguments). Requires ``VLLM_ALLOW_INSECURE_SERIALIZATION=1``.

    from functools import partial
    from echokv_vllm import control
    llm.apply_model(partial(control.set_frozen, frozen=True))
    state = llm.apply_model(control.get_state)[0]   # rank 0
"""

from __future__ import annotations

import re
from collections.abc import Iterator

import torch

from echokv_vllm.backend import EchoKVAttentionImpl
from echokv_vllm.quant import SUPPORTED_BITS

_LAYER_IDX = re.compile(r"layers\.(\d+)\.")


def iter_layers(model: torch.nn.Module) -> Iterator[tuple[str, EchoKVAttentionImpl]]:
    """(layer_name, impl) for every EchoKV attention layer, in layer order."""
    found = []
    for name, mod in model.named_modules():
        impl = getattr(mod, "impl", None)
        if isinstance(impl, EchoKVAttentionImpl):
            layer_name = getattr(mod, "layer_name", name)
            m = _LAYER_IDX.search(layer_name)
            found.append((int(m.group(1)) if m else len(found), layer_name, impl))
    for _, layer_name, impl in sorted(found, key=lambda t: t[0]):
        yield layer_name, impl


def get_state(model: torch.nn.Module) -> dict:
    """Per-layer tier, EMAs and storage counters, plus model-wide totals."""
    layers = []
    written = {b: 0 for b in SUPPORTED_BITS}
    for name, impl in iter_layers(model):
        c = impl.controller
        entry = {
            "layer": name,
            "bits": c.bits,
            "ema": {str(b): v for b, v in c.ema.items()},
            "fires": c.fires,
            "transitions": c.transitions,
        }
        s = impl.store
        if s is not None:
            entry["tokens_written"] = {str(b): v for b, v in s.tokens_written.items()}
            entry["blocks_opened"] = {str(b): v for b, v in s.blocks_opened.items()}
            entry["downgraded_blocks"] = s.downgraded_blocks
            entry["demoted_blocks"] = s.demoted_blocks
            entry["sealed_groups"] = s.sealed_groups
            entry["pool_free"] = s.pool_free
            entry["pool_size"] = s.pool_size
            for b, v in s.tokens_written.items():
                written[b] += v
        layers.append(entry)
    total = sum(written.values())
    avg_bits = sum(b * v for b, v in written.items()) / total if total else None
    return {
        "tier_vector": [e["bits"] for e in layers],
        "tokens_written": {str(b): v for b, v in written.items()},
        "avg_stored_bits": avg_bits,
        "layers": layers,
    }


def get_tier_vector(model: torch.nn.Module) -> list[int]:
    return [impl.controller.bits for _, impl in iter_layers(model)]


def set_frozen(model: torch.nn.Module, frozen: bool) -> int:
    """Stop (or resume) tier changes; the canary keeps updating EMAs."""
    n = 0
    for _, impl in iter_layers(model):
        impl.controller.params.frozen = frozen
        n += 1
    return n


def set_tier_vector(model: torch.nn.Module, bits: list[int] | int) -> int:
    """Set each layer's current tier. Applies to blocks opened from now on."""
    layers = list(iter_layers(model))
    vec = [bits] * len(layers) if isinstance(bits, int) else list(bits)
    if len(vec) != len(layers):
        raise ValueError(f"got {len(vec)} tiers for {len(layers)} layers")
    for (_, impl), b in zip(layers, vec):
        if b not in SUPPORTED_BITS:
            raise ValueError(f"unsupported tier {b}")
        impl.controller.bits = b
    return len(layers)


def set_tau0(model: torch.nn.Module, tau0: float) -> int:
    n = 0
    for _, impl in iter_layers(model):
        impl.controller.params.tau0 = tau0
        n += 1
    return n


def reset_counters(model: torch.nn.Module) -> int:
    """Zero storage counters (e.g. between warmup and measurement)."""
    n = 0
    for _, impl in iter_layers(model):
        if impl.store is not None:
            impl.store.reset_counters()
        n += 1
    return n
