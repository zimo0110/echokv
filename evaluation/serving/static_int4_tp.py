"""TP-aware static-INT4 hook installer.

Used by the 70B (and any future TP > 1) diagnostics to reproduce the
KIVI-INT4 baseline across all tensor-parallel ranks. The function in
this module is shipped to each vLLM worker via ``llm.apply_model``;
keeping it in a regular importable module (not ``__main__``) is
required so pickle can resolve the function reference on the worker
side. (Functions defined in ``__main__`` cannot be pickled cleanly:
``PicklingError: ... it's not the same object as __main__.foo``.)

The single-GPU path lives in ``canary_vllm.py`` (``CanaryHarness``
with ``force_bits=4``); this module is the TP-aware standalone for
the static-baseline cell only — it doesn't observe δ̂ or do tier
control. For the 70B diagnostic that's all we need.
"""

from __future__ import annotations

from typing import Any

GROUP_SIZE = 32
BITS = 4


def install_static_int4_hooks(model: Any) -> int:
    """Worker-side hook installer.

    Walks the worker's local model, finds every Attention module, and
    installs a forward-pre-hook that quantizes K and V to KIVI-style
    group-32 INT4 before the cached-attention call sees them.

    Returns the number of attention modules hooked. The caller
    (``llm.apply_model``) collects one count per TP rank; they should
    all match (same model architecture on every rank).
    """
    import torch  # local imports — picklable

    from evaluation.motivation.kivi import simulate_quantization

    n_hooked = 0
    for name, mod in model.named_modules():
        if type(mod).__name__ != "Attention":
            continue
        num_kv_heads = (
            getattr(mod, "num_kv_heads", None)
            or getattr(mod, "num_attention_heads", None)
        )
        if num_kv_heads is None:
            continue

        def make_hook(num_kv_heads):
            def pre_hook(module, args):
                if len(args) < 3:
                    return None
                query, key, value = args[0], args[1], args[2]
                if key is None or value is None:
                    return None
                try:
                    with torch.no_grad():
                        if key.dim() == 2:
                            T, hd = key.shape
                            head_size = hd // num_kv_heads
                            k_view = key.view(T, num_kv_heads, head_size)
                            v_head_size = value.shape[-1] // num_kv_heads
                            v_view = value.view(T, num_kv_heads, v_head_size)
                            k_q = simulate_quantization(
                                k_view, bits=BITS, group_size=GROUP_SIZE,
                                group_axis=-1,
                            ).reshape(key.shape).to(key.dtype)
                            v_q = simulate_quantization(
                                v_view, bits=BITS, group_size=GROUP_SIZE,
                                group_axis=-1,
                            ).reshape(value.shape).to(value.dtype)
                        else:
                            k_q = simulate_quantization(
                                key, bits=BITS, group_size=GROUP_SIZE,
                                group_axis=-1,
                            ).to(key.dtype)
                            v_q = simulate_quantization(
                                value, bits=BITS, group_size=GROUP_SIZE,
                                group_axis=-1,
                            ).to(value.dtype)
                    return (query, k_q, v_q) + tuple(args[3:])
                except Exception:
                    return None

            return pre_hook

        mod.register_forward_pre_hook(make_hook(num_kv_heads))
        n_hooked += 1
    return n_hooked
