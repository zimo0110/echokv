"""TP-aware NVFP4 (E2M1, group-16) hook installer.

Parallels ``static_int4_tp.py`` and ``static_int8_tp.py``. The function in
this module is shipped to each vLLM worker via ``llm.apply_model``;
keeping it in a regular importable module (not ``__main__``) is
required so pickle can resolve the function reference on the worker
side.

The single-GPU equivalent for NVFP4 lives in
``evaluation/motivation/nvfp4.py`` (``simulate_nvfp4``). This file
wraps that quantizer in a forward-pre-hook on each ``Attention``
module so the K and V tensors get round-tripped through NVFP4 before
the cached-attention call sees them. Like the other static-quant TP
modules, this is **for the static-baseline cell only** — it doesn't
observe δ̂ or do tier control. EchoKV's controller still picks
INT2/INT4/INT8; NVFP4 is a *separate static comparison baseline* the
paper compares EchoKV against.

Group axis convention matches KIVI: -1 (the head-dim axis) for both
K and V, mirroring ``static_int4_tp.py``. Group size 16 is the
NVFP4 default (Blackwell tensor-core microbatch).
"""

from __future__ import annotations

from typing import Any

GROUP_SIZE = 16


def install_nvfp4_hooks(model: Any) -> int:
    """Worker-side hook installer for NVFP4.

    Walks the worker's local model, finds every Attention module, and
    installs a forward-pre-hook that quantizes K and V to NVFP4
    (E2M1, group-16, per-group max-abs FP scale) before the cached-
    attention call sees them.

    Returns the number of attention modules hooked.
    """
    import torch  # local imports — picklable

    from evaluation.motivation.nvfp4 import simulate_nvfp4

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
                            k_q = simulate_nvfp4(
                                k_view, group_size=GROUP_SIZE,
                                group_axis=-1,
                            ).reshape(key.shape).to(key.dtype)
                            v_q = simulate_nvfp4(
                                v_view, group_size=GROUP_SIZE,
                                group_axis=-1,
                            ).reshape(value.shape).to(value.dtype)
                        else:
                            k_q = simulate_nvfp4(
                                key, group_size=GROUP_SIZE,
                                group_axis=-1,
                            ).to(key.dtype)
                            v_q = simulate_nvfp4(
                                value, group_size=GROUP_SIZE,
                                group_axis=-1,
                            ).to(value.dtype)
                    return (query, k_q, v_q) + tuple(args[3:])
                except Exception:
                    return None

            return pre_hook

        mod.register_forward_pre_hook(make_hook(num_kv_heads))
        n_hooked += 1
    return n_hooked
