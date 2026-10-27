"""EchoKV settings, read from ``ECHOKV_*`` environment variables.

Environment variables reach every vLLM worker process, including the
spawned ones, so the backend can read them wherever vLLM constructs it.

    ECHOKV_ENABLE           1 to install the backend (see plugin.py)
    ECHOKV_MEM_BITS         memory budget, bits/element: 2, 4, 6 or 8 (default 4,
                            or ECHOKV_FIXED_BITS when that is set)
    ECHOKV_INIT_BITS        starting tier for every layer (default: highest tier
                            the budget gives every block, i.e. 8, 4, 4, 2 for
                            budgets 8, 6, 4, 2)
    ECHOKV_FIXED_BITS       pin every layer to this tier, controller off (static baselines)
    ECHOKV_FROZEN           1 to keep tiers fixed while the canary keeps measuring
    ECHOKV_TAU0             per-layer canary KL budget (default 0.01)
    ECHOKV_L0               reference context length for τ scaling (default 4096)
    ECHOKV_TAU_SCALING      none (default) or sqrt: τ0·sqrt(max(L, L0)/L0)
    ECHOKV_ALPHA            EMA weight of the newest sample (default 0.1)
    ECHOKV_CANARY_INTERVAL  fire every N forward steps per layer (default 64)
    ECHOKV_KREF_WINDOW      K_ref ring size in tokens (default 64)
    ECHOKV_CANARY_MAX_Q     decode queries sampled per fire (default 8)
    ECHOKV_K_GROUP          tokens per per-channel K group (default 32; divides block size)
    ECHOKV_V_GROUP          head_dim elements per per-token V group (default 32)
    ECHOKV_RESIDUAL_SLOTS   open-K-group slots per layer (default max_num_seqs + 16)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from echokv_vllm.controller import ControllerParams
from echokv_vllm.quant import SUPPORTED_BITS


@dataclass(frozen=True)
class EchoKVConfig:
    mem_bits: int = 4
    k_group: int = 32
    v_group: int = 32
    residual_slots: int | None = None
    canary_interval: int = 64
    kref_window: int = 64
    canary_max_q: int = 8
    controller: ControllerParams = field(default_factory=ControllerParams)


def _get(name: str, default, cast):
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else cast(raw)


def default_init_bits(mem_bits: int) -> int:
    """Highest tier every block can hold under the budget (INT4 is +0.7% perplexity at 8B)."""
    return max((b for b in SUPPORTED_BITS if b <= mem_bits), default=2)


def load_config() -> EchoKVConfig:
    fixed = _get("ECHOKV_FIXED_BITS", None, int)
    # A pinned tier with no explicit budget gets a budget that holds it.
    mem_bits = _get("ECHOKV_MEM_BITS", 4 if fixed is None else max(fixed, 2), int)
    params = ControllerParams(
        tau0=_get("ECHOKV_TAU0", 0.01, float),
        l0=_get("ECHOKV_L0", 4096, int),
        tau_scaling=_get("ECHOKV_TAU_SCALING", "none", str),
        alpha=_get("ECHOKV_ALPHA", 0.1, float),
        init_bits=_get("ECHOKV_INIT_BITS", default_init_bits(mem_bits), int),
        fixed_bits=fixed,
        frozen=_get("ECHOKV_FROZEN", "0", str) == "1",
    )
    for name, b in (("ECHOKV_INIT_BITS", params.init_bits), ("ECHOKV_FIXED_BITS", fixed)):
        if b is not None and b not in SUPPORTED_BITS:
            raise ValueError(f"{name} must be one of {SUPPORTED_BITS}, got {b}")
    cfg = EchoKVConfig(
        mem_bits=mem_bits,
        k_group=_get("ECHOKV_K_GROUP", 32, int),
        v_group=_get("ECHOKV_V_GROUP", 32, int),
        residual_slots=_get("ECHOKV_RESIDUAL_SLOTS", None, int),
        canary_interval=_get("ECHOKV_CANARY_INTERVAL", 64, int),
        kref_window=_get("ECHOKV_KREF_WINDOW", 64, int),
        canary_max_q=_get("ECHOKV_CANARY_MAX_Q", 8, int),
        controller=params,
    )
    if params.tau_scaling not in ("none", "sqrt"):
        raise ValueError("ECHOKV_TAU_SCALING must be none or sqrt")
    if cfg.mem_bits not in (2, 4, 6, 8):
        raise ValueError(f"ECHOKV_MEM_BITS must be 2, 4, 6 or 8, got {cfg.mem_bits}")
    if cfg.canary_interval < 1:
        raise ValueError("ECHOKV_CANARY_INTERVAL must be >= 1")
    if cfg.kref_window < cfg.k_group:
        raise ValueError("ECHOKV_KREF_WINDOW must be >= ECHOKV_K_GROUP")
    return cfg
