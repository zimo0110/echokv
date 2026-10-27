"""Absolute canary: full-precision K_ref ring buffer + per-tier attention KL.

δ̂_b = KL( softmax(Q · K̃_bᵀ) ‖ softmax(Q · K_refᵀ) ),  K̃_b = storage round-trip of K_ref at b bits

K_ref holds the most recent ``window`` keys in full precision, captured
before they are quantized into the cache. The measurement never reads the
cache, so the controller's own tier decisions cannot feed back into it.
K̃_b uses the cache's exact K format: per channel over consecutive groups of
``k_group`` tokens, in token order (``fake_quant_channels``).
"""

from __future__ import annotations

import torch

from echokv_vllm.quant import fake_quant_channels


class KRefRing:
    """Circular buffer of the last ``window`` keys, ``[window, H, D]`` float32."""

    def __init__(self, window: int) -> None:
        if window < 2:
            raise ValueError("K_ref window must be >= 2")
        self.window = window
        self.buf: torch.Tensor | None = None
        self.fill = 0
        self.pos = 0

    def push(self, keys: torch.Tensor) -> None:
        keys = keys[-self.window :].float()
        t = keys.shape[0]
        if t == 0:
            return
        if self.buf is None:
            self.buf = torch.zeros(self.window, *keys.shape[1:], device=keys.device)
        # At most two slice copies (wrap-around), no index kernel.
        first = min(t, self.window - self.pos)
        self.buf[self.pos : self.pos + first] = keys[:first]
        if first < t:
            self.buf[: t - first] = keys[first:]
        self.pos = (self.pos + t) % self.window
        self.fill = min(self.window, self.fill + t)

    def view(self) -> torch.Tensor | None:
        """Filled keys, oldest first; None until at least 2 are present."""
        if self.buf is None or self.fill < 2:
            return None
        if self.fill < self.window:
            return self.buf[: self.fill]
        return torch.roll(self.buf, -self.pos, dims=0)


def canary_kl(
    query: torch.Tensor,
    kref: torch.Tensor,
    bits: tuple[int, ...],
    k_group: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Mean per-head, per-query KL at each tier in ``bits``; float32 ``[len(bits)]``.

    query: ``[nq, Hq, D]``; kref: ``[W, H, D]`` oldest first, ``Hq % H == 0``.
    Only whole ``k_group`` groups of K_ref are scored (a partial group would
    sit in the cache's full-precision residual and contribute no error).
    """
    w = (kref.shape[0] // k_group) * k_group
    if w == 0:
        raise ValueError(f"K_ref has {kref.shape[0]} keys, fewer than k_group={k_group}")
    kref = kref[kref.shape[0] - w :]
    q = query.float().transpose(0, 1)  # [Hq, nq, D]
    rep = q.shape[0] // kref.shape[1]

    def log_weights(k: torch.Tensor) -> torch.Tensor:
        kh = k.transpose(0, 1).repeat_interleave(rep, dim=0)  # [Hq, W, D]
        return torch.log_softmax(q @ kh.transpose(1, 2) * softmax_scale, dim=-1)

    logp_ref = log_weights(kref)
    out = []
    for b in bits:
        logp_q = log_weights(fake_quant_channels(kref, b, k_group))
        out.append((logp_q.exp() * (logp_q - logp_ref)).sum(-1).mean())
    return torch.stack(out).clamp_min(0.0)
