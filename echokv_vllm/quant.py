"""Group-asymmetric quantization with storage numerics.

Groups of ``group_size`` along the last dim. Each group stores an fp16
``scale = (max - min) / (2**b - 1)`` and an fp16 ``min``; codes are
``round((x - min) / scale)``, reconstruction is ``code * scale + min``.

The offset is a float, not an integer zero-point. An integer zero-point
clamped to ``[0, 2**b - 1]`` (as in ``evaluation/motivation/kivi.py``) can
only represent groups that span zero: a group of all-positive or
all-negative values clamps and reconstructs with errors as large as the
values themselves. K outlier channels are exactly such groups.

``scale`` and ``min`` are rounded to fp16 *before* the codes are computed,
so ``fake_quant`` is bit-identical to a write followed by a read of the
cache. The canary uses ``fake_quant``; the cache uses ``quantize`` + ``pack``.
"""

from __future__ import annotations

import torch

SUPPORTED_BITS = (2, 4, 8)
_FP16_TINY = float(torch.finfo(torch.float16).tiny)


def quantize(
    x: torch.Tensor, bits: int, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize ``x[..., D]`` per group of ``group_size`` along the last dim.

    Returns ``(codes uint8 [..., D], scale fp16 [..., G], min fp16 [..., G])``
    with ``G = D // group_size`` and codes in ``[0, 2**bits - 1]``.
    """
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"unsupported bits: {bits}")
    d = x.shape[-1]
    if d % group_size != 0:
        raise ValueError(f"head dim {d} not divisible by group_size {group_size}")
    levels = (1 << bits) - 1
    xg = x.float().reshape(*x.shape[:-1], d // group_size, group_size)
    mn16 = xg.amin(dim=-1).to(torch.float16)
    mn = mn16.float()
    scale16 = ((xg.amax(dim=-1) - mn) / levels).clamp_min(_FP16_TINY).to(torch.float16)
    q = torch.round((xg - mn.unsqueeze(-1)) / scale16.float().unsqueeze(-1)).clamp_(0, levels)
    return q.reshape(x.shape).to(torch.uint8), scale16, mn16


def dequantize(
    codes: torch.Tensor, scale: torch.Tensor, mn: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Inverse of ``quantize``; returns float32 ``[..., D]``."""
    d = codes.shape[-1]
    qg = codes.float().reshape(*codes.shape[:-1], d // group_size, group_size)
    out = qg * scale.float().unsqueeze(-1) + mn.float().unsqueeze(-1)
    return out.reshape(codes.shape)


def fake_quant(x: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    """Round-trip through the cache's storage format; returns float32."""
    return dequantize(*quantize(x, bits, group_size), group_size)


def stats_along(x: torch.Tensor, bits: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """fp16 ``(scale, min)`` of ``x`` reduced over ``dim`` (dim removed)."""
    levels = (1 << bits) - 1
    xf = x.float()
    mn16 = xf.amin(dim=dim).to(torch.float16)
    scale16 = ((xf.amax(dim=dim) - mn16.float()) / levels).clamp_min(_FP16_TINY).to(torch.float16)
    return scale16, mn16


def quantize_with(x: torch.Tensor, scale: torch.Tensor, mn: torch.Tensor, bits: int) -> torch.Tensor:
    """Codes for ``x`` under given (broadcastable) fp16 stats; out-of-range values clamp."""
    levels = (1 << bits) - 1
    return torch.round((x.float() - mn.float()) / scale.float()).clamp_(0, levels).to(torch.uint8)


def dequantize_with(codes: torch.Tensor, scale: torch.Tensor, mn: torch.Tensor) -> torch.Tensor:
    return codes.float() * scale.float() + mn.float()


def fake_quant_channels(k: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    """K storage round-trip: per channel over consecutive groups of ``group`` tokens.

    ``k`` is ``[T, H, D]`` in token order; a trailing partial group is left
    in full precision, as the cache keeps it in the residual buffer.
    """
    t = k.shape[0]
    full = (t // group) * group
    out = k.float().clone()
    if full:
        kg = out[:full].unflatten(0, (full // group, group))  # [n, g, H, D]
        scale, mn = stats_along(kg, bits, dim=1)
        codes = quantize_with(kg, scale.unsqueeze(1), mn.unsqueeze(1), bits)
        out[:full] = dequantize_with(codes, scale.unsqueeze(1), mn.unsqueeze(1)).flatten(0, 1)
    return out


def pack(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack uint8 codes ``[..., D]`` into ``[..., D * bits // 8]`` bytes.

    Little-endian within a byte: element ``i`` of a byte occupies bits
    ``[i * bits, (i + 1) * bits)``.
    """
    if bits == 8:
        return codes
    per = 8 // bits
    c = codes.reshape(*codes.shape[:-1], codes.shape[-1] // per, per).to(torch.int32)
    shifts = torch.arange(per, device=codes.device, dtype=torch.int32) * bits
    return (c << shifts).sum(dim=-1).to(torch.uint8)


def unpack(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Inverse of ``pack``; returns uint8 codes ``[..., n * 8 // bits]``."""
    if bits == 8:
        return packed
    per = 8 // bits
    shifts = torch.arange(per, device=packed.device, dtype=torch.int32) * bits
    c = (packed.to(torch.int32).unsqueeze(-1) >> shifts) & ((1 << bits) - 1)
    return c.reshape(*packed.shape[:-1], packed.shape[-1] * per).to(torch.uint8)
