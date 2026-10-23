"""KIVI-style asymmetric per-group INT4 quantization (reference).

Used by the motivation study to simulate INT4 KV storage without the
production paged kernel. Pure PyTorch, CPU + GPU, differentiable on
neither path (quantization is non-differentiable; we only need
forward correctness for the motivation study).

What "KIVI-style" means here
----------------------------
KIVI quantizes K and V separately. For a tensor of shape
``(..., group_axis, ...)`` we partition the chosen axis into groups
of fixed ``group_size`` (default 32, matching KIVI and EchoKV).
Within each group we compute:

    mn = group.min();  mx = group.max()
    scale     = (mx - mn) / (2**bits - 1)      # FP16 in production
    zero_pt   = round(-mn / scale)             # INTb
    q         = round(x / scale + zero_pt).clamp(0, 2**bits - 1)
    dequant   = (q - zero_pt) * scale

This is KIVI's asymmetric per-group scheme: one FP16 scale and one
INTb zero-point per group; in production these are the structural
metadata that EchoKV's canary reads (no auxiliary buffer).

Note on V quantization: KIVI groups along the token axis for V and
along the head-dim axis for K. This module takes the grouping axis
as an argument; the delta logger picks the right axis per tensor.

Invariants this module must uphold
----------------------------------
INV-1  Nothing beyond (scale, zero_point, q_int) is produced. No
       additional shadow tensor. The dequantized view is derived
       from these three on demand.

The quantize/dequantize round-trip is lossy by construction. A
round-trip through INT4 must never fall outside the span
[group_min, group_max + scale] of the original group.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


DEFAULT_GROUP_SIZE: int = 32
DEFAULT_BITS: int = 4


@dataclass
class QuantizedTensor:
    """A group-quantized tensor plus its scales and zero-points.

    ``q_int`` and ``zero_point`` are integer-valued but stored in
    floating dtypes for torch ergonomics; they stay within
    ``[0, 2**bits - 1]``. Reconstruction is
    ``(q_int - zero_point) * scale`` with correct broadcasting.
    """

    q_int: torch.Tensor          # shape == original, dtype float for math
    scale: torch.Tensor          # one per group
    zero_point: torch.Tensor     # one per group
    bits: int
    group_size: int
    group_axis: int
    original_shape: torch.Size


def _reshape_for_grouping(
    x: torch.Tensor, group_axis: int, group_size: int
) -> tuple[torch.Tensor, torch.Size]:
    """Split ``group_axis`` into (num_groups, group_size).

    Returns the reshaped tensor and the original shape. Requires the
    grouped axis length to be divisible by ``group_size``.
    """
    axis = group_axis % x.ndim
    shape = x.shape
    n = shape[axis]
    if n % group_size != 0:
        raise ValueError(
            f"length of axis {axis} ({n}) is not divisible by "
            f"group_size ({group_size})"
        )
    new_shape = (
        *shape[:axis],
        n // group_size,
        group_size,
        *shape[axis + 1 :],
    )
    return x.reshape(new_shape), shape


def quantize(
    x: torch.Tensor,
    *,
    bits: int = DEFAULT_BITS,
    group_size: int = DEFAULT_GROUP_SIZE,
    group_axis: int = -1,
) -> QuantizedTensor:
    """KIVI-style asymmetric per-group INTb quantization.

    Parameters
    ----------
    x:
        Floating-point tensor.
    bits:
        Bits per element. Typical values: 2, 4, 8.
    group_size:
        Elements per quantization group along ``group_axis``.
    group_axis:
        Axis along which to form groups. Negative indices allowed.
    """
    if bits < 1 or bits > 16:
        raise ValueError(f"unsupported bits: {bits}")
    if group_size < 1:
        raise ValueError("group_size must be >= 1")

    levels = (1 << bits) - 1  # 2**bits - 1
    reshaped, original_shape = _reshape_for_grouping(
        x.to(torch.float32), group_axis, group_size
    )
    axis = group_axis % x.ndim
    # group dim is now at `axis + 1` after reshape; reduce over it.
    reduce_dim = axis + 1
    mn = reshaped.amin(dim=reduce_dim, keepdim=True)
    mx = reshaped.amax(dim=reduce_dim, keepdim=True)
    span = mx - mn
    is_const = span == 0
    # Constant groups (span == 0) need a non-standard encoding: the
    # usual ``scale = span/levels`` is zero and the round-trip
    # collapses the group to 0. We fold the group's constant value
    # into the scale instead, using ``q = levels`` (if mn >= 0) or
    # ``q = 0`` (if mn < 0). Reconstructs the constant exactly.
    abs_mn = mn.abs()
    standard_scale = (span / levels).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    const_scale = torch.where(
        abs_mn > 0, abs_mn / levels, torch.ones_like(mn)
    )
    scale = torch.where(is_const, const_scale, standard_scale)
    zero_point_standard = torch.round(-mn / scale).clamp_(0, levels)
    zero_point_const = torch.where(
        mn < 0, torch.full_like(mn, float(levels)), torch.zeros_like(mn)
    )
    zero_point = torch.where(is_const, zero_point_const, zero_point_standard)

    q = torch.round(reshaped / scale + zero_point).clamp_(0, levels)
    q_int = q.reshape(original_shape)

    # Squeeze group-member dim out of scale/zero_point so they broadcast
    # against the *unexpanded* shape along ``group_axis``.
    scale = scale.squeeze(reduce_dim)
    zero_point = zero_point.squeeze(reduce_dim)
    return QuantizedTensor(
        q_int=q_int,
        scale=scale,
        zero_point=zero_point,
        bits=bits,
        group_size=group_size,
        group_axis=axis,
        original_shape=original_shape,
    )


def dequantize(qt: QuantizedTensor) -> torch.Tensor:
    """Reconstruct a floating tensor from a QuantizedTensor."""
    axis = qt.group_axis
    # Broadcast scale/zero_point over the grouped dim: insert a length-
    # ``group_size`` dim at ``axis + 1`` of size 1, let broadcasting
    # handle the rest, then reshape back.
    shape = qt.q_int.shape
    n = shape[axis]
    new_shape = (
        *shape[:axis],
        n // qt.group_size,
        qt.group_size,
        *shape[axis + 1 :],
    )
    q_grouped = qt.q_int.reshape(new_shape).to(torch.float32)
    scale = qt.scale.unsqueeze(axis + 1)
    zero_point = qt.zero_point.unsqueeze(axis + 1)
    dequant = (q_grouped - zero_point) * scale
    return dequant.reshape(qt.original_shape)


def simulate_quantization(
    x: torch.Tensor,
    *,
    bits: int = DEFAULT_BITS,
    group_size: int = DEFAULT_GROUP_SIZE,
    group_axis: int = -1,
) -> torch.Tensor:
    """Round-trip ``x`` through INTb and back. Preserves dtype."""
    qt = quantize(x, bits=bits, group_size=group_size, group_axis=group_axis)
    return dequantize(qt).to(x.dtype)
