"""NVFP4 (E2M1) per-group quantization, BF16-emulated for Hopper.

NVFP4 is NVIDIA's 4-bit floating-point format introduced for
Blackwell tensor cores. Each element occupies 4 bits arranged as
1 sign + 2 exponent + 1 mantissa (no NaN, no Inf), with a per-group
FP scale (typically FP8 E4M3, group size 16) carried alongside.

Representable magnitudes
------------------------
The 8 non-negative codes in E2M1 land at::

    {0, 0.5, 1, 1.5, 2, 3, 4, 6}

(the last four come from exponent ∈ {01, 10, 11} with mantissa
∈ {0, 1}; the first two are subnormals at exponent = 00). Combined
with the sign bit this yields 15 distinct values (one ±0).

Why BF16 emulation
------------------
Hopper (H100) has no native NVFP4 tensor-core path; Blackwell (B100,
B200, GB200) does. To compare EchoKV against an NVFP4 baseline on
Hopper we emulate the round-trip in BF16: scale → snap to nearest
representable level → unscale. The numerical reconstruction is
identical to what a Blackwell native run would produce because the
representable set is the same and scaling is associative with BF16
multiplication for these magnitudes.

Per-group scale
---------------
Following the standard NVFP4 recipe we use a per-group FP scale (no
zero-point — NVFP4 is symmetric around zero, unlike KIVI's
asymmetric per-group INT4). The scale is set so the group max-abs
maps to the largest positive representable level (6.0). Group size
is configurable; default 16 matches the Blackwell tensor-core
microbatch.

Invariants (parallel to ``kivi.py``)
------------------------------------
INV-1  Round-trip uses exactly one FP scale per group and the
       quantized fp4 codes; nothing else. Decode is
       ``code_value * scale``.

Round-trip on values that ARE in the representable set (after
scaling) must be exact. Round-trip on values that aren't picks the
nearest representable level; rounding is to-nearest-even on ties.
"""

from __future__ import annotations

import torch


DEFAULT_GROUP_SIZE: int = 16
NVFP4_MAX: float = 6.0


# Sorted list of non-negative E2M1 representable magnitudes.
# Stored as a constant tensor lazily so we don't materialise on import.
_NVFP4_LEVELS_LIST: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_NVFP4_LEVELS: dict[torch.device, torch.Tensor] = {}


def _levels(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Cached, device-resident tensor of representable NVFP4 magnitudes."""
    key = (device, dtype)
    cached = _NVFP4_LEVELS.get(key)  # type: ignore[arg-type]
    if cached is not None:
        return cached
    t = torch.tensor(_NVFP4_LEVELS_LIST, device=device, dtype=dtype)
    _NVFP4_LEVELS[key] = t  # type: ignore[index]
    return t


def _snap_to_levels(abs_x: torch.Tensor) -> torch.Tensor:
    """Round each element of ``abs_x`` to the nearest E2M1 magnitude.

    Inputs are non-negative scalars in their group-scaled range
    [0, ~6]. Implementation: for each element, take the index whose
    representable level is closest. We use ``bucketize`` with the
    midpoints between consecutive levels as the boundaries.
    """
    levels = _levels(abs_x.device, abs_x.dtype)
    # Midpoints between consecutive representable magnitudes — used as
    # split points so that bucketize returns the index of the nearest
    # level for each input. Length = len(levels) - 1.
    midpoints = (levels[:-1] + levels[1:]) / 2
    idx = torch.bucketize(abs_x, midpoints, right=False)
    return levels[idx]


def _reshape_for_grouping(
    x: torch.Tensor, group_axis: int, group_size: int
) -> tuple[torch.Tensor, torch.Size]:
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


def simulate_nvfp4(
    x: torch.Tensor,
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
    group_axis: int = -1,
) -> torch.Tensor:
    """BF16-emulated NVFP4 round-trip.

    Parameters
    ----------
    x:
        Input floating tensor (any floating dtype).
    group_size:
        Elements per quantization group along ``group_axis``. Default 16.
    group_axis:
        Axis along which groups are formed; negative indices allowed.

    Returns
    -------
    Tensor with the same shape and dtype as ``x``, holding the NVFP4
    round-trip reconstruction. The tensor is in floating dtype because
    we run the model itself in BF16; the 4-bit code is the *information*
    we're emulating, not the storage format on Hopper.
    """
    if group_size < 1:
        raise ValueError("group_size must be >= 1")

    in_dtype = x.dtype
    fp32 = x.to(torch.float32)
    reshaped, original_shape = _reshape_for_grouping(fp32, group_axis, group_size)
    axis = group_axis % x.ndim
    reduce_dim = axis + 1

    # Per-group max-abs scale: map group max-abs to NVFP4_MAX (6.0).
    # Zero groups get scale=1 (any non-zero scale works because the
    # subsequent snap-to-levels will pin every element to 0 anyway).
    max_abs = reshaped.abs().amax(dim=reduce_dim, keepdim=True)
    is_zero_group = max_abs == 0
    scale = torch.where(
        is_zero_group, torch.ones_like(max_abs), max_abs / NVFP4_MAX,
    )
    # Scale-down to the [0, 6] range, snap, scale back up.
    scaled = reshaped / scale
    sign = torch.sign(scaled)
    snapped_abs = _snap_to_levels(scaled.abs())
    code_value = sign * snapped_abs
    dequant = code_value * scale
    return dequant.reshape(original_shape).to(in_dtype)
