"""Native KIVI-style per-group INT4 KV-cache quantization kernels.

Foundation for the vLLM fork (Phase 11). Three public functions:

  * ``round_trip_int4_per_group(x, group_size)`` — quantize then
    dequantize in one Triton kernel; equivalent to
    ``simulate_quantization`` but in a single launch. Used as the hook
    replacement in the static-quant forward-pre-hook path.

  * ``pack_int4_per_group(x, group_size)`` — quantize and emit the
    packed cache layout (``data: uint8``, ``scale: float16``,
    ``zp: uint8``). One Triton kernel; bit-for-bit equivalent to running
    ``round_trip_int4_per_group`` on the same input then re-quantizing
    and packing in PyTorch. This is the write-path kernel for vLLM's
    paged cache.

  * ``unpack_int4_per_group(data, scale, zp, group_size, out_dtype)`` —
    dequantize from the packed cache layout back to a float tensor.
    The read-path counterpart used for in-process A/B comparison
    against ``round_trip_int4_per_group`` and for the eventual fused
    inline-dequant attention kernel (Phase D).

Storage layout (matches what we'll emit into vLLM's paged cache):
  data:  uint8,   shape (..., head_dim // 2)
         each byte holds two INT4 codes; low nibble = even idx, high nibble = odd idx
  scale: float16, shape (..., n_groups)
  zp:    uint8,   shape (..., n_groups)   # 0..15

Numerical contract (must match ``evaluation/motivation/kivi.py``):
  levels = 2**bits - 1 = 15
  mn = group.min(); mx = group.max()
  scale = (mx - mn) / levels         (clamped to fp32_tiny to avoid /0)
  zp    = round(-mn / scale).clamp(0, levels)
  q     = round(x / scale + zp).clamp(0, levels)
  recon = (q - zp) * scale

Tested in tests/test_native_int4_kv.py.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda.libdevice import rint  # round-half-to-even, IEEE 754


BITS: int = 4
LEVELS: int = (1 << BITS) - 1  # = 15


# ---------------------------------------------------------------------------
# Triton: fused quantize-dequantize round-trip (the hook replacement)
# ---------------------------------------------------------------------------

@triton.jit
def _round_trip_int4_kernel(
    x_ptr,                # input: float, shape (n_outer, n_groups, group_size)
    out_ptr,              # output: same dtype as input, same shape
    n_outer,              # i32: product of all dims before the grouped axis
    n_groups,             # i32
    group_size: tl.constexpr,
    LEVELS_F: tl.constexpr,        # 15.0 — passed as constexpr so it's known at compile time
    FP32_TINY: tl.constexpr,       # ~1.18e-38
):
    """Fused per-group INT4 quant + dequant in one kernel launch.

    Each program processes one group (one outer × one group). All
    arithmetic is in fp32 to match the reference; the result is cast
    back to the input dtype at write time.
    """
    pid_outer = tl.program_id(axis=0)
    pid_group = tl.program_id(axis=1)

    if pid_outer >= n_outer:
        return

    # Compute the base offset for this group within the flat tensor.
    base = pid_outer * (n_groups * group_size) + pid_group * group_size
    offs = base + tl.arange(0, group_size)
    x = tl.load(x_ptr + offs).to(tl.float32)

    mn = tl.min(x, axis=0)
    mx = tl.max(x, axis=0)
    span = mx - mn

    # Standard asymmetric scale; clamped to FP32_TINY to avoid div-by-zero
    # for non-constant groups.
    standard_scale = tl.maximum(span / LEVELS_F, FP32_TINY)
    # For constant groups: fold the constant value into the scale so
    # the round-trip reconstruction equals the constant exactly.
    abs_mn = tl.where(mn < 0.0, -mn, mn)
    is_const = span == 0.0
    const_scale = tl.where(abs_mn > 0.0, abs_mn / LEVELS_F, 1.0)
    scale = tl.where(is_const, const_scale, standard_scale)

    # Standard zero-point: clamp(round(-mn/scale), 0, LEVELS).
    zp_standard = tl.clamp(rint(-mn / scale), 0.0, LEVELS_F)
    # Constant-group zero-point: LEVELS if mn<0 else 0.
    zp_const = tl.where(mn < 0.0, LEVELS_F, 0.0)
    zp = tl.where(is_const, zp_const, zp_standard)

    # Quantize: q = clamp(round(x/scale + zp), 0, LEVELS).
    q = tl.clamp(rint(x / scale + zp), 0.0, LEVELS_F)
    # Dequantize: (q - zp) * scale.
    recon = (q - zp) * scale

    tl.store(out_ptr + offs, recon)


def round_trip_int4_per_group(
    x: torch.Tensor,
    *,
    group_size: int = 32,
) -> torch.Tensor:
    """Fused per-group INT4 round-trip via a single Triton kernel.

    Equivalent to ``simulate_quantization(x, bits=4, group_size=group_size,
    group_axis=-1)`` from ``evaluation/motivation/kivi.py``. The output's
    last dim is grouped; all other dims are treated as a single outer
    dim for kernel-launch purposes.

    Parameters
    ----------
    x : torch.Tensor
        Input on CUDA. Last dim must be divisible by ``group_size``.
        Any floating dtype.
    group_size : int
        Group size along the last dim (default 32, matches KIVI).

    Returns
    -------
    torch.Tensor
        Same shape and dtype as ``x``, holding the round-trip
        reconstruction.
    """
    if not x.is_cuda:
        raise ValueError("round_trip_int4_per_group requires a CUDA tensor")
    if x.shape[-1] % group_size != 0:
        raise ValueError(
            f"last dim {x.shape[-1]} not divisible by group_size {group_size}"
        )

    in_dtype = x.dtype
    # Compute path is fp32; we cast the input up and the output back.
    x_fp32 = x.to(torch.float32).contiguous()
    out_fp32 = torch.empty_like(x_fp32)

    n_groups = x_fp32.shape[-1] // group_size
    n_outer = x_fp32.numel() // (n_groups * group_size)

    grid = (n_outer, n_groups)
    _round_trip_int4_kernel[grid](
        x_fp32,
        out_fp32,
        n_outer,
        n_groups,
        group_size=group_size,
        LEVELS_F=float(LEVELS),
        FP32_TINY=torch.finfo(torch.float32).tiny,
    )
    return out_fp32.to(in_dtype)


# ---------------------------------------------------------------------------
# Phase B: write-path kernel (quantize + pack to vLLM cache layout)
# ---------------------------------------------------------------------------

@triton.jit
def _pack_int4_kernel(
    x_ptr,                # input: float32, shape (n_outer * n_groups * group_size,)
    data_ptr,             # output: uint8,   shape (n_outer * n_groups * (group_size // 2),)
    scale_ptr,            # output: float16, shape (n_outer * n_groups,)
    zp_ptr,               # output: uint8,   shape (n_outer * n_groups,)
    n_outer,              # i32
    n_groups,             # i32
    group_size: tl.constexpr,
    half_size: tl.constexpr,       # group_size // 2
    LEVELS_F: tl.constexpr,
    FP32_TINY: tl.constexpr,
):
    """Per-group INT4 quantize + nibble-pack in one launch.

    Each program processes one group: loads the even and odd halves of
    the group separately, computes scale/zp from their combined min/max
    (matches the reference's contiguous-group reduction), quantizes each
    half to a uint8 INT4 code, packs adjacent (even, odd) pairs into
    bytes (low nibble = even, high nibble = odd), and writes the byte +
    scale + zp to their respective cache buffers.
    """
    pid_outer = tl.program_id(axis=0)
    pid_group = tl.program_id(axis=1)

    if pid_outer >= n_outer:
        return

    # Strided loads: even-indexed and odd-indexed halves of the group.
    base = pid_outer * (n_groups * group_size) + pid_group * group_size
    arange_h = tl.arange(0, half_size)
    even_offs = base + arange_h * 2
    odd_offs  = base + arange_h * 2 + 1

    x_even = tl.load(x_ptr + even_offs).to(tl.float32)
    x_odd  = tl.load(x_ptr + odd_offs).to(tl.float32)

    # mn/mx must be computed over the WHOLE group; combining per-half
    # reductions gives the same result for min/max.
    mn = tl.minimum(tl.min(x_even, axis=0), tl.min(x_odd, axis=0))
    mx = tl.maximum(tl.max(x_even, axis=0), tl.max(x_odd, axis=0))
    span = mx - mn

    # Same scale/zp formula as the round-trip kernel — keeps bit-for-bit
    # parity with simulate_quantization on common-case groups.
    standard_scale = tl.maximum(span / LEVELS_F, FP32_TINY)
    abs_mn = tl.where(mn < 0.0, -mn, mn)
    is_const = span == 0.0
    const_scale = tl.where(abs_mn > 0.0, abs_mn / LEVELS_F, 1.0)
    scale = tl.where(is_const, const_scale, standard_scale)

    zp_standard = tl.clamp(rint(-mn / scale), 0.0, LEVELS_F)
    zp_const = tl.where(mn < 0.0, LEVELS_F, 0.0)
    zp = tl.where(is_const, zp_const, zp_standard)

    # Quantize each half to uint8 codes in [0, 15].
    q_even = tl.clamp(rint(x_even / scale + zp), 0.0, LEVELS_F).to(tl.uint8)
    q_odd  = tl.clamp(rint(x_odd  / scale + zp), 0.0, LEVELS_F).to(tl.uint8)

    # Pack: byte = (q_odd << 4) | q_even.
    packed = (q_odd << 4) | q_even

    # Write packed bytes (half_size per group).
    data_base = pid_outer * (n_groups * half_size) + pid_group * half_size
    tl.store(data_ptr + data_base + arange_h, packed)

    # Write scale (fp16) and zp (uint8). One element per program.
    side_idx = pid_outer * n_groups + pid_group
    tl.store(scale_ptr + side_idx, scale.to(tl.float16))
    tl.store(zp_ptr    + side_idx, zp.to(tl.uint8))


def pack_int4_per_group(
    x: torch.Tensor,
    *,
    group_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize ``x`` per-group along its last dim and emit the
    vLLM-paged-cache layout in one Triton launch.

    Parameters
    ----------
    x : torch.Tensor
        Input on CUDA. Last dim must be divisible by ``group_size`` and
        ``group_size`` must be even (so we can pack pairs of nibbles).
        Any floating dtype.
    group_size : int
        Group size along the last dim (default 32, matches KIVI).

    Returns
    -------
    data : torch.Tensor of dtype uint8
        Shape ``x.shape[:-1] + (x.shape[-1] // 2,)``. Each byte holds two
        INT4 codes; low nibble = even index, high nibble = odd index
        (within the same group).
    scale : torch.Tensor of dtype float16
        Shape ``x.shape[:-1] + (n_groups,)`` where
        ``n_groups = x.shape[-1] // group_size``.
    zp : torch.Tensor of dtype uint8
        Same shape as ``scale``. Holds the per-group zero-point in
        ``[0, 15]``.
    """
    if not x.is_cuda:
        raise ValueError("pack_int4_per_group requires a CUDA tensor")
    if x.shape[-1] % group_size != 0:
        raise ValueError(
            f"last dim {x.shape[-1]} not divisible by group_size {group_size}"
        )
    if group_size % 2 != 0:
        raise ValueError(f"group_size must be even (got {group_size})")

    x_fp32 = x.to(torch.float32).contiguous()
    n_groups = x_fp32.shape[-1] // group_size
    half_size = group_size // 2
    n_outer = x_fp32.numel() // (n_groups * group_size)

    data_shape = x.shape[:-1] + (x.shape[-1] // 2,)
    side_shape = x.shape[:-1] + (n_groups,)
    data  = torch.empty(data_shape, dtype=torch.uint8,   device=x.device)
    scale = torch.empty(side_shape, dtype=torch.float16, device=x.device)
    zp    = torch.empty(side_shape, dtype=torch.uint8,   device=x.device)

    grid = (n_outer, n_groups)
    _pack_int4_kernel[grid](
        x_fp32,
        data,
        scale,
        zp,
        n_outer,
        n_groups,
        group_size=group_size,
        half_size=half_size,
        LEVELS_F=float(LEVELS),
        FP32_TINY=torch.finfo(torch.float32).tiny,
    )
    return data, scale, zp


# ---------------------------------------------------------------------------
# Phase B: read-path kernel (unpack + dequantize from cache layout)
# ---------------------------------------------------------------------------

@triton.jit
def _unpack_int4_kernel(
    data_ptr,             # input: uint8,   shape (n_outer * n_groups * half_size,)
    scale_ptr,            # input: float16, shape (n_outer * n_groups,)
    zp_ptr,               # input: uint8,   shape (n_outer * n_groups,)
    out_ptr,              # output: float32, shape (n_outer * n_groups * group_size,)
    n_outer,
    n_groups,
    group_size: tl.constexpr,
    half_size: tl.constexpr,
):
    """Per-group INT4 dequantize from the packed cache layout. Each
    program processes one group, reads ``half_size`` bytes plus one
    scale/zp pair, unpacks the nibbles, and writes ``group_size`` fp32
    reconstructions back to ``out_ptr`` in the original interleaved
    order (even at slot 2j, odd at slot 2j+1)."""
    pid_outer = tl.program_id(axis=0)
    pid_group = tl.program_id(axis=1)

    if pid_outer >= n_outer:
        return

    # Load packed bytes for this group.
    data_base = pid_outer * (n_groups * half_size) + pid_group * half_size
    arange_h = tl.arange(0, half_size)
    bytes_ = tl.load(data_ptr + data_base + arange_h)  # uint8

    side_idx = pid_outer * n_groups + pid_group
    scale = tl.load(scale_ptr + side_idx).to(tl.float32)
    zp    = tl.load(zp_ptr    + side_idx).to(tl.float32)

    # Unpack: even = byte & 0xF, odd = (byte >> 4) & 0xF.
    q_even = (bytes_ & 0x0F).to(tl.float32)
    q_odd  = ((bytes_ >> 4) & 0x0F).to(tl.float32)

    # Dequantize.
    recon_even = (q_even - zp) * scale
    recon_odd  = (q_odd  - zp) * scale

    # Interleave back to original order: out[base + 2j]   = recon_even[j]
    #                                    out[base + 2j+1] = recon_odd[j]
    out_base = pid_outer * (n_groups * group_size) + pid_group * group_size
    tl.store(out_ptr + out_base + arange_h * 2,     recon_even)
    tl.store(out_ptr + out_base + arange_h * 2 + 1, recon_odd)


def unpack_int4_per_group(
    data: torch.Tensor,
    scale: torch.Tensor,
    zp: torch.Tensor,
    *,
    group_size: int = 32,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dequantize from the packed cache layout. Inverse of
    ``pack_int4_per_group``.

    Parameters
    ----------
    data, scale, zp : torch.Tensor
        Output of ``pack_int4_per_group``. Must all be on the same CUDA
        device. ``data`` is uint8, ``scale`` is float16, ``zp`` is uint8.
    group_size : int
        Must match the value used during packing.
    out_dtype : torch.dtype
        Output floating dtype (default float16).

    Returns
    -------
    recon : torch.Tensor of dtype ``out_dtype``
        Reconstructed tensor with the original (pre-pack) shape:
        ``data.shape[:-1] + (data.shape[-1] * 2,)``.
    """
    if not (data.is_cuda and scale.is_cuda and zp.is_cuda):
        raise ValueError("unpack_int4_per_group requires CUDA tensors")
    if group_size % 2 != 0:
        raise ValueError(f"group_size must be even (got {group_size})")
    half_size = group_size // 2
    if data.shape[-1] % half_size != 0:
        raise ValueError(
            f"data last dim {data.shape[-1]} not divisible by half_size {half_size}"
        )

    data_c  = data.contiguous()
    scale_c = scale.contiguous()
    zp_c    = zp.contiguous()

    n_groups = data_c.shape[-1] // half_size
    last_dim = n_groups * group_size
    n_outer = data_c.numel() // (n_groups * half_size)

    out_shape = data_c.shape[:-1] + (last_dim,)
    out_fp32 = torch.empty(out_shape, dtype=torch.float32, device=data_c.device)

    grid = (n_outer, n_groups)
    _unpack_int4_kernel[grid](
        data_c,
        scale_c,
        zp_c,
        out_fp32,
        n_outer,
        n_groups,
        group_size=group_size,
        half_size=half_size,
    )
    return out_fp32.to(out_dtype)
