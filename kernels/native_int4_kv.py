"""Native KIVI-style per-group INT4 KV-cache quantization kernels.

Foundation for the vLLM fork (Phase 11). Three pieces:

  * ``round_trip_int4_per_group(x, group_size)``      — quantize then
    dequantize in one Triton kernel; numerically identical to running
    quantize + dequantize separately, but ~10× fewer kernel launches.
    This is the kernel that will replace ``simulate_quantization`` in
    the existing forward-pre-hook static-quant path.

  * (later) split quantize / dequantize for the in-cache packed path.

Storage layout (matches what we'll emit into vLLM's paged cache):
  data:  uint8, shape (..., n_groups, group_size // 2)
         each byte holds two INT4 codes; low nibble = even idx, high nibble = odd idx
  scale: float32, shape (..., n_groups)
  zp:    int8,    shape (..., n_groups)   # 0..15

Numerical contract (must match ``evaluation/motivation/kivi.py``):
  levels = 2**bits - 1 = 15
  mn = group.min(); mx = group.max()
  scale = (mx - mn) / levels         (clamped to fp32_tiny to avoid /0)
  zp    = round(-mn / scale).clamp(0, levels)
  q     = round(x / scale + zp).clamp(0, levels)
  recon = (q - zp) * scale

Tested in tests/test_native_int4_kv.py; round-trip output must match
the reference simulate_quantization to within bf16 ULP for any input.
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
