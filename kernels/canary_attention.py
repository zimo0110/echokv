"""Canary-augmented paged-decode attention.

This file has two layers:

  1. ``dense_canary_attention`` — pure-PyTorch dense reference.
     Correctness oracle for the paged kernel; runnable on CPU.
     Signal: KL(softmax(Q · dequant(quant(K))) ‖ softmax(Q · K_fp16)).

  2. ``canary_paged_attention`` — production entry point.
     Two-stream INT4-paged attention with optional canary divergence
     measurement. Operates on dense tensors today; the ``paged`` naming and
     legacy keyword-arg shape (``k_cache``, ``v_cache``, ``block_tables``,
     KIVI scales/zero-points) are preserved for the eventual FlashInfer
     integration. ``block_tables`` and the scale/zero-point arguments are
     accepted but ignored on this code path — when FlashInfer is pinned
     they will route through the kernel instead. The ``is_canary=False``
     fast path is byte-for-byte the same code as a direct call to the
     internal ``_kivi4_attention_dense`` helper, so INV-4 (``no-canary
     path == KIVI-4 static``) holds by construction.

Design constraints (from ARCHITECTURE.md § 4.1)
------------------------------------------------
INV-1  Dequantization reads only structural KIVI metadata (per-group FP16
       scale + INT4 zero-point). No per-block auxiliary buffer.

INV-4  When ``is_canary=False``, output is bit-identical to KIVI-4 static.
       Both this kernel's no-canary path and the test's KIVI-4 reference
       call ``_kivi4_attention_dense``; bit-exactness is by construction.

Two-stream structure (is_canary=True)
--------------------------------------
* Stream 0 (default CUDA stream): production INT4 path. Computes
  ``ȧ = softmax(Q · dequant(quant(K)) / √d)`` and the attention output
  used by the rest of the model.
* Stream 1 (canary CUDA stream): FP16 reference path. Computes
  ``ā = softmax(Q · K_fp16 / √d)`` from the original FP16 K — this is the
  motivation-study δ, which v8.1 validated as the canary's per-step signal.
* After both streams complete, δ(ℓ) = mean KL(ȧ ‖ ā) is reduced via the
  ``_kl_rowwise_kernel`` Triton kernel and accumulated to
  ``canary_scratchpad`` and/or ``estimator``.

The two streams share no scratch tensors, so the canary stream's FP16
reference can run in parallel with the production INT4 path; we sync via
``Stream.wait_stream`` before the reduction.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover — Triton is in the scalable env
    _TRITON_AVAILABLE = False

from evaluation.motivation.kivi import simulate_quantization


_KL_EPS: float = 1e-9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _attention_weights(
    q: torch.Tensor,
    k: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """Softmax attention weights, computed in float32 for stability.

    q: (B, H, Lq, D)
    k: (B, H, Lk, D)
    → (B, H, Lq, Lk)
    """
    d_k = q.shape[-1]
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / (d_k ** 0.5)
    if causal:
        lq, lk = scores.shape[-2], scores.shape[-1]
        if lq == lk:
            mask = torch.triu(
                torch.ones(lq, lk, dtype=torch.bool, device=scores.device),
                diagonal=1,
            )
            scores = scores.masked_fill(mask, float("-inf"))
    return F.softmax(scores, dim=-1)


def _kl_divergence(
    p: torch.Tensor, q: torch.Tensor, eps: float = _KL_EPS
) -> torch.Tensor:
    """KL(p ‖ q), summed over the last dim. Returns shape p.shape[:-1]."""
    p_safe = p.clamp_min(eps)
    q_safe = q.clamp_min(eps)
    return (p_safe * (p_safe.log() - q_safe.log())).sum(dim=-1)


# ---------------------------------------------------------------------------
# Triton row-wise KL kernel
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:

    @triton.jit
    def _kl_rowwise_kernel(
        p_ptr, q_ptr, out_ptr,
        n_cols,
        p_row_stride, q_row_stride,
        EPS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """One block per row; reduces the last dim into ``out_ptr[row]``.

        Computes  out[r] = Σ_c p[r,c] · (log max(p[r,c], EPS) − log max(q[r,c], EPS))
        over c in [0, n_cols).  ``BLOCK_SIZE`` must be a power-of-two
        ≥ ``n_cols``; the launcher picks the smallest such size.
        """
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        p = tl.load(p_ptr + row * p_row_stride + cols, mask=mask, other=0.0)
        q = tl.load(q_ptr + row * q_row_stride + cols, mask=mask, other=0.0)
        eps = EPS
        p_safe = tl.maximum(p, eps)
        q_safe = tl.maximum(q, eps)
        kl_terms = tl.where(mask, p_safe * (tl.log(p_safe) - tl.log(q_safe)), 0.0)
        kl_row = tl.sum(kl_terms, axis=0)
        tl.store(out_ptr + row, kl_row)


_TRITON_MAX_BLOCK = 32768  # power-of-two cap — beyond this fall back to PyTorch


def _kl_rowwise(
    p: torch.Tensor, q: torch.Tensor, eps: float = _KL_EPS
) -> torch.Tensor:
    """Row-wise KL(p ‖ q), reducing the last dim.

    Triton on CUDA when ``n_cols ≤ 32768`` and Triton is available;
    PyTorch fallback otherwise. Returns a tensor of shape ``p.shape[:-1]``
    in float32.
    """
    assert p.shape == q.shape, f"shape mismatch: {p.shape} vs {q.shape}"
    n_cols = p.shape[-1]
    if not (
        _TRITON_AVAILABLE
        and p.is_cuda
        and q.is_cuda
        and n_cols > 0
        and n_cols <= _TRITON_MAX_BLOCK
    ):
        return _kl_divergence(p.float(), q.float(), eps=eps)

    leading = p.shape[:-1]
    n_rows = 1
    for s in leading:
        n_rows *= int(s)
    p2 = p.reshape(n_rows, n_cols).contiguous().float()
    q2 = q.reshape(n_rows, n_cols).contiguous().float()
    out = torch.empty(n_rows, dtype=torch.float32, device=p.device)
    block_size = 1
    while block_size < n_cols:
        block_size *= 2
    grid = (n_rows,)
    _kl_rowwise_kernel[grid](
        p2, q2, out,
        n_cols,
        p2.stride(0), q2.stride(0),
        EPS=eps,
        BLOCK_SIZE=block_size,
    )
    return out.reshape(leading)


# ---------------------------------------------------------------------------
# Scratchpad (plain Python list — the production path will use a pinned buffer)
# ---------------------------------------------------------------------------


class CanaryScratchpad:
    """Append-only scratchpad for canary KL samples.

    In production this maps to a pinned CUDA buffer ring. Here it is a
    plain Python list so that the dense reference is runnable in unit
    tests without a GPU.
    """

    def __init__(self) -> None:
        self._samples: list[dict] = []

    def write(self, layer_id: int, step: int, kl: float) -> None:
        self._samples.append({"layer_id": layer_id, "step": step, "kl": kl})

    def read_all(self) -> list[dict]:
        return list(self._samples)

    def clear(self) -> None:
        self._samples.clear()


# ---------------------------------------------------------------------------
# Dense reference implementation (non-paged, no FlashInfer)
# ---------------------------------------------------------------------------


def dense_canary_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_canary: bool = False,
    canary_scratchpad: Optional[CanaryScratchpad] = None,
    layer_id: int = -1,
    step: int = -1,
    quant_bits: int = 4,
    group_size: int = 32,
    causal: bool = True,
) -> torch.Tensor:
    """Dense (non-paged) canary attention — correctness reference.

    Computes scaled dot-product attention on dense (B, H, L, D) tensors
    (FP16/BF16 Q, K, V). When ``is_canary=True``, the canary path runs
    alongside the standard path and δ(ℓ) is written to
    ``canary_scratchpad``.

    This function is the testable oracle for the Triton paged kernel.
    It does not require FlashInfer.

    Canary signal (when is_canary=True)
    ------------------------------------
    δ(ℓ) = mean KL(ȧ_ℓ ‖ ā_ℓ) over (batch, heads, query positions)

    where:
      ā_ℓ = softmax(Q · K_fp16 / √d)               — FP16 reference
      ȧ_ℓ = softmax(Q · dequant(quant(K)) / √d)    — simulated INT path

    This matches the motivation study's δ definition exactly.
    """
    if is_canary:
        if canary_scratchpad is None:
            raise ValueError("canary_scratchpad is required when is_canary=True")

        # ā path (stream 1 reference): original FP16 K.
        w_fp16 = _attention_weights(q, k, causal=causal)

        # ȧ path (stream 0 INT4 simulation): quantize K then dequantize.
        k_quant = simulate_quantization(
            k, bits=quant_bits, group_size=group_size, group_axis=-1
        )
        w_int4 = _attention_weights(q, k_quant, causal=causal)

        kl = _kl_divergence(w_int4, w_fp16)  # (B, H, Lq)
        delta = float(kl.mean().item())
        canary_scratchpad.write(layer_id=layer_id, step=step, kl=delta)

        # Attention output uses the FP16 K (ā path is the reference).
        attn_out = torch.matmul(w_fp16, v.float()).to(q.dtype)
    else:
        # Standard path — no canary overhead.
        w = _attention_weights(q, k, causal=causal)
        attn_out = torch.matmul(w, v.float()).to(q.dtype)

    return attn_out


# ---------------------------------------------------------------------------
# KIVI-4 reference (the bit-exact target for INV-4)
# ---------------------------------------------------------------------------


def _kivi4_attention_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    quant_bits: int = 4,
    group_size: int = 32,
    causal: bool = True,
) -> torch.Tensor:
    """Reference KIVI-4 production attention on dense tensors.

    Quantises K and V via ``simulate_quantization`` (the same KIVI primitive
    used in the motivation study), then runs FP16 softmax-attention over
    the dequantised K and V. Returns the attention output in ``q.dtype``.

    INV-4 contract: ``canary_paged_attention(..., is_canary=False)`` returns
    *exactly this tensor*, byte-for-byte, on the same inputs. Both paths
    call this helper so bit-exactness is by construction.
    """
    k_int4 = simulate_quantization(
        k, bits=quant_bits, group_size=group_size, group_axis=-1
    )
    v_int4 = simulate_quantization(
        v, bits=quant_bits, group_size=group_size, group_axis=-1
    )
    w = _attention_weights(q, k_int4, causal=causal)
    return torch.matmul(w, v_int4.float()).to(q.dtype)


def _kivi4_attention_with_weights(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    quant_bits: int = 4,
    group_size: int = 32,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same as ``_kivi4_attention_dense`` but also returns the INT4 attention
    weights — the canary path needs them for the KL computation."""
    k_int4 = simulate_quantization(
        k, bits=quant_bits, group_size=group_size, group_axis=-1
    )
    v_int4 = simulate_quantization(
        v, bits=quant_bits, group_size=group_size, group_axis=-1
    )
    w = _attention_weights(q, k_int4, causal=causal)
    out = torch.matmul(w, v_int4.float()).to(q.dtype)
    return out, w


# ---------------------------------------------------------------------------
# Production entry point (paged, two CUDA streams, Triton KL reduction)
# ---------------------------------------------------------------------------


def canary_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: Any = None,
    k_scales: Any = None,
    k_zero_points: Any = None,
    v_scales: Any = None,
    v_zero_points: Any = None,
    *,
    is_canary: bool = False,
    canary_scratchpad: Optional[CanaryScratchpad] = None,
    estimator: Any = None,
    layer_id: int = -1,
    step: int = -1,
    quant_bits: int = 4,
    group_size: int = 32,
    causal: bool = True,
) -> torch.Tensor:
    """Canary-augmented paged INT4 attention with optional δ measurement.

    Operates on dense (B, H, L, D) Q/K/V tensors. The legacy keyword-arg
    shape (``k_cache``, ``v_cache``, ``block_tables``, KIVI scales / zero-
    points) is preserved for the eventual FlashInfer integration; on this
    code path ``k_cache`` and ``v_cache`` *are* the dense K and V tensors
    and ``block_tables``/``k_scales``/``v_scales``/``k_zero_points``/
    ``v_zero_points`` are accepted but unused — they will be wired through
    the FlashInfer kernel once it is pinned (CHANGELOG note).

    Behaviour
    ---------
    ``is_canary=False``:
        Returns ``_kivi4_attention_dense(q, k_cache, v_cache, ...)``. Output
        is bit-identical to a direct call to the same helper — INV-4 holds
        by construction. No δ recorded, no scratchpad write.

    ``is_canary=True``:
        Two CUDA streams (when on CUDA; serial fallback on CPU):
          * Stream 0 (default): production INT4 path — quantises K and V,
            runs FP16 softmax over (Q · K_int4) and (w · V_int4) → output
            and INT4 attention weights ȧ.
          * Stream 1 (canary stream): FP16 reference path — softmax over
            (Q · K_fp16) → reference attention weights ā.
        After ``Stream.wait_stream`` synchronises the two streams,
        δ(ℓ) = mean KL(ȧ ‖ ā) is reduced via the ``_kl_rowwise_kernel``
        Triton kernel (PyTorch fallback off-CUDA or beyond the BLOCK cap)
        and accumulated to ``canary_scratchpad`` and/or ``estimator``.
        Returns the production INT4 output (Stream 0).

    Either ``canary_scratchpad`` or ``estimator`` (or both) must be
    provided when ``is_canary=True``; the kernel does not silently drop
    the δ measurement.

    Parameters
    ----------
    q, k_cache, v_cache : torch.Tensor
        Dense (B, H, L, D). On the production paged path these are paged
        INT4 tensors plus a block table; the dense interface here uses
        FP16 K/V and quantises internally to drive the canary.
    block_tables, k_scales, k_zero_points, v_scales, v_zero_points :
        Reserved for FlashInfer integration; currently ignored.
    is_canary : bool
        Triggers the two-stream path. Sampling at rate ``p_c = 1/64`` is
        the caller's responsibility (``CanaryEstimator.is_canary_step``).
    canary_scratchpad : CanaryScratchpad, optional
        If provided, writes ``{layer_id, step, kl}`` per canary call.
    estimator : core.canary_estimator.CanaryEstimator, optional
        If provided, calls ``estimator.observe(layer_id, step, kl,
        is_canary=True)`` per canary call. Multiple consumers are fine —
        scratchpad and estimator are independent sinks.
    layer_id, step : int
        Labels recorded alongside δ.
    quant_bits, group_size : int
        KIVI quantisation parameters; default INT4 group-32.
    causal : bool
        Apply causal mask if Lq == Lk.
    """
    if not is_canary:
        # INV-4 path: identical to KIVI-4 static.
        return _kivi4_attention_dense(
            q, k_cache, v_cache,
            quant_bits=quant_bits,
            group_size=group_size,
            causal=causal,
        )

    if canary_scratchpad is None and estimator is None:
        raise ValueError(
            "canary_paged_attention requires canary_scratchpad or estimator "
            "when is_canary=True"
        )

    use_streams = q.is_cuda
    if use_streams:
        device = q.device
        canary_stream = torch.cuda.Stream(device=device)
        # Make canary_stream wait for any pending ops on the default stream
        # (so it sees a consistent view of inputs), then run the FP16
        # reference path on it concurrently with the production path.
        canary_stream.wait_stream(torch.cuda.current_stream(device=device))
        with torch.cuda.stream(canary_stream):
            w_fp16 = _attention_weights(q, k_cache, causal=causal)
        attn_out, w_int4 = _kivi4_attention_with_weights(
            q, k_cache, v_cache,
            quant_bits=quant_bits,
            group_size=group_size,
            causal=causal,
        )
        # Sync before reducing across streams.
        torch.cuda.current_stream(device=device).wait_stream(canary_stream)
    else:
        w_fp16 = _attention_weights(q, k_cache, causal=causal)
        attn_out, w_int4 = _kivi4_attention_with_weights(
            q, k_cache, v_cache,
            quant_bits=quant_bits,
            group_size=group_size,
            causal=causal,
        )

    # Per-position KL via Triton (CUDA + small Lk) or PyTorch fallback.
    kl_per_pos = _kl_rowwise(w_int4, w_fp16)
    delta = float(kl_per_pos.mean().item())

    if canary_scratchpad is not None:
        canary_scratchpad.write(layer_id=layer_id, step=step, kl=delta)
    if estimator is not None:
        estimator.observe(
            layer_id=layer_id, step=step, kl=delta, is_canary=True
        )

    return attn_out
