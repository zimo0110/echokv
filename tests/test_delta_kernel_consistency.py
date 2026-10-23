"""Cross-validation: motivation study δ == canary kernel δ on identical inputs.

The critic's concern (Session 3 review):
  "The dense PyTorch reference kernel quantizes K internally on the canary
   path. The motivation study's δ also only quantizes K. These need to be
   exactly consistent — same group size, same asymmetric INT4 scheme, same
   handling of the constant-group edge case."

What this test verifies
-----------------------
For the same (Q, K, V) tensors and the same quantisation parameters:

    compute_layer_delta(AttentionTrace(q, k, v), bits=b, group_size=g)
                   ==
    dense_canary_attention δ (written to scratchpad)

Specifically:
  - Same ``simulate_quantization`` call (both use group_axis=-1).
  - Same KL direction: KL(ȧ ‖ ā) = KL(w_quant ‖ w_fp16).
  - Same floating-point reduction: mean over (B, H, Lq) before
    converting to a Python float.
  - Same epsilon clamping (1e-9) applied inside ``_kl_divergence``.

If these diverge, the gate result won't validate the canary kernel.

Edge cases from Session 2
--------------------------
- Constant-group K (span = 0 within a group): the kivi module handles
  this by folding the constant into the scale so the round-trip is
  exact. Both paths use ``simulate_quantization`` which routes through
  the same code, so consistency is structural. The test still verifies
  it numerically.
- Very short K sequences where group_size > seq_len would cause an
  error in both paths; tested separately (short-seq guard).
"""

from __future__ import annotations

import pytest
import torch

from evaluation.motivation.delta import (
    AttentionTrace,
    compute_layer_delta,
)
from kernels.canary_attention import (
    CanaryScratchpad,
    dense_canary_attention,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run_both(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    bits: int = 4,
    group_size: int = 32,
    causal: bool = True,
) -> tuple[float, float]:
    """Return (delta_from_motivation_study, delta_from_canary_kernel)."""
    # Motivation study path.
    trace = AttentionTrace(q=q, k=k, v=v)
    delta_ms = float(
        compute_layer_delta(
            trace,
            bits=bits,
            group_size=group_size,
            k_group_axis=-1,
            causal=causal,
            reduction="mean",
        ).item()
    )

    # Canary kernel dense reference path.
    sp = CanaryScratchpad()
    dense_canary_attention(
        q, k, v,
        is_canary=True,
        canary_scratchpad=sp,
        quant_bits=bits,
        group_size=group_size,
        causal=causal,
    )
    delta_kernel = sp.read_all()[0]["kl"]

    return delta_ms, delta_kernel


# ---------------------------------------------------------------------------
# Core consistency tests
# ---------------------------------------------------------------------------


def test_delta_identical_for_int4_default_params():
    """INT4, group_size=32 (production default) — outputs must be bit-identical."""
    torch.manual_seed(0)
    q = torch.randn(1, 4, 32, 64)
    k = torch.randn(1, 4, 32, 64)
    v = torch.randn(1, 4, 32, 64)
    d_ms, d_k = _run_both(q, k, v, bits=4, group_size=32)
    assert d_ms == pytest.approx(d_k, abs=1e-6), (
        f"motivation_study δ={d_ms:.8f} != canary_kernel δ={d_k:.8f}"
    )


def test_delta_identical_for_int2():
    """INT2 — coarser quantisation, larger δ, same consistency requirement."""
    torch.manual_seed(1)
    q = torch.randn(1, 4, 32, 64)
    k = torch.randn(1, 4, 32, 64)
    v = torch.randn(1, 4, 32, 64)
    d_ms, d_k = _run_both(q, k, v, bits=2, group_size=32)
    assert d_ms == pytest.approx(d_k, abs=1e-6), (
        f"INT2 motivation_study δ={d_ms:.8f} != canary_kernel δ={d_k:.8f}"
    )


def test_delta_identical_for_int8():
    """INT8 — near-lossless quantisation, δ near zero, still must agree."""
    torch.manual_seed(2)
    q = torch.randn(1, 2, 32, 64)
    k = torch.randn(1, 2, 32, 64)
    v = torch.randn(1, 2, 32, 64)
    d_ms, d_k = _run_both(q, k, v, bits=8, group_size=32)
    assert d_ms == pytest.approx(d_k, abs=1e-6)


def test_delta_identical_multi_batch_and_heads():
    """Batch > 1, heads > 4 — reduction is over all (B, H, Lq), must agree."""
    torch.manual_seed(3)
    q = torch.randn(2, 8, 32, 64)
    k = torch.randn(2, 8, 32, 64)
    v = torch.randn(2, 8, 32, 64)
    d_ms, d_k = _run_both(q, k, v, bits=4, group_size=32)
    assert d_ms == pytest.approx(d_k, abs=1e-6)


def test_delta_identical_without_causal_mask():
    """Non-causal (encoder) attention — both paths must agree."""
    torch.manual_seed(4)
    q = torch.randn(1, 4, 16, 64)
    k = torch.randn(1, 4, 16, 64)
    v = torch.randn(1, 4, 16, 64)
    d_ms, d_k = _run_both(q, k, v, bits=4, group_size=32, causal=False)
    assert d_ms == pytest.approx(d_k, abs=1e-6)


def test_delta_identical_with_non_square_qk():
    """Decode-style: Lq=1 attending over Lk=64. Causal mask is skipped
    (only applied when Lq==Lk), so both paths must still agree.
    """
    torch.manual_seed(5)
    q = torch.randn(1, 4, 1, 64)
    k = torch.randn(1, 4, 64, 64)
    v = torch.randn(1, 4, 64, 64)
    d_ms, d_k = _run_both(q, k, v, bits=4, group_size=32, causal=True)
    assert d_ms == pytest.approx(d_k, abs=1e-6)


# ---------------------------------------------------------------------------
# Constant-group edge case (Session 2 catch)
# ---------------------------------------------------------------------------


def test_delta_consistent_on_constant_group_K():
    """K with a constant group (all elements equal within a group): the kivi
    module's constant-group handling must fire on BOTH paths and produce
    the same round-trip result → same δ.

    This guards the specific bug found in Session 2 where the naive
    ``scale = span/levels = 0`` collapsed constant groups to zero.
    """
    torch.manual_seed(6)
    q = torch.randn(1, 2, 32, 64)
    k = torch.randn(1, 2, 32, 64)
    v = torch.randn(1, 2, 32, 64)
    # Force one entire head's K to be constant — all groups are constant.
    k[:, 0, :, :] = 1.23456
    d_ms, d_k = _run_both(q, k, v, bits=4, group_size=32)
    assert d_ms == pytest.approx(d_k, abs=1e-6), (
        f"constant-group δ mismatch: ms={d_ms:.8f} kernel={d_k:.8f}"
    )


def test_delta_consistent_on_all_zeros_K():
    """All-zero K: every group has span=0 and value=0.
    The kivi constant-group path handles this (abs_mn=0 → scale=1, q=0).
    Both paths must produce the same δ (likely near zero since
    quantisation of zeros is lossless).
    """
    torch.manual_seed(7)
    q = torch.randn(1, 2, 32, 64)
    k = torch.zeros(1, 2, 32, 64)
    v = torch.randn(1, 2, 32, 64)
    d_ms, d_k = _run_both(q, k, v, bits=4, group_size=32)
    assert d_ms == pytest.approx(d_k, abs=1e-6)


# ---------------------------------------------------------------------------
# Numerical magnitude sanity
# ---------------------------------------------------------------------------


def test_delta_is_non_negative_in_both_paths():
    torch.manual_seed(8)
    q = torch.randn(1, 4, 32, 64)
    k = torch.randn(1, 4, 32, 64)
    v = torch.randn(1, 4, 32, 64)
    d_ms, d_k = _run_both(q, k, v)
    assert d_ms >= 0.0, f"motivation study δ={d_ms} is negative"
    assert d_k >= 0.0, f"canary kernel δ={d_k} is negative"


def test_delta_grows_with_coarser_quantisation_in_both_paths():
    """INT4 should give larger δ than INT8 in both paths, consistently."""
    torch.manual_seed(9)
    q = torch.randn(1, 4, 32, 64)
    k = torch.randn(1, 4, 32, 64)
    v = torch.randn(1, 4, 32, 64)
    d_ms_int4, d_k_int4 = _run_both(q, k, v, bits=4)
    d_ms_int8, d_k_int8 = _run_both(q, k, v, bits=8)
    assert d_ms_int4 >= d_ms_int8, "motivation study: INT4 δ should be >= INT8 δ"
    assert d_k_int4 >= d_k_int8, "canary kernel: INT4 δ should be >= INT8 δ"
