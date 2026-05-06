"""Tests for evaluation.motivation.nvfp4 — BF16-emulated NVFP4 round-trip.

Coverage:
  - Representable-set fidelity: any input that already equals
    `level * scale` for some representable level reconstructs exactly.
  - Round-to-nearest semantics on midpoints and just-off-midpoint values.
  - Per-group scale: independent groups don't pollute each other's
    rounding decisions.
  - Group axis other than -1 (sanity for the K/V token-axis case).
  - Determinism: two calls with the same input produce the same output.
  - Magnitude ceiling: outputs never exceed group_max_abs after the
    round-trip; group max-abs should reconstruct exactly because it
    maps to the highest representable level (6.0).
  - Zero-group handling.
  - Group-size divisibility error.
"""

from __future__ import annotations

import pytest
import torch

from evaluation.motivation.nvfp4 import (
    DEFAULT_GROUP_SIZE,
    NVFP4_MAX,
    _NVFP4_LEVELS_LIST,
    simulate_nvfp4,
)


def test_representable_set_roundtrips_exactly():
    """An input whose elements are exactly `level * scale` after scaling
    must round-trip without error."""
    levels = torch.tensor(_NVFP4_LEVELS_LIST)
    scale = 1.5
    # Build a single group of size 8: exactly the 8 non-negative levels
    # multiplied by `scale`. After per-group max-abs scaling the group's
    # max element (6 * 1.5 = 9) maps back to NVFP4_MAX = 6, scale = 1.5.
    group = (levels * scale).unsqueeze(0)  # shape (1, 8)
    # Pad with another group element so length is divisible by 8.
    x = group.repeat(2, 1)                  # shape (2, 8)
    y = simulate_nvfp4(x, group_size=8)
    assert torch.allclose(y, x, atol=1e-6), \
        f"expected exact reconstruction; got max diff {(y - x).abs().max().item()}"


def test_round_to_nearest_above_midpoint():
    """A value just above the midpoint between two levels should round up."""
    # After scaling, midpoint between 1.0 and 1.5 is 1.25. Use scale=1
    # by making group max-abs = 6 (so scale = 6/6 = 1).
    x = torch.tensor([[6.0, 1.26, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    y = simulate_nvfp4(x, group_size=8)
    # 1.26 should snap to 1.5 (nearest level above midpoint 1.25)
    assert y[0, 1].item() == pytest.approx(1.5, abs=1e-5)
    # Group max should be exactly 6 (representable)
    assert y[0, 0].item() == pytest.approx(6.0, abs=1e-5)


def test_round_to_nearest_below_midpoint():
    """A value just below the midpoint between two levels should round down."""
    # Midpoint between 1.0 and 1.5 is 1.25. 1.24 should snap to 1.0.
    x = torch.tensor([[6.0, 1.24, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    y = simulate_nvfp4(x, group_size=8)
    assert y[0, 1].item() == pytest.approx(1.0, abs=1e-5)


def test_negative_values_round_symmetrically():
    """The negative side of the representable set should mirror the positive."""
    x = torch.tensor([[-6.0, -3.5, -1.26, 0.0, 0.0, 0.0, 0.0, 0.0]])
    y = simulate_nvfp4(x, group_size=8)
    # -3.5 is the midpoint between -3 and -4 → ties go to even (snaps
    # to either -4 or to whichever bucketize boundary lands on; the
    # contract here is "nearest level," not "exactly tie-to-even")
    assert y[0, 0].item() == pytest.approx(-6.0, abs=1e-5)
    assert y[0, 1].item() in (-3.0, -4.0)
    assert y[0, 2].item() == pytest.approx(-1.5, abs=1e-5)


def test_groups_are_independent():
    """Per-group scaling: a giant element in one group must not affect
    another group's reconstruction."""
    # Group 1: small values (max-abs = 0.5). Group 2: large values
    # (max-abs = 100). Scaled, group 1 reconstructs as multiples of
    # 0.5/6, group 2 as multiples of 100/6.
    g1 = torch.tensor([0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    g2 = torch.tensor([100.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x = torch.stack([g1, g2], dim=0)  # shape (2, 8)
    y = simulate_nvfp4(x, group_size=8)
    # Group 1's 0.5 should map to its own group max (NVFP4_MAX = 6 in
    # scaled space → 0.5 in unscaled space); 0.25 should snap to 0.25
    # (the 3-level scaled value in group 1) — let's just assert the
    # reconstruction stays within the group's range.
    assert y[0].abs().max().item() <= 0.5 + 1e-5
    # Group 2's 100 should reconstruct exactly (group max-abs = 100,
    # which always lands on the highest representable level).
    assert y[1, 0].item() == pytest.approx(100.0, abs=1e-3)


def test_group_axis_other_than_last():
    """Quantization along a non-last axis (e.g. token axis for V) works."""
    x = torch.randn(8, 4)
    y = simulate_nvfp4(x, group_size=4, group_axis=0)
    assert y.shape == x.shape
    # Group max-abs values along axis 0 should be preserved
    g_max_in = x.abs().amax(dim=0)
    g_max_out = y.abs().amax(dim=0)
    assert torch.allclose(g_max_in, g_max_out, atol=1e-5)


def test_determinism():
    """Same input → same output, twice."""
    torch.manual_seed(0)
    x = torch.randn(4, 32)
    y1 = simulate_nvfp4(x, group_size=16)
    y2 = simulate_nvfp4(x, group_size=16)
    assert torch.equal(y1, y2)


def test_group_max_reconstructs_exactly():
    """The group max-abs element always maps to the highest representable
    level (6.0) after scaling, so it should round-trip exactly."""
    torch.manual_seed(1)
    x = torch.randn(4, 16) * 5
    y = simulate_nvfp4(x, group_size=16)
    # Find each group's max-abs element and check the reconstruction
    # matches up to floating-point precision.
    for row in range(4):
        i = x[row].abs().argmax()
        assert y[row, i].item() == pytest.approx(x[row, i].item(), abs=1e-3)


def test_zero_group():
    """A group of all zeros must reconstruct to all zeros."""
    x = torch.zeros(2, 16)
    y = simulate_nvfp4(x, group_size=16)
    assert torch.equal(y, x)


def test_group_size_divisibility_error():
    x = torch.randn(2, 13)  # 13 is not divisible by 16
    with pytest.raises(ValueError, match="not divisible"):
        simulate_nvfp4(x, group_size=16)


def test_no_extreme_values_outside_group_range():
    """The reconstruction of every element should be within [-group_max,
    +group_max] of its group."""
    torch.manual_seed(2)
    x = torch.randn(4, 16) * 7.5
    y = simulate_nvfp4(x, group_size=16)
    for row in range(4):
        gmax = x[row].abs().max().item()
        assert y[row].abs().max().item() <= gmax + 1e-3, \
            f"row {row}: group max {gmax}, recon max {y[row].abs().max().item()}"


def test_dtype_preserved():
    """Output dtype matches input dtype."""
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        x = torch.randn(2, 16, dtype=dtype)
        y = simulate_nvfp4(x, group_size=16)
        assert y.dtype == dtype


def test_default_group_size_is_16():
    assert DEFAULT_GROUP_SIZE == 16
    x = torch.randn(2, 16)
    y = simulate_nvfp4(x)  # no kwargs
    assert y.shape == x.shape
