"""System-level invariant tests — INV-1 through INV-5.

Each test encodes one invariant from ARCHITECTURE.md § 8. Tests that
cannot yet be fully validated (kernel bit-exactness, TP broadcasting)
are marked ``xfail`` with the reason. A test that flips from xfail to
pass is the acceptance signal for the associated implementation.

INV-1  No per-block auxiliary buffer.
INV-2  No in-place modification of an allocated block's contents.
INV-3  Under TP, precision updates are broadcast before the next
       decode step on any rank.
INV-4  The w/o-Shadow-Canary ablation must reproduce KIVI-4 static
       bit-for-bit.
INV-5  All benchmark numbers in REPORT.md come from real experiments
       (no hypothetical values are committed).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from core import block_allocator as ba
from core import canary_estimator as ce
from core import precision_controller as pc
from core import slo_budget as sb
from experimental import write_ahead_precision as wap


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# INV-1 — no per-block auxiliary buffer
# ---------------------------------------------------------------------------


def test_inv1_block_has_only_minimal_fields():
    fields = set(
        vars(
            ba.PhysicalTokenBlock(
                block_id=0,
                precision=ba.Precision.INT4,
                layer_id=0,
            )
        )
    )
    assert fields == {"block_id", "precision", "layer_id", "ref_count"}


def test_inv1_no_shadow_buffer_identifiers_in_core():
    """No source file on the core path defines an FP8 shadow buffer
    or page-refresh worker identifier — these were the Issue-1
    constructs that carried ≥50% memory overhead and were excised
    from the design.

    Only code-form identifiers are forbidden (snake_case or
    CamelCase). Docstrings and comments are free to *discuss* the
    excised constructs; what this test guards against is an actual
    implementation creeping back in.
    """
    # Underscore-joined snake_case and camel/Pascal case identifier
    # forms only. Plain-English prose (e.g. "fp8 shadow" in a
    # docstring that denies its existence) is not matched.
    forbidden_patterns = [
        r"\bfp8_shadow\b",
        r"\bshadow_buffer\b",
        r"\bpage_refresh\b",
        r"\brefresh_worker\b",
        r"\bFp8Shadow\w*\b",
        r"\bShadowBuffer\w*\b",
        r"\bPageRefresh\w*\b",
        r"\bRefreshWorker\w*\b",
    ]
    core_files = list((REPO_ROOT / "core").glob("*.py"))
    core_files += [REPO_ROOT / "kernels" / "canary_attention.py"]
    for path in core_files:
        text = path.read_text()
        for pat in forbidden_patterns:
            assert not re.search(pat, text), (
                f"forbidden identifier '{pat}' found in {path.name}"
            )


# ---------------------------------------------------------------------------
# INV-2 — no in-place block modification
# ---------------------------------------------------------------------------


_FORBIDDEN_NAMES = {"refresh", "repair", "retroactive", "rewrite", "migrate"}


def test_inv2_no_rewrite_surface_on_allocator():
    public = {n for n in dir(ba.BlockAllocator) if not n.startswith("_")}
    assert not (public & _FORBIDDEN_NAMES)


def test_inv2_no_rewrite_surface_on_canary():
    public = {n for n in dir(ce.CanaryEstimator) if not n.startswith("_")}
    assert not (public & _FORBIDDEN_NAMES)


def test_inv2_no_rewrite_surface_on_controller():
    public = {
        n for n in dir(pc.PIDPrecisionController) if not n.startswith("_")
    }
    assert not (public & _FORBIDDEN_NAMES)


def test_inv2_no_rewrite_surface_on_slo_budget():
    public = {n for n in dir(sb.SLOBudget) if not n.startswith("_")}
    assert not (public & _FORBIDDEN_NAMES)


def test_inv2_no_rewrite_surface_on_wap():
    public = {n for n in dir(wap.SinkPredictor) if not n.startswith("_")}
    assert not (public & _FORBIDDEN_NAMES)


# ---------------------------------------------------------------------------
# INV-3 — TP precision broadcast before decode step
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason=(
        "SLOBudget.broadcast_precision is stubbed; full TP harness is a "
        "session-3+ deliverable. This test must pass once the scheduler "
        "coupling is in place."
    ),
    strict=True,
)
def test_inv3_slo_budget_broadcast_is_synchronous():
    budget = sb.SLOBudget()
    # The real test will spin up a fake TP world and assert that all
    # ranks observe the updated precision vector before any rank
    # admits the next decode step. For now we only assert the API
    # point exists and is callable — which will remain xfail until
    # it is implemented.
    budget.broadcast_precision(levels=[ba.Precision.INT4], tp_world_size=1)


# ---------------------------------------------------------------------------
# INV-4 — w/o Shadow Canary == KIVI-4
# ---------------------------------------------------------------------------


def test_inv4_no_canary_path_is_bitexact_with_kivi4():
    """When ``is_canary=False``, ``canary_paged_attention`` output must be
    byte-for-byte equal to a direct KIVI-4 reference computation on the
    same inputs.

    Both paths route through ``_kivi4_attention_dense`` so the equality is
    by construction; this test pins that contract. If either path drifts
    (e.g. someone introduces a fast-path that quantises differently), this
    test fails and the invariant is violated.

    On CUDA this also exercises the two-stream-shaped no-canary path
    (which still reduces to a single-stream call on this branch — INV-4
    forbids the canary stream from launching when ``is_canary=False``).
    """
    import torch
    from kernels.canary_attention import (
        _kivi4_attention_dense,
        canary_paged_attention,
    )

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    for device in devices:
        g = torch.Generator(device=device).manual_seed(0)
        q = torch.randn(1, 4, 16, 64, device=device, generator=g)
        k = torch.randn(1, 4, 16, 64, device=device, generator=g)
        v = torch.randn(1, 4, 16, 64, device=device, generator=g)
        out = canary_paged_attention(q, k, v, is_canary=False)
        ref = _kivi4_attention_dense(q, k, v)
        assert torch.equal(out, ref), (
            f"INV-4 violated on {device}: max diff = "
            f"{(out - ref).abs().max().item():.2e}"
        )


# ---------------------------------------------------------------------------
# INV-5 — REPORT.md contains no hypothetical numbers
# ---------------------------------------------------------------------------


def test_inv5_report_uses_tbd_not_hypothetical_numbers():
    """Every non-header, non-decision row in REPORT.md's numeric tables
    is ``TBD`` until a real experiment runs.

    Heuristic: inside the marked benchmark/ablation/overhead table
    sections, every table cell that is not a header/separator and not
    a system name column must be either ``TBD`` or one of the known
    labels (for the notes column). This guards against the
    temptation to paste in hypothetical numbers before experiments
    complete.
    """
    report = (REPO_ROOT / "REPORT.md").read_text().splitlines()
    in_number_section = False
    saw_any_row = False
    for line in report:
        stripped = line.strip()
        # Start a guarded section at the three numeric tables.
        if stripped.startswith("### Accuracy") or stripped.startswith(
            "### Serving"
        ):
            in_number_section = True
            continue
        if stripped.startswith("## Ablation Table") or stripped.startswith(
            "## Overhead Table"
        ):
            in_number_section = True
            continue
        # Leave the section at the next top-level or sub-level heading.
        if in_number_section and stripped.startswith("## "):
            in_number_section = False
            continue
        if not in_number_section:
            continue
        if not stripped.startswith("|"):
            continue
        # header/separator rows like |---|---|
        if set(stripped.replace("|", "").strip()) <= set("-: "):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        # header rows that describe column names contain words, not numbers
        # and don't include TBD; skip them if the first cell matches a
        # header label we emitted.
        header_first_cells = {
            "System",
            "Trace",
            "Variant",
            "Component",
        }
        if cells and cells[0] in header_first_cells:
            continue
        saw_any_row = False or saw_any_row
        saw_any_row = True
        # Data cells after the first column: check that none is a bare
        # number. "TBD", "≈ 0", notes, and "0 (…)" are all fine.
        for cell in cells[1:]:
            if cell == "":
                continue
            if cell == "TBD":
                continue
            # allow qualitative zero annotations from the overhead table
            if "0" in cell and ("(" in cell or "≈" in cell or cell == "0"):
                continue
            # allow the INV-4 notes column entry
            if "INV-4" in cell:
                continue
            # allow the notes column (anything with letters) as long as
            # it doesn't start with a digit — numeric claims start with
            # a digit or a minus sign.
            if cell and cell[0].isdigit():
                pytest.fail(
                    f"REPORT.md has a numeric value where TBD is required: "
                    f"{cell!r} in line {line!r}"
                )
    assert saw_any_row, (
        "test_inv5 did not find any numeric-table rows to audit — "
        "the parser may be out of sync with REPORT.md structure."
    )
