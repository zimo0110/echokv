"""Tests for evaluation.motivation.analyze — stats + plotting."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from evaluation.motivation.analyze import (
    LOW_VARIANCE_STDEV,
    RHO_THRESHOLD,
    WorkloadReport,
    _filter_finite_scores,
    _mean_delta,
    _rank_avg_ties,
    analyze,
    compute_spearman_by_workload,
    compute_spearman_pooled,
    gate_decision,
    load_records,
    per_workload_reports,
    spearman_rho,
)


# ---------------------------------------------------------------------------
# Spearman ρ numeric tests
# ---------------------------------------------------------------------------


def test_spearman_perfect_monotone_is_plus_one():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert spearman_rho(xs, ys) == pytest.approx(1.0)


def test_spearman_perfect_antimonotone_is_minus_one():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [50.0, 40.0, 30.0, 20.0, 10.0]
    assert spearman_rho(xs, ys) == pytest.approx(-1.0)


def test_spearman_undefined_on_constant_series():
    xs = [1.0, 2.0, 3.0]
    ys = [4.0, 4.0, 4.0]
    assert spearman_rho(xs, ys) is None


def test_spearman_rank_avg_ties_matches_known_case():
    # xs = [10, 20, 20, 30] -> ranks [1, 2.5, 2.5, 4]
    ranks = _rank_avg_ties([10.0, 20.0, 20.0, 30.0])
    assert ranks == [1.0, 2.5, 2.5, 4.0]


def test_spearman_undefined_on_singleton():
    assert spearman_rho([1.0], [1.0]) is None


# ---------------------------------------------------------------------------
# record filtering and mean-delta helpers
# ---------------------------------------------------------------------------


def test_mean_delta_averages_per_layer():
    assert _mean_delta({"delta": [0.1, 0.2, 0.3]}) == pytest.approx(0.2)


def test_filter_finite_scores_drops_nan_and_none():
    rows = [
        {"nll_fp16": 1.0, "nll_int4": 1.1},               # kept
        {"nll_fp16": None, "nll_int4": 1.1},               # dropped: fp16 None
        {"nll_fp16": 1.0, "nll_int4": None},               # dropped: int4 None
        {"nll_fp16": float("nan"), "nll_int4": 1.1},       # dropped: fp16 NaN
        {"nll_fp16": 1.0, "nll_int4": float("nan")},       # dropped: int4 NaN
        {"nll_fp16": 2.0, "nll_int4": 2.3},                # kept
        {"nll_fp16": 1.0},                                  # dropped: no int4
        {},                                                 # dropped: missing both
    ]
    kept = _filter_finite_scores(rows)
    assert len(kept) == 2
    assert kept[0]["nll_fp16"] == 1.0
    assert kept[1]["nll_fp16"] == 2.0


# ---------------------------------------------------------------------------
# end-to-end analyze() with a small synthetic log
# ---------------------------------------------------------------------------


def _write_synthetic_log(path: Path) -> None:
    # ruler_32k: higher δ → higher ΔNLL → ρ(δ, ΔNLL) = +1
    rows = []
    for i in range(10):
        rows.append(
            {
                "request_id": f"ruler_32k:{i}",
                "workload": "ruler_32k",
                "num_layers": 4,
                "delta": [0.01 * (i + 1)] * 4,
                "task_score": 1.0,
                "nll_fp16": 1.0,
                "nll_int4": 1.0 + i * 0.1,  # increases → ΔNLL increases
                "prompt_tokens": 16,
                "completion_tokens": 8,
            }
        )
    # gsm8k_cot: also higher δ → higher ΔNLL → ρ = +1
    for i in range(8):
        rows.append(
            {
                "request_id": f"gsm8k_cot:{i}",
                "workload": "gsm8k_cot",
                "num_layers": 4,
                "delta": [0.02 * (i + 1)] * 4,
                "task_score": 1.0,
                "nll_fp16": 1.0,
                "nll_int4": 1.0 + i * 0.1,
                "prompt_tokens": 32,
                "completion_tokens": 16,
            }
        )
    # sharegpt: no NLL (violin-only)
    for i in range(5):
        rows.append(
            {
                "request_id": f"sharegpt:{i}",
                "workload": "sharegpt",
                "num_layers": 4,
                "delta": [0.05] * 4,
                "task_score": None,
                "nll_fp16": None,
                "nll_int4": None,
                "prompt_tokens": 64,
                "completion_tokens": 64,
            }
        )
    path.write_text("\n".join(json.dumps(r) for r in rows))


def test_analyze_end_to_end(tmp_path: Path):
    log = tmp_path / "delta_records.jsonl"
    _write_synthetic_log(log)

    out_dir = tmp_path / "analysis"
    result = analyze(log, out_dir)

    assert result.num_records == 23
    assert result.violin_path.exists()
    assert result.heatmap_path.exists()
    rho = result.spearman_rho
    assert rho["ruler_32k"] == pytest.approx(1.0)
    assert rho["gsm8k_cot"] == pytest.approx(1.0)
    summary = json.loads((out_dir / "spearman.json").read_text())
    assert "per_workload" in summary
    assert summary["n"] == 23


def test_compute_spearman_by_workload_ignores_no_nll_workloads(tmp_path: Path):
    log = tmp_path / "delta_records.jsonl"
    _write_synthetic_log(log)
    records = load_records(log)
    per_wk = compute_spearman_by_workload(records)
    assert "sharegpt" not in per_wk  # no NLL values
    assert set(per_wk.keys()) <= {"ruler_32k", "gsm8k_cot", "humaneval"}


# ---------------------------------------------------------------------------
# Variance-aware gate
# ---------------------------------------------------------------------------


def _mk_record(
    workload: str,
    idx: int,
    delta: float,
    score: float | None,
    nll_fp16: float | None = None,
    nll_int4: float | None = None,
    layers: int = 4,
) -> dict:
    return {
        "request_id": f"{workload}:{idx}",
        "workload": workload,
        "num_layers": layers,
        "delta": [delta] * layers,
        "task_score": score,
        "nll_fp16": nll_fp16,
        "nll_int4": nll_int4,
        "prompt_tokens": 16,
        "completion_tokens": 4,
    }


def _linear_records(workload: str, n: int) -> list[dict]:
    """Higher δ → higher ΔNLL → ρ(δ, ΔNLL) = +1."""
    out = []
    for i in range(n):
        d = 0.01 * (i + 1)
        nll_fp16 = 1.0
        nll_int4 = 1.0 + i / (n - 1)  # increases → ΔNLL = i/(n-1)
        out.append(_mk_record(workload, i, d, 1.0, nll_fp16=nll_fp16, nll_int4=nll_int4))
    return out


def _saturated_records(workload: str, n: int, nll: float = 1.0) -> list[dict]:
    """NLL_FP16 = NLL_INT4 → ΔNLL = 0 for all → ΔNLL stdev = 0."""
    return [
        _mk_record(workload, i, 0.01 * (i + 1), 1.0, nll_fp16=nll, nll_int4=nll)
        for i in range(n)
    ]


def _noisy_records(workload: str, n: int) -> list[dict]:
    """δ increases but ΔNLL alternates → ρ(δ, ΔNLL) ≈ 0, stdev > LOW_VARIANCE_STDEV."""
    # ΔNLL alternates between +1.0 and -1.0 → stdev = 1.0 >> 0.10
    int4_vals = [2.0, 0.0] * (n // 2 + 1)
    return [
        _mk_record(workload, i, 0.01 * (i + 1), 1.0, nll_fp16=1.0, nll_int4=int4_vals[i])
        for i in range(n)
    ]


def _inverted_records(workload: str, n: int) -> list[dict]:
    """Higher δ → lower ΔNLL → ρ = -1 (inverted signal)."""
    out = []
    for i in range(n):
        d = 0.01 * (i + 1)
        nll_int4 = 1.0 - i / (n - 1)  # decreases as δ increases
        out.append(_mk_record(workload, i, d, 1.0, nll_fp16=1.0, nll_int4=nll_int4))
    return out


def test_gate_pass_when_all_workloads_have_good_rho():
    records = (
        _linear_records("ruler_32k", 20)
        + _linear_records("gsm8k_cot", 20)
        + _linear_records("humaneval", 20)
    )
    reports = per_workload_reports(records)
    decision, rationale = gate_decision(reports)
    assert decision == "pass", rationale
    assert all(r.outcome == "informative" for r in reports)


def test_gate_fail_when_a_variance_workload_has_low_rho():
    records = (
        _linear_records("ruler_32k", 20)
        + _noisy_records("humaneval", 40)
    )
    reports = per_workload_reports(records)
    decision, rationale = gate_decision(reports)
    assert decision == "fail", rationale
    assert "humaneval" in rationale


def test_gate_fail_on_inverted_signal():
    records = (
        _linear_records("ruler_32k", 20)
        + _inverted_records("gsm8k_cot", 20)
    )
    reports = per_workload_reports(records)
    decision, rationale = gate_decision(reports)
    assert decision == "fail", rationale
    assert "INVERTED" in rationale
    by_wk = {r.workload: r for r in reports}
    assert by_wk["gsm8k_cot"].outcome == "inverted"


def test_gate_selective_when_some_pass_and_others_low_variance():
    records = (
        _linear_records("ruler_32k", 20)
        + _linear_records("gsm8k_cot", 20)
        + _saturated_records("humaneval", 20)
    )
    reports = per_workload_reports(records)
    decision, rationale = gate_decision(reports)
    assert decision == "selective", rationale
    assert "humaneval" in rationale
    by_wk = {r.workload: r for r in reports}
    assert by_wk["ruler_32k"].outcome == "informative"
    assert by_wk["humaneval"].outcome == "undetermined"


def test_gate_no_signal_when_every_workload_is_saturated():
    records = (
        _saturated_records("ruler_32k", 20)
        + _saturated_records("gsm8k_cot", 20)
        + _saturated_records("humaneval", 20)
    )
    reports = per_workload_reports(records)
    decision, rationale = gate_decision(reports)
    assert decision == "no_signal", rationale


def test_low_variance_threshold_is_conservative():
    """ΔNLL stdev below threshold classified as undetermined."""
    n = 20
    records = []
    for i in range(n):
        # 18 records: ΔNLL = 0.05, 2 records: ΔNLL = 0.0 → pstdev ≈ 0.015
        nll_int4 = 1.05 if i < 18 else 1.0
        records.append(_mk_record("ruler_32k", i, 0.01 * (i + 1), 1.0,
                                  nll_fp16=1.0, nll_int4=nll_int4))
    reports = per_workload_reports(records)
    assert len(reports) == 1
    r = reports[0]
    assert r.delta_nll_stdev is not None
    assert r.delta_nll_stdev < LOW_VARIANCE_STDEV
    assert r.outcome == "undetermined"


def test_per_workload_reports_exclude_no_nll_workloads():
    records = (
        _linear_records("ruler_32k", 10)
        + [_mk_record("sharegpt", i, 0.05, None) for i in range(5)]
    )
    reports = per_workload_reports(records)
    assert {r.workload for r in reports} == {"ruler_32k"}


def test_analyze_result_exposes_gate_decision(tmp_path: Path):
    log = tmp_path / "delta_records.jsonl"
    records = _linear_records("ruler_32k", 15) + _linear_records("gsm8k_cot", 15)
    log.write_text("\n".join(json.dumps(r) for r in records))
    result = analyze(log, tmp_path / "analysis")
    assert result.gate_decision in {"pass", "selective"}
    assert result.gate_rationale
    summary = json.loads((tmp_path / "analysis" / "spearman.json").read_text())
    assert "gate" in summary
    assert summary["gate"]["decision"] == result.gate_decision
    assert summary["gate"]["rho_threshold"] == RHO_THRESHOLD
    assert summary["gate"]["low_variance_stdev"] == LOW_VARIANCE_STDEV
