"""Motivation-study analysis → the three deliverables.

Inputs: a JSONL file of :class:`~delta.DeltaRecord` rows.

Outputs:
    1. violin plot of δ by workload type          → ``violin.png``
    2. layer sensitivity heatmap                   → ``heatmap.png``
    3. Spearman ρ between mean δ and task accuracy → printed + JSON

Gate (REPORT.md § Motivation Study):
    Correlates δ against ΔNLL = NLL_INT4 − NLL_FP16, where both
    NLL values are computed via teacher-forcing on the FP16
    completion. Expected direction: ρ(δ, ΔNLL) ≥ 0 (higher
    quantization error predicts higher NLL degradation).
    Gate threshold: ρ ≥ 0.4 (signed, positive direction only).

Matplotlib is imported lazily so that `run.py` can be imported on
boxes without a matplotlib install (headless CI, e.g.).
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

from .workloads import ACCURACY_WORKLOADS, WORKLOADS


# Decision threshold for the research-validity gate (REPORT.md).
# With ΔNLL = NLL_INT4 − NLL_FP16 as the Y-axis, the expected direction
# is *positive*: higher δ should predict higher NLL degradation. The
# gate checks signed ρ ≥ threshold (not |ρ|). A strongly negative ρ
# means the pipeline is inverted (low δ predicts high degradation) and
# is treated as a separate failure mode, not a pass.
RHO_THRESHOLD: float = 0.4

# Accuracy-variance threshold below which ρ is considered
# undetermined. stdev on a [0, 1] accuracy scale; if the workload's
# scores barely move, rank correlation has nothing to attach to.
# This is what the Session-2 critic's refinement turns on.
LOW_VARIANCE_STDEV: float = 0.10


WorkloadOutcome = Literal[
    "informative",   # ΔNLL stdev >= LOW_VARIANCE_STDEV and ρ >= RHO_THRESHOLD
    "inverted",      # ΔNLL stdev >= LOW_VARIANCE_STDEV and ρ <= -RHO_THRESHOLD
    "uninformative", # ΔNLL stdev >= LOW_VARIANCE_STDEV and |ρ| < RHO_THRESHOLD
    "undetermined",  # ΔNLL stdev < LOW_VARIANCE_STDEV — ceiling/floor effect
    "insufficient",  # n < 2 or ρ undefined
]


@dataclass
class WorkloadReport:
    workload: str
    n: int
    rho: Optional[float]
    accuracy_mean: Optional[float]   # FP16 accuracy, for monitoring only
    accuracy_stdev: Optional[float]
    delta_nll_mean: Optional[float]   # mean ΔNLL = NLL_INT4 − NLL_FP16
    delta_nll_stdev: Optional[float]  # stdev of ΔNLL
    outcome: WorkloadOutcome


# Top-level gate classification (three outcomes, per the Session-2
# critic's refinement — see REPORT.md § Motivation Study).
GateDecision = Literal[
    "pass",       # every workload with variance has ρ ≥ threshold
    "selective",  # some workloads pass; the rest are undetermined
                  # (low variance → KV compression doesn't matter there,
                  # so δ uninformativeness is expected, not a failure)
    "fail",       # at least one workload with variance has ρ < threshold
    "no_signal",  # no workload had sufficient variance to evaluate
]


@dataclass
class AnalysisResult:
    violin_path: Path
    heatmap_path: Path
    spearman_rho: dict[str, float]           # per-workload ρ (informative or uninformative only)
    spearman_pooled: Optional[float]         # ρ pooled across informative + uninformative
    per_workload: list[WorkloadReport]       # full per-workload report
    gate_decision: GateDecision
    gate_rationale: str
    num_records: int


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_records(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Spearman ρ — implemented locally to avoid a SciPy dep on lean boxes
# ---------------------------------------------------------------------------


def _rank_avg_ties(xs: Sequence[float]) -> list[float]:
    """Rank with tie-averaging (same as ``scipy.stats.rankdata``)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # ranks are 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation. Returns None if undefined (n<2 or
    zero variance in either series).
    """
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    n = len(xs)
    if n < 2:
        return None
    rx = _rank_avg_ties(xs)
    ry = _rank_avg_ties(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


# ---------------------------------------------------------------------------
# Analysis driver
# ---------------------------------------------------------------------------


def _mean_delta(record: dict) -> float:
    deltas = record["delta"]
    return sum(deltas) / len(deltas) if deltas else float("nan")


def _delta_nll(record: dict) -> Optional[float]:
    """ΔNLL = NLL_INT4 − NLL_FP16, or None if either value is missing."""
    nll_fp16 = record.get("nll_fp16")
    nll_int4 = record.get("nll_int4")
    if nll_fp16 is None or nll_int4 is None:
        return None
    try:
        fp16 = float(nll_fp16)
        int4 = float(nll_int4)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(fp16) and math.isfinite(int4)):
        return None
    return int4 - fp16  # positive = INT4 finds completion less likely


def _filter_finite_scores(records: Iterable[dict]) -> list[dict]:
    """Keep only records with finite nll_fp16 and nll_int4."""
    out: list[dict] = []
    for r in records:
        if _delta_nll(r) is not None:
            out.append(r)
    return out


def compute_spearman_by_workload(
    records: Sequence[dict],
) -> dict[str, float]:
    """Spearman ρ(δ, ΔNLL) per workload. ΔNLL = NLL_INT4 − NLL_FP16."""
    out: dict[str, float] = {}
    by_wk: dict[str, list[dict]] = {}
    for r in records:
        wk = r["workload"]
        by_wk.setdefault(wk, []).append(r)
    for wk, rs in by_wk.items():
        rs = _filter_finite_scores(rs)
        xs = [_mean_delta(r) for r in rs]
        ys = [_delta_nll(r) for r in rs]  # type: ignore[misc]
        rho = spearman_rho(xs, ys)  # type: ignore[arg-type]
        if rho is not None:
            out[wk] = rho
    return out


def compute_spearman_pooled(records: Sequence[dict]) -> Optional[float]:
    """Spearman ρ(δ, ΔNLL) pooled across all workloads with finite NLL."""
    rs = _filter_finite_scores(records)
    xs = [_mean_delta(r) for r in rs]
    ys = [_delta_nll(r) for r in rs]  # type: ignore[misc]
    return spearman_rho(xs, ys)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Variance-aware per-workload classification + gate decision
# ---------------------------------------------------------------------------


def _classify_outcome(
    n: int, rho: Optional[float], stdev: Optional[float]
) -> WorkloadOutcome:
    if n < 2 or stdev is None:
        return "insufficient"
    if stdev < LOW_VARIANCE_STDEV:
        return "undetermined"
    if rho is None:
        return "insufficient"
    if rho >= RHO_THRESHOLD:
        return "informative"
    if rho <= -RHO_THRESHOLD:
        return "inverted"  # backwards — likely a pipeline bug
    return "uninformative"


def per_workload_reports(records: Sequence[dict]) -> list[WorkloadReport]:
    """Per-workload {n, ρ(δ, ΔNLL), accuracy stats, ΔNLL stats, outcome}.

    All workloads with finite NLL values are reported (not just
    ACCURACY_WORKLOADS), because NLL is defined even without a task scorer.
    """
    by_wk: dict[str, list[dict]] = {}
    for r in records:
        by_wk.setdefault(r["workload"], []).append(r)

    reports: list[WorkloadReport] = []
    for wk in sorted(by_wk):
        rs = _filter_finite_scores(by_wk[wk])
        if not rs:
            continue  # workload has no finite NLL records — skip entirely
        xs = [_mean_delta(r) for r in rs]
        dnlls = [_delta_nll(r) for r in rs]  # all finite by construction
        rho = spearman_rho(xs, dnlls) if len(rs) >= 2 else None  # type: ignore[arg-type]

        # FP16 accuracy (monitoring only — may be None for HumanEval)
        fp16_scores = [
            float(r["task_score"])
            for r in rs
            if r.get("task_score") is not None
            and math.isfinite(float(r["task_score"]))
        ]
        acc_mean = sum(fp16_scores) / len(fp16_scores) if fp16_scores else None
        acc_stdev = statistics.pstdev(fp16_scores) if len(fp16_scores) >= 2 else None

        # ΔNLL stats — gate variance check
        dnll_mean = sum(dnlls) / len(dnlls) if dnlls else None  # type: ignore[arg-type]
        dnll_stdev = statistics.pstdev(dnlls) if len(dnlls) >= 2 else None  # type: ignore[arg-type]

        reports.append(
            WorkloadReport(
                workload=wk,
                n=len(rs),
                rho=rho,
                accuracy_mean=acc_mean,
                accuracy_stdev=acc_stdev,
                delta_nll_mean=dnll_mean,
                delta_nll_stdev=dnll_stdev,
                outcome=_classify_outcome(len(rs), rho, dnll_stdev),
            )
        )
    return reports


def gate_decision(
    reports: Sequence[WorkloadReport],
) -> tuple[GateDecision, str]:
    """Apply the three-outcome gate (pass / selective / fail / no_signal).

    The ``selective`` case is the Session-2 critic's refinement: if
    δ predicts accuracy on workloads that carry meaningful accuracy
    variance, and is undetermined on workloads that don't, that is
    not a failure — it's evidence that δ is selectively predictive
    on precisely the workloads where KV compression matters.
    """
    informative = [r for r in reports if r.outcome == "informative"]
    inverted = [r for r in reports if r.outcome == "inverted"]
    uninformative = [r for r in reports if r.outcome == "uninformative"]
    undetermined = [r for r in reports if r.outcome == "undetermined"]

    if inverted:
        wks = ", ".join(f"{r.workload} (ρ={r.rho:+.2f})" for r in inverted)
        return (
            "fail",
            f"INVERTED signal on {wks}: low δ predicts high ΔNLL. "
            f"Likely a pipeline bug (wrong hook target, sign flip, or δ/NLL mismatch). Halt.",
        )

    if uninformative:
        wks = ", ".join(
            f"{r.workload} (ρ={r.rho:+.2f}, ΔNLL_σ={r.delta_nll_stdev:.3f})"
            for r in uninformative
        )
        return (
            "fail",
            f"δ failed to predict ΔNLL on workload(s) with variance: {wks}. "
            f"Pivot: evaluate attention entropy / per-head kurtosis.",
        )

    if not informative:
        return (
            "no_signal",
            "no workload produced enough ΔNLL variance to evaluate ρ. "
            "Check INT4KVHook placement and that completions are non-trivial.",
        )

    if undetermined:
        info_wks = ", ".join(f"{r.workload} (ρ={r.rho:+.2f})" for r in informative)
        und_wks = ", ".join(
            f"{r.workload} (ΔNLL_σ={r.delta_nll_stdev:.3f}<{LOW_VARIANCE_STDEV})"
            for r in undetermined
        )
        return (
            "selective",
            f"δ predictive on {info_wks}; undetermined on {und_wks} due to "
            f"low ΔNLL variance (INT4 and FP16 equally likely there — "
            f"KV compression doesn't affect these completions).",
        )

    info_wks = ", ".join(f"{r.workload} (ρ={r.rho:+.2f})" for r in informative)
    return ("pass", f"δ predictive on all evaluable workloads: {info_wks}.")


def plot_violin(records: Sequence[dict], out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series: dict[str, list[float]] = {}
    for r in records:
        series.setdefault(r["workload"], []).append(_mean_delta(r))

    labels = [w for w in WORKLOADS if w in series and series[w]]
    data = [series[w] for w in labels]
    if not data:
        raise ValueError("no records to plot")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    parts = ax.violinplot(data, showmedians=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("mean δ(ℓ, r)  —  KL(w_INT4 ‖ w_FP16), per request")
    ax.set_title("EchoKV motivation: KV quantization error by workload")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_layer_heatmap(records: Sequence[dict], out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Group by workload, then average per-layer δ across requests.
    by_wk: dict[str, list[list[float]]] = {}
    for r in records:
        by_wk.setdefault(r["workload"], []).append(r["delta"])

    workload_order = [w for w in WORKLOADS if w in by_wk]
    if not workload_order:
        raise ValueError("no records to plot")
    num_layers = max(len(d) for d in by_wk[workload_order[0]])

    matrix = np.zeros((len(workload_order), num_layers), dtype=float)
    for i, wk in enumerate(workload_order):
        stacked = [d for d in by_wk[wk] if len(d) == num_layers]
        if not stacked:
            continue
        matrix[i] = np.mean(np.asarray(stacked, dtype=float), axis=0)

    fig, ax = plt.subplots(figsize=(10, 1 + 0.4 * len(workload_order)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(workload_order)))
    ax.set_yticklabels(workload_order)
    ax.set_xlabel("layer ℓ")
    ax.set_title("EchoKV motivation: layer sensitivity to KV quantization")
    fig.colorbar(im, ax=ax, label="mean δ(ℓ) over requests")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def analyze(records_path: Path, out_dir: Path) -> AnalysisResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(records_path)

    violin_path = plot_violin(records, out_dir / "violin.png")
    heatmap_path = plot_layer_heatmap(records, out_dir / "heatmap.png")

    per_wk_rho = compute_spearman_by_workload(records)
    pooled = compute_spearman_pooled(records)
    reports = per_workload_reports(records)
    decision, rationale = gate_decision(reports)

    (out_dir / "spearman.json").write_text(
        json.dumps(
            {
                "per_workload": per_wk_rho,
                "pooled": pooled,
                "n": len(records),
                "reports": [
                    {
                        "workload": r.workload,
                        "n": r.n,
                        "rho": r.rho,
                        "accuracy_mean": r.accuracy_mean,
                        "accuracy_stdev": r.accuracy_stdev,
                        "delta_nll_mean": r.delta_nll_mean,
                        "delta_nll_stdev": r.delta_nll_stdev,
                        "outcome": r.outcome,
                    }
                    for r in reports
                ],
                "gate": {
                    "decision": decision,
                    "rationale": rationale,
                    "rho_threshold": RHO_THRESHOLD,
                    "low_variance_stdev": LOW_VARIANCE_STDEV,
                },
            },
            indent=2,
        )
    )

    return AnalysisResult(
        violin_path=violin_path,
        heatmap_path=heatmap_path,
        spearman_rho=per_wk_rho,
        spearman_pooled=pooled,
        per_workload=reports,
        gate_decision=decision,
        gate_rationale=rationale,
        num_records=len(records),
    )


# ---------------------------------------------------------------------------
# Summary printing — for the session log and/or stdout
# ---------------------------------------------------------------------------


def format_summary(result: AnalysisResult) -> str:
    lines = [
        f"n = {result.num_records}",
        f"violin: {result.violin_path}",
        f"heatmap: {result.heatmap_path}",
        "",
        "Per-workload report (ρ(δ,ΔNLL) | ΔNLL mean | ΔNLL σ | outcome):",
    ]
    for r in result.per_workload:
        rho_s = f"{r.rho:+.3f}" if r.rho is not None else "   na"
        dm_s = f"{r.delta_nll_mean:+.4f}" if r.delta_nll_mean is not None else "     na"
        ds_s = f"{r.delta_nll_stdev:.4f}" if r.delta_nll_stdev is not None else "   na"
        lines.append(
            f"  {r.workload:15s}  ρ={rho_s}   ΔNLL={dm_s}   σ={ds_s}   n={r.n:4d}   {r.outcome}"
        )
    if result.spearman_pooled is not None:
        lines.append(f"\nSpearman ρ(δ, ΔNLL) pooled: {result.spearman_pooled:+.3f}")
    lines.append("")
    lines.append(f"GATE DECISION: {result.gate_decision.upper()}")
    lines.append(f"rationale: {result.gate_rationale}")
    return "\n".join(lines)
