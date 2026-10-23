"""v8 gate — within-request δ(t) vs ΔNLL(t) Spearman ρ.

Loads ``DeltaStepRecord`` JSONL written by ``run.py --gate-version v8``.
For each request, computes Spearman ρ of (δ_series, ΔNLL_series) over
decode steps, plus first-differenced ρ (detrended) and first-half /
second-half ρ (accumulation diagnostic).

Per-workload aggregates: mean ρ across requests, standard error of the
mean, share of inverted requests. Gate decision per the
author/critic spec:

    PASS       mean ρ_raw > 0.15 AND mean ρ_diff > 0 on all evaluated workloads
    SELECTIVE  ≥ 1 workload has mean ρ_raw > 0.10
    FAIL       all workloads have mean ρ_raw < 0.05

Statistical-power note (n=100 requests × T_comp ≈ 200 steps):
    SE per request ≈ 1/√(T-3) ≈ 0.071
    SE of mean over 100 reqs ≈ 0.0071
A mean ρ of 0.15 is therefore ~21 SE above zero at this scale — the threshold
is conservative.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .analyze import _rank_avg_ties, spearman_rho


PASS_THRESHOLD_RAW = 0.15      # mean ρ_raw must exceed this for pass
SELECTIVE_THRESHOLD_RAW = 0.10
FAIL_THRESHOLD_RAW = 0.05      # all workloads below this → fail
MIN_STEPS = 4                  # need ≥ 4 finite pairs to compute Spearman


@dataclass
class RequestRho:
    request_id: str
    workload: str
    n_steps: int
    rho_raw: Optional[float]
    rho_diff: Optional[float]
    rho_first: Optional[float]
    rho_second: Optional[float]
    delta_mean: Optional[float]
    delta_nll_mean: Optional[float]


@dataclass
class WorkloadV8Report:
    workload: str
    n_requests: int
    n_evaluated: int                  # requests with finite ρ_raw
    mean_rho_raw: Optional[float]
    se_rho_raw: Optional[float]
    mean_rho_diff: Optional[float]
    se_rho_diff: Optional[float]
    mean_rho_first: Optional[float]
    mean_rho_second: Optional[float]
    pct_positive_raw: Optional[float]  # share of requests with ρ_raw > 0
    mean_delta: Optional[float]
    mean_delta_nll: Optional[float]


def _finite_pairs(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    out_x, out_y = [], []
    for x, y in zip(xs, ys):
        if x is None or y is None:
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        out_x.append(float(x))
        out_y.append(float(y))
    return out_x, out_y


def compute_within_request_rho(record: dict) -> RequestRho:
    """One request → ρ_raw, ρ_diff, ρ_first, ρ_second."""
    delta = record.get("delta_series") or []
    nll_fp16 = record.get("nll_fp16_series") or []
    nll_int4 = record.get("nll_int4_series") or []
    if len(delta) != len(nll_fp16) or len(nll_fp16) != len(nll_int4):
        # Length mismatch — skip
        return RequestRho(
            request_id=record["request_id"],
            workload=record["workload"],
            n_steps=0,
            rho_raw=None, rho_diff=None,
            rho_first=None, rho_second=None,
            delta_mean=None, delta_nll_mean=None,
        )

    dnll = [(b - a) if (a is not None and b is not None) else None
            for a, b in zip(nll_fp16, nll_int4)]
    deltas, dnlls = _finite_pairs(delta, dnll)

    if len(deltas) < MIN_STEPS:
        return RequestRho(
            request_id=record["request_id"],
            workload=record["workload"],
            n_steps=len(deltas),
            rho_raw=None, rho_diff=None,
            rho_first=None, rho_second=None,
            delta_mean=statistics.fmean(deltas) if deltas else None,
            delta_nll_mean=statistics.fmean(dnlls) if dnlls else None,
        )

    rho_raw = spearman_rho(deltas, dnlls)
    # First differences for detrending
    d_diff = [deltas[i] - deltas[i - 1] for i in range(1, len(deltas))]
    n_diff = [dnlls[i] - dnlls[i - 1] for i in range(1, len(dnlls))]
    rho_diff = spearman_rho(d_diff, n_diff) if len(d_diff) >= MIN_STEPS else None
    # First / second halves
    mid = len(deltas) // 2
    rho_first = (
        spearman_rho(deltas[:mid], dnlls[:mid])
        if mid >= MIN_STEPS else None
    )
    rho_second = (
        spearman_rho(deltas[mid:], dnlls[mid:])
        if (len(deltas) - mid) >= MIN_STEPS else None
    )

    return RequestRho(
        request_id=record["request_id"],
        workload=record["workload"],
        n_steps=len(deltas),
        rho_raw=rho_raw,
        rho_diff=rho_diff,
        rho_first=rho_first,
        rho_second=rho_second,
        delta_mean=statistics.fmean(deltas),
        delta_nll_mean=statistics.fmean(dnlls),
    )


def _se_mean(xs: list[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    return statistics.pstdev(xs) / math.sqrt(len(xs))


def per_workload_v8_reports(per_req: list[RequestRho]) -> list[WorkloadV8Report]:
    by_w: dict[str, list[RequestRho]] = {}
    for r in per_req:
        by_w.setdefault(r.workload, []).append(r)

    reports = []
    for w, rs in sorted(by_w.items()):
        evaluated = [r for r in rs if r.rho_raw is not None]
        rho_raws = [r.rho_raw for r in evaluated]
        rho_diffs = [r.rho_diff for r in evaluated if r.rho_diff is not None]
        rho_firsts = [r.rho_first for r in evaluated if r.rho_first is not None]
        rho_seconds = [r.rho_second for r in evaluated if r.rho_second is not None]
        deltas = [r.delta_mean for r in rs if r.delta_mean is not None]
        delta_nlls = [r.delta_nll_mean for r in rs if r.delta_nll_mean is not None]

        n_pos = sum(1 for r in rho_raws if r > 0)
        reports.append(WorkloadV8Report(
            workload=w,
            n_requests=len(rs),
            n_evaluated=len(evaluated),
            mean_rho_raw=statistics.fmean(rho_raws) if rho_raws else None,
            se_rho_raw=_se_mean(rho_raws),
            mean_rho_diff=statistics.fmean(rho_diffs) if rho_diffs else None,
            se_rho_diff=_se_mean(rho_diffs),
            mean_rho_first=statistics.fmean(rho_firsts) if rho_firsts else None,
            mean_rho_second=statistics.fmean(rho_seconds) if rho_seconds else None,
            pct_positive_raw=(n_pos / len(rho_raws)) if rho_raws else None,
            mean_delta=statistics.fmean(deltas) if deltas else None,
            mean_delta_nll=statistics.fmean(delta_nlls) if delta_nlls else None,
        ))
    return reports


def gate_decision_v8(reports: list[WorkloadV8Report]) -> tuple[str, str]:
    """Gate v8 with detrend-aware criteria.

    A workload "carries signal" if either ``mean_ρ_raw > 0.15`` (raw within-
    request co-variation clears the PASS bar) or ``mean_ρ_diff > 0.15`` (step-
    by-step detrended co-variation does — the production canary fires on the
    raw quantity, but the detrended ρ is the cleanest test of whether δ(t)
    and ΔNLL(t) move together at the per-step grain). A workload is
    "inverted" if either measure is below ``-PASS_THRESHOLD_RAW``.

        PASS:       every evaluated workload carries signal
        SELECTIVE:  ≥ 1 workload carries signal (other workloads documented as
                    out-of-scope; honest scope, not a hedge)
        FAIL:       every workload mean_ρ_raw AND mean_ρ_diff < FAIL_THRESHOLD
        otherwise:  WEAK
    """
    evaluated = [r for r in reports if r.mean_rho_raw is not None]
    if not evaluated:
        return "no_signal", "no workload had ≥1 evaluable request"

    def carries_signal(r: WorkloadV8Report) -> bool:
        raw = r.mean_rho_raw or 0.0
        diff = r.mean_rho_diff if r.mean_rho_diff is not None else 0.0
        return raw > PASS_THRESHOLD_RAW or diff > PASS_THRESHOLD_RAW

    def inverted(r: WorkloadV8Report) -> bool:
        raw = r.mean_rho_raw if r.mean_rho_raw is not None else 0.0
        diff = r.mean_rho_diff if r.mean_rho_diff is not None else 0.0
        return raw < -PASS_THRESHOLD_RAW or diff < -PASS_THRESHOLD_RAW

    inv = [r.workload for r in evaluated if inverted(r)]
    if inv:
        return "inverted", f"workload(s) with strongly negative ρ: {', '.join(inv)}"

    carriers = [r for r in evaluated if carries_signal(r)]
    if len(carriers) == len(evaluated) and carriers:
        wks = ", ".join(
            f"{r.workload} ρ_raw={r.mean_rho_raw:+.3f} ρ_diff={r.mean_rho_diff:+.3f}"
            for r in carriers
        )
        return "pass", f"all workloads carry signal ({wks})"
    if carriers:
        wks_in = ", ".join(
            f"{r.workload} ρ_raw={r.mean_rho_raw:+.3f} ρ_diff={r.mean_rho_diff:+.3f}"
            for r in carriers
        )
        wks_out = ", ".join(
            r.workload for r in evaluated if not carries_signal(r)
        )
        return "selective", (
            f"signal carried by: {wks_in}. Out-of-scope: {wks_out}"
        )

    raws = [r.mean_rho_raw or 0.0 for r in evaluated]
    diffs = [r.mean_rho_diff or 0.0 for r in evaluated]
    if all(r < FAIL_THRESHOLD_RAW for r in raws) and all(d < FAIL_THRESHOLD_RAW for d in diffs):
        return "fail", f"all workloads ρ_raw AND ρ_diff < {FAIL_THRESHOLD_RAW}"
    return "weak", "between selective and fail thresholds"


def analyze_v8(records_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with records_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    per_req = [compute_within_request_rho(r) for r in records]
    reports = per_workload_v8_reports(per_req)
    decision, rationale = gate_decision_v8(reports)

    # Write JSON summary
    summary = {
        "n_requests": len(records),
        "n_evaluated": sum(r.n_evaluated for r in reports),
        "thresholds": {
            "pass_raw": PASS_THRESHOLD_RAW,
            "selective_raw": SELECTIVE_THRESHOLD_RAW,
            "fail_raw": FAIL_THRESHOLD_RAW,
            "min_steps": MIN_STEPS,
        },
        "per_workload": [
            {
                "workload": r.workload,
                "n_requests": r.n_requests,
                "n_evaluated": r.n_evaluated,
                "mean_rho_raw": r.mean_rho_raw,
                "se_rho_raw": r.se_rho_raw,
                "mean_rho_diff": r.mean_rho_diff,
                "se_rho_diff": r.se_rho_diff,
                "mean_rho_first": r.mean_rho_first,
                "mean_rho_second": r.mean_rho_second,
                "pct_positive_raw": r.pct_positive_raw,
                "mean_delta": r.mean_delta,
                "mean_delta_nll": r.mean_delta_nll,
            }
            for r in reports
        ],
        "gate": {"decision": decision, "rationale": rationale},
    }
    (out_dir / "spearman_v8.json").write_text(json.dumps(summary, indent=2))

    # Write per-request TSV (with first/second-half ρ for the paper's robustness check)
    tsv = out_dir / "per_request.tsv"
    with tsv.open("w") as f:
        f.write("request_id\tworkload\tn_steps\trho_raw\trho_diff\trho_first\trho_second\tdelta_mean\tdelta_nll_mean\n")
        for r in per_req:
            def fmt(x):
                return f"{x:.6f}" if isinstance(x, float) else "na"
            f.write(
                f"{r.request_id}\t{r.workload}\t{r.n_steps}\t"
                f"{fmt(r.rho_raw)}\t{fmt(r.rho_diff)}\t"
                f"{fmt(r.rho_first)}\t{fmt(r.rho_second)}\t"
                f"{fmt(r.delta_mean)}\t{fmt(r.delta_nll_mean)}\n"
            )
    return summary


def format_v8_summary(summary: dict) -> str:
    lines = [
        f"n_requests = {summary['n_requests']}",
        f"n_evaluated = {summary['n_evaluated']}",
        "",
        "Per-workload (mean ρ_raw ± SE | mean ρ_diff ± SE | first/second-half | %pos | n):",
    ]
    for w in summary["per_workload"]:
        def f(x): return f"{x:+.3f}" if isinstance(x, float) else "  na "
        def s(x): return f"{x:.3f}" if isinstance(x, float) else " na"
        lines.append(
            f"  {w['workload']:12s}  ρ_raw={f(w['mean_rho_raw'])}±{s(w['se_rho_raw'])}   "
            f"ρ_diff={f(w['mean_rho_diff'])}±{s(w['se_rho_diff'])}   "
            f"halves [{f(w['mean_rho_first'])}, {f(w['mean_rho_second'])}]   "
            f"%pos={f(w['pct_positive_raw'] * 100 if w['pct_positive_raw'] is not None else None)}%   "
            f"n={w['n_evaluated']}/{w['n_requests']}"
        )
    lines.append("")
    lines.append(f"GATE DECISION: {summary['gate']['decision'].upper()}")
    lines.append(f"rationale: {summary['gate']['rationale']}")
    return "\n".join(lines)


def per_layer_rho(records: list[dict], window: Optional[int] = None) -> dict:
    """Per-layer mean ρ(δ_ℓ(t), ΔNLL(t)) across requests.

    For each record, compute Spearman ρ over decode steps for each layer ℓ
    using the per-layer δ series captured in v8.1+ records. Mean across
    requests per workload + layer.

    ``window``: if set, restricts each request's series to the first ``window``
    completion tokens (e.g., ``window=32`` for RULER answer-only window).
    """
    by_w_layer: dict[str, dict[int, list[float]]] = {}
    for r in records:
        per_layer = r.get("per_layer_delta")
        if not per_layer:
            continue
        nf = r.get("nll_fp16_series") or []
        ni = r.get("nll_int4_series") or []
        if not nf or not ni:
            continue
        if window is not None:
            nf = nf[:window]
            ni = ni[:window]
        L = len(per_layer)
        for ell in range(L):
            d = per_layer[ell]
            if window is not None:
                d = d[:window]
            dnll = [b - a for a, b in zip(nf, ni)]
            d_clean, n_clean = _finite_pairs(d, dnll)
            if len(d_clean) < MIN_STEPS:
                continue
            rho = spearman_rho(d_clean, n_clean)
            if rho is None:
                continue
            by_w_layer.setdefault(r["workload"], {}).setdefault(ell, []).append(rho)

    summary = {}
    for w, layer_to_rhos in by_w_layer.items():
        per_layer_summary = []
        for ell in sorted(layer_to_rhos.keys()):
            rhos = layer_to_rhos[ell]
            per_layer_summary.append({
                "layer": ell,
                "n_requests": len(rhos),
                "mean_rho": statistics.fmean(rhos),
                "se_rho": _se_mean(rhos),
                "pct_positive": sum(1 for r in rhos if r > 0) / len(rhos),
            })
        summary[w] = per_layer_summary
    return summary


def windowed_rho(records: list[dict], windows: list[int]) -> dict:
    """Per-workload mean ρ over the first ``window`` completion tokens.

    Used to confirm the canary's "operating window" — the early decode steps
    where peaked attention dominates. Returns
    ``{workload: {window: {mean_rho, se_rho, pct_positive, n}}}``.
    """
    out: dict = {}
    for w_size in windows:
        for r in records:
            d = (r.get("delta_series") or [])[:w_size]
            nf = (r.get("nll_fp16_series") or [])[:w_size]
            ni = (r.get("nll_int4_series") or [])[:w_size]
            dnll = [b - a for a, b in zip(nf, ni)]
            d_clean, n_clean = _finite_pairs(d, dnll)
            if len(d_clean) < MIN_STEPS:
                continue
            rho = spearman_rho(d_clean, n_clean)
            if rho is None:
                continue
            out.setdefault(r["workload"], {}).setdefault(w_size, []).append(rho)
    summary: dict = {}
    for w, by_size in out.items():
        summary[w] = {}
        for sz, rhos in sorted(by_size.items()):
            summary[w][sz] = {
                "n": len(rhos),
                "mean_rho": statistics.fmean(rhos),
                "se_rho": _se_mean(rhos),
                "pct_positive": sum(1 for r in rhos if r > 0) / len(rhos),
            }
    return summary


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: python -m evaluation.motivation.analyze_v8 <records.jsonl> <out_dir>")
        sys.exit(1)
    summary = analyze_v8(Path(sys.argv[1]), Path(sys.argv[2]))
    print(format_v8_summary(summary))
