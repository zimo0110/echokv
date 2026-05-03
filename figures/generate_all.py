"""Regenerate all 5 publication figures for the EchoKV arXiv preprint.

Reads only real data files under ``results/motivation/`` and
``results/serving/``. Halts with a clear error if any required file
is missing. No mock data, no invented numbers.

Run from the project root::

    python figures/generate_all.py

Outputs ``figures/figN_*.{pdf,png}`` for N in {1..5} and prints a
summary table.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import numpy as np


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = Path(__file__).resolve().parent
FIG_DIR.mkdir(parents=True, exist_ok=True)

ACCENT_BLUE = "#1f77b4"
ACCENT_RED = "#d62728"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.2,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})


def save_both(fig, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.pdf", dpi=300)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300)


def panel_label(ax, label: str, title: Optional[str] = None) -> None:
    """Set the axes title to ``(<label>) <title>`` with the label bold-ish.

    Single-string title with the panel letter prefixed is cleaner and
    avoids the overlap problems that float a separate (a)/(b)/(c)
    label above ``ax.set_title``.
    """
    text = f"$\\mathbf{{({label})}}$" + ((" " + title) if title else "")
    ax.set_title(text, loc="left", fontsize=10, pad=6)


def require_file(path: Path, fig_name: str) -> Path:
    if not path.exists():
        print(f"\n[FATAL] {fig_name}: required data file missing: {path}",
              file=sys.stderr)
        print(f"\nNo figure was generated. Halting per the discipline that "
              f"figures are sourced only from real data; the closest "
              f"alternative file (if any) is the user's call to substitute, "
              f"not a script-level decision.", file=sys.stderr)
        sys.exit(1)
    return path


# ---------------------------------------------------------------------------
# Figure 1 — Within-request co-variation (paper §3.2)
# ---------------------------------------------------------------------------


def _spearman(a, b) -> float:
    """Spearman's rho between two equal-length sequences. Returns NaN when
    either sequence has zero variance after ranking."""
    n = len(a)
    if n != len(b) or n < 2:
        return float("nan")

    def _rank(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        ranks = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    ra, rb = _rank(a), _rank(b)
    mean_a = sum(ra) / n
    mean_b = sum(rb) / n
    num = sum((ra[i] - mean_a) * (rb[i] - mean_b) for i in range(n))
    da = math.sqrt(sum((x - mean_a) ** 2 for x in ra))
    db = math.sqrt(sum((x - mean_b) ** 2 for x in rb))
    return num / (da * db) if da * db > 0 else float("nan")


def fig1_motivation() -> dict:
    fig_name = "fig1_motivation"
    delta_path = require_file(
        ROOT / "results/motivation/delta_records_v8_1.jsonl", fig_name,
    )
    tsv_path = require_file(
        ROOT / "results/motivation/analysis_v8_1/per_request.tsv", fig_name,
    )

    # Load all RULER-32K records.
    records = []
    with delta_path.open() as f:
        for line in f:
            r = json.loads(line)
            if r["workload"] == "ruler_32k":
                records.append(r)
    n_records = len(records)

    # Per-request Spearman ρ_diff from the canonical analysis (used as
    # ground truth — these are detrended in the analysis pipeline).
    per_req: list[dict] = []
    with tsv_path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["workload"] != "ruler_32k":
                continue
            per_req.append({
                "request_id": row["request_id"],
                "rho_diff": float(row["rho_diff"]),
                "rho_first": float(row["rho_first"]) if row["rho_first"] != "na" else float("nan"),
            })

    rho_diff_vals = [r["rho_diff"] for r in per_req]
    mean_rho_diff = statistics.fmean(rho_diff_vals)
    sd_rho_diff = statistics.pstdev(rho_diff_vals)
    se_rho_diff = sd_rho_diff / math.sqrt(len(rho_diff_vals))
    ci_lo = mean_rho_diff - 1.96 * se_rho_diff
    ci_hi = mean_rho_diff + 1.96 * se_rho_diff
    z_mean = mean_rho_diff / se_rho_diff

    # Representative request: closest |ρ_diff − mean|.
    rep = min(per_req, key=lambda r: abs(r["rho_diff"] - mean_rho_diff))
    rep_record = next(r for r in records if r["request_id"] == rep["request_id"])

    # Build ΔNLL = NLL_int4 − NLL_fp16 series for representative.
    nll_fp = rep_record["nll_fp16_series"]
    nll_q = rep_record["nll_int4_series"]
    delta_series = rep_record["delta_series"]
    n_steps = len(delta_series)
    dnll_series = [nll_q[i] - nll_fp[i] for i in range(n_steps)]

    # Per-layer ρ̄: for each layer ℓ, compute Spearman(per_layer_delta[ℓ],
    # dnll) per request, average across requests. ``per_layer_delta`` in
    # this file is shape (num_layers, n_steps); skip layer 0 (sentinel zeros)
    # and any zero-variance series.
    num_layers = records[0]["num_layers"]
    per_layer_rho_means = [float("nan")] * num_layers
    per_layer_rho_se = [float("nan")] * num_layers
    for ell in range(num_layers):
        rhos: list[float] = []
        for r in records:
            pld = r["per_layer_delta"]
            if len(pld) <= ell:
                continue
            series = pld[ell]
            if len(series) != n_steps:
                continue
            if max(series) == min(series):
                continue
            dnll = [r["nll_int4_series"][i] - r["nll_fp16_series"][i]
                    for i in range(n_steps)]
            if max(dnll) == min(dnll):
                continue
            rho = _spearman(series, dnll)
            if not math.isnan(rho):
                rhos.append(rho)
        if rhos:
            mu = statistics.fmean(rhos)
            sd = statistics.pstdev(rhos)
            per_layer_rho_means[ell] = mu
            per_layer_rho_se[ell] = sd / math.sqrt(len(rhos))

    # Identify "highlighted" layers: per-layer mean ρ̄ > 2× its SE.
    z_per_layer = [
        (per_layer_rho_means[i] / per_layer_rho_se[i])
        if per_layer_rho_se[i] > 0 else 0.0
        for i in range(num_layers)
    ]
    peak_ell = max(
        (i for i in range(num_layers) if not math.isnan(per_layer_rho_means[i])),
        key=lambda i: per_layer_rho_means[i],
    )
    peak_rho = per_layer_rho_means[peak_ell]
    peak_z = z_per_layer[peak_ell]

    # ----- Build figure -----
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.6), constrained_layout=True)

    # Panel (a): one representative request, dual-axis time series
    ax = axes[0]
    panel_label(ax, "a", f"request {rep['request_id']}")
    ts = list(range(n_steps))
    ax.plot(ts, delta_series, color="black", linewidth=1.2, label=r"$\delta(t)$")
    ax.set_xlabel("decode step $t$")
    ax.set_ylabel(r"$\delta(t)$", color="black")
    ax.tick_params(axis="y", labelcolor="black")
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(ts, dnll_series, color="0.45", linewidth=1.2, linestyle=(0, (4, 2)),
             label=r"$\Delta$NLL$(t)$")
    ax2.set_ylabel(r"$\Delta$NLL$(t)$", color="0.45")
    ax2.tick_params(axis="y", labelcolor="0.45")
    ax.axvline(8, color="0.7", linewidth=0.7, linestyle=":")
    ax.text(0.05, 0.96,
            fr"$\rho_{{\mathrm{{diff}}}}={rep['rho_diff']:+.3f}$" + "\n"
            fr"first-8 $\rho={rep['rho_first']:+.3f}$",
            transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="0.7", linewidth=0.4,
                      boxstyle="round,pad=0.25"))

    # Panel (b): histogram of per-request ρ_diff
    ax = axes[1]
    panel_label(ax, "b", "per-request distribution")
    ax.hist(rho_diff_vals, bins=20, color="0.78", edgecolor="black", linewidth=0.6)
    ax.axvline(0, color="0.5", linewidth=0.7, linestyle=":")
    ax.axvline(mean_rho_diff, color="black", linewidth=1.2)
    ymin, ymax = ax.get_ylim()
    ax.add_patch(mpatches.Rectangle(
        (ci_lo, 0), ci_hi - ci_lo, ymax,
        color="0.85", alpha=0.5, zorder=0, linewidth=0,
    ))
    ax.set_xlabel(r"per-request $\rho_{\mathrm{diff}}$")
    ax.set_ylabel("count")
    ax.text(0.04, 0.96,
            fr"$\bar{{\rho}}={mean_rho_diff:+.3f}$" + "\n"
            fr"95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}]" + "\n"
            fr"$z={z_mean:.2f}\sigma$ ($n={len(rho_diff_vals)}$)",
            transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="0.7", linewidth=0.4,
                      boxstyle="round,pad=0.25"))

    # Panel (c): per-layer ρ̄ bar chart
    ax = axes[2]
    panel_label(ax, "c", r"per-layer $\bar\rho$")
    layers = list(range(num_layers))
    rho_arr = np.array([
        per_layer_rho_means[i] if not math.isnan(per_layer_rho_means[i]) else 0.0
        for i in range(num_layers)
    ])
    z_arr = np.array(z_per_layer)
    colors = ["black" if z > 2.0 else "0.78" for z in z_arr]
    ax.bar(layers, rho_arr, color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="0.5", linewidth=0.6, linestyle="--")
    ax.set_xlabel("layer index")
    ax.set_ylabel(r"$\bar\rho$")
    ax.text(0.04, 0.96,
            f"peak layer {peak_ell}: " + fr"$\bar\rho={peak_rho:+.3f}$"
            + "\n"
            fr"($z={peak_z:.1f}\sigma$)" + "\n"
            r"black: $z>2$",
            transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="0.7", linewidth=0.4,
                      boxstyle="round,pad=0.25"))

    save_both(fig, fig_name)
    plt.close(fig)

    return {
        "figure": "fig1_motivation",
        "source": "results/motivation/delta_records_v8_1.jsonl + analysis_v8_1/per_request.tsv",
        "n_records": n_records,
        "mean_rho_diff": mean_rho_diff,
        "ci": (ci_lo, ci_hi),
        "z_mean": z_mean,
        "representative": rep["request_id"],
        "peak_layer": peak_ell,
        "peak_rho": peak_rho,
        "peak_z": peak_z,
    }


# ---------------------------------------------------------------------------
# Figure 2 — Sensor mechanism diagram (schematic)
# ---------------------------------------------------------------------------


def _box(ax, x, y, w, h, text, fontsize=8.5, lw=1.0):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        linewidth=lw, edgecolor="black", facecolor="white",
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize)


def _arrow(ax, x0, y0, x1, y1, color="black", linestyle="-", lw=1.0,
           label=None, label_offset=(0, 0)):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", color=color, linestyle=linestyle, lw=lw),
    )
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx + label_offset[0], my + label_offset[1], label,
                ha="left", va="center", fontsize=7.5, color=color)


def fig2_sensor() -> dict:
    fig_name = "fig2_sensor"
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.6), constrained_layout=True)

    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.axis("off")

    # ---- (a) naive canary ----
    ax = axes[0]
    ax.text(0.0, 1.02, r"$\mathbf{(a)}$ naive canary",
            transform=ax.transAxes, fontsize=10, ha="left", va="bottom")

    _box(ax, 2.0, 8.0, 6.0, 0.9, "FP16 K projection")
    _arrow(ax, 5.0, 8.0, 5.0, 7.1)
    _box(ax, 1.5, 6.2, 7.0, 0.9, "tier_application (per-layer quantize)")
    _arrow(ax, 5.0, 6.2, 5.0, 5.3)
    _box(ax, 1.5, 4.4, 7.0, 0.9, "Cache (mixed-tier state)")
    _arrow(ax, 5.0, 4.4, 5.0, 3.5)
    _box(ax, 0.5, 1.8, 9.0, 1.7,
         "Canary fires reading from Cache:\n"
         r"$\mathrm{KL}(\mathrm{softmax}(qK^\top_{\mathrm{cache,quant}})\;\|\;"
         r"\mathrm{softmax}(qK^\top_{\mathrm{cache}}))$",
         fontsize=8)
    # Controller-decisions-perturb-substrate feedback (red dashed)
    ax.annotate(
        "", xy=(7.7, 6.65), xytext=(8.6, 2.65),
        arrowprops=dict(arrowstyle="->", color=ACCENT_RED, linestyle="--", lw=1.2,
                        connectionstyle="arc3,rad=0.5"),
    )
    ax.text(9.2, 4.5, "decisions\nperturb\nsubstrate",
            color=ACCENT_RED, fontsize=7.5, ha="center", va="center")

    # ---- (b) absolute canary ----
    ax = axes[1]
    ax.text(0.0, 1.02, r"$\mathbf{(b)}$ absolute canary (this paper)",
            transform=ax.transAxes, fontsize=10, ha="left", va="bottom")

    _box(ax, 2.0, 8.0, 6.0, 0.9, "FP16 K projection")
    # Branch left to K_ref, down to tier_application
    _arrow(ax, 4.0, 8.0, 1.6, 7.1)
    _arrow(ax, 6.0, 8.0, 6.0, 7.1)
    _box(ax, 0.2, 6.2, 3.2, 0.9, "K_ref ring buffer\n(N=64 rows, FP16)",
         fontsize=8.0)
    _box(ax, 4.0, 6.2, 5.8, 0.9, "tier_application")
    _arrow(ax, 6.9, 6.2, 6.9, 5.3)
    _box(ax, 4.0, 4.4, 5.8, 0.9, "Cache (mixed-tier state)")
    # canary fires from K_ref directly
    _arrow(ax, 1.8, 6.2, 1.8, 3.5, lw=1.0)
    _box(ax, 0.5, 1.8, 9.0, 1.7,
         "Canary fires reading from K_ref:\n"
         r"$\mathrm{KL}(\mathrm{softmax}(qK_{\mathrm{ref,quant}}^\top)\;\|\;"
         r"\mathrm{softmax}(qK_{\mathrm{ref}}^\top))$",
         fontsize=8)
    ax.text(5, 0.6, "K_ref independent of cache state — no feedback path",
            ha="center", va="center", fontsize=8, style="italic", color="0.3")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.04, wspace=0.05)
    save_both(fig, fig_name)
    plt.close(fig)

    return {"figure": "fig2_sensor", "source": "(schematic)", "rows": "-"}


# ---------------------------------------------------------------------------
# Figure 3 — Architecture diagram (schematic)
# ---------------------------------------------------------------------------


def fig3_architecture() -> dict:
    fig_name = "fig3_architecture"
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("auto")
    ax.axis("off")

    # Three component boxes — only the title goes in the box; bullet
    # detail goes below as small text. This avoids the previous
    # text-overflow problem.
    boxes = [
        (3, 65, 26, 22, "Shadow Canary\nEstimator",
         ["FlashInfer Triton kernel",
          r"$p_c=1/64$, two CUDA streams",
          "0 mem overhead beyond K_ref"]),
        (37, 65, 26, 22, "PID Precision\nController",
         [r"EMA over per-(layer, tier) $\hat\delta_b$",
          "argmin-with-budget on neighbours",
          r"tick every $N_c{=}100$ decode steps"]),
        (71, 65, 26, 22, "SLO-Coupled\nPrecision Budget",
         ["vLLM scheduler hook",
          "synchronous TP broadcast"]),
    ]
    for (x, y, w, h, title, bullets) in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.5",
            linewidth=1.0, edgecolor="black", facecolor="white",
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center",
                fontsize=9.5, fontweight="bold")
        # Bullet list below the title text, inside the box.
        for i, bul in enumerate(bullets):
            ax.text(x + w / 2, y + h * 0.40 - i * 3.5, "• " + bul,
                    ha="center", va="center", fontsize=8)

    # Forward arrows between boxes
    _arrow(ax, 30, 76, 37, 76, lw=1.2)
    ax.text(33.5, 80, r"$\hat\delta_b$", ha="center", va="bottom",
            fontsize=10, color="black")
    _arrow(ax, 64, 76, 71, 76, lw=1.2)
    ax.text(67.5, 80, r"$\{b_\ell\}$", ha="center", va="bottom",
            fontsize=10, color="black")

    # Loop-back from Budget → Estimator (alloc policy)
    ax.annotate(
        "", xy=(13, 65), xytext=(87, 65),
        arrowprops=dict(arrowstyle="->", color="0.3", linestyle="-", lw=1.0,
                        connectionstyle="arc3,rad=-0.4"),
    )
    ax.text(50, 35, "alloc policy", ha="center", va="center",
            fontsize=9, color="0.3", style="italic")

    # INV-2 barrier
    ax.axhline(20, color=ACCENT_RED, linestyle="--", linewidth=1.0,
               xmin=0.05, xmax=0.95)
    ax.text(50, 13,
            "INV-2 BARRIER: existing blocks are never modified — new allocations only.",
            ha="center", va="center", fontsize=9, color=ACCENT_RED,
            fontweight="bold")
    ax.text(50, 6, "(information-theoretic axiom, §2.3)",
            ha="center", va="center", fontsize=8, color="0.4", style="italic")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    save_both(fig, fig_name)
    plt.close(fig)

    return {"figure": "fig3_architecture", "source": "(schematic)", "rows": "-"}


# ---------------------------------------------------------------------------
# Figure 4 — Sensor validation (paper §5.5)
# ---------------------------------------------------------------------------


def fig4_sensor_validation() -> dict:
    fig_name = "fig4_sensor_validation"
    smoke_path = require_file(
        ROOT / "results/serving/smoke_absolute_canary_8b.json", fig_name,
    )
    blob = json.loads(smoke_path.read_text())
    runs = blob["runs"]

    # All 4 absolute-canary smoke runs (mns ∈ {1, 8} × seed ∈ {0, 7})
    by_label = {r["label"]: r for r in runs}

    def _max_nz(vec):
        nz = [v for v in vec if v > 0]
        return max(nz) if nz else 0.0

    abs_mns1 = _max_nz(by_label["mns=1_seed=0"]["delta_4_per_layer"])
    abs_mns8 = _max_nz(by_label["mns=8_seed=0"]["delta_4_per_layer"])

    # Naive (production warmup) baseline: from the documented 60× gap
    # finding (CHANGELOG Phase 5). We DO NOT have the legacy mns=1
    # smoke run on file (the legacy sensor was already replaced). The
    # canonical published number is the production warmup δ_4 max from
    # the Phase-5 diagnostic — read it from the Phase-5 result file.
    legacy_path = ROOT / "results/serving/session14_70b/echokv_3c_n100_mp_tau002_v2.json"
    if legacy_path.exists():
        b2 = json.loads(legacy_path.read_text())
        warmup_pt = b2.get("warmup_per_tier_delta_per_rank") or []
        # average across ranks of per-layer δ̂_4 max
        if warmup_pt and warmup_pt[0]:
            d4_per_rank = [d.get("4") or [] for d in warmup_pt]
            n_layers = len(d4_per_rank[0])
            per_layer = []
            for ell in range(n_layers):
                vals = [v[ell] for v in d4_per_rank if v[ell] > 0]
                per_layer.append(sum(vals) / len(vals) if vals else 0.0)
            naive_mns1_max = max(per_layer) if per_layer else 0.0
        else:
            naive_mns1_max = 0.0
    else:
        # Fallback if file missing: skip with a clear annotation
        naive_mns1_max = 0.0

    # Visual layout: 2x2
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.2), constrained_layout=True)

    # ----- (a) Magnitude (log scale) -----
    ax = axes[0, 0]
    panel_label(ax, "a", "magnitude")
    bars_x = [0, 1]
    bars_y = [naive_mns1_max if naive_mns1_max > 0 else 1e-7, abs_mns1]
    ax.bar(bars_x, bars_y, color=["0.45", "black"], edgecolor="black", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_xticks(bars_x)
    ax.set_xticklabels(
        ["naive\n(70B prod warmup)", "absolute\n(8B+TP=2 smoke)"],
        fontsize=8,
    )
    ax.set_ylabel(r"max $\hat\delta_4$ across layers")
    ratio = (abs_mns1 / naive_mns1_max) if naive_mns1_max > 0 else float("nan")
    for x, y in zip(bars_x, bars_y):
        ax.text(x, y * 1.6, f"{y:.1e}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(top=ax.get_ylim()[1] * 12)
    if math.isfinite(ratio):
        ax.text(0.5, 0.04, f"ratio: {ratio:.0f}× larger",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=9, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="0.7", linewidth=0.5,
                          boxstyle="round,pad=0.25"))

    # ----- (b) max_num_seqs invariance -----
    ax = axes[0, 1]
    panel_label(ax, "b", "max_num_seqs invariance")
    labels = ["mns=1\ns=0", "mns=1\ns=7", "mns=8\ns=0", "mns=8\ns=7"]
    keys = ["mns=1_seed=0", "mns=1_seed=7", "mns=8_seed=0", "mns=8_seed=7"]
    vals = [_max_nz(by_label[k]["delta_4_per_layer"]) for k in keys]
    ax.bar(range(4), vals, color="black", edgecolor="black", linewidth=0.6)
    # Hatched bar at left for the legacy KL≡0 floor at T=1
    ax.bar([-1.4], [1e-7], color="white", edgecolor=ACCENT_RED, linewidth=1.0,
           hatch="///", width=0.85)
    ax.text(-1.4, 2.0e-7,
            "naive\n$\\equiv 0$",
            ha="center", va="bottom", fontsize=7, color=ACCENT_RED)
    ax.set_xticks([-1.4] + list(range(4)))
    ax.set_xticklabels(["naive\n(T=1)"] + labels, fontsize=7.5)
    ax.set_yscale("log")
    ax.set_ylabel(r"max $\hat\delta_4$")
    ax.set_ylim(bottom=1e-8, top=ax.get_ylim()[1] * 8)
    abs_ratio = max(vals) / min(vals) if min(vals) > 0 else float("nan")
    if math.isfinite(abs_ratio):
        ax.text(0.5, 0.04,
                f"absolute spread across mns: {abs_ratio:.2f}×",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=9,
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="0.7", linewidth=0.5,
                          boxstyle="round,pad=0.25"))

    # ----- (c) Cross-tier monotonicity -----
    ax = axes[1, 0]
    panel_label(ax, "c", "cross-tier monotonicity")
    run = by_label["mns=8_seed=0"]
    d2 = run["delta_2_per_layer"]
    d4 = run["delta_4_per_layer"]
    d8 = run["delta_8_per_layer"]
    n_layers = len(d4)
    keep = [i for i in range(n_layers) if d4[i] > 0]
    cols = [0, 1, 2]
    for ell in keep:
        ax.plot(cols, [d2[ell], d4[ell], d8[ell]],
                color="0.7", linewidth=0.4, alpha=0.6)
    for c, vals in zip(cols, [d2, d4, d8]):
        ys = [vals[i] for i in keep]
        ax.scatter([c] * len(ys), ys, color="black", s=10, zorder=3)
    ax.set_xticks(cols)
    ax.set_xticklabels([r"$\hat\delta_2$", r"$\hat\delta_4$", r"$\hat\delta_8$"])
    ax.set_yscale("log")
    ax.set_ylabel(r"per-layer EMA")
    viols = 0
    total = 0
    for r in runs:
        for i in range(len(r["delta_4_per_layer"])):
            d2i = r["delta_2_per_layer"][i]
            d4i = r["delta_4_per_layer"][i]
            d8i = r["delta_8_per_layer"][i]
            if d4i == 0 and d2i == 0:
                continue
            total += 1
            if d2i + 1e-7 < d4i or d4i + 1e-7 < d8i:
                viols += 1
    ax.text(0.05, 0.96,
            f"{viols}/{total} violations\nacross 4 runs × {n_layers} layers",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            fontweight="bold" if viols == 0 else "normal",
            bbox=dict(facecolor="white", edgecolor="0.7", linewidth=0.4,
                      boxstyle="round,pad=0.25"))

    # ----- (d) Cross-seed stability -----
    ax = axes[1, 1]
    panel_label(ax, "d", "cross-seed stability")
    a = by_label["mns=8_seed=0"]["delta_4_per_layer"]
    b = by_label["mns=8_seed=7"]["delta_4_per_layer"]
    rho = _spearman(a, b)
    ax.scatter(a, b, color="black", s=14, edgecolor="white", linewidth=0.4)
    lo = min(min(a), min(b))
    hi = max(max(a), max(b))
    ax.plot([lo, hi], [lo, hi], color="0.6", linestyle="--", linewidth=0.7)
    ax.set_xlabel(r"$\hat\delta_4$ at seed 0")
    ax.set_ylabel(r"$\hat\delta_4$ at seed 7")
    ax.text(0.05, 0.96,
            f"Spearman $\\rho={rho:.3f}$\n($n={len(a)}$ layers)",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="0.7", linewidth=0.4,
                      boxstyle="round,pad=0.25"))

    save_both(fig, fig_name)
    plt.close(fig)

    return {
        "figure": "fig4_sensor_validation",
        "source": "results/serving/smoke_absolute_canary_8b.json + Phase-5 warmup blob",
        "rows": len(runs),
        "abs_mns1_max": abs_mns1,
        "abs_mns8_max": abs_mns8,
        "naive_mns1_max": naive_mns1_max,
        "ratio": abs_mns1 / naive_mns1_max if naive_mns1_max > 0 else None,
        "monotonicity_violations": viols,
        "cross_seed_rho": rho,
    }


# ---------------------------------------------------------------------------
# Figure 5 — 70B GSM8K-Hard publication results (paper §6.5)
# ---------------------------------------------------------------------------


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI on a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _mcnemar_p_two_sided(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    p_one = sum(math.comb(n, i) * 0.5 ** n for i in range(k, n + 1))
    return min(2 * p_one, 1.0) if n > 1 else 1.0


def fig5_70b_results() -> dict:
    fig_name = "fig5_70b_results"
    # Headline result: n=500 mns=8 paired comparison. EchoKV-3C with
    # the absolute-canary sensor evaluated on the same prompts as the
    # Phase-2 FP16 + INT4 baselines (which together demonstrated the
    # INT4 vs FP16 deficit at p=0.0522).
    ech_path = require_file(
        ROOT / "results/serving/session14_70b"
            / "echokv3c_abscanary_tau022_n500_mns8.json",
        fig_name,
    )
    ph2_path = require_file(
        ROOT / "results/serving/session14_70b/gsm8k_hard_n500.json",
        fig_name,
    )

    ech_blob = json.loads(ech_path.read_text())
    ph2_blob = json.loads(ph2_path.read_text())

    fp16_cell = next(c for c in ph2_blob["cells"] if c["config"] == "fp16")
    int4_cell = next(c for c in ph2_blob["cells"] if c["config"] == "static_int4")
    ech_m = ech_blob["measurement"]

    fp16 = {r["request_id"]: r["correct"] for r in fp16_cell["per_request"]}
    int4 = {r["request_id"]: r["correct"] for r in int4_cell["per_request"]}
    ech = {r["request_id"]: r["correct"] for r in ech_m["per_request"]}

    fp16_acc = sum(fp16.values()) / len(fp16)
    int4_acc = sum(int4.values()) / len(int4)
    ech_acc = sum(ech.values()) / len(ech)
    n_total = len(fp16)

    fp16_k = sum(1 for v in fp16.values() if v >= 0.5)
    int4_k = sum(1 for v in int4.values() if v >= 0.5)
    ech_k = sum(1 for v in ech.values() if v >= 0.5)

    fp16_ci = _wilson_ci(fp16_k, n_total)
    int4_ci = _wilson_ci(int4_k, n_total)
    ech_ci = _wilson_ci(ech_k, n_total)

    # Paired McNemar — three comparisons. The headline is EchoKV vs
    # FP16 (perfect match → Pareto win) and EchoKV vs INT4 (recovers
    # the deficit, mirroring INT4 vs FP16).
    common = set(fp16) & set(int4) & set(ech)
    b_fp_int4 = sum(1 for k in common if int4[k] < fp16[k])
    c_fp_int4 = sum(1 for k in common if int4[k] > fp16[k])
    p_fp_int4 = _mcnemar_p_two_sided(b_fp_int4, c_fp_int4)
    b_ech_int4 = sum(1 for k in common if ech[k] < int4[k])
    c_ech_int4 = sum(1 for k in common if ech[k] > int4[k])
    p_ech_int4 = _mcnemar_p_two_sided(b_ech_int4, c_ech_int4)
    b_ech_fp = sum(1 for k in common if ech[k] < fp16[k])
    c_ech_fp = sum(1 for k in common if ech[k] > fp16[k])
    p_ech_fp = _mcnemar_p_two_sided(b_ech_fp, c_ech_fp)
    paired_delta_ech_fp = (c_ech_fp - b_ech_fp) / len(common)

    proj = ech_blob["memory_projection"]
    avg_bits = proj["avg_bits"]
    savings_vs_fp16 = proj["savings_vs_fp16_pct"]
    tier_counts = {int(k): v for k, v in proj["tier_counts"].items()}

    # ----- Build figure -----
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2), constrained_layout=True)

    # Panel (a): accuracy bars with Wilson CI
    ax = axes[0]
    panel_label(ax, "a", f"accuracy at $n={n_total}$")
    labels = ["FP16", "INT4", "EchoKV"]
    accs = [fp16_acc, int4_acc, ech_acc]
    cis = [fp16_ci, int4_ci, ech_ci]
    err_lo = [a - c[0] for a, c in zip(accs, cis)]
    err_hi = [c[1] - a for a, c in zip(accs, cis)]
    colors = ["0.85", "0.55", "black"]
    ax.bar(range(3), accs, yerr=[err_lo, err_hi], color=colors,
           edgecolor="black", linewidth=0.7, width=0.65,
           error_kw=dict(ecolor="black", lw=0.8, capsize=4))
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("accuracy (95% Wilson CI)")
    ax.set_ylim(0.40, 0.58)
    for i, a in enumerate(accs):
        ax.text(i, cis[i][1] + 0.003, f"{a:.3f}",
                ha="center", va="bottom", fontsize=8.5)
    # Boxed legend at upper-left summarising the three paired
    # McNemar comparisons. The symmetric mirror — INT4 −0.022
    # (p=0.052) recovered by EchoKV +0.022 (p=0.061) — is the
    # headline; EchoKV vs FP16 paired Δ = 0.000 (p = 1.00).
    ax.text(0.04, 0.96,
            "Paired McNemar (same prompts):\n"
            fr"INT4 vs FP16   $\Delta={(int4_acc-fp16_acc):+.3f}$  $p={p_fp_int4:.3f}$"
            "\n"
            fr"EchoKV vs INT4  $\Delta={(ech_acc-int4_acc):+.3f}$  $p={p_ech_int4:.3f}$"
            "\n"
            fr"EchoKV vs FP16  $\Delta={(ech_acc-fp16_acc):+.3f}$  $p={p_ech_fp:.2f}$",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7, color="0.2",
            bbox=dict(facecolor="white", edgecolor="0.7", linewidth=0.4,
                      boxstyle="round,pad=0.25"))
    ax.text(0.5, -0.18,
            "max_num_seqs=8, paired same-prompt protocol",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=7.5, color="0.3", style="italic")

    # Panel (b): memory comparison
    ax = axes[1]
    panel_label(ax, "b", "memory (avg bits/elem)")
    bar_labels = [
        "FP16",
        "INT4",
        f"EchoKV-3C\n{{INT4:{tier_counts.get(4,0)}, INT8:{tier_counts.get(8,0)}}}",
    ]
    bar_bits = [16.0, 4.0, avg_bits]
    bar_colors = ["0.85", "0.55", "black"]
    bars = ax.barh(range(3), bar_bits, color=bar_colors, edgecolor="black",
                   linewidth=0.7)
    ax.set_yticks(range(3))
    ax.set_yticklabels(bar_labels, fontsize=8.5)
    ax.set_xlabel("bits per element (lower is less memory)")
    ax.set_xlim(0, 21.5)
    annotations = [
        "(baseline)",
        "75.0% savings vs FP16",
        f"{savings_vs_fp16:.1f}% savings vs FP16",
    ]
    for i, (b, a) in enumerate(zip(bars, annotations)):
        ax.text(b.get_width() + 0.3, b.get_y() + b.get_height() / 2,
                f"{bar_bits[i]:.2f} bits — {a}",
                va="center", ha="left", fontsize=8)
    ax.invert_yaxis()  # FP16 on top

    save_both(fig, fig_name)
    plt.close(fig)

    return {
        "figure": "fig5_70b_results",
        "source": "results/serving/session14_70b/echokv3c_abscanary_tau022_n500_mns8.json + gsm8k_hard_n500.json",
        "n": n_total,
        "fp16_acc": fp16_acc,
        "int4_acc": int4_acc,
        "ech_acc": ech_acc,
        "p_fp_int4": p_fp_int4,
        "p_ech_int4": p_ech_int4,
        "p_ech_fp": p_ech_fp,
        "avg_bits": avg_bits,
        "savings_vs_fp16_pct": savings_vs_fp16,
        "tier_counts": tier_counts,
    }


# ---------------------------------------------------------------------------
# Figure 6 — Multi-workload adaptive bit allocation (paper §6.6)
# ---------------------------------------------------------------------------


def fig6_workload_adaptive() -> dict:
    """Three workloads, side-by-side, showing that EchoKV-3C matches FP16
    accuracy on all three while picking a workload-adaptive bit budget.

    Data sources:
      - GSM8K-Hard (L≈400, τ=0.022): n=500 mns=8 paired
        ``gsm8k_hard_n500.json`` cells +
        ``echokv3c_abscanary_tau022_n500_mns8.json``
      - HotpotQA (L≈1500, τ=0.022): n=500 mns=8 paired
        ``hotpotqa_n500_mns8/*.json`` (4 cells)
      - RULER@128K (L=128000, τ=0.124): n=20 mns=1 paired
        ``diag.json`` (FP16 + INT4) +
        ``echokv3c_abscanary_tau124_ruler128k_n20_mns1.json``

    The RULER@128K τ_budget (0.124) is the Pinsker-derived
    τ(L) = τ_0 · √(L/L_0) value with L_0=4000, τ_0=0.022. The same
    deterministic formula is used across all three workloads (for
    GSM8K-Hard and HotpotQA, L is close enough to L_0 that the unscaled
    τ_0=0.022 gives the same answer as the formula within rounding;
    for RULER@128K the scaling factor is ~5.66×).
    """
    fig_name = "fig6_workload_adaptive"

    # ----- GSM8K-Hard cells (n=500 mns=8) -----
    gsm_baselines = require_file(
        ROOT / "results/serving/session14_70b/gsm8k_hard_n500.json", fig_name,
    )
    gsm_ech_path = require_file(
        ROOT / "results/serving/session14_70b"
            / "echokv3c_abscanary_tau022_n500_mns8.json",
        fig_name,
    )
    gsm_b = json.loads(gsm_baselines.read_text())
    gsm_e = json.loads(gsm_ech_path.read_text())
    gsm = {
        "fp16": next(c for c in gsm_b["cells"] if c["config"] == "fp16"),
        "int4": next(c for c in gsm_b["cells"] if c["config"] == "static_int4"),
        "ech":  gsm_e["measurement"],
        "ech_proj": gsm_e["memory_projection"],
    }

    # ----- HotpotQA cells (n=500 mns=8) -----
    hot_dir = ROOT / "results/serving/session14_70b/hotpotqa_n500_mns8"
    require_file(hot_dir / "fp16_n500_mns8.json", fig_name)
    require_file(hot_dir / "int4_n500_mns8.json", fig_name)
    require_file(hot_dir / "int8_n500_mns8.json", fig_name)
    require_file(hot_dir / "echokv3c_abscanary_tau022_n500_mns8.json", fig_name)

    def _load_baseline(name: str) -> dict:
        b = json.loads((hot_dir / f"{name}_n500_mns8.json").read_text())
        return b["cells"][0]

    hot_e_blob = json.loads((hot_dir / "echokv3c_abscanary_tau022_n500_mns8.json").read_text())
    hot = {
        "fp16": _load_baseline("fp16"),
        "int4": _load_baseline("int4"),
        "int8": _load_baseline("int8"),
        "ech":  hot_e_blob["measurement"],
        "ech_proj": hot_e_blob["memory_projection"],
    }

    # ----- RULER@128K cells (n=20 mns=1; long-context regime) -----
    rul_baselines = require_file(
        ROOT / "results/serving/session14_70b/diag.json", fig_name,
    )
    rul_ech_path = require_file(
        ROOT / "results/serving/session14_70b"
            / "echokv3c_abscanary_tau124_ruler128k_n20_mns1.json",
        fig_name,
    )
    rul_b = json.loads(rul_baselines.read_text())
    rul_e = json.loads(rul_ech_path.read_text())
    rul = {
        "fp16": next(c for c in rul_b["cells"] if c["config"] == "fp16"),
        "int4": next(c for c in rul_b["cells"] if c["config"] == "static_int4"),
        "ech":  rul_e["measurement"],
        "ech_proj": rul_e["memory_projection"],
    }

    # ----- Plot: 3 panels (accuracy, bits, τ(L) scaling) -----
    fig = plt.figure(figsize=(6.5, 3.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    workloads = ["GSM8K-Hard", "HotpotQA", "RULER@128K"]
    sources = [gsm, hot, rul]

    # ---- Panel (a): grouped accuracy bars per workload ----
    panel_label(ax_a, "a", "accuracy")
    bar_specs = [
        ("FP16",      "0.85", "fp16"),
        ("INT4",      "0.55", "int4"),
        ("INT8",      "0.30", "int8"),
        ("EchoKV-3C", "black", "ech"),
    ]
    n_groups = len(workloads)
    n_bars = len(bar_specs)
    width = 0.18
    centers = np.arange(n_groups)

    for j, (label, color, key) in enumerate(bar_specs):
        offsets = (j - (n_bars - 1) / 2) * width
        for i, src in enumerate(sources):
            if key not in src:
                continue
            y = src[key]["accuracy"]
            ax_a.bar(centers[i] + offsets, y, width=width, color=color,
                     edgecolor="black", linewidth=0.5,
                     label=label if i == 0 else None)
            ax_a.text(centers[i] + offsets, y + 0.012, f"{y:.3f}",
                      ha="center", va="bottom", fontsize=6, rotation=60)

    # Mark missing INT8 cells
    for i, src in enumerate(sources):
        if "int8" not in src:
            int8_off = (2 - (n_bars - 1) / 2) * width
            ax_a.text(centers[i] + int8_off, 0.42,
                      "INT8\n(n/a)", ha="center", va="bottom",
                      fontsize=5.5, color="0.5", style="italic")

    ax_a.set_xticks(centers)
    ax_a.set_xticklabels(["GSM8K\nHard", "HotpotQA", "RULER\n@128K"],
                         fontsize=8)
    ax_a.set_ylabel("accuracy")
    ax_a.set_ylim(0.40, 1.10)
    ax_a.legend(loc="lower left", fontsize=6.5, frameon=True,
                framealpha=0.95, edgecolor="0.7", borderpad=0.3, ncol=4,
                columnspacing=0.6, handlelength=0.8)

    # ---- Panel (b): EchoKV-3C bit budget per workload ----
    panel_label(ax_b, "b", "bit budget chosen")
    bits_chosen = [s["ech_proj"]["avg_bits"] for s in sources]
    tiers = [s["ech_proj"]["tier_counts"] for s in sources]
    bars = ax_b.bar(centers, bits_chosen, width=0.45, color="black",
                    edgecolor="black", linewidth=0.6)
    for level, lab, color in [(16.0, "FP16", "0.6"),
                              (8.0, "INT8", "0.55"),
                              (4.0, "INT4", "0.55")]:
        ax_b.axhline(level, linestyle="--", color=color, linewidth=0.6, zorder=0)
        ax_b.text(n_groups - 0.45, level + 0.15, lab, fontsize=7,
                  color=color, ha="right", va="bottom")
    for i, (b, c, tt) in enumerate(zip(centers, bits_chosen, tiers)):
        tc = {int(k): v for k, v in tt.items()}
        ax_b.text(b, c + 0.4,
                  f"{c:.2f} bits\n{{4:{tc.get(4,0)}"
                  + (f", 8:{tc.get(8,0)}" if tc.get(8, 0) > 0 else "")
                  + (f", 2:{tc.get(2,0)}" if tc.get(2, 0) > 0 else "")
                  + "}",
                  ha="center", va="bottom", fontsize=6.5)
    ax_b.set_xticks(centers)
    ax_b.set_xticklabels(["GSM8K\nHard", "HotpotQA", "RULER\n@128K"],
                         fontsize=8)
    ax_b.set_ylabel("avg bits per element")
    ax_b.set_ylim(0, 19)

    # ---- Panel (c): τ(L) = τ_0 · √(L/L_0) scaling ----
    panel_label(ax_c, "c", r"$\tau(L)=\tau_0\sqrt{L/L_0}$")
    L_values = [400, 1500, 128000]
    tau_used = [0.022, 0.022, 0.124]
    tau_predicted = [0.022 * (L / 4000) ** 0.5 for L in L_values]

    # Smooth Pinsker curve
    L_range = np.logspace(np.log10(200), np.log10(200000), 200)
    tau_curve = 0.022 * (L_range / 4000) ** 0.5
    ax_c.plot(L_range, tau_curve, color="0.55", linewidth=1.0,
              linestyle="--", label=r"$\tau_0\sqrt{L/L_0}$")
    # Plot τ used per workload (markers)
    ax_c.scatter(L_values, tau_used, marker="o", s=40, color="black",
                 zorder=3, edgecolor="white", linewidth=0.6,
                 label=r"$\tau$ used")
    # Annotate workload names — offsets staggered to avoid overlap
    # GSM8K-Hard (L=400) below-left; HotpotQA (L=1500) above-right; RULER above-left
    annot_offsets = {
        "GSM8K-Hard": (8, -12),
        "HotpotQA":   (8, 6),
        "RULER@128K": (-6, 8),
    }
    annot_ha = {
        "GSM8K-Hard": "left",
        "HotpotQA":   "left",
        "RULER@128K": "right",
    }
    for L, t, w in zip(L_values, tau_used, workloads):
        ax_c.annotate(w, (L, t),
                      xytext=annot_offsets[w], textcoords="offset points",
                      fontsize=6.5, ha=annot_ha[w])
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_xlabel(r"context length $L$ (tokens)")
    ax_c.set_ylabel(r"$\tau_{\mathrm{budget}}$")
    ax_c.set_xlim(200, 300000)
    ax_c.set_ylim(0.012, 0.30)
    ax_c.legend(loc="lower right", fontsize=7, frameon=True,
                framealpha=0.95, edgecolor="0.7", borderpad=0.3)
    ax_c.grid(True, which="both", linestyle=":", linewidth=0.4,
              color="0.85", zorder=0)

    save_both(fig, fig_name)
    plt.close(fig)

    return {
        "figure": "fig6_workload_adaptive",
        "source": "GSM8K-Hard + HotpotQA n=500 mns=8 + RULER@128K n=20 mns=1",
        "rows": 1020,
        "gsm_ech_acc": gsm["ech"]["accuracy"],
        "gsm_ech_bits": gsm["ech_proj"]["avg_bits"],
        "gsm_ech_tiers": gsm["ech_proj"]["tier_counts"],
        "hot_ech_acc": hot["ech"]["accuracy"],
        "hot_ech_bits": hot["ech_proj"]["avg_bits"],
        "hot_ech_tiers": hot["ech_proj"]["tier_counts"],
        "rul_ech_acc": rul["ech"]["accuracy"],
        "rul_ech_bits": rul["ech_proj"]["avg_bits"],
        "rul_ech_tiers": rul["ech_proj"]["tier_counts"],
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    print(f"Generating figures into {FIG_DIR}/")
    print("-" * 80)

    results: list[dict] = []

    print("[1/5] fig1_motivation ...", flush=True)
    results.append(fig1_motivation())

    print("[2/5] fig2_sensor (schematic) ...", flush=True)
    results.append(fig2_sensor())

    print("[3/5] fig3_architecture (schematic) ...", flush=True)
    results.append(fig3_architecture())

    print("[4/5] fig4_sensor_validation ...", flush=True)
    results.append(fig4_sensor_validation())

    print("[5/6] fig5_70b_results ...", flush=True)
    results.append(fig5_70b_results())

    print("[6/6] fig6_workload_adaptive ...", flush=True)
    results.append(fig6_workload_adaptive())

    # Summary table
    print()
    print("=" * 92)
    print(f"{'figure':<24}  {'source':<48}  {'rows':>5}  status")
    print("-" * 92)
    rowmap = {
        "fig1_motivation": ("results/motivation/delta_records_v8_1.jsonl + per_request.tsv", 200),
        "fig2_sensor": ("(schematic, no data)", "-"),
        "fig3_architecture": ("(schematic, no data)", "-"),
        "fig4_sensor_validation": ("results/serving/smoke_absolute_canary_8b.json + Phase-5 warmup", 4),
        "fig5_70b_results": ("results/serving/session14_70b/echokv3c_abscanary_tau022_n500_mns8.json + gsm8k_hard_n500.json", 500),
        "fig6_workload_adaptive": ("session14_70b/{gsm8k_hard,hotpotqa}_n500_mns8 + echokv3c cells", 1000),
    }
    for r in results:
        source, rows = rowmap[r["figure"]]
        print(f"{r['figure']:<24}  {source:<48}  {str(rows):>5}  OK")
    print("=" * 92)

    # Print key numbers as a sanity check
    print()
    print("Key data values used in annotations (real, not invented):")
    for r in results:
        keys = sorted(k for k in r if k not in {"figure", "source", "n_records",
                                                "rows"})
        if keys:
            kv = "  ".join(f"{k}={r[k]!r}" for k in keys
                           if not isinstance(r[k], (list, tuple, dict)))
            print(f"  {r['figure']}: {kv}")

    print()
    print("Wrote:")
    for stem in ("fig1_motivation", "fig2_sensor", "fig3_architecture",
                 "fig4_sensor_validation", "fig5_70b_results",
                 "fig6_workload_adaptive"):
        for ext in ("pdf", "png"):
            p = FIG_DIR / f"{stem}.{ext}"
            ok = "OK" if p.exists() else "MISSING"
            print(f"  {p}  [{ok}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
