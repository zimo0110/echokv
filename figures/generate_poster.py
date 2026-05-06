"""Poster figures for EchoKV — colorful, large-typeface, one PNG per file.

Outputs to figures/poster/:
    01_architecture.png       — vertical-stack control-flow diagram
    02_throughput.png         — concurrent requests on 8×H100 + TP=8
    03_benchmarks.png         — accuracy + memory side-by-side
    04_layer_history.png      — GSM8K-Hard tier evolution warmup → final
    05_delta_distributions.png — per-layer δ̂_4 distributions per workload

Run:
    /opt/conda/envs/scalable/bin/python figures/generate_poster.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np


# --------------------------------------------------------------------------
# Style — poster-readable typography, generous spacing
# --------------------------------------------------------------------------

POSTER_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 15,
    "axes.labelweight": "semibold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.4,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "legend.fontsize": 12,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#CBD5E1",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
}

# Color palette — red + blue poster theme
# Baselines (FP16/INT8/INT4) span blue from dark to light;
# EchoKV-3C is red so it pops as the "this paper" line.
COLORS = {
    "FP16":      "#1E3A8A",   # navy blue — coldest baseline
    "INT8":      "#3B82F6",   # medium blue
    "INT4":      "#93C5FD",   # light blue
    "EchoKV-3C": "#DC2626",   # red — highlight
}
ECHOKV_EDGE = "#7F1D1D"        # dark red, used to outline EchoKV bars
DARK_TEXT = "#1F2937"
GRID_GRAY = "#E5E7EB"


# --------------------------------------------------------------------------
# Data sources
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
SESSION = ROOT / "results/serving/session14_70b"

PATHS = {
    "gsm_dyn":  SESSION / "echokv3c_abscanary_tau022_n500_mns8_continuous_dynamic.json",
    "gsm_frz":  SESSION / "echokv3c_abscanary_tau022_n500_mns8.json",
    "gsm_base": SESSION / "gsm8k_hard_n500.json",
    "hot_dyn":  SESSION / "hotpotqa_n500_mns8/echokv3c_abscanary_tau022_n500_mns8_continuous_dynamic.json",
    "hot_frz":  SESSION / "hotpotqa_n500_mns8/echokv3c_abscanary_tau022_n500_mns8.json",
    "hot_fp16": SESSION / "hotpotqa_n500_mns8/fp16_n500_mns8.json",
    "hot_int4": SESSION / "hotpotqa_n500_mns8/int4_n500_mns8.json",
    "hot_int8": SESSION / "hotpotqa_n500_mns8/int8_n500_mns8.json",
    "rul_dyn":  SESSION / "echokv3c_abscanary_tau124_ruler128k_n20_mns1_continuous_dynamic.json",
    "rul_frz":  SESSION / "echokv3c_abscanary_tau124_ruler128k_n20_mns1.json",
    "rul_diag": SESSION / "diag.json",
}


def load(p):
    return json.loads(Path(p).read_text())


def per_rank_avg_bits(blob):
    """Average per-rank bits/element from final_tiers_per_rank."""
    pr = blob.get("final_tiers_per_rank")
    if not pr:
        return blob["memory_projection"]["avg_bits"]
    return sum(sum(r) / len(r) for r in pr) / len(pr)


def consol_max(per_rank):
    L = len(per_rank[0])
    return [max(per_rank[r][li] for r in range(len(per_rank))) for li in range(L)]


# --------------------------------------------------------------------------
# Figure 1 — architecture (vertical stack)
# --------------------------------------------------------------------------

def fig1_architecture():
    """Vertical-stack architecture diagram. Each box: number+title (top line),
    subtitle in italics on the second line, detail text below. Forward arrows
    between consecutive boxes; a curved red feedback arrow runs from box 5
    on the right side back up to box 2."""
    fig, ax = plt.subplots(figsize=(11, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14.5)
    ax.axis("off")

    # Box geometry
    box_w = 8.4
    box_h = 1.85
    box_x = 1.5                  # left edge — leaves room on the right for the feedback arrow
    spacing = 0.85

    boxes = [
        {
            "n": "1",
            "title": "Calibration & Warmup",
            "subtitle": "offline · choose τ-budget",
            "detail": "τ₀ = 0.022 calibrated on 8B + RULER@4K\nτ(L) = τ₀ · max(1, √(L/L₀))  for serving context L",
            "color": "#FEF3C7",
            "edge": "#D97706",
        },
        {
            "n": "2",
            "title": "Decode + Canary Fires",
            "subtitle": "per request · p_c = 1/64 decode steps",
            "detail": "FlashInfer Triton kernel computes  KL(p_FP16, q_quant)\non a separate CUDA stream — overlaps with attention",
            "color": "#DBEAFE",
            "edge": "#2563EB",
        },
        {
            "n": "3",
            "title": "Per-Tier δ̂ EMAs",
            "subtitle": "per layer · per bit-width",
            "detail": "δ̂_b(ℓ) ← α · δ_new + (1−α) · δ̂_b(ℓ)\nabsolute K_ref buffer breaks cache-state feedback",
            "color": "#CFFAFE",
            "edge": "#0891B2",
        },
        {
            "n": "4",
            "title": "Controller Decision",
            "subtitle": "argmin-with-budget · every 10 decode steps",
            "detail": "b_ℓ ← min { b : δ̂_b(ℓ) ≤ τ }\n→ workload-conditional per-layer tier vector",
            "color": "#D1FAE5",
            "edge": "#059669",
        },
        {
            "n": "5",
            "title": "Apply Quantization",
            "subtitle": "new block allocations only · INV-2",
            "detail": "vLLM block-allocator reads per-layer b_ℓ\nexisting blocks are never re-quantized",
            "color": "#FCE7F3",
            "edge": "#BE185D",
        },
    ]

    n = len(boxes)
    total_h = n * box_h + (n - 1) * spacing
    y_top = 13.2

    for i, b in enumerate(boxes):
        y = y_top - box_h - i * (box_h + spacing)

        # Outer rounded rectangle
        rect = FancyBboxPatch(
            (box_x, y), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.20",
            linewidth=2.6,
            edgecolor=b["edge"],
            facecolor=b["color"],
            zorder=2,
        )
        ax.add_patch(rect)

        # Step number — large circular badge on the left
        badge_cx = box_x + 0.55
        badge_cy = y + box_h - 0.55
        circle = mpatches.Circle(
            (badge_cx, badge_cy), 0.32,
            facecolor=b["edge"], edgecolor="white", linewidth=2.0, zorder=3,
        )
        ax.add_patch(circle)
        ax.text(badge_cx, badge_cy, b["n"],
                ha="center", va="center",
                fontsize=15, fontweight="bold", color="white",
                zorder=4)

        # Title (top line, left aligned, after the badge)
        ax.text(
            box_x + 1.10, y + box_h - 0.44,
            b["title"],
            fontsize=17, fontweight="bold",
            color=DARK_TEXT,
            va="center",
            zorder=3,
        )
        # Subtitle (just below the title, italic, smaller)
        ax.text(
            box_x + 1.10, y + box_h - 0.95,
            b["subtitle"],
            fontsize=12, style="italic",
            color="#4B5563",
            va="center",
            zorder=3,
        )
        # Detail block at the bottom of the box
        ax.text(
            box_x + 0.30, y + 0.30,
            b["detail"],
            fontsize=11,
            color="#374151",
            va="bottom",
            zorder=3,
        )

        # Forward arrow to next box
        if i < n - 1:
            arrow_y_top = y
            arrow_y_bot = y - spacing + 0.05
            ax.add_patch(FancyArrowPatch(
                (box_x + box_w / 2, arrow_y_top),
                (box_x + box_w / 2, arrow_y_bot),
                arrowstyle="-|>",
                mutation_scale=24,
                linewidth=2.8,
                color="#1F2937",
                zorder=4,
            ))

    # Closed-loop feedback arrow: from box 5 right-edge curving up to box 2 right-edge
    y_step5 = y_top - 5 * box_h - 4 * spacing + box_h / 2
    y_step2 = y_top - 2 * box_h - 1 * spacing + box_h / 2
    feedback_x = box_x + box_w + 0.05
    feedback_xr = feedback_x + 1.6
    # Up-going curve on the right side
    ax.add_patch(FancyArrowPatch(
        (feedback_x, y_step5),
        (feedback_x, y_step2),
        arrowstyle="-|>",
        mutation_scale=24,
        linewidth=2.6,
        color="#DC2626",
        connectionstyle="arc3,rad=0.40",
        zorder=4,
    ))
    # Label, placed clear of the curve
    ax.text(
        feedback_xr, (y_step5 + y_step2) / 2,
        "closed-loop\nfeedback",
        fontsize=12, fontweight="bold", color="#DC2626",
        ha="center", va="center",
        zorder=5,
    )

    # Figure title
    ax.text(
        6.0, 14.05,
        "EchoKV Closed-Loop Architecture",
        fontsize=24, fontweight="bold",
        color=DARK_TEXT,
        ha="center", va="center",
    )
    ax.text(
        6.0, 13.65,
        "Per-layer KV-cache bit-widths chosen at runtime — no offline tuning per workload",
        fontsize=13, style="italic",
        color="#6B7280",
        ha="center", va="center",
    )

    out = ROOT / "figures/poster/01_architecture.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.name}")


# --------------------------------------------------------------------------
# Figure 2 — throughput (concurrent requests on 8×H100, TP=8)
# --------------------------------------------------------------------------

def fig2_throughput():
    """Memory-bounded concurrent request capacity, 8×H100 80GB + TP=8.

    Llama-3.1-70B BF16 weights ≈ 140 GB → ~17.5 GB/GPU at TP=8.
    Allow ~5 GB activation buffer per GPU. Available KV per GPU ≈ 57.5 GB →
    cluster-aggregate KV ≈ 460 GB.

    Per-token KV bytes (70B uses GQA: 8 KV heads × 128 dim × 80 layers × 2 K/V):
        BF16: 80*2*8*128*2 = 327,680 B/token = 320 KB
        INT8: 160 KB
        INT4: 80 KB
        EchoKV-3C @ ~4.4 bits: 88 KB

    At 8K context per request:
        FP16:   2.5 GB/req → ~184 concurrent
        INT8:   1.25       → ~368
        INT4:   0.625      → ~736
        EchoKV: 0.6875     → ~669
    """
    bytes_per_token = {
        "FP16/BF16":     320 * 1024,
        "static INT8":   160 * 1024,
        "static INT4":    80 * 1024,
        "EchoKV":        (4.4 / 16) * 320 * 1024,  # = 88 KB
    }
    cluster_kv_gb = 460
    context = 8192
    bytes_per_req = {k: v * context for k, v in bytes_per_token.items()}
    reqs = {k: int(cluster_kv_gb * 1024**3 / v) for k, v in bytes_per_req.items()}

    labels = list(reqs.keys())
    values = [reqs[k] for k in labels]
    colors = [COLORS["FP16"], COLORS["INT8"], COLORS["INT4"], COLORS["EchoKV-3C"]]

    with plt.rc_context(POSTER_STYLE):
        fig, ax = plt.subplots(figsize=(11, 6.5))
        bars = ax.bar(
            labels, values,
            color=colors,
            edgecolor="white",
            linewidth=2,
            zorder=3,
        )

        # Annotations on bars
        ymax = max(values) * 1.15
        for bar, val, lab in zip(bars, values, labels):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.012,
                f"{val:,}",
                ha="center", va="bottom",
                fontsize=18, fontweight="bold",
                color=DARK_TEXT,
            )
            # Speedup vs FP16 inside the bar
            if lab != "FP16/BF16":
                speedup = val / values[0]
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2,
                    f"{speedup:.1f}× FP16",
                    ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white",
                )

        ax.set_ylim(0, ymax)
        ax.set_ylabel("Concurrent prompts (KV-memory bound)")
        ax.set_title("Cluster capacity at 8K context — 8×H100 + Llama-3.1-70B + TP=8")

        # Subtitle / footer
        fig.text(
            0.5, -0.02,
            "Cluster-aggregate KV budget ≈ 460 GB after 70B BF16 weights & activations.   "
            "EchoKV bit/elem from GSM8K-Hard dynamic (≈4.4 bits/rank).",
            ha="center", fontsize=11, style="italic", color="#4B5563",
        )

        # Highlight EchoKV bar with a dark-red outline
        bars[-1].set_edgecolor(ECHOKV_EDGE)
        bars[-1].set_linewidth(3.5)

        ax.grid(axis="y", linestyle="--", color=GRID_GRAY, zorder=1)
        ax.set_axisbelow(True)

        out = ROOT / "figures/poster/02_throughput.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"  wrote {out.name}")


# --------------------------------------------------------------------------
# Figure 3 — accuracy + memory (two panels side-by-side)
# --------------------------------------------------------------------------

def fig3_benchmarks():
    # Workload data — three configs only (FP16 / INT4 / EchoKV-3C); INT8 omitted.
    gsm = load(PATHS["gsm_base"])
    fp16_g = next(c for c in gsm["cells"] if c["config"] == "fp16")["accuracy"]
    int4_g = next(c for c in gsm["cells"] if c["config"] == "static_int4")["accuracy"]
    ech_g = load(PATHS["gsm_dyn"])
    ech_g_acc = ech_g["measurement"]["accuracy"]
    ech_g_bits = per_rank_avg_bits(ech_g)

    fp16_h = load(PATHS["hot_fp16"])["cells"][0]["accuracy"]
    int4_h = load(PATHS["hot_int4"])["cells"][0]["accuracy"]
    ech_h = load(PATHS["hot_dyn"])
    ech_h_acc = ech_h["measurement"]["accuracy"]
    ech_h_bits = per_rank_avg_bits(ech_h)

    rul = load(PATHS["rul_diag"])
    fp16_r = next(c for c in rul["cells"] if c["config"] == "fp16")["accuracy"]
    int4_r = next(c for c in rul["cells"] if c["config"] == "static_int4")["accuracy"]
    ech_r = load(PATHS["rul_dyn"])
    ech_r_acc = ech_r["measurement"]["accuracy"]
    ech_r_bits = per_rank_avg_bits(ech_r)

    workloads = ["GSM8K-Hard", "HotpotQA", "RULER@128K"]
    accs = {
        "FP16":      [fp16_g, fp16_h, fp16_r],
        "INT4":      [int4_g, int4_h, int4_r],
        "EchoKV-3C": [ech_g_acc, ech_h_acc, ech_r_acc],
    }
    bits_data = {
        "FP16":      [16, 16, 16],
        "INT4":      [4, 4, 4],
        "EchoKV-3C": [ech_g_bits, ech_h_bits, ech_r_bits],
    }

    config_colors = {
        "FP16":      COLORS["FP16"],
        "INT4":      COLORS["INT4"],
        "EchoKV-3C": COLORS["EchoKV-3C"],
    }

    with plt.rc_context(POSTER_STYLE):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(17, 7))

        x = np.arange(len(workloads)) * 1.8   # workload-group spacing
        n_cfg = 3
        width = 0.36
        # Multiply by 1.35 so adjacent bars in a group leave a visible gap
        # (bar-edge spacing ≈ 0.13 units) — keeps centered labels breathable.
        offsets = np.linspace(-1, 1, n_cfg) * width * 1.35

        # --- Panel A: accuracy ---
        # Every label sits directly on top of its bar at a uniform offset.
        # Horizontal separation between bars is enough to keep labels from
        # colliding even when accuracies are nearly identical (HotpotQA).
        cfg_order = list(accs.keys())  # FP16, INT4, EchoKV-3C
        A_OFFSET = 0.020
        for i, cfg in enumerate(cfg_order):
            y = [v if v is not None else 0 for v in accs[cfg]]
            mask = [v is not None for v in accs[cfg]]
            bars = ax_a.bar(
                x + offsets[i], y, width,
                color=config_colors[cfg],
                edgecolor="white",
                linewidth=1.4,
                label=cfg,
                zorder=3,
            )
            for bar, v, present in zip(bars, accs[cfg], mask):
                if not present:
                    continue
                ax_a.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + A_OFFSET,
                    f"{v:.3f}",
                    ha="center", va="bottom",
                    fontsize=11, fontweight="bold",
                    color=config_colors[cfg],
                )
            if cfg == "EchoKV-3C":
                for bar in bars:
                    bar.set_edgecolor(ECHOKV_EDGE)
                    bar.set_linewidth(2.4)

        ax_a.set_xticks(x)
        ax_a.set_xticklabels(workloads, fontsize=14, fontweight="semibold")
        ax_a.set_ylabel("Accuracy")
        ax_a.set_ylim(0, 1.25)
        ax_a.set_title("(a)  Accuracy by workload", loc="left")
        ax_a.legend(loc="upper left", ncol=3, fontsize=11)
        ax_a.grid(axis="y", linestyle="--", color=GRID_GRAY, zorder=1)
        ax_a.set_axisbelow(True)

        # --- Panel B: bits/element ---
        # Same uniform-offset rule. Bars are horizontally separated, so even
        # when EchoKV ≈ INT4 the labels don't overlap.
        B_OFFSET = 0.30
        for i, cfg in enumerate(cfg_order):
            y = [v if v is not None else 0 for v in bits_data[cfg]]
            mask = [v is not None for v in bits_data[cfg]]
            bars = ax_b.bar(
                x + offsets[i], y, width,
                color=config_colors[cfg],
                edgecolor="white",
                linewidth=1.4,
                label=cfg,
                zorder=3,
            )
            for bar, v, present in zip(bars, bits_data[cfg], mask):
                if not present:
                    continue
                txt = f"{v:.2f}" if cfg == "EchoKV-3C" else f"{int(v)}"
                ax_b.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + B_OFFSET,
                    txt,
                    ha="center", va="bottom",
                    fontsize=11, fontweight="bold",
                    color=config_colors[cfg],
                )
            if cfg == "EchoKV-3C":
                for bar in bars:
                    bar.set_edgecolor(ECHOKV_EDGE)
                    bar.set_linewidth(2.4)

        ax_b.set_xticks(x)
        ax_b.set_xticklabels(workloads, fontsize=14, fontweight="semibold")
        ax_b.set_ylabel("Bits per element  (per-TP-rank avg)")
        ax_b.set_ylim(0, 19)
        ax_b.set_title("(b)  Memory cost by workload", loc="left")
        ax_b.legend(loc="upper right", ncol=3, fontsize=11)
        ax_b.grid(axis="y", linestyle="--", color=GRID_GRAY, zorder=1)
        ax_b.set_axisbelow(True)

        # Overall title
        fig.suptitle(
            "Three-workload Pareto picture — 70B + TP=4, continuously-dynamic protocol",
            fontsize=18, fontweight="bold", color=DARK_TEXT, y=1.02,
        )

        out = ROOT / "figures/poster/03_benchmarks.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"  wrote {out.name}")


# --------------------------------------------------------------------------
# Figure 4 — per-layer tier history (GSM8K-Hard dynamic warmup → final)
# --------------------------------------------------------------------------

def fig4_layer_history():
    blob = load(PATHS["gsm_dyn"])
    seeded = blob["consolidated_tiers"]            # per-layer warmup max-across-ranks
    final_pr = blob["final_tiers_per_rank"]
    final_consol = consol_max(final_pr)
    L = len(seeded)

    with plt.rc_context(POSTER_STYLE):
        fig, ax = plt.subplots(figsize=(17, 7.5))

        # Draw a vertical "stem" from warmup to final per layer; markers at each end.
        for li in range(L):
            w = seeded[li]
            f = final_consol[li]
            x_pos = li
            if w == f:
                # Unchanged — single faded marker
                ax.plot(x_pos, w, marker="o", markersize=9,
                        markerfacecolor=TIER_COLORS[w],
                        markeredgecolor="white", markeredgewidth=1.4,
                        alpha=0.55, zorder=3)
            else:
                color = "#DC2626" if f > w else "#059669"  # red up / green down
                ax.plot(
                    [x_pos, x_pos], [w, f],
                    color=color, linewidth=2.5, alpha=0.85, zorder=2,
                )
                # Warmup marker (open circle)
                ax.plot(x_pos, w, marker="o", markersize=10,
                        markerfacecolor="white",
                        markeredgecolor=TIER_COLORS[w], markeredgewidth=2.0,
                        zorder=3)
                # Final marker (filled square)
                ax.plot(x_pos, f, marker="s", markersize=11,
                        markerfacecolor=TIER_COLORS[f],
                        markeredgecolor="white", markeredgewidth=1.4,
                        zorder=4)

        # Tier reference lines
        for tier in (2, 4, 8):
            ax.axhline(tier, color=TIER_COLORS[tier], linestyle=":", alpha=0.25, zorder=1)

        # Extend y-axis upward so the legend sits above the data, not over it
        ax.set_xlim(-1, L)
        ax.set_ylim(1, 11.5)
        ax.set_yticks([2, 4, 8])
        ax.set_yticklabels(["INT2 (2 bits)", "INT4 (4 bits)", "INT8 (8 bits)"])
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Bit-width  (consolidated max-across-ranks)")
        ax.set_title(
            "Per-layer tier evolution — GSM8K-Hard, continuously-dynamic protocol",
            loc="left", pad=18,
        )

        # Custom legend — placed in the top reserved band, above all data
        legend_elems = [
            Line2D([0], [0], marker="o", linestyle="", markersize=11,
                   markerfacecolor="white", markeredgecolor="#374151", markeredgewidth=2.0,
                   label="warmup tier"),
            Line2D([0], [0], marker="s", linestyle="", markersize=11,
                   markerfacecolor="#374151", markeredgecolor="white", markeredgewidth=1.4,
                   label="final tier"),
            Line2D([0], [0], color="#059669", linewidth=2.5, label="downgraded (↓ memory)"),
            Line2D([0], [0], color="#DC2626", linewidth=2.5, label="upgraded (↑ memory)"),
            Line2D([0], [0], marker="o", linestyle="", markersize=9,
                   markerfacecolor=TIER_COLORS[4], markeredgecolor="white",
                   markeredgewidth=1.4, alpha=0.55, label="unchanged"),
        ]
        ax.legend(
            handles=legend_elems,
            loc="upper center", bbox_to_anchor=(0.5, 1.02),
            ncol=5, fontsize=11, frameon=True,
        )

        # Summary stats annotation — top-right corner inside the headroom band
        n_changes = sum(1 for li in range(L) if final_consol[li] != seeded[li])
        n_down    = sum(1 for li in range(L) if final_consol[li] < seeded[li])
        n_up      = sum(1 for li in range(L) if final_consol[li] > seeded[li])
        seed_bits = sum(seeded) / L
        final_bits = sum(final_consol) / L
        ax.text(
            L - 1, 10.4,
            f"warmup avg = {seed_bits:.2f}  →  final avg = {final_bits:.2f} bits/elem\n"
            f"{n_changes} of {L} layers changed  ({n_down} ↓,  {n_up} ↑)",
            fontsize=12, fontweight="bold", color=DARK_TEXT,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.40", facecolor="#F9FAFB",
                      edgecolor="#CBD5E1", linewidth=1.0),
        )

        ax.grid(axis="x", linestyle="--", color=GRID_GRAY, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

        out = ROOT / "figures/poster/04_layer_history.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"  wrote {out.name}")


# --------------------------------------------------------------------------
# Figure 5 — per-layer δ̂_4 distributions per workload
# --------------------------------------------------------------------------

def fig5_delta_distributions():
    """For each workload's EchoKV-3C run, average δ̂_4 across ranks per layer
    (80 values per workload), plot the distribution + τ-budget threshold."""

    workloads = [
        ("GSM8K-Hard",   PATHS["gsm_dyn"], 0.022),
        ("HotpotQA",     PATHS["hot_dyn"], 0.022),
        ("RULER@128K",   PATHS["rul_dyn"], 0.124),
    ]
    data = {}
    for name, p, tau in workloads:
        blob = load(p)
        delta_pr = blob["warmup_per_tier_delta_per_rank"]   # list of 4 dicts {2:[80], 4:[80], 8:[80]}
        # Average across ranks per layer for δ̂_4
        per_layer = []
        for li in range(80):
            vals = [delta_pr[r]["4"][li] for r in range(len(delta_pr))]
            per_layer.append(np.mean(vals))
        data[name] = (np.array(per_layer), tau)

    with plt.rc_context(POSTER_STYLE):
        fig, ax = plt.subplots(figsize=(13, 7.5))

        # Wider lane spacing (1.6 per workload) so τ labels stay inside
        # their workload's lane and don't bleed into the next violin.
        lane_step = 1.6
        positions = np.arange(len(workloads)) * lane_step
        # Replace zeros with floor so log-scale doesn't blow up
        eps = 1e-7

        for i, (name, _, _) in enumerate(workloads):
            arr, tau = data[name]
            arr_clip = np.maximum(arr, eps)

            # Strip + violin combo
            parts = ax.violinplot(
                [arr_clip],
                positions=[positions[i]],
                widths=0.85,
                showmedians=False,
                showextrema=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(WORKLOAD_COLORS[name])
                pc.set_edgecolor(WORKLOAD_COLORS[name])
                pc.set_alpha(0.30)

            # Jittered scatter
            jitter = (np.random.RandomState(0 + i).rand(len(arr_clip)) - 0.5) * 0.24
            ax.scatter(
                positions[i] + jitter, arr_clip,
                s=24, alpha=0.55,
                color=WORKLOAD_COLORS[name],
                edgecolor="white", linewidth=0.5,
                zorder=3,
            )

            # Median line
            med = np.median(arr_clip)
            ax.plot(
                [positions[i] - 0.42, positions[i] + 0.42], [med, med],
                color=WORKLOAD_COLORS[name],
                linewidth=3, zorder=4, solid_capstyle="round",
            )

            # τ_budget threshold marker — horizontal segment over the violin lane,
            # label centered above the line so it stays inside the lane.
            ax.plot(
                [positions[i] - 0.55, positions[i] + 0.55], [tau, tau],
                color="#1F2937", linewidth=2.6, linestyle="--", zorder=5,
            )
            # Label slightly above the dashed line, log space
            ax.text(
                positions[i], tau * 1.45,
                f"τ = {tau:.3f}",
                fontsize=12, fontweight="bold",
                color=DARK_TEXT,
                va="bottom", ha="center",
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="#CBD5E1", linewidth=0.8),
            )

        ax.set_yscale("log")
        ax.set_xticks(positions)
        ax.set_xticklabels([w[0] for w in workloads], fontsize=14, fontweight="semibold")
        ax.set_ylabel(r"$\hat{\delta}_4$  (per-layer warmup-end EMA, log scale)")
        ax.set_xlim(-lane_step * 0.7, (len(workloads) - 1) * lane_step + lane_step * 0.7)
        ax.set_ylim(1e-7, 1.0)
        ax.set_title(
            "Per-layer δ̂_4 distribution by workload\n"
            "(layers below the dashed τ admit INT4; above τ get upgraded to INT8)",
            loc="left",
        )

        # Legend
        legend_elems = [
            Line2D([0], [0], marker="o", linestyle="", markersize=10,
                   markerfacecolor=WORKLOAD_COLORS[w], markeredgecolor="white",
                   markeredgewidth=0.6, alpha=0.75, label=w)
            for w in [n for n, _, _ in workloads]
        ]
        legend_elems.append(Line2D(
            [0], [0], color="#1F2937", linewidth=2.3, linestyle="--",
            label="τ-budget threshold (per workload)"
        ))
        ax.legend(handles=legend_elems, loc="lower left", fontsize=11)

        ax.grid(axis="y", which="both", linestyle="--", color=GRID_GRAY, alpha=0.6, zorder=0)
        ax.set_axisbelow(True)

        out = ROOT / "figures/poster/05_delta_distributions.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"  wrote {out.name}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    out_dir = ROOT / "figures/poster"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating poster figures into {out_dir.relative_to(ROOT)}/")
    print("-" * 60)
    plt.rcParams.update(POSTER_STYLE)
    # Only Figs 2 and 3 per the current poster spec; the other functions
    # (fig1_architecture, fig4_layer_history, fig5_delta_distributions)
    # remain defined for future re-runs but are not invoked here.
    fig2_throughput()
    fig3_benchmarks()
    print("-" * 60)
    for p in sorted(out_dir.glob("*.png")):
        print(f"  {p.relative_to(ROOT)}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
