"""Generate paper figures: architecture diagram + cross-workload Pareto
+ sensor validation. Red/blue palette with light, paper-friendly tones.

Outputs to paper/figures/:
    architecture.pdf
    cross_workload.pdf
    sensor_validation.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


# ───────── style ─────────
PAPER_STYLE = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}

# Light red/blue palette
C_OFFLINE = "#DBEAFE"   # very light blue   (calibration / warmup)
C_OFFLINE_E = "#3B82F6" # medium blue       (offline-box edges)
C_ONLINE = "#FEE2E2"    # very light red    (online-loop boxes)
C_ONLINE_E = "#DC2626"  # medium red        (online-loop edges + feedback arrow)
C_ACCENT = "#1E40AF"    # deep blue accent
C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"

# Per-config colors for cross-workload bars
COLORS = {
    "FP16":      "#1E3A8A",  # deep navy
    "INT8":      "#3B82F6",  # medium blue
    "INT4":      "#93C5FD",  # light blue
    "NVFP4":     "#BFDBFE",  # very light blue
    "EchoKV":    "#DC2626",  # red
}

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper/figures"
OUT.mkdir(parents=True, exist_ok=True)


# ───────────────────────────────────────────────────────────────────────────
# Figure 1: architecture
# ───────────────────────────────────────────────────────────────────────────

def fig_architecture():
    """Two-phase loop: offline (calibration → warmup) flows into the
    online closed loop (canary → controller → allocator/quant → canary)."""
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 5.6)
        ax.axis("off")

        def box(x, y, w, h, label, sub, fill, edge):
            r = FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                linewidth=1.4, edgecolor=edge, facecolor=fill, zorder=2,
            )
            ax.add_patch(r)
            ax.text(x + w/2, y + h*0.62, label,
                    ha="center", va="center",
                    fontsize=10, fontweight="bold", color=C_TEXT, zorder=3)
            ax.text(x + w/2, y + h*0.22, sub,
                    ha="center", va="center",
                    fontsize=7.2, style="italic", color="#4B5563", zorder=3)

        # ─── offline row (top) ───
        BOX_W, BOX_H = 2.4, 1.0
        OFFLINE_Y = 4.0
        # Calibration
        box(0.6, OFFLINE_Y, BOX_W, BOX_H,
            "Calibration",
            r"per-layer  $\hat{\delta}_4 \to \tau_0$",
            C_OFFLINE, C_OFFLINE_E)
        # Warmup
        box(3.6, OFFLINE_Y, BOX_W, BOX_H,
            "Warmup",
            r"controller settles  $\widehat{E}_{\ell,b}$",
            C_OFFLINE, C_OFFLINE_E)
        # Arrow: calibration → warmup
        ax.add_patch(FancyArrowPatch(
            (0.6 + BOX_W, OFFLINE_Y + BOX_H/2),
            (3.6, OFFLINE_Y + BOX_H/2),
            arrowstyle="-|>", mutation_scale=12,
            linewidth=1.5, color=C_OFFLINE_E, zorder=2,
        ))
        # Arrow: warmup → online (down to first online box)
        ax.add_patch(FancyArrowPatch(
            (3.6 + BOX_W/2, OFFLINE_Y),
            (1.8 + BOX_W/2, 2.45),  # land at canary's top-center
            arrowstyle="-|>", mutation_scale=12, linewidth=1.5,
            color=C_OFFLINE_E,
            connectionstyle="arc3,rad=-0.2",
            zorder=2,
        ))

        # offline-phase label
        ax.text(0.5, OFFLINE_Y + BOX_H + 0.15,
                "OFFLINE",
                fontsize=8, fontweight="bold", color=C_OFFLINE_E)

        # ─── online row (bottom): closed loop ───
        ONLINE_Y = 1.4
        # Canary
        box(0.6, ONLINE_Y, BOX_W, BOX_H,
            "Canary",
            r"$\hat{\delta}_{\ell,b}$ vs FP16 ref",
            C_ONLINE, C_ONLINE_E)
        # Controller
        box(4.8, ONLINE_Y, BOX_W, BOX_H,
            "Controller",
            r"argmin $\widehat{E}_{\ell,b} \leq \tau$",
            C_ONLINE, C_ONLINE_E)
        # Allocator / quant
        box(9.0, ONLINE_Y, BOX_W, BOX_H,
            "Quantization",
            r"per-tier free pools",
            C_ONLINE, C_ONLINE_E)

        # forward arrows (online)
        for x_from, x_to in [(0.6 + BOX_W, 4.8), (4.8 + BOX_W, 9.0)]:
            ax.add_patch(FancyArrowPatch(
                (x_from, ONLINE_Y + BOX_H/2),
                (x_to, ONLINE_Y + BOX_H/2),
                arrowstyle="-|>", mutation_scale=12,
                linewidth=1.5, color=C_ONLINE_E, zorder=2,
            ))

        # feedback arrow: quantization → canary (curved below)
        feedback_y = ONLINE_Y - 0.6
        ax.add_patch(FancyArrowPatch(
            (9.0 + BOX_W/2, ONLINE_Y),
            (0.6 + BOX_W/2, ONLINE_Y),
            arrowstyle="-|>", mutation_scale=14,
            linewidth=1.8, color=C_ONLINE_E,
            connectionstyle=f"arc3,rad=-0.18",
            zorder=1,
        ))
        # feedback label, centered below the loop
        ax.text(6.0, 0.45,
                "feedback: K/V from new blocks → canary observes the new tier",
                ha="center", va="center",
                fontsize=8, style="italic", color=C_ONLINE_E)

        # online-phase label
        ax.text(0.5, ONLINE_Y + BOX_H + 0.15,
                "ONLINE  (per decode step)",
                fontsize=8, fontweight="bold", color=C_ONLINE_E)

        # τ_0 label on the warmup → online arrow
        ax.text(5.7, 3.55, r"$\tau_0$",
                fontsize=10, color=C_OFFLINE_E,
                fontweight="bold")

        # invariants pseudo-legend at the right edge
        # (kept small so it doesn't crowd the diagram)

        out = OUT / "architecture.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"  wrote {out.relative_to(ROOT)}")


# ───────────────────────────────────────────────────────────────────────────
# Figure 2: cross-workload (stacked vertically)
# ───────────────────────────────────────────────────────────────────────────

def _load_cell(p, key=None):
    p = Path(p)
    if not p.exists(): return None
    b = json.load(open(p))
    if "measurement" in b:
        return b["measurement"]
    if key is not None:
        for c in b["cells"]:
            if c["config"] == key: return c
    return b["cells"][0]


def fig_cross_workload():
    """Stacked-vertical 2-panel figure:
       (a) accuracy across 4 workloads × 5 configs
       (b) bits/elem across the same.
    Tier-vector composition annotated above the EchoKV bar in panel (b)."""
    base = ROOT / "results/serving/session14_70b"
    base15 = ROOT / "results/serving/session15_70b"

    # ─── pull data ───
    # GSM8K-Hard
    gsm = json.load(open(base / "gsm8k_hard_n500.json"))
    fp16_g = next(c for c in gsm["cells"] if c["config"] == "fp16")
    int4_g = next(c for c in gsm["cells"] if c["config"] == "static_int4")
    nv_g = json.load(open(base / "gsm8k_hard_n500_nvfp4_tp8.json"))["cells"][0]
    ech_g = json.load(open(base / "echokv3c_abscanary_tau022_n500_mns8.json"))
    ech_g_acc = ech_g["measurement"]["accuracy"]
    ech_g_bits = ech_g["memory_projection"]["avg_bits"]
    ech_g_tiers = ech_g["memory_projection"]["tier_counts"]

    # HotpotQA
    hot_dir = base / "hotpotqa_n500_mns8"
    fp16_h = json.load(open(hot_dir / "fp16_n500_mns8.json"))["cells"][0]
    int4_h = json.load(open(hot_dir / "int4_n500_mns8.json"))["cells"][0]
    int8_h = json.load(open(hot_dir / "int8_n500_mns8.json"))["cells"][0]
    nv_h = json.load(open(hot_dir / "nvfp4_n500_mns8_tp8.json"))["cells"][0]
    ech_h = json.load(open(hot_dir / "echokv3c_abscanary_tau022_n500_mns8.json"))
    ech_h_acc = ech_h["measurement"]["accuracy"]
    ech_h_bits = ech_h["memory_projection"]["avg_bits"]
    ech_h_tiers = ech_h["memory_projection"]["tier_counts"]

    # RULER@128K
    rul = json.load(open(base / "diag.json"))
    fp16_r = next(c for c in rul["cells"] if c["config"] == "fp16")
    int4_r = next(c for c in rul["cells"] if c["config"] == "static_int4")
    ech_r = json.load(open(base / "echokv3c_abscanary_tau124_ruler128k_n20_mns1.json"))
    ech_r_acc = ech_r["measurement"]["accuracy"]
    ech_r_bits = ech_r["memory_projection"]["avg_bits"]
    ech_r_tiers = ech_r["memory_projection"]["tier_counts"]

    # τ-Bench airline (frozen-tier protocol)
    fp16_t = json.load(open(base15 / "taubench_airline_fp16_n50.json"))["cells"][0]
    int4_t = json.load(open(base15 / "taubench_airline_int4_n50.json"))["cells"][0]
    int8_t = json.load(open(base15 / "taubench_airline_int8_n50.json"))["cells"][0]
    nv_t = json.load(open(base15 / "taubench_airline_nvfp4_n50.json"))["cells"][0]
    ech_t = json.load(open(base15 / "taubench_airline_echokv_frozen_n50.json"))
    ech_t_acc = ech_t["measurement"]["accuracy"]
    ech_t_bits = ech_t["memory_projection"]["avg_bits"]
    ech_t_tiers = ech_t["memory_projection"]["tier_counts"]

    workloads = ["GSM8K-Hard", "HotpotQA", "RULER@128K", r"$\tau$-Bench"]
    accs = {
        "FP16":   [fp16_g["accuracy"], fp16_h["accuracy"], fp16_r["accuracy"], fp16_t["accuracy"]],
        "INT4":   [int4_g["accuracy"], int4_h["accuracy"], int4_r["accuracy"], int4_t["accuracy"]],
        "INT8":   [None,                int8_h["accuracy"], None,              int8_t["accuracy"]],
        "NVFP4":  [nv_g["accuracy"],    nv_h["accuracy"],   None,              nv_t["accuracy"]],
        "EchoKV": [ech_g_acc, ech_h_acc, ech_r_acc, ech_t_acc],
    }
    bits = {
        "FP16":   [16.0, 16.0, 16.0, 16.0],
        "INT4":   [4.0, 4.0, 4.0, 4.0],
        "INT8":   [None, 8.0, None, 8.0],
        "NVFP4":  [4.0, 4.0, None, 4.0],
        "EchoKV": [ech_g_bits, ech_h_bits, ech_r_bits, ech_t_bits],
    }
    tiers = [ech_g_tiers, ech_h_tiers, ech_r_tiers, ech_t_tiers]

    with plt.rc_context(PAPER_STYLE):
        fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(6.5, 5.6),
                                          gridspec_kw={"hspace": 0.35})
        x = np.arange(len(workloads))
        n_cfg = 5
        width = 0.16
        offsets = (np.arange(n_cfg) - (n_cfg - 1) / 2) * width
        cfg_order = ["FP16", "INT4", "INT8", "NVFP4", "EchoKV"]

        # Panel (a): accuracy
        for i, cfg in enumerate(cfg_order):
            y = [v if v is not None else 0 for v in accs[cfg]]
            mask = [v is not None for v in accs[cfg]]
            bars = ax_a.bar(x + offsets[i], y, width=width,
                            color=COLORS[cfg], edgecolor="white",
                            linewidth=0.5, label=cfg, zorder=3)
            for bar, v, m in zip(bars, accs[cfg], mask):
                if not m: continue
                ax_a.text(bar.get_x() + bar.get_width()/2,
                          bar.get_height() + 0.012,
                          f"{v:.3f}",
                          ha="center", va="bottom",
                          fontsize=6.5, color=COLORS[cfg],
                          rotation=70 if cfg != "EchoKV" else 0,
                          fontweight="bold" if cfg == "EchoKV" else "normal")
            if cfg == "EchoKV":
                for bar, m in zip(bars, mask):
                    if m: bar.set_edgecolor("#7F1D1D"); bar.set_linewidth(0.8)
        # mark missing INT8/NVFP4 cells
        for i in range(len(workloads)):
            for j_miss, key_miss in [(2, "INT8"), (3, "NVFP4")]:
                if accs[key_miss][i] is None:
                    miss_off = (j_miss - (n_cfg - 1)/2) * width
                    ax_a.text(i + miss_off, 0.05,
                              f"{key_miss}\n(n/a)", ha="center", va="bottom",
                              fontsize=5.5, color="0.55", style="italic")
        ax_a.set_xticks(x); ax_a.set_xticklabels(workloads)
        ax_a.set_ylabel("Accuracy")
        ax_a.set_ylim(0, 1.18)
        ax_a.set_title("(a)  Task accuracy (paired protocol; same prompts across cells)",
                       loc="left")
        ax_a.legend(loc="upper left", ncol=5, frameon=True, fontsize=7,
                    columnspacing=0.6, handlelength=1.0, handletextpad=0.3,
                    bbox_to_anchor=(0.0, 1.02))
        ax_a.grid(axis="y", linestyle=":", color=C_GRID, zorder=1)
        ax_a.set_axisbelow(True)

        # Panel (b): bits / element
        for i, cfg in enumerate(cfg_order):
            y = [v if v is not None else 0 for v in bits[cfg]]
            mask = [v is not None for v in bits[cfg]]
            bars = ax_b.bar(x + offsets[i], y, width=width,
                            color=COLORS[cfg], edgecolor="white",
                            linewidth=0.5, label=cfg, zorder=3)
            for bar, v, m in zip(bars, bits[cfg], mask):
                if not m: continue
                txt = f"{v:.2f}" if cfg == "EchoKV" else f"{int(v)}"
                ax_b.text(bar.get_x() + bar.get_width()/2,
                          bar.get_height() + 0.4,
                          txt,
                          ha="center", va="bottom",
                          fontsize=6.5, color=COLORS[cfg],
                          fontweight="bold" if cfg == "EchoKV" else "normal")
            if cfg == "EchoKV":
                for bar, m in zip(bars, mask):
                    if m: bar.set_edgecolor("#7F1D1D"); bar.set_linewidth(0.8)
        # tier-vector annotation under each EchoKV bar
        ech_off = (4 - (n_cfg - 1)/2) * width
        for i, t in enumerate(tiers):
            tc = {int(k): v for k, v in t.items()}
            parts = []
            for k in (2, 4, 8):
                if k in tc and tc[k] > 0:
                    parts.append(f"{tc[k]}×INT{k}")
            label = ",  ".join(parts)
            ax_b.text(i + ech_off, -2.0, label,
                      ha="center", va="top",
                      fontsize=6.0, color=COLORS["EchoKV"])
        ax_b.set_xticks(x); ax_b.set_xticklabels(workloads)
        ax_b.set_ylabel("Bits per element")
        ax_b.set_ylim(-3.0, 19)
        ax_b.set_title("(b)  Cache memory cost; tier-vector composition annotated under EchoKV bars",
                       loc="left")
        ax_b.grid(axis="y", linestyle=":", color=C_GRID, zorder=1)
        ax_b.set_axisbelow(True)

        out = OUT / "cross_workload.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"  wrote {out.relative_to(ROOT)}")


# ───────────────────────────────────────────────────────────────────────────
# Figure 3: sensor validation (red-blue palette)
# ───────────────────────────────────────────────────────────────────────────

def fig_sensor_validation():
    """4-panel: signal magnitude / batch invariance / monotonicity /
    cross-seed stability. Recoloured to the red-blue palette."""
    # Pull from existing smoke + warmup data
    smoke_path = ROOT / "results/serving/smoke_absolute_canary_8b.json"
    warmup_path = ROOT / "results/serving/session14_70b/echokv_3c_n100_mp_tau002_v2.json"
    if not (smoke_path.exists() and warmup_path.exists()):
        print(f"  skipping sensor validation (missing source files)")
        return

    smoke = json.load(open(smoke_path))
    warmup = json.load(open(warmup_path))

    runs = smoke["runs"]
    runs_by = {(r["max_num_seqs"], r["seed"]): r for r in runs}
    abs_mns1 = max(runs_by[(1, 0)]["delta_4_per_layer"])
    abs_mns8_s0 = max(runs_by[(8, 0)]["delta_4_per_layer"])
    abs_mns8_s7 = max(runs_by[(8, 7)]["delta_4_per_layer"])

    # Naive (legacy production) max from warmup blob
    def naive_max(blob):
        per_rank = blob.get("warmup_per_tier_delta_per_rank", [])
        if not per_rank: return None
        n_layers = len(per_rank[0]["4"])
        layer_avg = [
            sum(per_rank[r]["4"][li] for r in range(len(per_rank))) / len(per_rank)
            for li in range(n_layers)
        ]
        return max(layer_avg)
    naive = naive_max(warmup) or 5.7e-5

    cross_seed = [
        (a, b) for a, b in zip(runs_by[(8, 0)]["delta_4_per_layer"],
                                runs_by[(8, 7)]["delta_4_per_layer"])
    ]

    with plt.rc_context(PAPER_STYLE):
        fig, axs = plt.subplots(1, 4, figsize=(7.5, 1.9))

        # (a) magnitude: log scale, two bars
        a = axs[0]
        a.bar([0, 1], [naive, abs_mns1],
              color=[COLORS["INT8"], COLORS["EchoKV"]],
              edgecolor="white", linewidth=0.5)
        a.set_yscale("log")
        a.set_xticks([0, 1]); a.set_xticklabels(["legacy", "EchoKV"])
        a.set_title("(a)  signal magnitude", loc="left", fontsize=9)
        a.set_ylabel(r"max $\hat{\delta}_4$")
        a.text(1, abs_mns1 * 1.5, f"{abs_mns1/naive:.0f}×",
               ha="center", fontsize=8, color=COLORS["EchoKV"], fontweight="bold")

        # (b) batch invariance: mns=1 vs mns=8
        b = axs[1]
        b.bar([0, 1], [abs_mns1, abs_mns8_s0],
              color=[COLORS["INT4"], COLORS["EchoKV"]],
              edgecolor="white", linewidth=0.5)
        b.set_xticks([0, 1]); b.set_xticklabels(["mns=1", "mns=8"])
        b.set_title("(b)  batch invariance", loc="left", fontsize=9)
        b.set_ylabel(r"max $\hat{\delta}_4$")
        b.text(0.5, max(abs_mns1, abs_mns8_s0)*1.05,
               f"ratio {max(abs_mns1, abs_mns8_s0)/min(abs_mns1, abs_mns8_s0):.2f}×",
               ha="center", fontsize=8, color=COLORS["EchoKV"])

        # (c) monotonicity: scatter δ̂_2 vs δ̂_4 (must be ≥)
        c = axs[2]
        d2 = runs_by[(8, 0)]["delta_2_per_layer"]
        d4 = runs_by[(8, 0)]["delta_4_per_layer"]
        d8 = runs_by[(8, 0)]["delta_8_per_layer"]
        c.scatter(d4, d2, s=12, color=COLORS["EchoKV"],
                  edgecolor="white", linewidth=0.4, alpha=0.8)
        mx = max(max(d2), max(d4)) * 1.1
        c.plot([0, mx], [0, mx], color=COLORS["FP16"], linestyle="--",
               linewidth=0.8, alpha=0.6)
        c.set_xlim(0, mx); c.set_ylim(0, mx)
        c.set_xlabel(r"$\hat{\delta}_4$ per layer")
        c.set_ylabel(r"$\hat{\delta}_2$ per layer")
        c.set_title(r"(c)  monotonicity ($\hat{\delta}_2 \geq \hat{\delta}_4$)",
                    loc="left", fontsize=9)

        # (d) cross-seed stability
        d = axs[3]
        s0 = runs_by[(8, 0)]["delta_4_per_layer"]
        s7 = runs_by[(8, 7)]["delta_4_per_layer"]
        d.scatter(s0, s7, s=12, color=COLORS["EchoKV"],
                  edgecolor="white", linewidth=0.4, alpha=0.8)
        mx = max(max(s0), max(s7)) * 1.1
        d.plot([0, mx], [0, mx], color=COLORS["FP16"], linestyle="--",
               linewidth=0.8, alpha=0.6)
        # Spearman rho
        from scipy.stats import spearmanr
        try:
            rho, _ = spearmanr(s0, s7)
            d.text(mx * 0.05, mx * 0.92, fr"$\rho={rho:.3f}$",
                   fontsize=8, color=C_TEXT)
        except Exception: pass
        d.set_xlim(0, mx); d.set_ylim(0, mx)
        d.set_xlabel(r"$\hat{\delta}_4$ at seed 0")
        d.set_ylabel(r"$\hat{\delta}_4$ at seed 7")
        d.set_title("(d)  cross-seed stability", loc="left", fontsize=9)

        for ax in axs:
            ax.grid(True, linestyle=":", color=C_GRID, alpha=0.6)
            ax.set_axisbelow(True)

        fig.tight_layout(w_pad=1.0)
        out = OUT / "sensor_validation.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"  wrote {out.relative_to(ROOT)}")


# ───────────────────────────────────────────────────────────────────────────

def main():
    print(f"Generating paper figures into {OUT.relative_to(ROOT)}/")
    fig_architecture()
    fig_cross_workload()
    fig_sensor_validation()


if __name__ == "__main__":
    main()
