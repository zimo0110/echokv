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
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 5.6)
        ax.axis("off")

        def box(x, y, w, h, label, sub, fill, edge):
            r = FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                linewidth=1.4, edgecolor=edge, facecolor=fill, zorder=2,
            )
            ax.add_patch(r)
            ax.text(x + w/2, y + h*0.66, label,
                    ha="center", va="center",
                    fontsize=10, fontweight="bold", color=C_TEXT, zorder=3)
            ax.text(x + w/2, y + h*0.28, sub,
                    ha="center", va="center",
                    fontsize=7.2, style="italic", color="#4B5563", zorder=3)

        # ─── offline row (top) ───
        BOX_W, BOX_H = 3.4, 1.05
        OFFLINE_Y = 4.0
        # Calibration
        box(0.6, OFFLINE_Y, BOX_W, BOX_H,
            "Calibration",
            r"$\tau_0 = \mathrm{mean}(\widehat{E}_{\ell,4})$",
            C_OFFLINE, C_OFFLINE_E)
        # Warmup
        box(4.6, OFFLINE_Y, BOX_W, BOX_H,
            "Warmup",
            r"EMAs settle",
            C_OFFLINE, C_OFFLINE_E)
        # Arrow: calibration → warmup
        ax.add_patch(FancyArrowPatch(
            (0.6 + BOX_W, OFFLINE_Y + BOX_H/2),
            (4.6, OFFLINE_Y + BOX_H/2),
            arrowstyle="-|>", mutation_scale=12,
            linewidth=1.5, color=C_OFFLINE_E, zorder=2,
        ))
        # Arrow: warmup → online (down to first online box)
        ax.add_patch(FancyArrowPatch(
            (4.6 + BOX_W/2, OFFLINE_Y),
            (0.6 + BOX_W/2, 2.45),
            arrowstyle="-|>", mutation_scale=12, linewidth=1.5,
            color=C_OFFLINE_E,
            connectionstyle="arc3,rad=-0.22",
            zorder=2,
        ))

        # offline-phase label
        ax.text(0.5, OFFLINE_Y + BOX_H + 0.18,
                "OFFLINE",
                fontsize=8.5, fontweight="bold", color=C_OFFLINE_E)

        # ─── online row (bottom): closed loop ───
        ONLINE_Y = 1.4
        # Canary
        box(0.6, ONLINE_Y, BOX_W, BOX_H,
            "Canary",
            r"$\hat{\delta}_{\ell,b}$ vs FP16",
            C_ONLINE, C_ONLINE_E)
        # Controller
        box(5.3, ONLINE_Y, BOX_W, BOX_H,
            "Controller",
            r"EMA + threshold",
            C_ONLINE, C_ONLINE_E)
        # Allocator / quant
        box(10.0, ONLINE_Y, BOX_W, BOX_H,
            "Quantization",
            r"per-tier free pools",
            C_ONLINE, C_ONLINE_E)

        # forward arrows (online)
        for x_from, x_to in [(0.6 + BOX_W, 5.3), (5.3 + BOX_W, 10.0)]:
            ax.add_patch(FancyArrowPatch(
                (x_from, ONLINE_Y + BOX_H/2),
                (x_to, ONLINE_Y + BOX_H/2),
                arrowstyle="-|>", mutation_scale=12,
                linewidth=1.5, color=C_ONLINE_E, zorder=2,
            ))

        # feedback arrow: quantization → canary (curved below)
        ax.add_patch(FancyArrowPatch(
            (10.0 + BOX_W/2, ONLINE_Y),
            (0.6 + BOX_W/2, ONLINE_Y),
            arrowstyle="-|>", mutation_scale=14,
            linewidth=1.8, color=C_ONLINE_E,
            connectionstyle=f"arc3,rad=-0.16",
            zorder=1,
        ))
        # feedback label
        ax.text(6.7, 0.42,
                "feedback: new K/V blocks observed by canary",
                ha="center", va="center",
                fontsize=8, style="italic", color=C_ONLINE_E)

        # online-phase label
        ax.text(0.5, ONLINE_Y + BOX_H + 0.18,
                "ONLINE  (per decode step)",
                fontsize=8.5, fontweight="bold", color=C_ONLINE_E)

        # τ_0 label on the warmup → online arrow
        ax.text(3.0, 3.55, r"$\tau_0$",
                fontsize=10, color=C_OFFLINE_E,
                fontweight="bold")

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
    EchoKV runs in continuous-dynamic mode (controller live during serving).
    Bits are per-rank averages; tier-vector composition (consolidated
    max-across-ranks) annotated under EchoKV bars in panel (b)."""
    base = ROOT / "results/serving/session14_70b"
    base15 = ROOT / "results/serving/session15_70b"
    from collections import Counter

    def per_rank_bits(blob):
        pr = blob.get("final_tiers_per_rank")
        if not pr: return blob["memory_projection"]["avg_bits"]
        return sum(sum(r)/len(r) for r in pr) / len(pr)

    def consol(blob):
        pr = blob.get("final_tiers_per_rank")
        if not pr: return blob["memory_projection"]["tier_counts"]
        L = len(pr[0])
        cm = [max(pr[r][li] for r in range(len(pr))) for li in range(L)]
        return dict(Counter(cm))

    # ─── pull data ───
    # GSM8K-Hard
    gsm = json.load(open(base / "gsm8k_hard_n500.json"))
    fp16_g = next(c for c in gsm["cells"] if c["config"] == "fp16")
    int4_g = next(c for c in gsm["cells"] if c["config"] == "static_int4")
    nv_g = json.load(open(base / "gsm8k_hard_n500_nvfp4_tp8.json"))["cells"][0]
    ech_g_blob = json.load(open(base / "echokv3c_abscanary_tau022_n500_mns8_continuous_dynamic.json"))
    ech_g_acc = ech_g_blob["measurement"]["accuracy"]
    ech_g_bits = per_rank_bits(ech_g_blob)
    ech_g_tiers = consol(ech_g_blob)

    # HotpotQA
    hot_dir = base / "hotpotqa_n500_mns8"
    fp16_h = json.load(open(hot_dir / "fp16_n500_mns8.json"))["cells"][0]
    int4_h = json.load(open(hot_dir / "int4_n500_mns8.json"))["cells"][0]
    int8_h = json.load(open(hot_dir / "int8_n500_mns8.json"))["cells"][0]
    nv_h = json.load(open(hot_dir / "nvfp4_n500_mns8_tp8.json"))["cells"][0]
    ech_h_blob = json.load(open(hot_dir / "echokv3c_abscanary_tau022_n500_mns8_continuous_dynamic.json"))
    ech_h_acc = ech_h_blob["measurement"]["accuracy"]
    ech_h_bits = per_rank_bits(ech_h_blob)
    ech_h_tiers = consol(ech_h_blob)

    # RULER@128K
    rul = json.load(open(base / "diag.json"))
    fp16_r = next(c for c in rul["cells"] if c["config"] == "fp16")
    int4_r = next(c for c in rul["cells"] if c["config"] == "static_int4")
    ech_r_blob = json.load(open(base / "echokv3c_abscanary_tau124_ruler128k_n20_mns1_continuous_dynamic.json"))
    ech_r_acc = ech_r_blob["measurement"]["accuracy"]
    ech_r_bits = per_rank_bits(ech_r_blob)
    ech_r_tiers = consol(ech_r_blob)

    # τ-Bench (airline + retail, dynamic). Use merged accuracy + averaged
    # per-rank bits across the two splits.
    def _per_req(c): return [r["correct"] for r in c["per_request"]]
    fp16_ta = json.load(open(base15 / "taubench_airline_fp16_n50.json"))["cells"][0]
    fp16_tr = json.load(open(base15 / "taubench_retail_fp16_n50.json"))["cells"][0]
    int4_ta = json.load(open(base15 / "taubench_airline_int4_n50.json"))["cells"][0]
    int4_tr = json.load(open(base15 / "taubench_retail_int4_n50.json"))["cells"][0]
    int8_ta = json.load(open(base15 / "taubench_airline_int8_n50.json"))["cells"][0]
    int8_tr = json.load(open(base15 / "taubench_retail_int8_n50.json"))["cells"][0]
    nv_ta = json.load(open(base15 / "taubench_airline_nvfp4_n50.json"))["cells"][0]
    nv_tr = json.load(open(base15 / "taubench_retail_nvfp4_n50.json"))["cells"][0]
    ech_ta_blob = json.load(open(base15 / "taubench_airline_echokv_dynamic_n50.json"))
    ech_tr_blob = json.load(open(base15 / "taubench_retail_echokv_dynamic_n50.json"))

    def merge_acc(a, b):
        n = len(a["per_request"]) + len(b["per_request"])
        c = sum(_per_req(a)) + sum(_per_req(b))
        return c / max(n, 1)
    fp16_t = {"accuracy": merge_acc(fp16_ta, fp16_tr)}
    int4_t = {"accuracy": merge_acc(int4_ta, int4_tr)}
    int8_t = {"accuracy": merge_acc(int8_ta, int8_tr)}
    nv_t   = {"accuracy": merge_acc(nv_ta, nv_tr)}
    # EchoKV dynamic: average per-rank bits across the two domains
    ech_t_acc = (ech_ta_blob["measurement"]["accuracy"] * len(ech_ta_blob["measurement"]["per_request"])
                  + ech_tr_blob["measurement"]["accuracy"] * len(ech_tr_blob["measurement"]["per_request"])) / \
                 max(len(ech_ta_blob["measurement"]["per_request"]) + len(ech_tr_blob["measurement"]["per_request"]), 1)
    ech_t_bits = (per_rank_bits(ech_ta_blob) + per_rank_bits(ech_tr_blob)) / 2
    # Use airline tier counts for the annotation (retail is identical {INT4:1, INT8:79})
    ech_t_tiers = consol(ech_ta_blob)

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
# Figure 4: method comparison (4-panel side-by-side)
# ───────────────────────────────────────────────────────────────────────────

# Bit-tier colour map (matches cross_workload bar colours)
BIT_COLOR = {
    2:  "#FECACA",   # very light red — INT2
    4:  "#93C5FD",   # light blue — INT4
    8:  "#3B82F6",   # medium blue — INT8
    16: "#1E3A8A",   # deep navy — FP16
}


def fig_method_comparison():
    """4-panel side-by-side: how EchoKV's bit-allocation strategy differs
    from related KV-cache compression methods. Each panel shows one column
    of L=80 layers stacked vertically, coloured by the assigned bit-width
    under that strategy. Only the rightmost panel (EchoKV) carries a
    feedback arrow, signalling that its allocation is determined online."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import FancyArrowPatch
    from collections import Counter

    L = 80  # Llama-3.1-70B
    # Static cells
    uniform_int4 = [4] * L
    uniform_int8 = [8] * L
    # Static mixed-precision: deterministic offline schedule. As a
    # representative example we use a 50/50 split (low-half INT8,
    # high-half INT4) — the kind of fixed schedule offline methods
    # produce. Exact schedule not load-bearing; the panel is illustrative.
    static_mixed = [8 if i < L // 2 else 4 for i in range(L)]
    # EchoKV: actual HotpotQA tier vector (consolidated, dynamic).
    hot_blob = json.load(open(
        ROOT / "results/serving/session14_70b" /
        "hotpotqa_n500_mns8" /
        "echokv3c_abscanary_tau022_n500_mns8_continuous_dynamic.json"))
    pr = hot_blob["final_tiers_per_rank"]
    echokv = [max(pr[r][li] for r in range(len(pr))) for li in range(L)]

    panels = [
        ("(a) Uniform INT4",  "static; every layer 4 bits",   uniform_int4),
        ("(b) Uniform INT8",  "static; every layer 8 bits",   uniform_int8),
        ("(c) Static mixed",  "fixed offline schedule",       static_mixed),
        ("(d) EchoKV",        "online, canary-driven",        echokv),
    ]

    with plt.rc_context(PAPER_STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(7.2, 3.0),
                                  gridspec_kw={"wspace": 0.5})
        for ax, (title, sub, tv) in zip(axes, panels):
            arr = np.array(tv).reshape(-1, 1)
            cmap = ListedColormap([BIT_COLOR[2], BIT_COLOR[4], BIT_COLOR[8]])
            # Map bits → discrete level for imshow (0=INT2, 1=INT4, 2=INT8)
            level = np.where(arr == 2, 0, np.where(arr == 4, 1, 2))
            ax.imshow(level, aspect="auto", cmap=cmap, vmin=0, vmax=2,
                      origin="lower", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([0, L // 2, L - 1])
            ax.set_yticklabels(["0", f"{L//2}", f"{L-1}"], fontsize=7)
            ax.set_ylabel("layer index" if title.startswith("(a)") else "")
            ax.set_title(title, fontsize=9, loc="center", pad=14)
            # Tier composition counts (one line under the title, above strip)
            tc = Counter(tv)
            comp = ", ".join(f"{tc[k]}×INT{k}" for k in (2, 4, 8) if tc.get(k))
            ax.text(0.5, 1.015, comp, transform=ax.transAxes,
                    ha="center", va="bottom",
                    fontsize=6.8, color="#374151")
            # Sub-caption under panel
            ax.text(0.5, -0.07, sub, transform=ax.transAxes,
                    ha="center", va="top", fontsize=7,
                    style="italic", color="#4B5563")

        # Feedback arrow on panel (d): a curved loop on the right side
        ax_d = axes[3]
        ax_d.annotate("", xy=(1.18, 0.05), xycoords="axes fraction",
                      xytext=(1.18, 0.95), textcoords="axes fraction",
                      arrowprops=dict(arrowstyle="-|>",
                                      color=C_ONLINE_E,
                                      connectionstyle="arc3,rad=0.45",
                                      linewidth=1.3))
        ax_d.text(1.32, 0.5, "canary\nfeedback",
                  transform=ax_d.transAxes,
                  ha="left", va="center",
                  fontsize=7, style="italic",
                  color=C_ONLINE_E)

        # Discrete legend at bottom
        from matplotlib.patches import Patch
        handles = [
            Patch(facecolor=BIT_COLOR[2], edgecolor="white", label="INT2"),
            Patch(facecolor=BIT_COLOR[4], edgecolor="white", label="INT4"),
            Patch(facecolor=BIT_COLOR[8], edgecolor="white", label="INT8"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=3,
                   bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=8,
                   handlelength=1.2)
        fig.subplots_adjust(bottom=0.18)

        out = OUT / "method_comparison.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"  wrote {out.relative_to(ROOT)}")


# ───────────────────────────────────────────────────────────────────────────
# Figure 5: per-layer tier heatmap across workloads
# ───────────────────────────────────────────────────────────────────────────

def fig_tier_heatmap():
    """Per-layer chosen bit-width across workloads. Rows are workloads;
    columns are layers; colour is the consolidated (max-across-ranks) tier
    selected by the canary controller during continuous-dynamic serving.
    Reveals the workload-coherence claim visually: short reasoning workloads
    bias INT4, retrieval workloads bias INT8, RULER drops to INT2 on a
    single attention-sink layer."""
    from matplotlib.colors import ListedColormap

    base = ROOT / "results/serving/session14_70b"
    base15 = ROOT / "results/serving/session15_70b"

    def consol(blob):
        pr = blob["final_tiers_per_rank"]
        L = len(pr[0])
        return [max(pr[r][li] for r in range(len(pr))) for li in range(L)]

    rows = [
        ("GSM8K-Hard",
         json.load(open(base / "echokv3c_abscanary_tau022_n500_mns8_continuous_dynamic.json"))),
        ("HotpotQA",
         json.load(open(base / "hotpotqa_n500_mns8" /
                        "echokv3c_abscanary_tau022_n500_mns8_continuous_dynamic.json"))),
        ("RULER@128K",
         json.load(open(base / "echokv3c_abscanary_tau124_ruler128k_n20_mns1_continuous_dynamic.json"))),
        (r"$\tau$-Bench",
         json.load(open(base15 / "taubench_airline_echokv_dynamic_n50.json"))),
    ]
    L = len(consol(rows[0][1]))
    mat_levels = np.zeros((len(rows), L), dtype=int)
    for i, (_, blob) in enumerate(rows):
        tv = consol(blob)
        mat_levels[i] = [(0 if b == 2 else (1 if b == 4 else 2)) for b in tv]

    cmap = ListedColormap([BIT_COLOR[2], BIT_COLOR[4], BIT_COLOR[8]])

    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(7.0, 1.9))
        im = ax.imshow(mat_levels, aspect="auto", cmap=cmap, vmin=0, vmax=2,
                       interpolation="nearest")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows])
        ax.set_xticks([0, L // 4, L // 2, 3 * L // 4, L - 1])
        ax.set_xticklabels(["0", f"{L//4}", f"{L//2}", f"{3*L//4}", f"{L-1}"])
        ax.set_xlabel("Layer index")

        # Per-row composition (right margin)
        from collections import Counter
        for i, (_, blob) in enumerate(rows):
            tv = consol(blob)
            tc = Counter(tv)
            parts = [f"{tc[k]}×INT{k}" for k in (2, 4, 8) if tc.get(k)]
            avg = sum(tv) / len(tv)
            ax.text(L + 1.2, i, f"{', '.join(parts)} ({avg:.2f} bits)",
                    ha="left", va="center", fontsize=7.2, color="#374151")

        # Colour-bar legend at bottom
        from matplotlib.patches import Patch
        handles = [
            Patch(facecolor=BIT_COLOR[2], edgecolor="white", label="INT2"),
            Patch(facecolor=BIT_COLOR[4], edgecolor="white", label="INT4"),
            Patch(facecolor=BIT_COLOR[8], edgecolor="white", label="INT8"),
        ]
        ax.legend(handles=handles, loc="upper center",
                  bbox_to_anchor=(0.5, -0.35), ncol=3,
                  frameon=False, fontsize=8, handlelength=1.2)

        out = OUT / "tier_heatmap.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"  wrote {out.relative_to(ROOT)}")


# ───────────────────────────────────────────────────────────────────────────
# Figure 6: per-layer canary stress + tier selection
# ───────────────────────────────────────────────────────────────────────────

def fig_per_layer_stress():
    """Per-layer canary divergence \\hat{\\delta}_4 with the controller's
    chosen tier overlaid as colour. Inspired by PyramidKV's per-layer
    attention plots, but the y-axis here is the canary's quantization-error
    signal (the input the controller sees), and the bar colour is the
    bit-width the controller assigned that layer. Layers whose error
    exceeds the threshold $\\tau$ (dashed line) are upgraded from INT4 to
    INT8 — the closed-loop decision rule made visible."""
    base = ROOT / "results/serving/session14_70b"
    base15 = ROOT / "results/serving/session15_70b"

    def per_layer_delta_4(blob):
        ptd = blob["warmup_per_tier_delta_per_rank"]
        L = len(ptd[0]["4"])
        return [sum(r["4"][li] for r in ptd) / len(ptd) for li in range(L)]

    def consol(blob):
        pr = blob["final_tiers_per_rank"]
        L = len(pr[0])
        return [max(pr[r][li] for r in range(len(pr))) for li in range(L)]

    rows = [
        ("GSM8K-Hard", 0.022,
         json.load(open(base / "echokv3c_abscanary_tau022_n500_mns8_continuous_dynamic.json"))),
        ("HotpotQA", 0.022,
         json.load(open(base / "hotpotqa_n500_mns8" /
                        "echokv3c_abscanary_tau022_n500_mns8_continuous_dynamic.json"))),
        ("RULER@128K", 0.124,
         json.load(open(base / "echokv3c_abscanary_tau124_ruler128k_n20_mns1_continuous_dynamic.json"))),
        (r"$\tau$-Bench (airline)", 0.022,
         json.load(open(base15 / "taubench_airline_echokv_dynamic_n50.json"))),
    ]

    L = 80
    with plt.rc_context(PAPER_STYLE):
        fig, axes = plt.subplots(len(rows), 1, figsize=(7.0, 3.6),
                                  sharex=True,
                                  gridspec_kw={"hspace": 0.65})
        for ax, (name, tau, blob) in zip(axes, rows):
            d4 = per_layer_delta_4(blob)
            # Use seed (warmup-end) consolidated tiers — these are the
            # tiers the canary signal in d4 actually drove the controller
            # to pick. final_tiers_per_rank is end-of-measurement (after
            # dynamic drift), which would mismatch the warmup-end signal.
            tiers = blob["consolidated_tiers"]
            acc = blob["measurement"]["accuracy"]
            # Bar colours from chosen tier
            colors = [BIT_COLOR.get(t, "#888") for t in tiers]
            x = np.arange(L)
            ax.bar(x, d4, width=0.95, color=colors,
                   edgecolor="white", linewidth=0.2, zorder=3)
            # τ threshold line
            ax.axhline(tau, color=C_ONLINE_E, linestyle="--",
                       linewidth=1.0, alpha=0.85, zorder=2)
            # ymax: max of (data, 1.4×τ) so the τ line is visible
            ymax = max(max(d4) * 1.15, tau * 1.5)
            ax.set_ylim(0, ymax)
            ax.set_xlim(-0.5, L - 0.5)
            ax.set_ylabel(r"$\hat{\delta}_4$", fontsize=8)
            # workload + acc + bits annotation in upper-right of the panel
            from collections import Counter
            tc = Counter(tiers)
            avg = sum(tiers) / len(tiers)
            promotions = sum(v for k, v in tc.items() if k > 4)
            demotions = sum(v for k, v in tc.items() if k < 4)
            label = (
                f"{name}   "
                f"acc={acc:.3f}   "
                f"{avg:.2f} bits   "
                + (f"{promotions} INT8↑   " if promotions else "")
                + (f"{demotions} INT2↓   " if demotions else "")
            )
            ax.text(0.0, 1.05, label, transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=8.5,
                    fontweight="bold", color=C_TEXT)
            # τ label on the line (right side)
            ax.text(L - 1, tau, fr" $\tau{{=}}{tau:.3f}$",
                    ha="right", va="bottom", fontsize=7,
                    color=C_ONLINE_E, fontweight="bold")
            ax.grid(axis="y", linestyle=":", color=C_GRID, alpha=0.6, zorder=1)
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", labelsize=7)

        axes[-1].set_xlabel("Layer index", fontsize=9)
        axes[-1].tick_params(axis="x", labelsize=8)

        # Shared discrete legend at top
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        handles = [
            Patch(facecolor=BIT_COLOR[2], edgecolor="white", label="INT2 (demoted)"),
            Patch(facecolor=BIT_COLOR[4], edgecolor="white", label="INT4 (default)"),
            Patch(facecolor=BIT_COLOR[8], edgecolor="white", label="INT8 (upgraded)"),
            Line2D([0], [0], color=C_ONLINE_E, linestyle="--",
                   linewidth=1.0, label=r"$\tau$ threshold"),
        ]
        fig.legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, 1.005), ncol=4,
                   frameon=False, fontsize=8, handlelength=1.4,
                   columnspacing=1.4)
        out = OUT / "per_layer_stress.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"  wrote {out.relative_to(ROOT)}")


# ───────────────────────────────────────────────────────────────────────────

def main():
    print(f"Generating paper figures into {OUT.relative_to(ROOT)}/")
    fig_architecture()
    fig_cross_workload()
    fig_sensor_validation()
    fig_method_comparison()
    fig_tier_heatmap()
    fig_per_layer_stress()


if __name__ == "__main__":
    main()
