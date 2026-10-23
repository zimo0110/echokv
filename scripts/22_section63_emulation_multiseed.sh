#!/usr/bin/env bash
# §6.3 emulation reproducibility — multi-seed paired McNemar.
#
# Re-runs the paper's §6.3 emulation protocol at three seeds with n=500
# per seed, and pairs each EchoKV-3C run against FP16 on the same
# prompts. Tests whether the §6.3 headline (EchoKV-3C frozen Δ=0 vs
# FP16 at 5.45 bits, GSM8K-Hard 70B) reproduces across seeds, or
# whether seed=0 was a favorable-noise outlier.
#
# §6.3 protocol per seed:
#   * FP16 baseline (default BF16 cache, no canary) — n=500, mns=8.
#   * EchoKV-3C with absolute canary, tau_budget=0.022, warmup_n=16,
#     kref_window_size=64, multi_precision — n=500, mns=8.
#   * Same prompt seed for both phases of the seed, so the paired
#     McNemar is on identical prompts.
#
# Outputs:
#   results/serving/section63_repro/seed${S}_fp16_static.json   (FP16 cell)
#   results/serving/section63_repro/seed${S}_echokv.json        (EchoKV cell)
#   results/serving/section63_repro/aggregate.json              (multi-seed)
#
# Run as: bash scripts/22_section63_emulation_multiseed.sh
# Or in tmux:
#   tmux new -s repro63 'bash scripts/22_section63_emulation_multiseed.sh'
#   # Detach: Ctrl-b then d
#   # Reattach: tmux attach -t repro63
#
# Expected wall: ~3-4 hours on 70B + TP=4 (3 seeds × ~55-65 min/seed).
# Each seed: ~10 min FP16 + ~40-50 min EchoKV + 2× engine init.

set -euo pipefail
cd "$(dirname "$0")/.."

# 70B requires TP=4. Default to GPUs 0-3; override if needed.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

STAMP=$(date -u +%Y%m%d_%H%M%S)
LOG="/tmp/section63_multiseed_${STAMP}.log"
OUT_DIR="results/serving/section63_repro"
mkdir -p "${OUT_DIR}"

SEEDS=(0 1 2)
N_PROMPTS=500
WARMUP_N=16
TAU_BUDGET=0.022
KREF_WINDOW=64
MAX_NUM_SEQS=8

echo "================================================================" | tee "${LOG}"
echo "§6.3 emulation reproducibility — multi-seed" | tee -a "${LOG}"
echo "  seeds: ${SEEDS[*]}, n_prompts=${N_PROMPTS}, mns=${MAX_NUM_SEQS}" | tee -a "${LOG}"
echo "  tau_budget=${TAU_BUDGET}, warmup_n=${WARMUP_N}, kref=${KREF_WINDOW}" | tee -a "${LOG}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "${LOG}"
echo "  log: ${LOG}" | tee -a "${LOG}"
echo "  out: ${OUT_DIR}/" | tee -a "${LOG}"
echo "  expected wall: ~3-4 hours" | tee -a "${LOG}"
echo "  start: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG}"
echo "================================================================" | tee -a "${LOG}"

for S in "${SEEDS[@]}"; do
    FP16_OUT="${OUT_DIR}/seed${S}_fp16_static.json"
    ECHOKV_OUT="${OUT_DIR}/seed${S}_echokv.json"

    echo | tee -a "${LOG}"
    echo "================================================================" | tee -a "${LOG}"
    echo "Seed ${S} — Phase 1: FP16 + static_int4 baselines (n=${N_PROMPTS})" | tee -a "${LOG}"
    echo "  out: ${FP16_OUT}" | tee -a "${LOG}"
    echo "  start: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG}"
    echo "================================================================" | tee -a "${LOG}"
    /opt/conda/envs/scalable/bin/python \
        -m evaluation.serving.run_session14_70b_gsm8k_hard \
        --n-prompts "${N_PROMPTS}" \
        --configs fp16 static_int4 \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --seed "${S}" \
        --out "${FP16_OUT}" 2>&1 | tee -a "${LOG}"

    echo | tee -a "${LOG}"
    echo "================================================================" | tee -a "${LOG}"
    echo "Seed ${S} — Phase 2: EchoKV-3C abs-canary (n=${N_PROMPTS})" | tee -a "${LOG}"
    echo "  out: ${ECHOKV_OUT}" | tee -a "${LOG}"
    echo "  start: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG}"
    echo "================================================================" | tee -a "${LOG}"
    /opt/conda/envs/scalable/bin/python \
        -m evaluation.serving.run_session14_70b_gsm8k_hard_echokv \
        --n-prompts "${N_PROMPTS}" \
        --warmup-n "${WARMUP_N}" \
        --multi-precision \
        --absolute-canary \
        --kref-window-size "${KREF_WINDOW}" \
        --tau-budget "${TAU_BUDGET}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --seed "${S}" \
        --out "${ECHOKV_OUT}" 2>&1 | tee -a "${LOG}"

    echo "Seed ${S} done at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG}"
done

echo | tee -a "${LOG}"
echo "================================================================" | tee -a "${LOG}"
echo "Aggregating: paired McNemar per seed + cross-seed summary" | tee -a "${LOG}"
echo "================================================================" | tee -a "${LOG}"

/opt/conda/envs/scalable/bin/python <<EOF 2>&1 | tee -a "${LOG}"
import json
import math
from collections import Counter
from pathlib import Path

OUT_DIR = Path("${OUT_DIR}")
SEEDS = [int(s) for s in "${SEEDS[*]}".split()]

def mcnemar(a_correct: dict, b_correct: dict):
    """Paired McNemar with Yates' continuity correction.
    a_correct, b_correct are dicts: request_id -> 0/1.
    Returns (b_wins, a_wins, n_disc, chi2_yates, p_two_sided)."""
    common = sorted(set(a_correct) & set(b_correct))
    b_wins = sum(1 for k in common if a_correct[k] < b_correct[k])  # A wrong, B right
    a_wins = sum(1 for k in common if a_correct[k] > b_correct[k])  # A right, B wrong
    n_disc = a_wins + b_wins
    if n_disc == 0:
        return a_wins, b_wins, 0, 0.0, 1.0
    chi2 = (abs(a_wins - b_wins) - 1) ** 2 / n_disc
    try:
        from scipy.stats import chi2 as _chi2_dist
        p = float(_chi2_dist.sf(chi2, df=1))
    except ImportError:
        z = chi2 ** 0.5
        p = math.erfc(z / math.sqrt(2.0))
    return a_wins, b_wins, n_disc, chi2, p


def load_cell(path: Path, cfg: str) -> tuple[dict, float]:
    blob = json.loads(path.read_text())
    cells = blob.get("cells", [])
    cell = next(c for c in cells if c["config"] == cfg)
    return ({r["request_id"]: int(r["correct"]) for r in cell["per_request"]},
            float(cell["accuracy"]))


def load_echokv(path: Path) -> tuple[dict, float, dict]:
    blob = json.loads(path.read_text())
    m = blob.get("measurement", blob)
    proj = blob.get("memory_projection", {})
    return ({r["request_id"]: int(r["correct"]) for r in m["per_request"]},
            float(m["accuracy"]),
            proj)


per_seed = []
for s in SEEDS:
    fp16_d, fp16_acc = load_cell(OUT_DIR / f"seed{s}_fp16_static.json", "fp16")
    int4_d, int4_acc = load_cell(OUT_DIR / f"seed{s}_fp16_static.json", "static_int4")
    ech_d, ech_acc, ech_proj = load_echokv(OUT_DIR / f"seed{s}_echokv.json")

    e_vs_fp16 = mcnemar(fp16_d, ech_d)
    e_vs_int4 = mcnemar(int4_d, ech_d)
    int4_vs_fp16 = mcnemar(fp16_d, int4_d)

    print(f"\n--- Seed {s} ---")
    print(f"  FP16:           acc={fp16_acc:.3f}")
    print(f"  static_int4:    acc={int4_acc:.3f}")
    print(f"  EchoKV-3C:      acc={ech_acc:.3f}  avg_bits={ech_proj.get('avg_bits')}")
    print(f"  tier_counts:    {ech_proj.get('tier_counts')}")
    print(f"  INT4 vs FP16:   Δ={int4_acc - fp16_acc:+.4f}  "
          f"({int4_vs_fp16[0]}/{int4_vs_fp16[1]})  χ²={int4_vs_fp16[3]:.2f}  p={int4_vs_fp16[4]:.4f}")
    print(f"  EchoKV vs FP16: Δ={ech_acc - fp16_acc:+.4f}  "
          f"({e_vs_fp16[0]}/{e_vs_fp16[1]})  χ²={e_vs_fp16[3]:.2f}  p={e_vs_fp16[4]:.4f}")
    print(f"  EchoKV vs INT4: Δ={ech_acc - int4_acc:+.4f}  "
          f"({e_vs_int4[0]}/{e_vs_int4[1]})  χ²={e_vs_int4[3]:.2f}  p={e_vs_int4[4]:.4f}")
    per_seed.append({
        "seed": s,
        "fp16_acc": fp16_acc, "int4_acc": int4_acc, "echokv_acc": ech_acc,
        "echokv_avg_bits": ech_proj.get("avg_bits"),
        "echokv_tier_counts": ech_proj.get("tier_counts"),
        "int4_vs_fp16": {"a_wins": int4_vs_fp16[0], "b_wins": int4_vs_fp16[1],
                        "chi2": int4_vs_fp16[3], "p": int4_vs_fp16[4],
                        "delta": int4_acc - fp16_acc},
        "echokv_vs_fp16": {"a_wins": e_vs_fp16[0], "b_wins": e_vs_fp16[1],
                          "chi2": e_vs_fp16[3], "p": e_vs_fp16[4],
                          "delta": ech_acc - fp16_acc},
        "echokv_vs_int4": {"a_wins": e_vs_int4[0], "b_wins": e_vs_int4[1],
                          "chi2": e_vs_int4[3], "p": e_vs_int4[4],
                          "delta": ech_acc - int4_acc},
    })


print("\n--- Cross-seed summary ---")
fp16s = [r["fp16_acc"] for r in per_seed]
int4s = [r["int4_acc"] for r in per_seed]
echs  = [r["echokv_acc"] for r in per_seed]
bits  = [r["echokv_avg_bits"] for r in per_seed if r["echokv_avg_bits"] is not None]

def mean(xs): return sum(xs)/len(xs)
def spread(xs): return max(xs) - min(xs)
print(f"  FP16        mean={mean(fp16s):.3f}  spread={spread(fp16s):.3f}  raw={fp16s}")
print(f"  static_int4 mean={mean(int4s):.3f}  spread={spread(int4s):.3f}  raw={int4s}")
print(f"  EchoKV-3C   mean={mean(echs):.3f}  spread={spread(echs):.3f}  raw={echs}")
if bits:
    print(f"  EchoKV bits mean={mean(bits):.2f}  spread={spread(bits):.2f}  raw={bits}")

# Tier vector stability
tier_summaries = [r["echokv_tier_counts"] for r in per_seed]
print(f"  Tier counts across seeds: {tier_summaries}")

# Reproducibility verdict
ech_deltas = [r["echokv_vs_fp16"]["delta"] for r in per_seed]
within_pm02_pub = sum(1 for d in ech_deltas if abs(d - 0.0) < 0.02)
print(f"\n  Published §6.3: EchoKV-3C frozen Δ vs FP16 = 0.000 (5.45 bits)")
print(f"  Reproduced (|Δ| < 0.02 vs published 0): {within_pm02_pub}/{len(per_seed)} seeds")

agg_out = OUT_DIR / "aggregate.json"
agg_out.write_text(json.dumps({
    "per_seed": per_seed,
    "fp16_mean": mean(fp16s), "fp16_spread": spread(fp16s),
    "int4_mean": mean(int4s), "int4_spread": spread(int4s),
    "echokv_mean": mean(echs), "echokv_spread": spread(echs),
    "echokv_bits_mean": mean(bits) if bits else None,
    "echokv_bits_spread": spread(bits) if bits else None,
    "tier_counts_per_seed": tier_summaries,
    "reproduces_pub_delta_within_02": within_pm02_pub,
    "n_seeds": len(per_seed),
}, indent=2))
print(f"\nwrote {agg_out}")
EOF

echo | tee -a "${LOG}"
echo "================================================================" | tee -a "${LOG}"
echo "Done at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG}"
echo "Outputs in ${OUT_DIR}/" | tee -a "${LOG}"
echo "Log:     ${LOG}" | tee -a "${LOG}"
echo "================================================================" | tee -a "${LOG}"
