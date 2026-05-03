# Benchmarks — how to run them

Reproduction guide for every cell that lands in [PAPER_DRAFT.md](PAPER_DRAFT.md)
and [REPORT.md](REPORT.md). Each section says **what the cell is**,
**which script to run**, **expected wall**, and **where the JSON lands**.
If you only want the headline numbers, read [Quick start](#quick-start)
and skip the rest.

---

## Quick start

```bash
# 0) Smoke test (8B + TP=2, ~2 min): catches env/import bugs before 70B time
tmux new -s smoke 'bash scripts/00_smoke_hotpotqa.sh'

# 1) Three-workload Pareto headline (frozen-tier protocol):
tmux new -s gsm    'bash scripts/03_echokv_n500_mns8_abscanary.sh'   # GSM8K-Hard, ~40 min
tmux new -s hot    'bash scripts/04_hotpotqa_n500_mns8.sh'           # HotpotQA, ~50 min
tmux new -s ruler  'bash scripts/06_ruler128k_echokv_tau_sqrtL.sh'   # RULER@128K, ~7 min

# 2) Continuously-dynamic protocol on the same three workloads:
tmux new -s cd-gsm 'bash scripts/08_gsm8k_hard_continuous_dynamic.sh'         # ~20 min
tmux new -s cd-rest 'bash scripts/09_hotpotqa_ruler_continuous_dynamic.sh'    # ~10 min total
```

Each script prints a paired-McNemar comparison vs the matching
baseline cells when it finishes.

---

## Environment

- **GPUs**: 4×H100-SXM5 80 GB. Default scripts use `CUDA_VISIBLE_DEVICES=4,5,6,7`
  so they leave the first four cards free for other work; override by
  exporting the variable before invoking.
- **Python**: `/opt/conda/envs/scalable/bin/python` is the canonical
  interpreter. Pinned dependency versions are in [CHANGELOG.md](CHANGELOG.md)
  under "Upstream Pins."
- **Required env vars** (every script sets these — don't override unless
  you know what you're doing):
  - `VLLM_ALLOW_INSECURE_SERIALIZATION=1` — required for vLLM's
    sub-process worker IPC at TP > 1.
  - `FLASHINFER_DISABLE_VERSION_CHECK=1` — flashinfer-python 0.6.3 +
    flashinfer-cubin 0.6.8.post1 disagree on a version number for a
    feature path the canary harness doesn't touch; the check is bypassed.
- **TP layout**: 70B uses TP=4 (one full model across the 4-GPU group).
  The smoke test uses 8B + TP=2 (only needs 2 cards).

---

## Configurations and protocols

### Configurations under test

Each measurement cell is a tuple `(workload, model, config, protocol)`.
Configurations:

| config | bits/elem | what it is |
|--|--:|--|
| `fp16` | 16.00 | unquantized KV cache; baseline |
| `static_int4` | 4.00 | KIVI-style group-32 INT4 across all layers |
| `static_int8` | 8.00 | KIVI-style group-32 INT8 across all layers |
| `echokv_3c` | 4.0–8.0 | closed-loop per-layer adaptive (this paper) |

The static cells use forward-pre-hooks installed on every TP rank's
attention module via `static_int4_tp.py` / `static_int8_tp.py`. The
`echokv_3c` cell uses the absolute-canary sensor + multi-precision
controller (`--multi-precision --absolute-canary`).

### Measurement protocols

Two protocols matter for the paper:

| protocol | warmup canary | measurement canary | controller during measurement | use case |
|--|--|--|--|--|
| **frozen-tier** (default) | `~0.25` | `1e-6` (off) | frozen at the consolidated warmup vector | reproducible paired McNemar |
| **continuously-dynamic** | `~0.25` | `1/64` (production) | unfrozen — tiers continue to adapt | architecturally-faithful "closed-loop runtime" claim |

The `--continuous-dynamic` flag on every echokv driver toggles the
second protocol. INV-2 (no in-place block rewrite) holds in both
modes — tier changes only affect *new* block allocations.

The driver writes `measurement_protocol`, `measurement_canary_rate`,
`measurement_freeze_controller`, and `final_tiers_per_rank` into the
output JSON so analysis code can distinguish.

### τ-budget choice

The controller's per-tier divergence budget is set by `--tau-budget`.
Use the Pinsker-derived rule (see [PAPER_DRAFT.md §4.X](PAPER_DRAFT.md)):

> **τ_used(L) = τ₀ · max(1, √(L/L₀))** with τ₀ = 0.022, L₀ = 4 000.

Concrete values used in this repo:

| workload | L | τ_used | rationale |
|--|--:|--:|--|
| GSM8K-Hard | ≈400 | 0.022 | L < L₀; floor at τ₀ |
| HotpotQA | ≈1500 | 0.022 | L < L₀; floor at τ₀ |
| RULER@128K | 128 000 | 0.124 | L ≫ L₀; rescaled by √L |

τ values like 0.007 (GSM8K-Hard) and 0.013 (HotpotQA) come from
applying the unfloored √L formula and are kept on disk only as
empirical evidence that the floor matters (CHANGELOG Phase-7/8
diagnostic cells).

---

## Workload × cell matrix

### GSM8K-Hard (n=500, max_num_seqs=8)

`gsm-hard` (10–20 step arithmetic on 7–9 digit constants), prompts ≈ 400 tokens.

| cell | protocol | τ | script | output JSON | wall |
|--|--|--:|--|--|--:|
| FP16 | (n/a) | (n/a) | (already on disk) | `gsm8k_hard_n500.json` (Phase-2) | — |
| static_int4 | (n/a) | (n/a) | (already on disk) | `gsm8k_hard_n500.json` (Phase-2) | — |
| static_int8 | (n/a) | (n/a) | not yet run at n=500 mns=8 | (cosmetic gap, see Open Q 26) | ~18 min |
| EchoKV-3C frozen | frozen-tier | 0.022 | `scripts/03_echokv_n500_mns8_abscanary.sh` | `echokv3c_abscanary_tau022_n500_mns8.json` | ~40 min |
| EchoKV-3C frozen, tight τ | frozen-tier | 0.007 | `scripts/07_gsm8k_hotpotqa_echokv_tau_sqrtL.sh` (first half) | `echokv3c_abscanary_tau007_n500_mns8.json` | ~20 min |
| EchoKV-3C dynamic | continuous-dynamic | 0.022 | `scripts/08_gsm8k_hard_continuous_dynamic.sh` | `echokv3c_abscanary_tau022_n500_mns8_continuous_dynamic.json` | ~20 min |

All cells live in `results/serving/session14_70b/`.

### HotpotQA (n=500, max_num_seqs=8)

10-paragraph multi-hop QA, prompts ≈ 1500 tokens, completions ≈ 5 tokens.

| cell | protocol | τ | script | output JSON | wall |
|--|--|--:|--|--|--:|
| FP16 | (n/a) | (n/a) | `scripts/04_hotpotqa_n500_mns8.sh` (FP16 leg) | `fp16_n500_mns8.json` | ~1 min |
| static_int4 | (n/a) | (n/a) | `scripts/04_hotpotqa_n500_mns8.sh` (INT4 leg) | `int4_n500_mns8.json` | ~1 min |
| static_int8 | (n/a) | (n/a) | `scripts/04_hotpotqa_n500_mns8.sh` (INT8 leg) | `int8_n500_mns8.json` | ~1 min |
| EchoKV-3C frozen | frozen-tier | 0.022 | `scripts/04_hotpotqa_n500_mns8.sh` (echokv leg) | `echokv3c_abscanary_tau022_n500_mns8.json` | ~1 min |
| EchoKV-3C frozen, tight τ | frozen-tier | 0.013 | `scripts/07_*.sh` (second half) | `echokv3c_abscanary_tau013_n500_mns8.json` | ~1 min |
| EchoKV-3C dynamic | continuous-dynamic | 0.022 | `scripts/09_*.sh` (HotpotQA leg) | `echokv3c_abscanary_tau022_n500_mns8_continuous_dynamic.json` | ~1 min |
| EchoKV-3C dynamic, tight τ | continuous-dynamic | 0.013 | one-off (Phase-8 follow-up) | `echokv3c_abscanary_tau013_n500_mns8_continuous_dynamic.json` | ~1 min |

All cells live in `results/serving/session14_70b/hotpotqa_n500_mns8/`.
Total full-sweep wall ~5 min — HotpotQA is the cheapest sandbox.

### RULER@128K (n=20, max_num_seqs=1)

Single-needle haystack at 128 000 tokens; prompts pre-built by
`evaluation/motivation/workloads.py::make_ruler_prompts(target_tokens=128_000)`.

| cell | protocol | τ | script | output JSON | wall |
|--|--|--:|--|--|--:|
| FP16 | (n/a) | (n/a) | (already on disk) | `diag.json` (Phase-2) | — |
| static_int4 | (n/a) | (n/a) | (already on disk) | `diag.json` (Phase-2) | — |
| EchoKV-3C frozen, un-rescaled τ | frozen-tier | 0.022 | `scripts/05_ruler128k_echokv_n20_mns1.sh` | `echokv3c_abscanary_tau022_ruler128k_n20_mns1.json` | ~7 min |
| EchoKV-3C frozen, √L-rescaled τ | frozen-tier | 0.124 | `scripts/06_ruler128k_echokv_tau_sqrtL.sh` | `echokv3c_abscanary_tau124_ruler128k_n20_mns1.json` | ~7 min |
| EchoKV-3C dynamic, √L-rescaled | continuous-dynamic | 0.124 | `scripts/09_*.sh` (RULER leg) | `echokv3c_abscanary_tau124_ruler128k_n20_mns1_continuous_dynamic.json` | ~7 min |

The un-rescaled τ=0.022 cell is kept on disk to demonstrate the
need for the √L rescaling: it produces `{INT4:9, INT8:71}` = 7.55
bits/element with no accuracy benefit (RULER@128K saturates at
INT4 anyway).

The n=20 mns=1 protocol re-uses the existing Phase-2 baselines in
`diag.json`. Long-context prefill at 128 K on 70B+TP=4 is ~18 s/prompt
sequential, so n=500 mns=8 would be ~6 hours per cell.

---

## Script inventory

| script | purpose | wall |
|--|--|--:|
| `00_smoke_hotpotqa.sh` | 4-prompt FP16 sanity, Llama-3.1-8B + TP=2 | ~2 min |
| `01_int8_gsm8k_hard.sh` | static_int8 GSM8K-Hard n=200 mns=1 (deterministic publication protocol) | ~50 min |
| `02_hotpotqa_sweep.sh` | HotpotQA 4 configs at n=200 mns=1 | ~2.5 h |
| `03_echokv_n500_mns8_abscanary.sh` | **EchoKV-3C frozen, GSM8K-Hard n=500 mns=8 paired vs Phase-2** | ~40 min |
| `04_hotpotqa_n500_mns8.sh` | **HotpotQA full 4-config sweep n=500 mns=8** | ~50 min |
| `05_ruler128k_echokv_n20_mns1.sh` | EchoKV-3C frozen, RULER@128K, **un-rescaled τ=0.022** (diagnostic) | ~7 min |
| `06_ruler128k_echokv_tau_sqrtL.sh` | **EchoKV-3C frozen, RULER@128K, √L-rescaled τ=0.124** | ~7 min |
| `07_gsm8k_hotpotqa_echokv_tau_sqrtL.sh` | EchoKV-3C frozen at the un-floored √L τ on GSM8K-Hard + HotpotQA (diagnostic) | ~25 min |
| `08_gsm8k_hard_continuous_dynamic.sh` | **EchoKV-3C continuous-dynamic, GSM8K-Hard n=500 mns=8** | ~20 min |
| `09_hotpotqa_ruler_continuous_dynamic.sh` | **EchoKV-3C continuous-dynamic, HotpotQA + RULER@128K** | ~10 min |
| `run_paper_phase6.sh` | wrapper: 01 then 02 (Phase-6 publication sweep) | ~3.5 h |

Bold = headline-cell scripts the paper depends on.

---

## Running headline cells in order

If you have one fresh checkout and need every paper-relevant JSON
on disk:

```bash
# Phase-2 baselines for GSM8K-Hard at n=500 mns=8 are assumed already
# present (gsm8k_hard_n500.json with FP16 + INT4 cells). If not, the
# scripts/03 driver will warn at exit when paired-McNemar lookup fails.

# Phase-5b — frozen GSM8K-Hard headline:
tmux new -s p5b 'bash scripts/03_echokv_n500_mns8_abscanary.sh'

# Phase-6 — HotpotQA full sweep:
tmux new -s p6 'bash scripts/04_hotpotqa_n500_mns8.sh'

# Phase-7 — RULER@128K with √L-rescaled τ:
tmux new -s p7 'bash scripts/06_ruler128k_echokv_tau_sqrtL.sh'

# Phase-8a — GSM8K-Hard continuous-dynamic:
tmux new -s p8a 'bash scripts/08_gsm8k_hard_continuous_dynamic.sh'

# Phase-8b — HotpotQA + RULER continuous-dynamic:
tmux new -s p8b 'bash scripts/09_hotpotqa_ruler_continuous_dynamic.sh'
```

Total wall ≈ 2 hours when run sequentially on the same 4-GPU group.

---

## Reading the JSON

Every echokv-3c JSON has the same top-level shape:

```jsonc
{
  "session": "...",
  "model": "meta-llama/Llama-3.1-70B-Instruct",
  "tensor_parallel_size": 4,
  "n_prompts": 500,
  "warmup_n": 16,
  "max_num_seqs": 8,
  "phase": "measurement_complete",
  "measurement_protocol": "frozen-tiers" | "continuous-dynamic",  // Phase-8+
  "measurement_canary_rate": 1e-6 | 0.015625,                     // Phase-8+
  "measurement_freeze_controller": true | false,                  // Phase-8+
  "consolidated_tiers": [4, 8, 4, ...],                  // 80-element list
  "memory_projection": {
    "tier_counts": {"4": 51, "8": 29},
    "avg_bits": 5.45,
    "savings_vs_fp16_pct": 65.9,
    "savings_vs_kivi4_pct": -36.3
  },
  "warmup_per_tier_delta_per_rank": [...],   // δ̂_b EMA at end of warmup
  "final_tiers_per_rank": [...],             // post-measurement (Phase-8+)
  "measurement": {
    "accuracy": 0.488,
    "wall_clock_s": 1139,
    "per_request": [
      {"request_id": "...", "correct": 1, ...},  // 500 entries
      ...
    ]
  },
  "measurement_snapshots": [...]
}
```

Static-baseline JSONs (FP16/INT4/INT8) follow the multi-cell shape
used by `run_session14_70b_*.py`:

```jsonc
{
  "cells": [
    {
      "config": "fp16",
      "accuracy": 0.488,
      "per_request": [{"request_id": "...", "correct": 1}, ...],
      ...
    },
    {"config": "static_int4", ...},
    ...
  ]
}
```

### Paired McNemar between two JSONs

```python
import json, math
def per_req(blob, cell_name=None):
    if "measurement" in blob:
        m = blob["measurement"]
    else:
        m = next(c for c in blob["cells"] if c["config"] == cell_name)
    return {r["request_id"]: r["correct"] for r in m["per_request"]}

def mcnemar(a, b):
    common = set(a) & set(b)
    bw = sum(1 for k in common if b[k] < a[k])
    cw = sum(1 for k in common if b[k] > a[k])
    n = bw + cw
    if n == 0: return 0, 0, 1.0, 0.0
    k = max(bw, cw)
    p = sum(math.comb(n, i) * 0.5**n for i in range(k, n+1))
    return bw, cw, min(2*p, 1.0), (cw - bw) / max(len(common), 1)

fp16 = per_req(json.load(open("results/serving/session14_70b/gsm8k_hard_n500.json")), "fp16")
ech  = per_req(json.load(open("results/serving/session14_70b/echokv3c_abscanary_tau022_n500_mns8.json")))
b, c, p, d = mcnemar(fp16, ech)
print(f"Δ={d:+.4f}, b/c={b}/{c}, p={p:.4f}")
```

Every Phase-3+ script ends with a built-in version of this — output
the script's stdout for the paired-comparison summary it prints
automatically.

---

## Failure recovery

A killed driver process can leave 4 zombie `VLLM::Worker_TP*`
processes holding ~70 GB each on GPUs 4–7. To recover:

```bash
ps -ef | grep -E "Worker_TP|EngineCore" | grep "$(whoami)" | grep -v grep
kill -9 <pids>          # SIGKILL needed if SIGTERM doesn't take
nvidia-smi --query-gpu=index,memory.used --format=csv | sed -n '1p;6,9p'   # verify
```

Each cell writes its own JSON, so a mid-sweep failure leaves earlier
cells usable. Sweep scripts (03, 04, 09) run cells sequentially and
unconditionally — they don't skip cells whose JSON already exists. To
re-run only the missing cell, edit the script or invoke the driver
directly:

```bash
/opt/conda/envs/scalable/bin/python -m evaluation.serving.run_session14_70b_hotpotqa \
    --n-prompts 500 --configs static_int8 --max-num-seqs 8 --max-new-tokens 64 --seed 0 \
    --out results/serving/session14_70b/hotpotqa_n500_mns8/int8_n500_mns8.json
```

The flags every echokv driver accepts:

```
--n-prompts N          (default 100)
--warmup-n N           (default 8; 16 is the production setting)
--max-num-seqs N       (default 8; 1 for deterministic paired protocol)
--max-new-tokens N     (default 512 GSM8K-Hard, 64 HotpotQA, 32 RULER)
--multi-precision      (always set for paper cells)
--absolute-canary      (always set for paper cells)
--kref-window-size N   (default 64)
--tau-budget X         (0.022 short context, 0.124 RULER@128K)
--continuous-dynamic   (Phase-8+)
--measurement-canary-rate X  (default 1/64; only matters with --continuous-dynamic)
--seed N               (default 0)
--out PATH
```

---

## Wall-time budget

Approximate per-cell costs on 4×H100 + TP=4:

| operation | wall |
|--|--:|
| vLLM cold load 70B | ~3 min |
| 8B + TP=2 cold load | ~30 s |
| 1 prompt @ 4 K context, GSM8K-Hard CoT | ~2.3 s |
| 1 prompt @ 1.5 K context, HotpotQA short answer | ~0.1 s |
| 1 prompt @ 128 K context, RULER single-needle | ~18 s |
| Canary overhead at p_c=1/64 | +0.09% TPOT |
| Continuous-dynamic vs frozen (per cell) | within +1% wall |

n=500 mns=8 GSM8K-Hard ≈ 19 min measurement + 3 min load = ~22 min.
n=500 mns=8 HotpotQA ≈ 1 min measurement + 3 min load = ~4 min.
n=20 mns=1 RULER@128K ≈ 6 min measurement + 3 min load = ~9 min.

Multiply by the number of cells you're running.

---

## Where each result is used

| paper figure / claim | source JSON(s) | script |
|--|--|--|
| Figure 4 sensor validation | `smoke_absolute_canary_8b.json`, `echokv_3c_n100_mp_tau002_v2.json` | (already on disk; produced by the smoke + an early Phase-5 70B run) |
| Figure 5 GSM8K-Hard headline | `gsm8k_hard_n500.json`, `echokv3c_abscanary_tau022_n500_mns8.json` | Phase-2 + `scripts/03_*.sh` |
| Figure 6 panels (a)+(b) (workload-adaptive) | as above + `hotpotqa_n500_mns8/*` + `diag.json` + `echokv3c_abscanary_tau124_ruler128k_n20_mns1.json` | Phase-2 + `scripts/04_*.sh` + `scripts/06_*.sh` |
| Figure 6 panel (c) (τ(L) curve) | (annotation only — no JSON beyond τ values used) | (analytical) |
| §4.X τ₀ floor evidence | `echokv3c_abscanary_tau007_n500_mns8.json`, `hotpotqa_n500_mns8/echokv3c_abscanary_tau013_n500_mns8.json` | `scripts/07_*.sh` + Phase-8 follow-up |
| §6.X frozen vs dynamic | `*_continuous_dynamic.json` (all three workloads) | `scripts/08_*.sh`, `scripts/09_*.sh` |
| §6.6 overhead | (referenced from Sessions 8 + 13 measurements; no separate run for this paper) | (already on disk) |

---

## Known gaps

- **No INT8 cell for GSM8K-Hard at n=500 mns=8** (cosmetic — Open Q 26).
  Trivial fix; ~18 min via `01_int8_gsm8k_hard.sh` adapted from n=200.
- **n is at the threshold of significance** for the GSM8K-Hard mirror
  effect (p=0.052/0.061). n=1000 follow-up would resolve §6.7 #2 but
  is not currently required by any claim.
- **The √L exponent in §4.X is confirmed but not fitted** — RULER@128K
  saturates accuracy at INT4 already, so the test cannot distinguish
  L^0.5 from L^0.4 / L^0.6 (Open Q 28).

See [REPORT.md §Open Questions](REPORT.md) for the full list.
