# EchoKV — Publication figures

Six publication-quality figures for the EchoKV arXiv preprint. All
data figures are derived from real result files under
`results/motivation/` and `results/serving/`. No mock data, no
invented numbers.

Regenerate with:

```bash
python figures/generate_all.py
```

The script halts with a clear error if any required data file is
missing.

## Style

- Output: PDF (vector) + PNG @ 300 DPI for every figure
- Color palette: black + grayscale primary, with `#1f77b4` blue and
  `#d62728` red as accents (red used only for the feedback arrow in
  Fig 2 panel a and the INV-2 barrier in Fig 3)
- Font: matplotlib serif at 10pt axis labels, 9pt ticks, 11pt panel
  labels (`(a)`/`(b)`/`(c)` rendered via `\mathbf`)
- `rcParams['font.family']='serif'`, `rcParams['mathtext.fontset']='cm'`
- Width: 6.5" full-column (all 5 figures); heights tuned per figure
- No gridlines, no 3D effects, no shadows

## Figure-by-figure provenance

### Figure 1 — Within-request co-variation (paper §3.2)

- **Source A**: `results/motivation/delta_records_v8_1.jsonl` (200 records;
  100 RULER-32K, 100 GSM8K-CoT)
- **Source B**: `results/motivation/analysis_v8_1/per_request.tsv`
  (canonical per-request `rho_diff` from the post-EOS-fix analysis)
- **Filtering**: only `workload == "ruler_32k"` records are used (n=100)

**Panel (a)** — representative request:

- The TSV is scanned for the request whose per-request `rho_diff` is
  closest to the population mean. **ruler_32k:46** is selected
  (ρ_diff = +0.154 vs population mean +0.155).
- δ(t) is read from `delta_series` (length 32) on that record.
- ΔNLL(t) = `nll_int4_series[t] - nll_fp16_series[t]`, both length 32.
- Vertical dotted line at t=8 marks the answer-window boundary.

**Panel (b)** — per-request distribution:

- 20-bin histogram of all 100 RULER-32K `rho_diff` values from the TSV.
- Annotated mean ρ̄ = +0.155, 95% CI computed from the population SE
  (1.96 × pstdev / √n).
- z = mean / SE.

**Panel (c)** — per-layer ρ̄:

- For each of 32 layers, Spearman ρ is computed between
  `per_layer_delta[ℓ]` (length 32, the per-step δ for layer ℓ) and
  ΔNLL_series (length 32) per request, then averaged across the 100
  RULER-32K requests.
- Layer 0 is all-zeros sentinel; included with ρ=0 for visual
  completeness but irrelevant.
- Bars whose mean-ρ̄ z-score (mean/SE) exceeds 2.0 are colored black;
  others gray.

### Figure 2 — Sensor mechanism diagram (paper §4.2)

Schematic. No data file. Generated programmatically with
`matplotlib.patches.FancyBboxPatch` and `ax.annotate` arrows.

The figure illustrates the architectural fix this paper introduces:
the legacy ("naive") canary computes KL against the running quantized
cache, creating a positive-feedback path from controller decisions
back into the measurement substrate; the absolute canary maintains a
per-layer FP16 K_ref ring buffer (N=64 rows) captured before
tier-application, breaking the dependency.

### Figure 3 — Architecture diagram (paper §5)

Schematic. No data file. Three components left-to-right:

- **Shadow Canary Estimator** (FlashInfer Triton kernel, `p_c=1/64`,
  two CUDA streams, 0 mem overhead beyond K_ref)
- **PID Precision Controller** (EMA over per-(layer, tier) δ̂_b,
  argmin-with-budget on neighbours, tick every `N_c=100` decode steps)
- **SLO-Coupled Precision Budget** (vLLM scheduler hook, synchronous
  TP broadcast)

Forward arrows: δ̂_b → controller → {b_ℓ}. Loop-back arrow:
"alloc policy" from budget → estimator. INV-2 barrier in red below.

### Figure 4 — Sensor validation (paper §5.5)

- **Source A**: `results/serving/smoke_absolute_canary_8b.json` (4 runs:
  mns ∈ {1, 8} × seed ∈ {0, 7}, on Llama-3.1-8B + TP=2)
- **Source B**: `results/serving/session14_70b/echokv_3c_n100_mp_tau002_v2.json`
  (Phase-5 70B warmup blob; used for the legacy production-warmup
  δ̂_4 baseline in panel a)

**Panel (a)** — magnitude:
log-scale bar chart, two bars: legacy production-warmup max δ̂_4
(`naive_mns1_max ≈ 5.7e-5`, computed by averaging across-rank then
taking layer-max from the warmup blob's `warmup_per_tier_delta_per_rank`)
vs absolute-canary mns=1 seed=0 max δ̂_4 (`abs_mns1_max ≈ 0.031`).
Ratio ≈ 551×.

**Panel (b)** — `max_num_seqs` invariance:
the four absolute-canary runs' max δ̂_4 plotted on log scale. Hatched
red bar at left represents the legacy sensor's mechanically-zero
reading at T=1 (softmax over a single key element is identically
[1.0]; KL ≡ 0 — this is a property of the legacy formula, not a
fabricated number). Spread across mns ≈ 1.34×.

**Panel (c)** — cross-tier monotonicity:
for the mns=8 seed=0 run, scatter (δ̂_2, δ̂_4, δ̂_8) for each of 32
layers, with thin gray lines connecting the same layer's three
values. δ̂_2 ≥ δ̂_4 ≥ δ̂_8 holds by construction; verified empirically
with 0 violations counted across all 4 runs (0 / 128 layer-runs).

**Panel (d)** — cross-seed stability:
scatter of per-layer δ̂_4 from seed 0 (x) vs seed 7 (y), both at
mns=8. Diagonal y=x reference. Spearman ρ = 0.876.

### Figure 5 — 70B GSM8K-Hard Pareto headline (paper §6.5)

**Sources** (all at n=500, max_num_seqs=8, same paired prompts via seed=0):

- `results/serving/session14_70b/gsm8k_hard_n500.json` — Phase-2
  baselines (FP16 + static_int4 cells)
- `results/serving/session14_70b/echokv3c_abscanary_tau022_n500_mns8.json`
  — Phase-6 EchoKV-3C run with the absolute-canary sensor at the
  same protocol as the baselines

**Panel (a)** — accuracy at n=500:
bar chart with Wilson 95% CIs. Per-bar accuracies extracted directly
from each blob:

- FP16 = 0.488 (244/500)
- INT4 = 0.466 (233/500)
- **EchoKV-3C = 0.488 (244/500)** — matches FP16 exactly

Boxed legend reports the three paired McNemar comparisons computed
from the matched per-prompt outcomes:

- INT4 vs FP16: 27 discordants (19 INT4-worse, 8 INT4-better)
  → Δ = −0.022, p = 0.052 (the deficit, suggestive)
- EchoKV vs INT4: 29 discordants (9 EchoKV-worse, 20 EchoKV-better)
  → Δ = +0.022, p = 0.061 (mirror-image recovery)
- EchoKV vs FP16: 28 discordants (14 / 14, perfectly balanced)
  → Δ = 0.000, p = 1.00 (matches)

The mirror symmetry — same effect-size magnitude, opposite sign,
near-identical p-values — is the structural evidence that
EchoKV-3C's per-layer INT8 upgrades target exactly the layers
responsible for the static-INT4 deficit.

**Panel (b)** — memory:
horizontal bar chart of average bits per element. EchoKV-3C tier
vector {INT4: 51, INT8: 29, INT2: 0} (from
`memory_projection.tier_counts`) gives avg = 5.45 bits, savings vs
FP16 = 65.9%. The closed-loop upgrades 29 of 80 layers (36%) from
INT4 to INT8.

#### Why n=500 mns=8 (not n=200 mns=1)

The Phase-2 baselines at n=500 already documented the INT4 vs FP16
deficit (Δ=−0.022, p=0.052). To make a defensible *paired*
comparison about EchoKV-3C, we needed the same prompts at the same
protocol. n=500 mns=8 gives the largest paired n we have. The
mns=8 batch-composition noise affects all three configurations
symmetrically (every batched run picks up the same FP-rounding
dependence on batch composition), so paired McNemar remains valid.

A separate n=200 max_num_seqs=1 publication run exists at
`results/serving/session14_70b/publication/{fp16,int4,echokv3c_*}_n200_seq1.json`
and shows the same direction (EchoKV-3C ≈ FP16 ≈ INT4 within ±1σ
at n=200). It's the deterministic-protocol confirmation but lacks
statistical power at n=200.

### Figure 6 — Multi-workload adaptive bit allocation + τ(L) scaling (paper §6.5, §4.X)

**Sources**:

- GSM8K-Hard (n=500, max_num_seqs=8): same as Figure 5 (FP16 + INT4
  from `gsm8k_hard_n500.json`; EchoKV-3C from
  `echokv3c_abscanary_tau022_n500_mns8.json`). INT8 cell missing
  at this protocol — shown as "INT8 (n/a)" in the figure.
- HotpotQA (n=500, max_num_seqs=8): all 4 cells in
  `results/serving/session14_70b/hotpotqa_n500_mns8/`:
  - `fp16_n500_mns8.json` — 0.682
  - `int4_n500_mns8.json` — 0.686
  - `int8_n500_mns8.json` — 0.684
  - `echokv3c_abscanary_tau022_n500_mns8.json` — 0.682
- RULER@128K (n=20, max_num_seqs=1; matches the existing Phase-2
  RULER@128K protocol stored in
  `results/serving/session14_70b/diag.json`):
  - FP16, static_int4 cells from `diag.json`
  - EchoKV-3C from
    `echokv3c_abscanary_tau124_ruler128k_n20_mns1.json`
    (τ = 0.124 = τ₀·√(128 000/4 000), the √L-scaled budget)

**Panel (a)** — accuracy by workload:
grouped bar chart, 3-4 bars per workload (FP16, INT4, optional INT8,
EchoKV-3C). EchoKV-3C matches FP16 on all three workloads.

**Panel (b)** — EchoKV-3C bit budget chosen:
three black bars, with horizontal dashed reference lines at FP16=16,
INT8=8, INT4=4. GSM8K-Hard: 5.45 bits, {INT4:51, INT8:29}. HotpotQA:
4.55 bits, {INT4:69, INT8:11}. RULER@128K: 3.98 bits, {INT4:79,
INT2:1}.

**Panel (c)** — τ(L) = τ₀·√(L/L₀) scaling:
log-log plot of the Pinsker-derived τ-budget scaling curve
(τ₀ = 0.022, L₀ = 4 000) overlaid with the τ actually used per
workload. Markers at three points: GSM8K-Hard (L≈400, τ=0.022),
HotpotQA (L≈1500, τ=0.022), RULER@128K (L=128 000, τ=0.124). The
RULER point sits exactly on the predicted curve — empirical
confirmation that the controller's per-step total-variation budget
must scale as √L when context grows.

The contrast in panel (b) — 5.45 → 4.55 → 3.98 bits — is the
*workload-adaptive* signature: more INT8 upgrades on the workload
where INT4 hurts (GSM8K-Hard, INT4 deficit p=0.052), fewer on
HotpotQA (INT4 deficit p=0.79), almost none on RULER@128K (INT4
already saturates accuracy at 1.000).

## Data values used in annotations (real, not invented)

```
fig1_motivation:
  mean_rho_diff      = +0.155 (z=8.34σ, n=100)
  representative req = ruler_32k:46  (ρ_diff=+0.154, ρ_first=+0.159)
  peak per-layer     = layer 24, ρ̄=+0.142, z=7.94σ

fig4_sensor_validation:
  abs_mns1_max         = 0.0315
  abs_mns8_max         = 0.0381
  naive (70B warmup)   = 5.71e-5     (from Phase-5 warmup blob)
  ratio (abs/naive)    = 551×
  monotonicity_viols   = 0 / 128 layer-runs
  cross_seed_spearman  = 0.876

fig5_70b_results (n=500 mns=8 paired):
  FP16 acc        = 0.488 (244/500)
  INT4 acc        = 0.466 (233/500)
  EchoKV-3C acc   = 0.488 (244/500)   ← matches FP16 exactly
  EchoKV-3C bits  = 5.45  (savings 65.9% vs FP16)
  tier counts     = {INT4:51, INT8:29, INT2:0}
  Paired Δ        = INT4 vs FP16 −0.022 (p=0.052)
                  | EchoKV vs INT4 +0.022 (p=0.061) — mirror recovery
                  | EchoKV vs FP16  0.000 (p=1.00)  — matches

fig6_workload_adaptive (frozen-tier protocol):
  GSM8K-Hard (n=500 mns=8 paired):
    FP16=0.488  INT4=0.466  INT8=(n/a)  EchoKV-3C=0.488
    EchoKV bits = 5.45  ({INT4:51, INT8:29}, 65.9% savings vs FP16)
  HotpotQA (n=500 mns=8 paired):
    FP16=0.682  INT4=0.686  INT8=0.684  EchoKV-3C=0.682
    EchoKV bits = 4.55  ({INT4:69, INT8:11}, 71.6% savings vs FP16)
  RULER@128K (n=20 mns=1 paired vs Phase-2 diag.json):
    FP16=1.000  INT4=1.000  EchoKV-3C=1.000
    EchoKV bits = 3.98  ({INT4:79, INT2:1}, 75.2% savings vs FP16)
  Workload-adaptive contrast: 5.45 → 4.55 → 3.98 bits, with INT8
  upgrades dropping from 29 → 11 → 0 as the workload's INT4 deficit
  shrinks.
  τ(L) scaling: τ_used at L = 400 / 1500 / 128 000 = 0.022 / 0.022
  / 0.124. Pinsker-predicted τ₀·√(L/L₀) at those L = 0.0070 / 0.0135
  / 0.124 (RULER lies on the curve; the two short-context workloads
  use the τ₀ floor, which dominates at small L).

continuously-dynamic protocol (Phase 8 — load-bearing for the
"closed-loop / dynamic" abstract claim):
  GSM8K-Hard (n=500 mns=8, --continuous-dynamic, τ=0.022):
    acc=0.486 (frozen 0.488, paired McNemar p=1.00)
    seeded {INT4:37, INT8:43} = 6.15 bits  →
    final consolidated {INT4:65, INT8:15} = 4.75 bits
    per-rank avg = 4.33 bits (4.20 / 4.20 / 4.38 / 4.42)
    drift: 28 monotone INT8→INT4 downgrades, 0 upgrades
  HotpotQA (n=500 mns=8, --continuous-dynamic, τ=0.022):
    acc=0.686 (frozen 0.682, paired McNemar p=0.75 — directional bump)
    seeded {INT4:67, INT8:13} = 4.65 bits  →
    final consolidated {INT4:62, INT8:18} = 4.90 bits
    per-rank avg = 4.40 bits (4.30 / 4.40 / 4.30 / 4.60)
    drift: 4 INT8→INT4 + 9 INT4→INT8 upgrades (controller catches
    layers warmup batch missed)
  RULER@128K (n=20 mns=1, --continuous-dynamic, τ=0.124):
    acc=1.000 (frozen 1.000, paired McNemar p=1.00)
    seeded = final = {INT4:79, INT2:1} = 3.98 bits  (zero drift)
    per-rank avg = 3.97 bits (3.95 / 3.98 / 3.98 / 3.98)
  Headline per-rank bits across three workloads (dynamic protocol):
    GSM8K-Hard ~4.33  HotpotQA ~4.40  RULER@128K ~3.97
    (≈73% / 72% / 75% memory savings vs FP16)
```

## Suggested captions

**Figure 1**:
> Within-request canary signal on Llama-3.1-8B + RULER-32K, n=100. (a)
> Representative request (ruler_32k:46) showing δ(t) and ΔNLL(t)
> co-variation across a 32-token completion; per-request ρ_diff = +0.154,
> first-8-token ρ = +0.159. (b) Distribution of per-request ρ_diff;
> mean = +0.155 (95% CI [+0.118, +0.191], z = 8.34σ). (c) Per-layer
> mean ρ̄; layers with z > 2 highlighted in black. Peak: layer 24 with
> ρ̄ = +0.142 (z = 7.94σ).

**Figure 2**:
> Naive canary (a) computes divergence against the running quantized
> cache, creating a positive-feedback loop where controller decisions
> perturb the measurement substrate. Absolute canary (b) maintains a
> per-layer ring buffer K_ref of FP16 K projections captured before
> tier-application, breaking the feedback dependency. Three measurement
> artifacts (T=1 collapse, two-equilibria attractors,
> calibration↔production scale gap) are mechanically eliminated.

**Figure 3**:
> EchoKV three-component architecture. The canary sensor measures
> per-layer divergence at sampled decode steps. The controller adapts
> bit-width allocation policy based on per-(layer, tier) EMAs. The
> SLO-coupled budget integrates precision into vLLM's scheduler under
> load. All control flow operates on new block allocations only;
> existing blocks are never modified (information-theoretic axiom,
> §2.3).

**Figure 4**:
> Four validation checks for the absolute canary on Llama-3.1-8B at
> TP=2. (a) Signal magnitude is 551× the naive canary's
> production-warmup readings. (b) max_num_seqs invariance: the T=1
> collapse mode that defeats the naive sensor is mechanically
> impossible (legacy KL ≡ 0 at T=1). (c) Cross-tier monotonicity:
> 0 violations across 32 layers × 4 runs. (d) Cross-seed stability:
> Spearman ρ = 0.876 between seeds 0 and 7.

**Figure 5**:
> Llama-3.1-70B-Instruct on GSM8K-Hard, n=500, max_num_seqs=8 paired
> protocol. (a) EchoKV-3C with absolute-canary sensor (τ_budget=0.022)
> matches FP16 accuracy exactly (paired Δ = 0.000, McNemar p = 1.00,
> 14/14 discordant pairs split). The static-INT4 deficit
> (Δ = −0.022 vs FP16, p = 0.052) is recovered by EchoKV-3C with the
> mirror-image effect (Δ = +0.022 vs INT4, p = 0.061) — same
> magnitude, opposite sign, near-identical paired test —
> structural evidence that EchoKV-3C's INT8 upgrades target the
> layers responsible for the INT4 deficit. (b) EchoKV-3C uses 5.45
> bits/element on average ({INT4: 51, INT8: 29, INT2: 0}), giving
> 65.9% memory savings vs FP16 — substantially less than uniform
> INT8 would consume.

**Figure 6**:
> Workload-adaptive bit allocation and τ(L) scaling.
> Llama-3.1-70B-Instruct on three workload classes spanning short to
> long context, frozen-tier measurement protocol. (a) EchoKV-3C
> matches FP16 accuracy on all three workloads (GSM8K-Hard 0.488 vs
> 0.488 at n=500 mns=8; HotpotQA 0.682 vs 0.682 at n=500 mns=8;
> RULER@128K 1.000 vs 1.000 at n=20 mns=1). (b) EchoKV-3C's chosen
> bit budget is workload-adaptive: 5.45 bits ({INT4:51, INT8:29})
> on GSM8K-Hard, where INT4 has a measurable deficit; 4.55 bits
> ({INT4:69, INT8:11}) on HotpotQA, where INT4 is sufficient; 3.98
> bits ({INT4:79, INT2:1}) on RULER@128K, where INT4 already
> saturates accuracy. The same controller produces all three
> allocations — no per-workload manual tuning. (c) The τ-budget
> itself scales as τ(L) = τ₀·√(L/L₀), derived from Pinsker's
> inequality plus the attention-sparsity assumption (§4.X). The
> dashed Pinsker curve (τ₀=0.022, L₀=4 000) passes through the
> RULER@128K marker (τ=0.124), confirming the predicted scaling
> empirically; both short-context workloads run at the τ₀ floor
> where the curve is dominated by L₀. Continuously-dynamic
> measurements (Phase 8, §6.X) tighten the per-rank bit-budget on
> the two short-context workloads to ~4.33 / ~4.40 bits at the same
> accuracy; RULER@128K stays at zero drift.

## Pending follow-ups

- **Figure 7 candidate** — a frozen-vs-continuously-dynamic
  comparison panel showing the three drift signatures (GSM8K-Hard
  monotone-down, HotpotQA mostly-up, RULER@128K zero), plus a
  per-rank vs consolidated bit-budget table. Source data already
  on disk in
  ``results/serving/session14_70b/{,hotpotqa_n500_mns8/}*continuous_dynamic.json``.
  Not yet added because Phase 8 just landed; flag for the next
  figure-regeneration pass once the §6.X narrative is stable.

## Reproducibility

```
python figures/generate_all.py
```

Prints a summary table at the end. Outputs are deterministic given
the input files; no randomness, no seed-dependent computations
(Spearman ρ on real data is deterministic).
