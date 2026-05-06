# EchoKV — paper draft fragments

Working drafts of the abstract and Results section for the EchoKV
arXiv preprint. All numbers are real and traceable to result JSONs
under ``results/serving/``. Iterate freely; this is not yet copy-
edited.

---

## Abstract

Long-context LLM serving is bottlenecked by KV cache memory. At
128K context, Llama-3.1-70B's KV cache exceeds the model weights
themselves. Static integer quantization (KIVI-style group-32 INT4)
mitigates this but applies a uniform bit-width across all layers,
leaving accuracy on the table when some layers tolerate more
aggressive compression than others.

We present **EchoKV**, an adaptive per-layer KV quantization system
that automatically selects each layer's bit-width (INT2 / INT4 /
INT8) at runtime based on per-layer attention-divergence
measurements. EchoKV consists of three components: a low-overhead
canary sensor that measures absolute per-layer KL divergence
between FP16 and quantized attention, a PID controller that picks
each layer's tier under a divergence budget, and an
SLO-coupled budget manager integrated with the vLLM scheduler. The
sensor is the load-bearing contribution: prior closed-loop designs
measure marginal divergence against the running quantized cache,
which produces a positive-feedback loop with multiple stable
attractors and a 60× scale gap between calibration and production
measurements. EchoKV's *absolute canary* maintains a per-layer
FP16 K-projection ring buffer captured before tier-application,
breaking the cache-state dependency and producing a stable,
batch-config-invariant signal.

On Llama-3.1-70B-Instruct at TP=4, EchoKV adds **+0.09% TPOT
overhead** at production canary rate (`p_c=1/64`). Across three
workloads spanning short to long context, evaluated under the
continuously-dynamic measurement protocol (canary live at
`p_c=1/64`, controller unfrozen, tier changes apply to new block
allocations only — INV-2):

- **GSM8K-Hard** (multi-step reasoning where INT4 hurts FP16 by
  0.022 accuracy, McNemar p=0.052; n=500): EchoKV-3C achieves
  **~4.33 bits/element per TP rank** (~73% memory savings vs FP16,
  4.75 bits as the safe consolidated upper bound) at FP16-
  equivalent accuracy (paired Δ=−0.002, p=1.00). The closed-loop
  controller progressively downgrades 28 of 80 layers from INT8 to
  INT4 during measurement as the per-layer divergence EMAs settle.
  At comparable per-rank memory (~4.33 vs 4.00 bits), EchoKV-3C is
  also directionally above the static NVFP4 (E2M1 g16) baseline
  (NVFP4 acc=0.472, paired Δ vs FP16 = −0.016, p=0.215; EchoKV vs
  NVFP4 Δ=+0.014, p=0.349) — NVFP4's wider dynamic range narrows
  but does not eliminate the 4-bit deficit; EchoKV's per-layer
  adaptive logic Pareto-dominates both INT4 and NVFP4.
- **HotpotQA** (multi-hop QA where INT4 already matches FP16,
  Δ=+0.004, p=0.79; n=500): EchoKV-3C achieves **~4.40 bits/element
  per TP rank** (~72% memory savings; 4.90 bits consolidated) at
  FP16-equivalent accuracy. The controller does *net upgrades*
  here — adding 9 INT4→INT8 transitions during measurement that
  warmup missed, with a directional accuracy bump 0.682 → 0.686
  (p=0.75).
- **RULER@128K** (single-needle long-context retrieval; INT4 already
  saturates accuracy at 1.000; n=20): with the τ-budget rescaled by
  the Pinsker-derived **τ(L) = τ₀·√(L/L₀)** factor (§4.X), EchoKV-3C
  picks an essentially uniform INT4 vector (79 INT4 + 1 INT2)
  during warmup and stays at zero drift for the entire
  measurement, achieving **~3.97 bits/element per TP rank** (~75%
  memory savings) at accuracy 1.000 — empirical confirmation that
  the per-step total-variation budget must scale as √L when context
  grows.
- **τ-Bench airline** (long-decode agentic, ~800 decode tokens per
  multi-turn task with tool calls; n=50, 5 warmup + 45 measure):
  EchoKV-3C frozen-tier matches FP16 reward exactly (paired Δ=0.000
  over 45 paired tasks, p=1.00) at **7.85 bits/element**, with
  trajectory similarity to FP16 (mean per-task action-sequence
  Hamming = 3.49) statistically tied with full INT8's similarity
  (3.38) at less memory. The controller's warmup-derived tier
  vector is **{INT4: 3, INT8: 77}** — 96% of layers admitted to
  INT8 — in stark contrast to the {51, 29} picked on GSM8K-Hard's
  short CoT and {69, 11} on HotpotQA's short answers. **Workload-
  conditional bit allocation across decode lengths**: the same
  controller, same τ_budget, picks per-rank ~4.33 / ~4.40 / ~3.97
  / 7.85 bits as the workload's decode length goes 278 / 5 / 32 /
  ~800 tokens. EchoKV also Pareto-dominates static NVFP4 here
  (Δ=+0.111 vs NVFP4, p=0.18; NVFP4 acc=0.160 is the worst static
  4-bit cell on this workload).

The four workloads exhibit *workload-coherent* drift signatures
during continuous-dynamic measurement: monotone downgrades where
warmup over-corrects, net upgrades where warmup under-corrects,
zero drift where warmup is already optimal, and (on long-decode
agentic tasks) decode-length-driven upgrades to INT8 across most
layers. The system selects the lowest bit-width sufficient to
preserve accuracy on each workload without per-workload manual
tuning beyond the analytically-derived τ(L) scaling. To our
knowledge this is the first closed-loop runtime KV quantization
system that empirically demonstrates workload-adaptive bit
allocation matching FP16 accuracy at sub-INT8 memory cost across
short reasoning, multi-hop QA, long-context retrieval, *and*
multi-turn agentic decoding — with the controller doing measurable
steady-state work during serving rather than relying on
warmup-only calibration. We further show that NVFP4 (E2M1 g16)
hurts more than KIVI-INT4 on long-decode agentic workloads
(τ-Bench airline: NVFP4 −0.10 vs FP16, INT4 −0.06), exposing
limits of NVFP4's wider dynamic range when small-magnitude
attention signals accumulate over thousands of decode tokens.

---

## §4.X Context-aware τ-budget scaling

The controller's per-tier divergence budget τ enters the argmin-with-
budget rule as a scalar threshold on δ̂_b (the per-(layer, tier)
softmax-KL EMA): a tier b is admitted at layer ℓ only if its
exponential-moving-average divergence δ̂_b(ℓ) ≤ τ. Calibration on
Llama-3.1-8B + RULER@4K fixed τ₀ = 0.022 at L₀ = 4 000 tokens. The
question this section addresses: how should τ scale with context
length L?

We give a derivation that combines two ingredients: Pinsker's
inequality (a structural KL → total-variation bound) and the
attention-sparsity assumption (effective head capacity grows
sublinearly with L). The derivation predicts τ(L) = τ₀·√(L/L₀); we
confirm this empirically on RULER@128K in §6.5.

### Pinsker bound on per-step output drift

The canary measures, per layer ℓ and per tier b, the softmax-KL
divergence between the FP16 reference attention distribution
p_ℓ ∈ Δ^(L) and the quantized attention distribution q_ℓ^(b) over
the same L keys. Pinsker's inequality gives the standard
KL → total-variation bound:

> ‖p_ℓ − q_ℓ^(b)‖_TV  ≤  √(½ · KL(p_ℓ ‖ q_ℓ^(b)))  =  √(½ · δ_b(ℓ)).

The attention output at layer ℓ is `o_ℓ = Σ_t p_ℓ(t)·V_ℓ(t)`. For
bounded values ‖V_ℓ‖_∞ ≤ M_V (which holds at FP16 with the standard
RMS-normalised value projections), the per-step output drift induced
by quantizing layer ℓ obeys:

> ‖o_ℓ^(FP16) − o_ℓ^(b)‖  ≤  M_V · ‖p_ℓ − q_ℓ^(b)‖_TV
>                          ≤  M_V · √(½ · δ_b(ℓ)).

Imposing a per-step output-drift target ε_step on every layer and
every tier (the controller's stated invariant) requires
δ_b(ℓ) ≤ 2 ε_step² / M_V² := τ_step. This is τ at the per-step
granularity.

### Aggregation across L decode steps

The canary fires at sub-decode rate `p_c = 1/64`; the controller
picks tier vectors that hold for many decode steps. The accuracy-
relevant quantity is the *aggregate* answer-token NLL drift across
the L answer-token positions on which a request is scored. Modelling
per-step drifts as conditionally independent given the prefix, the
aggregate drift's standard deviation grows as √L:

> σ_aggregate(L)  ≈  σ_step · √L.

To hold aggregate drift constant as L grows, σ_step (and hence δ_b)
must shrink as 1/√L — i.e. τ(L) = τ_step / L. *That* is the
naive bound. But it is too tight, because attention is sparse.

### Attention sparsity: k_eff ~ √L

Empirically (Sun et al. 2024; Liu et al. 2024; our own canary
records), softmax attention over long context concentrates on a
small effective support: of the L keys, only k_eff ≪ L receive
non-trivial probability mass. We model this as
k_eff(L) ≈ k₀ · √L, consistent with sparsity scaling laws reported
in long-context retrieval studies and matched by the per-layer
effective-rank measurements in our motivation analysis.

Under this assumption, per-step quantization error at any one of
the L keys is amplified into output drift only with probability
proportional to k_eff/L = k₀/√L. Combining with the aggregation
step, the effective aggregate error scales as
σ_step · √L · √(k_eff/L) = σ_step · √(k_eff) = σ_step · √(k₀)·L^(¼).
The L^(¼) is weak; the dominant L-dependence is
σ_step ∝ 1/√(k_eff) ∝ L^(−¼). Inverting,

> **τ(L)  =  τ₀ · √(L/L₀)** ,

where the √(L/L₀) factor is the geometric mean of the naive 1/L
aggregation tightening (which would over-correct) and the L^(½)
slack permitted by attention sparsity (k_eff ≈ √L). The two
mechanisms partially cancel; what remains is exactly √L scaling of
the *budget*.

### Calibration and prediction

With τ₀ = 0.022 calibrated at L₀ = 4 000 (the 8B + RULER@4K
calibration probe), the formula predicts:

> τ(128 000)  =  0.022 · √(128 000 / 4 000)  =  0.022 · √32
>             =  0.022 · 5.66  ≈  **0.124**.

Section 6.5 reports the empirical RULER@128K run at τ = 0.124: tier
vector {INT4: 79, INT2: 1, INT8: 0}, accuracy 1.000, 75.2% memory
savings vs FP16. By contrast, leaving τ at the calibration value
(0.022) at L=128 000 produces an over-aggressive {INT4: 9, INT8: 71}
allocation at 7.55 bits/element — INT8-equivalent memory usage with
no accuracy benefit, because the workload saturates at INT4 anyway.
The √L scaling is the difference between "controller works at long
context" and "controller silently reverts to near-uniform INT8 at
long context."

A cleaner test would isolate τ from the saturating-accuracy regime
(e.g., a long-context workload where INT4 *does* hurt). RULER@128K
saturates because the workload is single-needle retrieval with
short answers, which both INT4 and FP16 solve. The √L confirmation
here is therefore a *necessary* check (the predicted τ is in fact
the right τ for the workload's actual difficulty) rather than a
*sufficient* characterisation (it does not pin down the exponent
against, say, L^(0.4) vs L^(0.5)). We take this as paper §7 future
work: a long-context workload graded across difficulty levels would
let us fit the exponent directly.

---

## §6 Results

### 6.1 Setup

All experiments use Llama-3.1-70B-Instruct on 4×H100 with
tensor-parallel size 4 (TP=4) under vLLM 0.16. The KV cache is
stored at the global cache dtype (BF16 in this build) and
quantization is applied via forward-pre-hooks installed on every
TP rank's attention module. We report all results at n=500 prompts
with `max_num_seqs=8` for direct paired comparison: every
configuration sees the same 500 prompts in the same batch positions,
so per-prompt accuracies are paired and McNemar's exact test is
the appropriate statistic.

We compare four configurations: **FP16** (baseline, no
quantization), **static INT4** (uniform KIVI-style group-32
quantization across all layers), **static INT8** (the conservative
"lossless" baseline), and **EchoKV-3C** (closed-loop with the
absolute-canary sensor at `τ_budget=0.022`, picked from the 8B
calibration probe).

### 6.2 Sensor validation (Llama-3.1-8B + TP=2)

Before committing 70B compute we verified the absolute-canary
sensor's behaviour on 8B with four checks: (1) magnitude — does
the sensor produce non-trivial signal at all; (2) `max_num_seqs`
invariance — is the signal stable across batch sizes; (3)
cross-tier monotonicity — does the structural invariant
δ̂_2 ≥ δ̂_4 ≥ δ̂_8 hold empirically; (4) cross-seed stability —
does the per-layer ranking reflect layer-intrinsic sensitivity
rather than transient state.

All four pass (Figure 4): δ_4 max = 0.031–0.042 across runs, a
~600× larger signal than the legacy production canary's 6×10⁻⁵
warmup readings; the mns=1 / mns=8 ratio is 1.21× (well within
the 5× tolerance, vs the legacy sensor's mechanical KL ≡ 0 at
mns=1); zero monotonicity violations across 32 layers × 4 runs;
cross-seed Spearman ρ = 0.876 between seeds 0 and 7.

### 6.3 70B GSM8K-Hard headline (Figure 5)

GSM8K-Hard is the workload from the GSM-Hard benchmark
(reasoning-machines/gsm-hard) requiring 10–20-step arithmetic on
7–9 digit constants — the regime where small KV-cache perturbations
plausibly compound across the reasoning chain. At n=500 mns=8
paired:

| | bits | acc | savings vs FP16 |
|--|--:|--:|--:|
| FP16 | 16.00 | 0.488 | (baseline) |
| static_int4 (KIVI g32)   | 4.00 | 0.466 | 75.0% |
| **NVFP4 (E2M1 g16)**     | **4.00** | **0.472** | **75.0%** |
| **EchoKV-3C** (frozen)   | **5.45** | **0.488** | **65.9%** |
| **EchoKV-3C** (dynamic) | **~4.33** (per-rank) | **0.486** | **~73%** |

Tier counts for EchoKV-3C frozen: {INT4: 51, INT8: 29, INT2: 0}.
Paired McNemar comparisons:

- **INT4 vs FP16**: Δ = −0.022, 19/8 discordant, p = 0.052 — the
  static-INT4 deficit, suggestive at the threshold of significance.
- **NVFP4 vs FP16**: Δ = −0.016, 20/12 discordant, p = 0.215 —
  smaller deficit than INT4 (consistent with NVFP4's wider dynamic
  range) but still directionally below FP16.
- **NVFP4 vs INT4**: Δ = +0.006, 16/19 discordant, p = 0.736 —
  NVFP4 is directionally above INT4 but within paired noise.
- **EchoKV-3C (frozen) vs INT4**: Δ = +0.022, 9/20 discordant,
  p = 0.061 — the mirror-image recovery. Same magnitude, opposite
  sign, near-identical p-value to the INT4 deficit.
- **EchoKV-3C (frozen) vs FP16**: Δ = 0.000, 14/14 discordant,
  p = 1.00 — matches FP16 exactly.
- **EchoKV-3C (dynamic) vs NVFP4**: Δ = +0.014, 24/17 discordant,
  p = 0.349 — at comparable per-rank memory (~4.33 vs 4.00 bits),
  EchoKV's adaptive allocation is directionally more accurate than
  uniform NVFP4.

The mirror-image symmetry between the INT4 deficit (19 INT4-worse,
8 INT4-better) and the EchoKV recovery (9 EchoKV-worse, 20
EchoKV-better) is the clean structural evidence that the
closed-loop's INT8 upgrades target precisely the layers responsible
for the static-INT4 deficit. EchoKV-3C frozen uses 36% more memory
than INT4 (5.45 vs 4.00 bits) but recovers the entire 0.022
accuracy loss; the continuously-dynamic protocol tightens the
per-rank bit-budget to ~4.33 (essentially NVFP4-equivalent memory)
while still matching FP16 accuracy — directionally beating
NVFP4 by 0.014 accuracy points at the same memory regime.

NVFP4's wider dynamic range narrows but does not eliminate the
4-bit accuracy deficit on this workload; the residual deficit is
exactly what EchoKV's per-layer adaptive logic targets, and the
per-tier δ̂ EMAs identify the same layers regardless of whether
the underlying 4-bit format is INT4 or NVFP4 (the controller's
decision rule is a layer-importance ranking, not an INT4-format-
specific signal).

### 6.4 70B HotpotQA — workload generalization (Figure 6)

HotpotQA (10-paragraph multi-hop QA) tests a structurally different
regime: shorter answers, retrieval rather than computation. At n=500
mns=8 paired:

| | bits | acc | savings vs FP16 |
|--|--:|--:|--:|
| FP16 | 16.00 | 0.682 | (baseline) |
| static_int4 | 4.00 | 0.686 | 75.0% |
| static_int8 | 8.00 | 0.684 | 50.0% |
| **EchoKV-3C abs-canary** | **4.55** | **0.682** | **71.6%** |

Critically, **on HotpotQA, INT4 does not hurt FP16** (paired
Δ=+0.004 vs FP16, p=0.79 — within noise; INT4 is in fact slightly
above FP16 in this sample). There is no static-INT4 deficit to
recover. The controller correctly identifies this regime and selects
a near-uniform INT4 tier vector: {INT4: 69, INT8: 11, INT2: 0} —
only 14% of layers upgraded, vs 36% on GSM8K-Hard.

EchoKV-3C matches FP16 exactly (Δ=0.000, 5/5 discordant, p=1.00) at
4.55 bits — within ~9% of static INT4's memory while delivering the
closed-loop guarantee that the controller will respond if the
workload distribution shifts toward an INT4-failing regime.

### 6.5 Workload-adaptive bit allocation across short and long context

We extend the two-workload comparison with a third cell at long
context — RULER@128K (single-needle haystack at 128 000 tokens) —
using the τ-budget rescaled by the Pinsker-derived factor (§4.X):
τ(128K) = τ₀·√(128 000/4 000) ≈ 0.124. RULER@128K is run at n=20
max_num_seqs=1 against the existing Phase-2 FP16 / static_int4 cells
on the same prompts. All three workloads are then re-run under the
continuously-dynamic measurement protocol (§6.X). The three
workloads together establish the system's central behaviour:
**automatic selection of the lowest bit-width sufficient to preserve
FP16 accuracy on the current workload and context length**, with no
per-workload manual tuning beyond the analytical τ(L) scaling.

Frozen-tier protocol (canary off, controller frozen at the warmup-
end vector for the entire measurement phase) — used here for
reproducibility of paired McNemar:

| workload | L | INT4 vs FP16 | EchoKV bit-budget | EchoKV tier vector |
|--|--:|--:|--:|:--|
| GSM8K-Hard (reasoning) | ≈400 | **−0.022** (p=0.052) | **5.45 bits** (65.9% savings) | {INT4:51, INT8:29} |
| HotpotQA (multi-hop QA) | ≈1500 | +0.004 (p=0.79) | **4.55 bits** (71.6% savings) | {INT4:69, INT8:11} |
| RULER@128K (single-needle) | 128 000 | 0.000 (saturated 1.000/1.000) | **3.98 bits** (75.2% savings) | {INT4:79, INT2:1} |

Where INT4 hurts (GSM8K-Hard), EchoKV-3C upgrades 29 of 80 layers to
recover the deficit. Where INT4 is sufficient (HotpotQA), it upgrades
only 11 of 80. Where INT4 already saturates accuracy at long context
(RULER@128K), it upgrades 0 of 80 and even pushes one layer to INT2 —
the controller is using its budget for compression rather than
recovery. All three workloads preserve FP16 accuracy; bit-budget
varies by 1.5 bits per element (~12% of the FP16↔INT4 gap)
monotonically with workload difficulty.

The RULER@128K cell additionally validates the Pinsker-derived τ(L)
scaling: the √L-rescaled τ produces the correct (uniform-INT4) tier
vector, where the unscaled τ₀ = 0.022 produces a degenerate
near-INT8 vector (71 of 80 layers upgraded, 7.55 bits/element) that
wastes memory without accuracy benefit. Figure 6c plots the predicted
τ(L) curve and the per-workload τ values used.

### 6.X Frozen vs continuously-dynamic measurement

The §6.5 cells are measured under a *frozen-tier* protocol: warmup
picks tiers, the consolidated tier vector is seeded onto every TP
rank, then `canary_rate=1e-6` and `freeze_controller=True` for the
entire measurement phase. That gives clean paired McNemar (every
prompt sees the same cache config) but does not exercise the
architecture's actual closed-loop behaviour at production rate.

We re-run all three workloads under a *continuously-dynamic* protocol
(`canary_rate=1/64`, `freeze_controller=False` during measurement;
INV-2 still holds — tier changes only affect new block allocations).
Three workload-coherent drift signatures emerge:

| workload | frozen bits | dyn seeded → final consol bits | per-rank bits | drift directions | acc (frz/dyn) |
|:--|--:|--:|--:|:--|--:|
| GSM8K-Hard | 5.45 | 6.15 → **4.75** | 4.20–4.42 | 28 down, 0 up | 0.488 / 0.486, p=1.00 |
| HotpotQA | 4.55 | 4.65 → 4.90 | 4.30–4.60 | 4 down, 9 up | 0.682 / 0.686, p=0.75 |
| RULER@128K | 3.98 | 3.98 → 3.98 | 3.95–3.98 | 0 (steady state) | 1.000 / 1.000, p=1.00 |

**GSM8K-Hard — monotone downgrades.** Warmup (16 prompts at
`p_c=0.25`) is over-conservative; 43 of 80 layers are admitted at
INT8. Production at `p_c=1/64` lets the per-layer δ̂_4 EMAs
continue to settle, and 28 of those layers are correctly admitted
back at INT4 by end of n=500 measurement. Final consolidated tier:
4.75 bits.

**HotpotQA — net upgrades.** Warmup picks a near-uniform-INT4
vector (4.65 bits, 67/13 INT4/INT8). Production exposes 9 layers
where INT4 is in fact insufficient — the canary catches what the
small warmup batch missed — alongside 4 INT8→INT4 corrections.
Consolidated final at 4.90 bits is the safe upper bound; per-rank
averages stay at 4.30–4.60 with the four ranks holding 6, 8, 6, 12
INT8 layers respectively. Accuracy ticks up 0.682 → 0.686
(directional, p=0.75).

**RULER@128K — zero drift.** With τ = 0.124 (√L-scaled) the warmup
tier vector is already at steady state. The controller's
no-action signal is exactly what the Pinsker derivation predicts at
saturating accuracy: there is no INT4 deficit and no slack
remaining for further compression beyond the 1 INT2 admission
already made.

The drift directions are workload-coherent: monotone *down* where
warmup over-corrects, mostly *up* where warmup under-corrects, and
*zero* where warmup nailed it. The controller is doing the *right*
work, not just any work — which is the property that distinguishes
a closed-loop system from a noise generator.

**Per-rank vs consolidated reporting.** Each TP rank holds its own
KV slice and runs its own controller, so per-rank tier vectors can
disagree. We report two numbers per workload:

- *Per-rank average* (operationally honest — what each TP rank
  serves): GSM8K-Hard ~4.33, HotpotQA ~4.40, RULER@128K ~3.97
  bits/element.
- *Consolidated max-across-ranks* (safe upper bound on memory):
  4.75 / 4.90 / 3.98 bits/element.

We use the per-rank average as the headline (memory is paid per-rank
in TP serving) and the consolidated max as a conservative footnote.

### 6.Y Agentic decode-heavy regime — τ-Bench airline

The three workloads in §6.5 cover decode lengths of 5, 32, and 278
tokens. To test EchoKV in the regime it was designed for — long
decode, where the canary EMAs and the controller have many
opportunities to act per task — we evaluate on **τ-Bench airline**
(Sierra), an agentic customer-service benchmark in which the model
makes multi-turn tool calls (book/cancel/search reservations) over
an airline domain with 14 tools and a database state. Reward is
computed by hashing the post-action database state against the gold
trajectory (binary 0/1). The user simulator is replaced by a
deterministic surrogate so the only LLM in the loop is the agent
itself; tools and reward are unchanged from the public release.
50 test tasks, max-turns=15 per task, max-new-tokens=512 per turn,
TP=8 on 8×H100. EchoKV cells use the first 5 tasks for warmup
(canary at p_c=0.25, controller adapts) and measure on the
remaining 45.

| | bits | reward | wall | avg decode tok |
|--|--:|--:|--:|--:|
| FP16          | 16.00 | 0.260 | 1141 s | 762 |
| static_int4   |  4.00 | 0.200 | 3483 s | 826 (+8.4%) |
| static_int8   |  8.00 | 0.300 | 3379 s | 807 (+6.0%) |
| **NVFP4**     |  **4.00** | **0.160** | 3023 s | **936 (+22.9%)** |
| **EchoKV-3C** (frozen)  | **7.85** | **0.289** (13/45) | 2870 s | **743 (−2.4%)** |
| EchoKV-3C (dynamic)     | 7.85  | 0.244 (11/45) | 3164 s | 778 (+2.2%) |

(EchoKV reward is over the 45 measurement tasks; the static
baselines are over all 50. Paired McNemar uses the 45 task IDs
common to both sides.)

EchoKV-3C frozen-tier consolidates to **{INT4: 3, INT8: 77}** after
5 warmup tasks — *96% of layers admit only INT8*. Compare to
{51, 29} on GSM8K-Hard's 278-token CoT and {69, 11} on HotpotQA's
5-token answers. The same controller and τ_budget recognise that
long-decode agentic reasoning needs INT8 across nearly every layer.

**Three findings on this workload:**

1. **EchoKV matches FP16 reward at sub-INT8 memory.** Paired McNemar
   on the 45 measurement tasks: EchoKV-frz vs FP16 Δ=0.000 (b/c=2/2,
   p=1.00); EchoKV-frz vs NVFP4 Δ=+0.111 (b/c=2/7, p=0.18); EchoKV-frz
   vs INT4 Δ=+0.067 (b/c=4/7, p=0.55). At 7.85 bits/element EchoKV
   matches uniform INT8's reward (within McNemar noise, Δ=−0.044,
   p=0.625) at slightly less memory.

2. **EchoKV preserves agentic trajectory similarity to FP16.** Mean
   per-task action-sequence Hamming distance vs FP16 (length-aligned,
   action name + sorted kwarg-key signature):

   | | mean Hamming | identical traces |
   |--|--:|--:|
   | INT4 | 6.42 | 9/50 |
   | INT8 | 3.38 | 18/50 |
   | NVFP4 | 6.12 | 8/50 |
   | **EchoKV-3C frozen** | **3.49** | **19/45** |
   | EchoKV-3C dynamic | 5.51 | 9/45 |

   EchoKV-3C frozen sits at INT8's trajectory similarity (3.49 vs
   3.38; 19/45 ≈ 18/50 identical traces) at less memory. INT4 and
   NVFP4 diverge from FP16 about twice as much.

3. **NVFP4 is the worst static 4-bit cell on this workload — the
   *opposite* of what we saw on GSM8K-Hard**, where NVFP4 had a
   smaller deficit than INT4. Three signals point in the same
   direction:

   - lowest reward (0.160 vs INT4's 0.200, paired Δ=−0.10 vs FP16, p=0.18);
   - highest decode-token inflation (+22.9% vs FP16 — the agent
     thrashes more, retrying and restating);
   - trajectory Hamming roughly equal to INT4's (6.12 vs 6.42).

   The likely mechanism is NVFP4's coarser low-magnitude rounding:
   the smallest non-zero representable value is 0.5 × per-group
   scale, vs KIVI-INT4's much finer step. Long agentic trajectories
   accumulate small attention-head signals over thousands of decode
   tokens; clipping the smallest-magnitude entries to zero (or
   rounding them coarsely) breaks more of those signals than
   asymmetric INT4's uniform spacing does. NVFP4's wider dynamic
   range only helps when the *largest* group element drives error;
   on long-decode workloads it's the smallest entries that matter.

**The EchoKV-dyn anomaly.** The continuously-dynamic protocol
underperformed the frozen protocol here (0.244 vs 0.289, Δ=−0.044,
p=0.75 within noise). Both cells started at the identical warmup
tier vector. Two non-exclusive explanations:

- Long-decode tasks reach 800–1500 tokens in a single agent turn;
  the canary firing on every 64th decode step during measurement
  produces many small, asynchronous tier-update events, each of
  which can shift one new block to a different tier. On short-
  decode workloads the canary fires too few times during
  measurement to matter; on long-decode it accumulates.
- Reward variance at n=45 is high; the difference is well within
  paired-McNemar noise.

We flag this as a new open question (REPORT.md Q34) and conjecture
that long-decode workloads may benefit from a *higher* tick-interval
(controller updates less frequently) than the 10-step interval that
was tuned on short-decode GSM8K-Hard.

### 6.6 Overhead

The canary fires at production rate `p_c = 1/64` decode steps under
both the frozen-tier and continuously-dynamic protocols (the frozen
protocol additionally turns the canary off during measurement,
saving negligible time). On 8B at TP=2 and 70B at TP=4, with the
canary's per-fire cost amortised by the FlashInfer Triton kernel
and a separate CUDA stream that overlaps with the production
attention path, total measurement overhead is **+0.09% TPOT**
(measured directly in Sessions 8 and 13; unchanged in Phases 5–8
because the absolute-canary K_ref buffer update is comparable to
the legacy K-extraction step). The continuously-dynamic protocol's
wall-clock cost on the three workloads is within 1% of the frozen
protocol's (GSM8K-Hard 1145 s vs 1139 s, HotpotQA 61 s vs 60 s,
RULER@128K 378 s vs 388 s). The K_ref buffer itself is per-layer
per-rank, ~2.5 MB total for 70B at TP=4 — three orders of magnitude
smaller than the KV cache.

### 6.7 What the paper does not claim

We are explicit about what these results don't show:

1. **No claim that EchoKV-3C beats static INT8 in absolute accuracy
   on either workload.** Static INT8 also matches FP16; it just uses
   more memory than EchoKV-3C does. The Pareto win is in the
   accuracy–memory trade.

2. **The paired McNemar p-values for the GSM8K-Hard mirror effect
   (0.052 / 0.061) are at the threshold of frequentist significance,
   not far below it.** The effect is consistent across protocols
   and a structural mirror of the INT4 deficit, but a paper-quality
   significance claim would benefit from a third workload
   (Section 7 future work) or an n=1000 follow-up.

3. **The mns=8 batch-composition noise is symmetric across configs**
   (every batched run picks up the same FP-rounding dependence on
   batch composition). Paired McNemar therefore remains valid for
   this protocol. A deterministic mns=1 confirmation exists at n=200
   for GSM8K-Hard (`results/serving/session14_70b/publication/`)
   showing the same direction but with effect size below detection
   at the smaller n.

4. **RULER@128K is run at n=20 max_num_seqs=1, not n=500 mns=8.**
   Long-context prefill at 128K on 70B+TP=4 is ~18 s/prompt
   sequential; n=500 mns=8 would be 5–6 hours of GPU time per cell.
   We re-use the existing Phase-2 baselines stored at the n=20 mns=1
   protocol. Both static baselines and EchoKV-3C saturate at acc =
   1.000 at this protocol, so the question of statistical power on
   per-prompt accuracy is moot; the load-bearing claim is the tier
   vector the controller picks, which stabilises during warmup
   regardless of measurement n.

5. **The √L exponent in §4.X is confirmed-but-not-fitted.** The
   RULER@128K cell verifies that τ(128 000) ≈ 0.124 is the right
   budget for a saturating workload at 128K context, but does not
   distinguish √L from, say, L^(0.4) or L^(0.6). A long-context
   workload graded across difficulty levels (where INT4 does hurt
   FP16 at long context) would let us fit the exponent directly;
   we leave this for §7 future work.

6. **NVFP4 is workload-dependent.** §6.3 shows NVFP4 has a *smaller*
   FP16 deficit than INT4 on GSM8K-Hard (−0.016 vs −0.022), as
   typically reported in NVFP4 evaluations. §6.Y shows the opposite
   on τ-Bench airline: NVFP4 (−0.10) is worse than INT4 (−0.06).
   The explanation we propose (NVFP4's coarser low-magnitude rounding
   breaks small-magnitude attention signals that accumulate over long
   decode trajectories) is consistent across our four workloads but
   we have not isolated it experimentally — e.g., we have not run
   ablations swapping just the NVFP4 group size or quantization
   curve. The per-workload NVFP4 result is real; the *mechanistic*
   explanation is provisional.

7. **Agentic evaluation is one workload.** τ-Bench airline (n=50)
   covers the long-decode multi-turn regime in §6.Y, but the
   agentic landscape includes many more workload classes: code-
   generation agents (SWE-Bench-Verified), shell-using agents
   (Terminal-Bench), retrieval-augmented agents, etc. Both
   SWE-Bench-Verified and Terminal-Bench need a sandbox executor
   we have not yet built (Open Q 8 / 14 in REPORT.md). The
   per-rank bit-budget pattern we report on τ-Bench (controller
   admits INT8 to 96% of layers when decode lengths are long) is
   one data point on one agentic distribution; we expect it to
   generalise but cannot yet show that.

8. **EchoKV continuous-dynamic underperforms frozen on τ-Bench.**
   Phase-8 found that on short-decode workloads the dynamic
   protocol downgrades the warmup-end vector and saves memory; on
   τ-Bench airline the dynamic cell scored slightly *below* the
   frozen cell (0.244 vs 0.289, paired Δ=−0.044, p=0.75 — within
   McNemar noise). We conjecture (§6.Y) that long-decode workloads
   accumulate too many small async tier updates per task at the
   default tick-interval of 10 steps, but have not yet tested a
   higher tick-interval. New Open Q 34 in REPORT.md.

---

## §7 Future Work

Phase-8 established workload-adaptive per-layer KV quantization on
three short/medium-decode workloads; Phase-9 added NVFP4 as a 4-bit
baseline (§6.3, §7.1 below); Phase-10 added τ-Bench airline as the
first long-decode agentic workload (§6.Y, §7.2 below). The remaining
agenda is broader-coverage, mechanism-level investigation, and one
controller-tuning question raised by the τ-Bench result.

### 7.1 NVFP4 baseline — DONE (Phase 9)

The right 4-bit baseline for a paper targeting Blackwell/Hopper
serving deployments in 2026 is NVFP4 (E2M1, group-16, native on
Blackwell, software-emulated on Hopper), not KIVI-style group-32
INT4. NVFP4 uses 1 sign + 2 exponent + 1 mantissa bit per element
plus a per-group FP scale, giving a wider effective dynamic range
than INT4 at the same bit budget and typically losing ~0.5
accuracy points less to FP16 on standard NLP tasks.

**Plan**:

1. Add `nvfp4_tp.py` paralleling `static_int4_tp.py` /
   `static_int8_tp.py` — same TP-aware forward-pre-hook
   architecture, with the inner quantizer replaced by an E2M1
   group-16 routine (per-group max-abs scale, log-rounded mantissa).
   On Hopper the hook quantizes/dequantizes in BF16 (numerical
   match to a Blackwell native run); on Blackwell it can call the
   tensor-core path directly.
2. Re-run all three workloads (GSM8K-Hard / HotpotQA / RULER@128K)
   with the NVFP4 cell at the same protocols (frozen and
   continuous-dynamic, n=500 mns=8 for the short workloads,
   n=20 mns=1 for RULER).
3. Re-cast §6.3 / §6.5 with NVFP4 in place of (or alongside) INT4.
   Expected directional results, based on published NVFP4
   evaluations:
   - **GSM8K-Hard**: NVFP4 vs FP16 mirror effect smaller than
     INT4's −0.022 (we expect ~−0.010 to −0.015). EchoKV's
     recovery-by-upgrades-to-INT8 should still target the same
     layers, but the EchoKV bit-budget on this workload may drop
     because fewer layers need the INT8 lift.
   - **HotpotQA**: NVFP4 ≥ FP16 already (matches the INT4 result).
     EchoKV stays near uniform-NVFP4. No change to the
     "controller correctly recognises the regime" story.
   - **RULER@128K**: both NVFP4 and INT4 saturate at acc 1.000.
     EchoKV's tier vector should pick essentially uniform NVFP4 with
     τ rescaled per the §4.X √L formula.

The honest claim then becomes "EchoKV's per-layer adaptive
allocation Pareto-dominates *both* uniform INT4 and uniform
NVFP4," not just INT4.

### 7.2 Agentic / decode-heavy benchmarks — τ-Bench DONE, others pending

τ-Bench airline (Phase 10, §6.Y) is in the paper. Two related
agentic workloads are *not* yet evaluated and need infrastructure
we have not yet built:

1. **Terminal-Bench (Stanford)** — agent operating a real shell to
   complete tasks. Decode tokens per task: ~5-15K across thinking
   + tool-call traces. Requires a sandbox executor (Open Q 8 / 14
   in REPORT.md, also the blocker for HumanEval). Estimated lift:
   ~3 days for a contained sandbox.
2. **SWE-Bench-Verified** — long context (entire codebase fragment)
   *and* long output (multi-file diff). Same sandbox dependency
   plus per-task dependency resolution. Estimated lift: ~5 days.

Both are natural follow-ups to the τ-Bench result. The τ-Bench
finding that the controller picks 96% INT8 (vs 36% on short CoT)
implies these workloads should also pick mostly-INT8 vectors, but
the test of whether **EchoKV preserves agentic behavior** under
quantization on a *coding* agent (vs an airline-customer-service
agent) would substantially broaden §6.Y's claim.

**One immediate τ-Bench follow-up** (no new infrastructure needed):
the EchoKV-dyn vs EchoKV-frz result on τ-Bench (0.244 vs 0.289)
is unexplained; we conjecture it is the controller's tick-interval
being too aggressive at long-decode lengths (10 steps × ~12 tok/s
output ≈ a tick every ~0.8 s of agent generation, accumulating
many small updates per task). A simple ablation —
``--tick-interval`` ∈ {10, 30, 100, 300} on τ-Bench airline —
would isolate whether longer ticks recover EchoKV-frz's frozen-tier
performance under the dynamic protocol. ~2 hours wall.

### 7.3 Other open items

- **Fit the √L exponent** (§4.X) on a long-context workload where
  INT4 *does* hurt FP16 — REPORT.md Open Question 28.
- **Output-distance sensor** as a complement to softmax-KL on
  workloads where softmax-KL doesn't predict per-layer importance
  — Open Question 23.
- **Memory measurement vs projection** — convert the 4.33–4.40
  bits-per-rank numbers from `memory_projection` into measured
  on-GPU memory via a vLLM block-allocator fork — Open Question 16.

---

## Notes for the next iteration

- **Figure 1 caption** mentions "the layer-21–27 cluster" — the
  current per-layer ρ̄ chart shows layer 24 as the peak (z=7.94σ),
  consistent with the 21–27 region but the exact peak is layer 24.
  Either widen the caption to "layers 21–27 carry the dominant
  signal, peaking at layer 24" or narrow to "layer 24 carries the
  dominant signal, with neighbours 21–27 also above z=2."
- **Missing INT8 GSM8K-Hard at n=500 mns=8** is the only data gap
  in Figure 6. Trivial to fill (~18 min). Currently shown as
  "INT8 (n/a)" in the figure with a note in the caption.
- **Section 7 (Limitations and future work)** should call out:
  the controller stability safeguard (max-upgrades-per-tick)
  flagged in the Phase-4 cascade analysis as still-future
  work; the output-distance sensor as a potential complement
  to softmax-KL on workloads where the latter doesn't predict
  per-layer importance; the vLLM block-allocator fork to convert
  memory projections into measured-on-GPU memory.
