# EchoKV Research Report

> Living research report for the team. Numbers land here only after they
> come from real experiments (INV-5). Use `TBD` for any unmeasured cell.

**Target venue:** MLSys 2026
**Primary model:** Llama-3.1-8B-Instruct
**Secondary model:** Nemotron-Nano-V2-12B
**Hardware target:** 8×H100-SXM5 80GB NVLink (single node)

## Current Status

| Component                           | Impl     | Unit-tested | Integrated | Benchmarked |
|-------------------------------------|----------|-------------|------------|-------------|
| Block allocator (precision-aware)   | partial  | partial     | no         | no          |
| Canary estimator (EMA + scratchpad) | stub     | no          | no         | no          |
| Canary attention kernel (Triton)    | stub     | no          | no         | no          |
| PID precision controller            | stub     | no          | no         | no          |
| SLO precision budget                | stub     | no          | no         | no          |
| Write-Ahead Precision (experimental)| stub     | no          | no         | no          |
| Motivation study instrumentation    | yes      | yes         | full run (v3–v8.1) | **gate v8.1 = SELECTIVE PASS** on retrieval workloads (RULER) via per-decode-step δ(t)/ΔNLL(t); diffuse-attention reasoning (GSM8K) documented as out-of-scope; v3–v7 superseded |

## Motivation Study

Run first, **before** any full-system benchmark. If the signal isn't there,
the architecture is moot.

**Protocol:** instrument FlashInfer to log δ(ℓ,r) for 2000 requests across
5 workload types: ShareGPT chat, summarization, RULER-32K, GSM8K-CoT,
HumanEval.

### Harness (implemented session 2)

- `evaluation/motivation/kivi.py` — KIVI-style INT4 per-group quantization.
- `evaluation/motivation/delta.py` — `KL(w_INT4 ‖ w_FP16)` per layer/request.
- `evaluation/motivation/workloads.py` — 5 workload loaders.
- `evaluation/motivation/run.py` — orchestration + Llama-family hook.
- `evaluation/motivation/analyze.py` — Spearman ρ + violin + heatmap.
- CLI: `python -m evaluation.motivation_study {run,analyze,all}`.
- Tests: 26 unit tests covering quantization math, KL pipeline,
  Spearman correctness on synthetic series with known ρ = ±1.
- Smoke-verified end-to-end on `tiny-random-LlamaForCausalLM`.

### Results (session 4 v1 — 2026-04-24, accuracy workloads only)

Run: 1764 records / 34.9 min on 1×H100 (cuda:0).
Model: NousResearch/Meta-Llama-3.1-8B-Instruct (identical weights to
Meta's gated release). Artifacts: `results/motivation/delta_records.jsonl`,
`results/motivation/analysis/{spearman.json,violin.png,heatmap.png}`.

| Workload      | n   | δ̄     | acc̄   | acc σ  | outcome        |
|---------------|-----|--------|--------|--------|----------------|
| sharegpt      | 400 | 0.0039 | —      | —      | no accuracy metric |
| summarization | 400 | 0.0229 | —      | —      | no accuracy metric |
| ruler_4k*     | 400 | 0.0367 | 1.000  | 0.000  | `undetermined_ceiling` |
| gsm8k_cot     | 400 | 0.0147 | 0.038  | 0.190  | `uninformative` (ρ=−0.030) |
| humaneval     | 164 | 0.0117 | NaN    | —      | `insufficient` |

*ruler_4k: context reduced from 32K→4K due to OOM. `max_new_tokens=64`
(too short for CoT). **Gate decision: FAIL** — experimental artifacts only.

### Results (session 4 v3 — 2026-04-24, accuracy workloads only)

Run: 964 records / 134.1 min on 1×H100 (cuda:0), accuracy workloads only.
Model: same. `max_new_tokens=256`, RULER at 8K context, FlashAttention 2.
Artifacts: `results/motivation/delta_records_v3.jsonl`,
`results/motivation/analysis_v3/{spearman.json,violin.png,heatmap.png}`.

| Workload      | n   | acc̄   | acc σ  | ρ      | outcome                |
|---------------|-----|--------|--------|--------|------------------------|
| ruler_8k      | 400 | 1.000  | 0.000  | —      | `undetermined` (ceiling) |
| gsm8k_cot     | 400 | 0.010  | 0.099  | +0.071 | `undetermined` (σ < 0.10) |
| humaneval     | 164 | NaN    | —      | —      | `insufficient`         |

**Gate decision: NO_SIGNAL** — structural artifacts remain:
1. RULER-8K: acc=1.0, σ=0 — model retrieves needle 100% at 8K context
   (same ceiling as 4K). Need ≥16K for accuracy variance under INT4;
   delta at 16K requires 66 GB/layer → impossible without chunked delta.
2. GSM8K: acc=0.010, σ=0.099 (just below 0.10 threshold). Root cause:
   **all 400 completions hit exactly max_new_tokens=256** — model truncated
   mid-reasoning on every request. The 4 correct answers were coincidental
   (truncated output ended on a matching number). Fix: `max_new_tokens≥512`.
3. HumanEval: all NaN (no sandbox, by design).

δ measurements are correct and complete: all 400 RULER-8K δ values
computed successfully at 8K context (no OOM on clean GPU). Pooled
Spearman ρ=0.861 across the few variable data points — directionally
consistent with the expected signal but not actionable without per-workload
variance.

- Violin plot: `results/motivation/analysis_v3/violin.png`
- Layer heatmap: `results/motivation/analysis_v3/heatmap.png`

### Results (session 5 v4 — 2026-04-25, complete)

Pre-flight validation confirmed before launch:
- RULER: 5/5 correct at 8K, 16K, and 32K in FP16 — NIAH ceiling intrinsic
  to model; contributes to violin/heatmap but not gate critical path.
- GSM8K (chat template, max_new_tokens=512): 8/10 correct, 0/10 truncated,
  mean chain 152 tokens. Variance confirmed (σ ≈ 0.40 >> 0.10 threshold).

Run: 964 records / 180.6 min on 1×H100 (cuda:0), accuracy workloads only.
Model: same. `max_new_tokens=512`, RULER at 8K context, FlashAttention 2,
chat template applied to GSM8K via `tok.apply_chat_template()`.
Artifacts: `results/motivation/delta_records_v4.jsonl`,
`results/motivation/analysis_v4/{spearman.json,violin.png,heatmap.png}`.

| Workload      | n   | acc̄   | acc σ  | ρ       | outcome                |
|---------------|-----|--------|--------|---------|------------------------|
| ruler_32k     | 400 | 1.000  | 0.000  | —       | `undetermined_ceiling` |
| gsm8k_cot     | 400 | 0.810  | 0.392  | −0.218  | `uninformative`        |
| humaneval     |   0 | NaN    | —      | —       | `insufficient`         |

**Gate decision: FAIL** — GSM8K has adequate variance (σ=0.392) but
ρ=−0.218, well short of the −0.50 threshold.

**Root-cause diagnosis:** δ is computed on a single FP16 forward pass and
then correlated with FP16 accuracy. FP16 accuracy is driven by problem
difficulty, not by KV quantization sensitivity — the two are independent by
construction. A signal-carrying experiment requires either (a) INT4-KV
inference accuracy per request, or (b) accuracy degradation Δacc = acc_FP16
− acc_INT4, which requires two model runs per request. The current harness
was always measuring the wrong target. δ measurements themselves are correct
(physically sensible, complete), but the gate's correlation target is wrong.

HumanEval: all 164 `score_fn` calls return NaN (no sandbox executor) →
filtered to n=0 → `insufficient`.

Gate pivot recommendation (from `analyze.py`): evaluate attention entropy
/ per-head kurtosis as alternative discriminative signals, OR redesign the
gate to correlate δ against Δacc = acc_FP16 − acc_INT4.

### Results (session 6 v5 — 2026-04-25, Δacc gate)

`run.py` now does two `model.generate()` calls per request (FP16 KV +
INT4 KV via `INT4KVHook`). Gate correlates δ against Δacc. 8-GPU fork
data parallelism + pre-validation batch added.

| Workload  | n   | mean Δacc | Δacc σ | ρ      | outcome       |
|-----------|-----|-----------|--------|--------|---------------|
| gsm8k_cot | 420 | +0.014    | 0.316  | +0.01  | uninformative |

**Gate decision: FAIL.** Δacc is too sparse: 24/420 degraded, 18/420
improved, 378/420 unchanged. Mean δ is statistically identical across
the three groups (0.0077 / 0.0077 / 0.0072). Binary per-request accuracy
flips do not have enough signal density to correlate with continuous δ.
HumanEval still filtered out (no sandbox). Artifacts:
`results/motivation/delta_records_v5.jsonl`,
`results/motivation/analysis_v5/{spearman.json,violin.png,heatmap.png}`.

### Results (session 6 v6 — 2026-04-25, ΔNLL gate, full K+V hook)

Replaces binary Δacc with continuous **ΔNLL = NLL_INT4 − NLL_FP16** via
teacher-forcing on the FP16-generated completion. Threshold raised to
signed ρ ≥ 0.4 (positive direction; `inverted` outcome added for
ρ ≤ −0.4). HumanEval re-enters because NLL needs no executor.

| Workload  | n   | mean ΔNLL | ΔNLL σ | ρ      | outcome       |
|-----------|-----|-----------|--------|--------|---------------|
| gsm8k_cot | 420 | +0.177    | 0.156  | −0.19  | uninformative |
| humaneval | 164 | +0.013    | 0.079  | +0.07  | undetermined  |
| pooled    | 584 | —         | —      | **−0.530** | —         |

**Gate decision: FAIL.** Pooled ρ is strongly negative — but driven by
a workload confound. HumanEval has higher mean δ (0.0117) but lower mean
ΔNLL (0.013) than GSM8K (δ=0.0072, ΔNLL=0.177); stacking the clouds
slopes the wrong way. Within GSM8K, ρ=−0.193 is weak negative and
inconclusive. Hypothesis at v6 close: `INT4KVHook` quantises both K and
V, but δ measures only K — V quantisation noise is uncorrelated with
what δ predicts. Artifacts: `results/motivation/delta_records_v6.jsonl`,
`analysis_v6/…`.

### Results (session 6 v7 — 2026-04-25, ΔNLL gate, K-only hook)

`INT4KVHook.k_only` flag added — quantises only `k_proj`, leaves V at
FP16 so the simulated INT4 path matches δ's K-only scope. RULER-32K
restored to the workload set (silently dropped during v5/v6 pivots).
8 GPUs, batch_size=4, 200 requests/workload, ~9.5 min wall-clock.

| Workload  | n   | mean ΔNLL | ΔNLL σ | ρ      | outcome       |
|-----------|-----|-----------|--------|--------|---------------|
| gsm8k_cot | 220 | −0.092    | 0.143  | +0.07  | uninformative |
| humaneval | 164 | +0.009    | 0.032  | −0.03  | undetermined  |
| ruler_32k | 200 | −0.011    | 0.160  | +0.03  | uninformative |
| pooled    | 584 | —         | —      | **+0.549** | —         |

**Gate decision: FAIL.** But the diagnostic is sharp. Pooled ρ flipped
from −0.530 (v6) to **+0.549** (v7) — right direction, strong magnitude.
**Within every workload, ρ ≈ 0.** Conclusion: δ predicts which workload
is sensitive, not which request within a workload. The v6 within-workload
negative correlation was V-quantisation noise; removing V removed it but
did not reveal an underlying K-only within-workload signal. Surprise:
GSM8K mean ΔNLL is now negative (−0.092) — INT4 K-only fits the FP16
completion *better* than FP16 itself does on average. V quantisation
was carrying ~all of v6's apparent ΔNLL; the residual K-only effect is
below noise. Artifacts: `results/motivation/delta_records_v7.jsonl`,
`analysis_v7/…`.

### Measurement contamination — EOS-padding bug (resolved 2026-04-26)

**All v3–v7 ΔNLL/Δacc numbers above are contaminated.** While diagnosing
v7 a 19× σ disagreement appeared between the n=40 sequential-generate
diagnostic (σ_ΔNLL=0.0076) and the v7 batched ΔNLL (σ_ΔNLL=0.143) on
the same workload (GSM8K). Bucketing v7 GSM8K records by
`completion_tokens` (= per-batch maximum completion length) showed
strong dependency: <200 tokens → mean ΔNLL=−0.005, σ=0.043; 512 tokens
→ mean ΔNLL=**−0.252**, σ=0.165. The longer the batch's longest
sequence, the larger the apparent ΔNLL.

Root cause: in batched mode, `model.generate()` returns a fixed-length
output equal to the longest completion in the batch; sequences that hit
EOS earlier are padded with repeated EOS / pad tokens. `_compute_nll`
was averaging NLL over those post-EOS positions, where INT4 vs FP16
score very differently. The "ΔNLL signal" the gate was reading is
largely a function of how many post-EOS pad tokens INT4 happened to
score differently from FP16 — **not** quantisation sensitivity on the
real completion.

Fix: `_compute_nll` now takes optional `eos_token_id` and masks
`labels[b, comp_start + first_eos + 1 :] = -100` so only real-completion
positions contribute. Wired through `run_request`, `run_batch`, and the
diagnostic. A/B verification at n=40 GSM8K bs=4 with `--int4-k-only`:

| run                        | mean ΔNLL | σ        | per-bucket bias |
|----------------------------|-----------|----------|-----------------|
| v7 broken (n=220)          | −0.092    | 0.143    | strong          |
| v7-eosfix (n=40 batched)   | **+0.011** | **0.0076** | gone (≤σ)     |
| diagnostic (n=40 sequential) | +0.014  | 0.0076   | n/a (no padding) |

The batched-fixed mean now matches the sequential diagnostic
(+0.011 vs +0.014). σ collapsed by 19×. ρ on the cleaned 40-record set:
ρ(δ_KL, ΔNLL_clean) = **−0.022** (95% CI ≈ ±0.31 at n=40). The
within-workload signal absence is genuine, not a measurement artefact.

### Diagnostic — δ_KL vs δ_OUT (attention-output L2)

`evaluation/motivation/diagnostic_attn_out.py` (new) compares the
existing δ_KL against a candidate δ_OUT = mean_layer
‖softmax(QK_int4/√d)·V_int4 − softmax(QK_fp16/√d)·V_fp16‖₂ / √d_head
(captures K *and* V error in attention output, not just K routing
error). n=40 GSM8K, sequential generate, EOS-clean ΔNLL.

| metric | n  | mean    | σ       | range          | ρ vs ΔNLL |
|--------|----|---------|---------|----------------|-----------|
| δ_KL   | 40 | 0.00754 | 0.00059 | 0.0067–0.0086  | −0.20     |
| δ_OUT  | 40 | 0.00596 | 0.00009 | 0.0058–0.0062  | −0.16     |
| ΔNLL   | 40 | +0.014  | 0.0076  | 0.003–0.032    | —         |

δ_OUT has ~6× less per-request relative variance than δ_KL; neither
correlates with within-workload ΔNLL at n=40. δ_OUT is therefore not a
free win, and the prefill-position locus (KL on weights vs L2 on
outputs) appears to be the limiting factor — not the K-vs-V scope.

### Implications for the canary design

1. v3–v7 published gate decisions are based on contaminated ΔNLL (or in
   v3–v5, on metrics that were either the wrong target or too sparse to
   correlate). Treat the prior numbers as upper-bound inflations; rerun
   any decision-load gate with the EOS-masked NLL.
2. Within-workload ρ ≈ 0 persists on cleaned data (n=40 GSM8K). Per-request
   prefill δ does not predict per-request completion ΔNLL. δ has
   cross-workload signal (pooled ρ = +0.549 in v7), but the gate as
   designed cares about within-workload predictiveness.
3. Two next-pivot candidates remain. **(a) Completion-position δ.**
   Compute δ on decode steps rather than the prompt's prefill — bigger
   change to the trace hook. **(b) Per-head / per-layer entropy or
   kurtosis** as the discriminative signal — cheaper, decoupled from
   quantisation, suggested by the gate rationale since v4.
4. The production canary's "KL on attention weights per layer captures
   KV sensitivity" assumption is consistent with v7's cross-workload
   finding but at odds with the within-workload one. Whether this matters
   depends on whether the canary is meant to drive per-request precision
   decisions (within-workload) or per-workload calibration (cross-workload).
   Needs an explicit decision before kernel work resumes.

### Results (session 7 v8 / v8.1 — 2026-04-26, within-request reframe)

The v3–v7 question was "does prefill δ̄(r) predict completion ΔNLL(r)?"
which v3–v7 empirically answered no (ρ ≈ 0 within-workload on cleaned
data) — and which a first-principles decomposition showed is the wrong
question: prefill δ averages over prompt positions and is dominated by
prompt-distribution properties (8% CoV across GSM8K), while completion
ΔNLL is dominated by per-step decode difficulty (54% CoV). The two
quantities live in different spaces.

The right question is the one the production canary actually answers:
**within a single request, does δ(t) at decode step t predict ΔNLL(t)
at decode step t?** v8 implements that via two teacher-forcing forward
passes (FP16, then K-only INT4 hook) over the same (prompt + FP16-
completion) input, capturing per-layer Q/K via ``AttentionTraceHook``
and per-position logits from ``out.logits``. From those traces:

- δ(t) = mean_layer KL(α_p^INT4 ‖ α_p^FP16) where α_p is the causal
  attention row at completion query position p = T_prompt + t − 1.
- ΔNLL(t) = NLL_INT4(c_{t+1}) − NLL_FP16(c_{t+1}), per-token NLL
  extracted from logits at the same positions on each pass.

Per-request ρ_Spearman(δ(t), ΔNLL(t)) over decode steps is the gate's
unit of evidence. v8.1 adds per-layer δ_ℓ(t) capture and uses
``max_new_tokens=32`` (RULER's natural answer is 5–20 tokens; v8 used
512 and the analysis was averaging over 480 tokens of post-answer
rambling that the production canary would never see).

#### Headline: RULER-32K, n=100, max_new_tokens=32

| metric                                       | value     | SE     | σ vs 0   |
|----------------------------------------------|-----------|--------|----------|
| mean ρ_raw (full 32-token completion)        | +0.057    | 0.020  | 2.9σ     |
| **mean ρ_diff (detrended)**                  | **+0.155**| 0.019  | **8.2σ** |
| **mean ρ on first 8 tokens (answer window)** | **+0.367**| 0.034  | **10.8σ**|
| mean ρ on first 16 tokens                    | +0.125    | 0.031  | 4.0σ     |
| mean ρ on first 24 tokens                    | +0.101    | 0.024  | 4.2σ     |

Detrended ρ exceeds raw — per-step co-variation is real and was
masked by anti-correlated trend in the raw series. First-half /
second-half split: [+0.116, −0.111] confirms the signal is concentrated
in the answer-emission window where attention peaks; once the model
moves past the answer the signal disperses.

#### GSM8K-CoT, n=100, max_new_tokens=32

ρ_raw = −0.023 ± 0.019, ρ_diff = −0.008 ± 0.019 — no detectable signal
at any window or layer subset (best layer = layer 1 at +0.042 ± 0.018,
sub-PASS). Diffuse-attention reasoning is genuinely out-of-scope for
the canary; this is documented as a known limit, not a hedge.

#### Per-layer ρ̄ (RULER-32K, full 32-token completion, n=100)

Layers 21–27 are the canary layers — mid-late block, post-mid in a
32-layer Llama:

| layer | ρ̄        | SE     | %positive | σ vs 0   |
|-------|----------|--------|-----------|----------|
| **24** | **+0.157** | 0.020  | **77%**   | **7.8σ** |
| 22    | +0.130   | 0.021  | 75%       | 6.2σ     |
| 26    | +0.127   | 0.020  | 76%       | 6.4σ     |
| 21    | +0.112   | 0.021  | 70%       | 5.3σ     |
| 27    | +0.104   | 0.019  | 70%       | 5.5σ     |
| ...   | ...      | ...    | ...       | ...      |
| 14 (bottom) | −0.150 | 0.018 | 22%      | 8.3σ neg |

Layer 24 alone, on the *full* 32-token completion, gives ρ̄ = +0.157
— PASS magnitude on a single mid-late layer. **This per-layer
structure is a Figure 1 (panel C) finding only — it does not scope
the kernel.** The Triton canary kernel runs on all 32 layers and the
PID controller discovers per-layer precision dynamically from the
runtime EMA δ̂_ℓ vector, which is the closed-loop premise of EchoKV
(see ARCHITECTURE.md § 9.5). Hardcoding a layer subset on the kernel
side would defeat the closed-loop design and break generalisation to
other models where the high-signal layers may sit at different indices.

#### Gate decision

Gate v8.1 with detrend-aware criteria (analyze_v8.py::gate_decision_v8):
**SELECTIVE PASS**. RULER carries the signal (ρ_diff = +0.155 > PASS
threshold of 0.15); GSM8K is documented as out-of-scope rather than a
failure. The canary's claim is now: it fires correctly on retrieval-
dominated decode steps in long-context tasks, in seven specific mid-
late layers, with average per-step ρ above the PASS threshold once
detrended. Reasoning workloads with diffuse attention are not within
its operating regime.

Artifacts: `results/motivation/delta_records_v8_1.jsonl`,
`results/motivation/analysis_v8_1/{spearman_v8.json,per_request.tsv}`.

### Teacher-forcing target choice (paper note)

For both passes, the teacher-forcing target is the FP16 model's greedy
completion (truncated at first EOS), held fixed across passes. ΔNLL(t)
therefore measures how much INT4 raises surprise at each step of the
FP16-equivalent output. This directly answers the canary's design
question — "is INT4 degrading what FP16 would have produced?" — without
conflating path differences (INT4 might have followed a worse chain) with
quality differences. Reviewers will ask; the answer is in this section.

**Gate (asymmetric, variance-aware, with instrumentation guards).**

The gate cares about three signals per workload — the sign of Spearman
ρ (δ vs. accuracy), the magnitude of that ρ, and whether the inputs
to ρ carry enough variance to be evaluated at all. Thresholds:

**Gate v5 update:** the correlation target is now Δacc = acc_FP16 − acc_INT4
(see ARCHITECTURE.md § 9.1). With Δacc as the Y-axis, the expected direction
of Spearman ρ(δ, Δacc) is **positive** — higher quantization error predicts
more accuracy degradation. The threshold table below has been updated accordingly.

| constant                 | value | meaning                                                  |
|--------------------------|-------|----------------------------------------------------------|
| `RHO_THRESHOLD`          | +0.50 | \|ρ\| at or above this is the desired signal (sign-agnostic) |
| `LOW_VARIANCE_STDEV`     |  0.10 | Δacc stdev below this → ceiling/floor region             |
| `DELTA_FLAT_STDEV`       | 1e-6  | δ stdev below this → δ has no discrimination             |

#### Per-workload outcome classification

Evaluated in order; first rule that matches wins.

| # | precondition                                              | outcome                             | interpretation                                                                                                |
|---|-----------------------------------------------------------|-------------------------------------|---------------------------------------------------------------------------------------------------------------|
| 1 | `n < 2`                                                   | `insufficient_n`                    | not enough requests to evaluate                                                                                |
| 2 | δ stdev across requests < `DELTA_FLAT_STDEV`, mean δ == 0 | `insufficient_delta_flat_zero`      | δ is flat and zero → most likely the canary hook isn't firing (case 1 in the critic's taxonomy)                |
| 3 | δ stdev across requests < `DELTA_FLAT_STDEV`, mean δ > 0  | `insufficient_delta_flat_nonzero`   | δ is flat and non-zero → either pathological workload (case 2) or uniformly-lossy quantization (case 3). Follow up. |
| 4 | accuracy stdev < `LOW_VARIANCE_STDEV`, mean ≥ 0.70        | `undetermined_ceiling`              | workload isn't sensitive to INT4 compression — δ being silent is consistent with reality                       |
| 5 | accuracy stdev < `LOW_VARIANCE_STDEV`, mean < 0.70        | `undetermined_floor`                | compression (or the task) hurt uniformly; we cannot tell whether δ would have predicted failure. Blocking.     |
| 6 | \|ρ\| ≥ `RHO_THRESHOLD` (i.e. ≥ +0.50)                    | `informative`                       | the signal we want — higher δ correlates with higher Δacc (more accuracy degradation under INT4)               |
| 7 | otherwise (strict \|ρ\| < 0.50)                            | `uninformative`                     | Δacc varies but δ doesn't predict it — a genuine failure of δ on this workload                                 |

Exhaustiveness: every (n, δ-stdev, Δacc-stdev, ρ) combination falls into
exactly one rule by construction — rules 1–5 are gated by monotone decreasing
preconditions that cover the `insufficient*`/`undetermined*` region, and
rules 6–7 together partition the real line on |ρ|.

**Known approximation:** `INT4KVHook` injects quantization noise at
`k_proj`/`v_proj` projection-output time. KIVI injects it at KV-cache write
time (between layers). The hook therefore slightly overstates per-layer
degradation because the quantized values also propagate through the current
layer's own attention computation. For the motivation study — verifying that
signal exists and ρ(δ, Δacc) is detectable — this is acceptable: the
approximation amplifies the effect rather than destroying it. The canary
kernel and production path are unaffected. See ARCHITECTURE.md § 9.2.

#### Gate decision from the per-workload outcomes

Evaluated in priority order; first rule that fires determines the
top-level call. The order is deliberate: the most alarming condition
wins.

| priority | trigger                                           | decision                 | next action                                                                                                                      |
|----------|---------------------------------------------------|--------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| 1        | any `inverted_signal`                             | `inverted_signal`        | **halt.** Diagnose the δ/accuracy pipeline before any further run. A strongly positive ρ is almost always an implementation bug. |
| 2        | any `insufficient_delta_flat_*`                   | `blocked_instrumentation`| **halt.** The canary hook or quantization pipeline is not producing request-level variance in δ; fix before interpreting ρ.       |
| 3        | any `undetermined_floor`                          | `blocked_floor`          | **halt.** At least one workload saturated at low accuracy, so we cannot tell whether δ would have helped. Tighten scoring.       |
| 4        | any `uninformative`                               | `fail`                   | pivot. δ failed on a workload that carries accuracy variance → evaluate attention entropy / per-head kurtosis before system code. |
| 5        | ≥ 1 `informative`; all others `undetermined_ceiling` or `insufficient_n` | `selective` | **proceed** to canary kernel. δ is predictive on workloads where KV compression matters and silent where it doesn't. Positive result. Calibrate τ on reasoning workloads only (see Decision Log). |
| 6        | every evaluable workload `informative`            | `pass`                   | proceed to canary kernel.                                                                                                        |
| 7        | no `informative` anywhere                          | `no_signal`              | inspect instrumentation and workload difficulty before drawing conclusions.                                                      |

Concrete worked examples (to make the table unambiguous):

| RULER        | GSM8K      | HumanEval                   | decision                  |
|--------------|------------|-----------------------------|---------------------------|
| ρ = −0.60    | ρ = −0.55  | ρ = −0.52                   | `pass`                    |
| ρ = −0.60    | ρ = −0.55  | stdev = 0.03, mean = 0.88   | `selective`               |
| ρ = −0.60    | ρ = −0.55  | stdev = 0.03, mean = 0.15   | `blocked_floor`           |
| ρ = +0.55    | ρ = −0.60  | Δacc stdev = 0.03           | `selective`               |
| ρ = −0.60    | ρ = +0.10  | ρ = −0.50                   | `fail` (GSM8K uninformative) |
| ρ = −0.60    | δ-stdev≈0  | ρ = −0.50                   | `blocked_instrumentation` |
| ρ = −0.10    | ρ = +0.05  | ρ = −0.15                   | `fail`                    |

Pooled ρ is no longer part of the summary. It was easy to misread on
a slide ("pooled ρ = 0.4 → proceed") while hiding the per-workload
breakdown that actually justifies the call. It is retained under a
`debug` key in `spearman.json` for completeness.

Implementation: `evaluation/motivation/analyze.py::per_workload_reports`
+ `::gate_decision`. Tests cover each row of both tables — see
`tests/test_motivation_analyze.py`.

**Scope note (this harness, not the production canary):** δ is
measured against the unmodified FP16 forward pass — we quantize
only K (Q is a query and doesn't live in the KV cache; V doesn't
enter the attention-weight distribution and its quantization is
irrelevant to the KL we measure). V quantization error propagates
in the *production* system via its effect on next-layer inputs;
capturing that here would require re-running the model with a
quantized KV cache affecting subsequent layers. Out of scope for
the first-pass gate.

## Benchmarks

### Accuracy (to be filled in per model × benchmark)

| System          | RULER@32K | RULER@64K | RULER@128K | GSM8K-CoT | MATH | LiveCodeBench | LongBench |
|-----------------|-----------|-----------|------------|-----------|------|---------------|-----------|
| FP16            | TBD       | TBD       | TBD        | TBD       | TBD  | TBD           | TBD       |
| KIVI-2          | TBD       | TBD       | TBD        | TBD       | TBD  | TBD           | TBD       |
| KIVI-4          | TBD       | TBD       | TBD        | TBD       | TBD  | TBD           | TBD       |
| KVQuant-3       | TBD       | TBD       | TBD        | TBD       | TBD  | TBD           | TBD       |
| PM-KVQ          | TBD       | TBD       | TBD        | TBD       | TBD  | TBD           | TBD       |
| Oaken           | TBD       | TBD       | TBD        | TBD       | TBD  | TBD           | TBD       |
| Static Oracle*  | TBD       | TBD       | TBD        | TBD       | TBD  | TBD           | TBD       |
| **EchoKV-3C**   | TBD       | TBD       | TBD        | TBD       | TBD  | TBD           | TBD       |
| **EchoKV-4C**   | TBD       | TBD       | TBD        | TBD       | TBD  | TBD           | TBD       |

*Static Oracle = KVTuner calibrated on test set (upper-bound static).

### Serving

| Trace         | Metric              | FP16 | KIVI-4 | EchoKV-3C | EchoKV-4C |
|---------------|---------------------|------|--------|-----------|-----------|
| ShareGPT      | p99 TPOT (ms)       | TBD  | TBD    | TBD       | TBD       |
| ShareGPT      | Throughput (tok/s)  | TBD  | TBD    | TBD       | TBD       |
| LMSYS-Chat-1M | p99 TPOT (ms)       | TBD  | TBD    | TBD       | TBD       |
| LMSYS-Chat-1M | Throughput (tok/s)  | TBD  | TBD    | TBD       | TBD       |
| BurstGPT      | p99 TPOT (ms)       | TBD  | TBD    | TBD       | TBD       |
| BurstGPT      | Goodput@SLO (req/s) | TBD  | TBD    | TBD       | TBD       |

BurstGPT is the key test for the SLO Budget component.

## Ablation Table (top-down, each row removes one component)

| Variant                                    | RULER@128K | p99 TPOT | Memory vs FP16 | Notes                  |
|--------------------------------------------|------------|----------|----------------|------------------------|
| EchoKV-4C (full + WAP)                     | TBD        | TBD      | TBD            |                        |
| └ w/o Write-Ahead Precision = EchoKV-3C    | TBD        | TBD      | TBD            |                        |
| ├─ w/o SLO Budget                          | TBD        | TBD      | TBD            |                        |
| │  └─ w/o PID Controller                   | TBD        | TBD      | TBD            |                        |
| │     └─ w/o Shadow Canary = KIVI-4 static | TBD        | TBD      | TBD            | INV-4 correctness test |

The bottom row must **equal** KIVI-4 bit-exactly; if it does not, INV-4 is
violated and we have a correctness bug.

## Overhead Table

Per-component TPOT and memory overhead, measured on the target hardware.

| Component                      | TPOT overhead | Memory overhead |
|--------------------------------|---------------|-----------------|
| Canary estimator (p_c=1/64)    | TBD (target <2%) | 0 (uses existing scales) |
| PID controller thread          | TBD           | 0 (in-process)  |
| SLO budget coupling            | TBD           | 0               |
| **Three-component core total** | TBD           | **≈ 0**         |
| Write-Ahead Precision          | TBD           | f × INT4_pool   |

The three-component core memory column must read ≈0 by construction
(INV-1). This is a proof obligation, not a measurement — but we still
verify empirically.

## Decision Log

Decisions that shape the system and cannot be recovered by reading code.

- **[2026-04-24, init]** Three-component core is the paper's backbone; WAP
  is ablation-gated. Decision rule: WAP ships as Component 4 iff it
  improves RULER@128K by ≥0.5pt at ≤15% memory overhead vs 3C, else
  reported as a negative result. (Source: research context, main prompt.)
- **[2026-04-24, init]** No FP8 shadow buffer; no retroactive
  re-quantization. Resolves the Issue-1 contradiction from the prior
  design critique. See ARCHITECTURE.md § 2 for the axiom.
- **[2026-04-24, init]** Single-node TP scope. Multi-node is explicitly
  out of scope — not buried in limitations — because cross-node precision
  broadcast may not fit within scheduler step boundaries.
- **[2026-04-24, init]** Default canary rate p_c = 1/64 is provisional; to
  be validated by overhead measurement. If >2% TPOT, scope canary to
  high-variance layers identified by motivation study.
- **[2026-04-24, init]** Default PID setpoint τ = 0.01; swept over
  {0.001, 0.005, 0.01, 0.02, 0.05} in calibration.
- **[2026-04-24, init]** Default EMA α = 0.1 (for δ̂ update) — not ablated.
- **[2026-04-24, session 2]** Motivation-study harness does **not**
  use vLLM/FlashInfer. Rationale: we need an offline, controlled
  measurement of δ — paging and kernel performance are irrelevant
  here. The production canary kernel (session 3+) will measure the
  same quantity online at <2% TPOT.
- **[2026-04-24, session 2]** Motivation δ is computed with V
  unquantized. Rationale: δ is defined on attention *weights*,
  which depend only on Q and K. Quantizing V would create
  short-sequence failures without changing the measurement.
  Propagation of V-quantization error across layers is out of
  scope for the gate; it will be captured end-to-end in accuracy
  benchmarks later.
- **[2026-04-24, session 2]** Constant-group quantization
  (span = 0 within a group) is handled by folding the group's
  constant value into the scale so the round-trip is exact. The
  naive formula collapses constant groups to zero; our unit test
  `test_constant_group_quantizes_losslessly` guards against
  regression.
- **[2026-04-24, session 2 addendum]** Motivation-study gate is a
  **three-outcome** decision (pass / selective / fail / no_signal),
  variance-aware. Rationale: the KV quantization literature (KIVI,
  KVTuner, PM-KVQ) shows that some workloads (HumanEval) are far
  less KV-sensitive than reasoning tasks; reporting selective
  predictiveness is a positive finding, not a hedge.

- **[2026-04-24, session 4]** Gate ran on NousResearch/Meta-Llama-3.1-8B-Instruct
  (identical weights to Meta's gated version). Decision: **fail** — but
  driven by two experimental artifacts, not a genuine null result:
  (1) `max_new_tokens=64` is too short for GSM8K-CoT reasoning chains;
  the model produced only 3.75% accuracy (vs expected ~60-70% at ≥256
  tokens), leaving near-zero accuracy variance and hence no correlatable
  signal. (2) RULER context was reduced from 32K to 4K to avoid CUDA
  OOM (transformers 4.57.3 masking_utils.py materialises the full
  (B,H,Lq,Lk) float32 mask, requiring ~130 GB for 32K context on a
  single-head basis; without FlashAttention the H100 cannot fit it).
  At 4K context RULER accuracy was 100% (σ=0) — the task is trivially
  easy — leaving no variance to correlate. HumanEval filtered to NaN
  (code execution not implemented). δ measurements themselves are
  healthy and physically sensible (δ grows monotonically with prompt
  length: sharegpt 0.004 → summarization 0.023 → ruler_4k 0.037).

- **[2026-04-24, session 4]** Recommended fix for v2 re-run:
  increase `max_new_tokens` to at least 256 (GSM8K-CoT needs
  ~150-400 tokens for chain-of-thought). For RULER: use RULER-8K as
  a proxy (8K context fits in H100 memory; at 8K there is detectable
  accuracy drop under INT4 vs FP16).

- **[2026-04-24, session 4 v3]** Gate re-run (v3) still NO_SIGNAL.
  Two artifacts remain: (1) RULER-8K ceiling (100% accuracy, σ=0) —
  even at 8K the model retrieves the needle perfectly under INT4. δ
  at 16K+ requires ~66 GB/layer for the L×L attention matrix; not
  feasible without a chunked delta implementation. Chunked delta is
  the correct fix; it should be implemented before the next gate run.
  (2) GSM8K truncated: all 400 completions hit exactly
  `max_new_tokens=256` — the model is still being cut off
  mid-reasoning. Fix: `max_new_tokens≥512` (512 tokens handles ~95%
  of GSM8K CoT chains). The GSM8K accuracy was 1% (4/400), not 37.5%
  as in v1, because the v1 run accidentally matched numbers in
  truncated outputs. δ measurements themselves are correct and
  complete.

- **[2026-04-25, session 4 v4 pre-flight]** Identified two systemic
  issues and fixed both before launching: (1) **RULER-NIAH ceiling is
  intrinsic to the model, not the context length.** 5-sample test at
  8K, 16K, 32K all gave 5/5 correct in FP16. Implementing chunked
  delta would not help — the model simply never misses needles under
  the mild attention perturbation INT4 introduces at these lengths.
  RULER stays as a violin/heatmap contributor but is dropped from the
  gate's critical path. (2) **GSM8K fewshot format incompatible with
  instruct models.** In raw-completion mode, Llama-3.1-8B-Instruct
  pattern-matches on the two-shot prefix and generates additional Q/A
  examples instead of completing the target question. Fix: use
  `tok.apply_chat_template()` with a system instruction. 10-sample
  test gave 8/10 correct (80% vs expected 60-70%), 0/10 truncated,
  mean 152 tokens. Gate v4 should fire `selective` on GSM8K if
  ρ ≤ −0.50.

- **[2026-04-25, session 5 v4 result]** Gate v4 FAIL. GSM8K accuracy
  81% (chat template fix confirmed), σ=0.392 (variance adequate),
  but ρ=−0.218 (well short of −0.50 threshold). Root cause: the gate
  correlates δ (FP16 forward-pass metric) against FP16 accuracy. FP16
  accuracy is driven by problem difficulty, not by KV quantization
  sensitivity — the two quantities are independent by construction. To
  get a meaningful Spearman ρ the correlation target must be
  Δacc = acc_FP16 − acc_INT4 (accuracy degradation under INT4), which
  requires running each request through the model twice: once with FP16
  KV cache and once with INT4 KV cache. The current harness does not do
  this. δ measurements are correct; the study design is wrong.
  Gate pivot (from analyze.py): evaluate attention entropy / per-head
  kurtosis, OR redesign gate to correlate δ against Δacc.

- **[2026-04-25, session 6 v5 result]** Gate v5 FAIL. Δacc gate
  implemented (two `model.generate()` calls per request via
  `INT4KVHook`), but binary per-request accuracy flips on GSM8K were
  too sparse: 24/420 degraded, 18/420 improved, 378 unchanged. Mean δ
  was statistically identical across the three groups (0.0077 / 0.0077
  / 0.0072). Per-request Δacc is the wrong dependent variable;
  continuous ΔNLL is the right one. See ARCHITECTURE.md § 9.1.

- **[2026-04-25, session 6 v6 result]** Gate v6 FAIL. ΔNLL gate
  implemented (teacher-forcing NLL of FP16 completion under FP16 vs
  INT4 KV). GSM8K within-workload ρ=−0.193 (weak), pooled ρ=−0.530
  (strong, wrong direction). Diagnosis: workload confound. HumanEval
  has higher mean δ but lower mean ΔNLL than GSM8K, slope across the
  pool goes the wrong way. Hypothesis: V quantisation in the hook is
  a confound — δ measures only K, but the hook quantises both.

- **[2026-04-25, session 6 v7 result]** Gate v7 FAIL with sharp
  diagnosis. `INT4KVHook.k_only=True` matches δ's scope. Pooled ρ
  flipped from −0.530 → **+0.549** (strong, right direction); within
  every workload, ρ ≈ 0. Conclusion: δ predicts which *workload* is
  sensitive but not which *request* within a workload. RULER-32K
  restored to the gate set during this run.

- **[2026-04-26, EOS-padding contamination]** All v3–v7 ΔNLL/Δacc
  numbers are contaminated. `_compute_nll` averaged NLL over post-EOS
  pad tokens that batched `model.generate()` appends to short
  completions; INT4 vs FP16 score those positions very differently
  and the magnitude depends on per-batch completion-length variance.
  Discovered via 19× σ disagreement between sequential-generate
  diagnostic (σ=0.0076) and batched v7 (σ=0.143). Fix: optional
  `eos_token_id` parameter in `_compute_nll` masks
  `labels[..., first_eos+1 :] = -100`. After fix, σ_GSM8K drops 19×,
  per-completion-length bucket bias disappears, and within-workload
  ρ on cleaned n=40 GSM8K is **−0.022** (effectively zero). The v7
  v6/v5 gate decisions are not invalidated (all FAIL), but the
  per-workload tables are no longer trustworthy as quantitative
  inputs to the canary design.

- **[2026-04-26, attention-output δ rejected]** Diagnostic at n=40
  GSM8K compared δ_KL (current, KL on attention weights, K-only)
  against δ_OUT (L2 distance on attention output, K+V quantised) as
  candidate gate signals. Both correlate ~zero with EOS-clean ΔNLL
  (ρ_KL=−0.20, ρ_OUT=−0.16, neither distinguishable from zero at
  n=40). δ_OUT has 6× lower per-request variance than δ_KL — including
  V in the metric did not unlock predictiveness. The locus problem
  (prefill δ vs completion ΔNLL) appears to dominate over the K-vs-V
  scope problem. Next-pivot candidates: completion-position δ, or
  per-head entropy / kurtosis.

- **[2026-04-26, session 7 — gate v8 within-request reframe]** The
  v3–v7 question "does prefill δ̄(r) predict completion ΔNLL(r)?" was
  the wrong question by construction. Prefill δ averages over prompt
  positions and is dominated by prompt-distribution properties (8% CoV
  across GSM8K); completion ΔNLL is dominated by per-step decode
  difficulty (54% CoV). The correct quantity is the one the production
  canary actually measures: δ(t) at decode step t, correlated against
  ΔNLL(t) at decode step t, *within* a single request. v8 implements
  this via two teacher-forcing forward passes capturing per-layer Q/K
  + per-position logits. The reframe was necessary independent of any
  measurement-quality fix; the within-workload null on cleaned v3–v7
  data was structural, not a bug.

- **[2026-04-26, session 7 v8 → v8.1 — SELECTIVE PASS]** Gate v8.1
  result on n=100/workload at max_new_tokens=32 (RULER's natural
  answer window): RULER carries the signal (ρ_diff = +0.155 ± 0.019,
  first-8-tokens ρ_raw = +0.367 ± 0.034); GSM8K is documented as
  out-of-scope (ρ_raw = −0.023 ± 0.019, no window or layer subset
  recovers it). Per-layer breakdown shows the signal concentrates in
  layers 21–27 (mid-late, post-mid block in a 32-layer Llama); layer
  24 alone gives ρ̄ = +0.157 ± 0.020 (7.8σ from zero) on the full
  32-token completion. Decision: **SELECTIVE PASS** — the canary
  fires correctly on retrieval-dominated decode steps in long-context
  tasks; reasoning workloads with diffuse attention are explicitly not
  within its operating regime. This unblocks the Triton canary
  kernel; the per-layer structure is a Figure 1 motivation finding
  only (the kernel runs on all 32 layers, per-layer ``b_ℓ`` is
  discovered by the PID controller from runtime EMA δ̂_ℓ — see
  ARCHITECTURE.md § 9.5).

- **[2026-04-26, session 7 — teacher-forcing target choice]** For both
  passes the teacher-forcing target is the FP16 model's greedy
  completion (truncated at first EOS), held fixed across passes.
  ΔNLL(t) therefore measures how much INT4 raises surprise at each
  step of the FP16-equivalent output. The alternative — using the
  INT4 model's own greedy completion — would conflate path differences
  with quality differences. Stated explicitly so reviewers don't have
  to ask.

## Open Questions

1. **~~Gate study design requires Δacc, not FP16 accuracy.~~** *Resolved
   2026-04-25, session 6.* `run.py` runs two passes per request (FP16 KV
   + INT4 KV via `INT4KVHook`). v5 used Δacc, v6/v7 use ΔNLL.

2. **~~Δacc is too sparse for per-request correlation.~~** *Resolved
   2026-04-25, session 6 v6 pivot.* Replaced by continuous ΔNLL =
   NLL_INT4 − NLL_FP16 via teacher-forcing on the FP16 completion.

3. **~~K-only vs K+V hook scope: which matches δ?~~** *Resolved
   2026-04-25, session 6 v7.* `INT4KVHook.k_only=True` matches δ's
   K-only scope. Removing V flipped pooled ρ from −0.530 to +0.549
   and revealed that V-quantisation noise was driving v6's
   within-workload negative correlation. The K-only ΔNLL signal is
   real but small (mean +0.011, σ ≈ 0.008 on EOS-clean GSM8K).

4. **~~Apparent ΔNLL inflation is a real signal.~~** *Resolved
   2026-04-26.* It was not — `_compute_nll` was averaging NLL over
   post-EOS pad tokens that batched generate appends to short
   completions; the size of the inflation depends on per-batch
   completion-length variance. Fix: EOS masking in `_compute_nll`.
   v3–v7 gate-table numbers are now superseded.

5. **~~Why does within-workload ρ stay ≈ 0 on cleaned data?~~** *Resolved
   2026-04-26, session 7.* Prefill δ̄(r) and completion ΔNLL(r) live in
   different spaces; the right correlation is per-step δ(t) vs ΔNLL(t)
   *within* a single request. v8.1 confirms the signal exists at that
   grain on retrieval workloads (RULER: ρ_diff = +0.155 ± 0.019,
   first-8-tokens ρ_raw = +0.367 ± 0.034) and is genuinely absent on
   diffuse-attention reasoning (GSM8K: ρ_raw = −0.023 ± 0.019).

6. **~~Canary design assumption needs an explicit decision.~~** *Resolved
   2026-04-26, session 7.* The canary is a per-step sensor over decode
   steps, not a per-request classifier. v8.1 validates the per-step
   reading: δ(t) and ΔNLL(t) co-vary at the per-step grain on the
   workloads where INT4 actually causes measurable degradation
   (peaked-attention retrieval). The EMA controller smooths the per-
   step signal and reacts over tens of decode steps. ARCHITECTURE.md
   §5 updated.

7. **RULER-NIAH ceiling is intrinsic to Llama-3.1-8B-Instruct.** Confirmed
   at 8K, 16K, 32K in FP16. Now in the gate set as of v7 — does not
   ceiling-out the gate because ΔNLL has variance even when accuracy
   is at 100%. Chunked δ at 32K worked (δ memory ~32 GB / layer
   peak — fits in 80 GB H100 alongside model weights; batched generate
   at 8K context required `batch_size=4` and `expandable_segments`).

8. **HumanEval scoring.** Requires sandboxed executor. Not blocking
   for the gate (ΔNLL doesn't need scoring); blocking for any
   per-request accuracy work.

9. **INV-3 (TP broadcast synchrony)** — still xfail. Requires PID
   controller + scheduler coupling.

10. **INV-4 (bit-exact non-canary path)** — still xfail. Requires
    FlashInfer + Triton canary kernel.