# Session 14 Phase 5 — Absolute-canary sensor redesign

**Date:** 2026-04-29
**Goal:** diagnose Phase-4's catastrophic
``max_num_seqs=1`` publication run (acc=0.365, 13pp below
static_int4) and fix the architectural cause, then re-validate the
EchoKV-3C result on 70B + GSM8K-Hard with the new sensor.

## What we found

Reading the canary code carefully revealed the legacy
``CanaryHarness`` measures
``KL(softmax(Q · quant(K)^T) ‖ softmax(Q · K^T))`` where Q and K
are the **just-projected FP16 tensors** for the current attention
call (before ``tier_application``). The KL is computed on a
``T × T`` attention matrix where T is the number of decode tokens
in the batch (= ``max_num_seqs`` at decode) or the prefill chunk
size.

Two distinct failure modes:

1. **T=1 collapse.** With ``max_num_seqs=1`` at pure decode, T=1.
   Softmax over a single element is ``[1.0]`` regardless. KL between
   two such distributions is identically 0. Controller sees
   ``δ̂_b = 0`` for every candidate b, argmin-with-budget picks the
   lowest tier (INT2). Result: uniform-INT2 cascade.
2. **Cache-state feedback.** Even at T > 1, the canary's
   measurement is indirectly coupled to the running tier mix via
   the K projection sequence processed so far. Phase 4's τ-sweep
   documented two attractors with 50× different δ̂_4 magnitudes
   (mixed-precision stable vs uniform-INT8 cascaded), depending on
   small initial-transient bias.

Both failure modes share a root cause: **the measurement is
coupled to runtime state**.

## The fix

A sliding-window FP16 K_ref buffer that's **always** FP16 and
**never** cache-dependent. At canary fires, the harness computes
absolute KL against this fixed FP16 reference rather than against
whatever the cache currently holds.

```
K_ref ∈ R^(N × num_kv_heads × head_size)   # always FP16
for b ∈ {2, 4, 8}:
    attn_b = softmax(Q · quant(K_ref, b)^T / √d)
attn_FP16 = softmax(Q · K_ref^T / √d)
δ_absolute(b) = KL(attn_b ‖ attn_FP16)
```

Implementation: feature-flagged via ``absolute_canary=True`` and
``kref_window_size=64`` on ``CanaryHarness``. Per-layer rolling
buffer; buffer-update happens before ``tier_application``. Memory
~2.5 MB per rank for 70B/TP=4. Legacy path preserved when the
flag is off.

T_key = N ≥ 2 always, so the T=1 collapse is mechanically
impossible. Both terms in the KL use the same FP16 reference, so
the cache's current bit-width doesn't enter the measurement —
the feedback loop breaks in both directions.

## Tests

9 new unit tests in ``tests/test_absolute_canary.py``:

- Buffer mechanics: lazy alloc, FIFO eviction, prefill-style
  large appends (T > window), decode-style 1-row appends.
- ``absolute_canary=False`` is a no-op (legacy preserved).
- Cross-tier monotonicity ``δ_2 ≥ δ_4 ≥ δ_8`` on real tensors.
- Non-trivial KL at T_query=1 when T_key > 1 — the property the
  redesign was specifically built for.
- Constructor parameter validation.

Full suite: **203 passed, 1 skipped** after the change (was 194
before).

## 8B + TP=2 smoke validation

Four checks before committing 70B time, all PASS:

| check | result |
|:--|:--|
| Magnitude (δ_4 above 1e-6 floor) | δ_4 max 0.031–0.042 across 4 runs |
| ``max_num_seqs`` invariance | mns=1 / mns=8 max-δ_4 ratio = 1.21× |
| Cross-tier monotonicity | 0 violations / 128 layer-runs |
| Cross-seed stability | Spearman ρ = 0.876 between seeds 0 & 7 |

The δ_4 magnitude (~0.04) is comparable to the legacy calibration-
probe scale (0.00356 max). The previous 60× calibration-vs-
production gap was an artifact of measuring different things; the
absolute canary closes it.

## 70B + GSM8K-Hard re-run

At τ_budget=0.022 (matched to the 8B smoke's δ_4 median):

### n=50 mns=1

Tier vector: {INT4:46, INT8:34, INT2:0}, avg 5.70 bits.
**acc=0.600.**

Compared to:
- static_int4 mns=1 n=200: 0.495
- FP16 mns=1 n=200: 0.510

This looked like a clean Pareto win at 5.70 bits — but n=50 has
SE ≈ 0.07, so 0.600 is consistent with anything from 0.46 to
0.74 at 95% confidence. Need n=200 to commit.

### n=200 mns=1 (publication validation)

Same tier vector {INT4:46, INT8:34} (reproducibility confirmed).
**acc=0.505.**

The n=50 was a lucky upward fluctuation. The n=200 tells us the
true workload-level effect is much smaller.

| run | bits | acc | Δ vs INT4 (95% CI) | McNemar p |
|:--|--:|--:|:--|--:|
| FP16        | 16.00 | 0.510 | +0.015 (±0.033) | 0.55 |
| static_int4 |  4.00 | 0.495 | (baseline)      | —    |
| EchoKV-3C abs-canary | 5.70 | 0.505 | **+0.010 (±0.037)** | **0.79** |

**Effect size below the noise floor at n=200.** Paired Δ vs
static_int4 is +0.010 with 95% CI [−0.027, +0.047]. McNemar
two-sided p=0.79.

## What changed about prior conclusions

- **Phase-3 "diffuse damage" verdict was premature.** The
  cascades that motivated that verdict were sensor artifacts.
  With the absolute-canary fix, EchoKV-3C produces a clean
  non-cascading mixed-precision tier vector.
- **Phase-4 "Pareto win" at τ=2e-5 mns=8 n=50 was inflated by
  the same artifacts.** The lucky 0.620 at n=50 was on top of
  a stable mixed-precision attractor that some τ-mns
  combinations land in by chance; the n=200 mns=1 absolute-
  canary measurement says the true effect is ~+0.010 over
  static_int4, below detection at n=200.
- **The static_int4 vs FP16 −0.022 finding (n=500 mns=8) is
  partially walked back.** At n=200 mns=1 the same comparison
  gives Δ=−0.015 (95% CI [−0.048, +0.018]; McNemar p=0.55) —
  same direction, no longer significant at the smaller paired n.

## What's intact after Phase 5

The architectural and engineering contributions hold cleanly:

1. **Sensor architecture.** Principled redesign that closes
   three artifacts at once: T=1 collapse, cache-state feedback,
   calibration-vs-production scale gap. Mathematically motivated
   (absolute KL against FP16 reference, not marginal KL against
   perturbed cache), minimally invasive (~150 LoC + a feature
   flag), and verified on 8B with a four-check smoke test.
2. **TP-aware harness.** ``apply_model``-based per-worker install
   path, ``TPCanaryHandle``, picklable helpers — all green on
   TP=2 (8B) and TP=4 (70B).
3. **Multi-precision canary** (Phase 2) correctly avoids the
   scalar-canary INT2 collapse on 70B GSM8K-Hard.
4. **Two-equilibria characterization.** Documented architectural
   finding: the legacy sensor has positive feedback that creates
   uniform-INT2 and uniform-INT8 attractors depending on initial
   transient.
5. **+0.09% TPOT measurement overhead** at production canary
   rate (Sessions 8-13, unchanged by Phase 5).

## What remains for the paper

The Pareto-improvement claim on 70B+GSM8K-Hard at p<0.05
**cannot be made from this n=200 mns=1 measurement alone**.
Three paths to substantiate it:

1. **Larger n.** n>500 mns=1 (~3 hours per cell) would tighten
   the CIs. The direction is consistent (+0.010 over INT4); a
   tighter CI might exclude 0.
2. **Workload with larger static-INT4 deficit.** LongBench
   multi-hop QA, 70B coding tasks, or harder reasoning datasets
   may give a bigger room to operate.
3. **More aggressive quantization.** INT3 as the static
   baseline (the unexplored middle ground between INT4 and
   the catastrophic INT2 floor); per-layer differentiation
   produces larger memory deltas at INT3 because the tier
   ladder is finer.

## Files

- ``evaluation/serving/canary_vllm.py`` — ``absolute_canary`` /
  ``kref_window_size`` flags + ``_kref_update`` / ``_kref_view``
  helpers + conditional path in ``_maybe_fire``.
- ``evaluation/serving/run_session14_70b_gsm8k_hard_echokv.py``
  — added ``--absolute-canary`` and ``--kref-window-size`` CLI
  flags.
- ``evaluation/serving/smoke_absolute_canary_8b.py`` — 8B+TP=2
  4-check smoke validation.
- ``tests/test_absolute_canary.py`` — 9 unit tests.
- ``results/serving/smoke_absolute_canary_8b.json``
- ``results/serving/session14_70b/echokv_3c_n50_abscanary_tau022_mns1.json``
  — n=50 lucky run (acc=0.600).
- ``results/serving/session14_70b/publication/echokv3c_abscanary_tau022_n200_seq1.json``
  — n=200 publication run (acc=0.505).
