# Session 14 Phase 3 — Closed-loop EchoKV-3C on 70B GSM8K-Hard (the Pareto test)

**Date:** 2026-04-29
**Goal:** test whether the closed-loop EchoKV-3C controller can
recover the 2.2pp INT4 accuracy loss documented in Phase 2 (n=500 on
70B+GSM8K-Hard) by selectively upgrading high-divergence layers to
INT8 — at lower memory cost than uniform INT8. This is the actual
Pareto test the project has been chasing for 14 sessions.

## What was built

### TP-aware ``CanaryHarness``

The Phase-2 static-INT4 path used ``llm.apply_model`` to install a
standalone K/V quantizing pre-hook on each TP worker. For Phase 3 we
need the full ``CanaryHarness`` (canary firing, controller, multi-
precision EMA, tier application) on each TP worker — not just the
quantization hook.

- **``CanaryHarness.__init__`` accepts ``model=`` directly.** When
  the harness is being constructed inside a worker process via
  ``apply_model``, the worker hands us its local model and we skip
  the LLM-engine unpack path. Backward-compatible: every existing
  callsite passes ``llm=`` positionally and uses kwargs for
  everything else.

- **``evaluation/serving/canary_tp.py``.** Picklable helpers
  (``install_canary_on_worker``, ``snapshot_canary_on_worker``,
  ``seed_tiers_on_worker``, ``remove_canary_on_worker``) shipped to
  workers via ``llm.apply_model``. The harness state is stashed on
  the worker's local model (via attribute) so successive RPC calls
  can find it. ``TPCanaryHandle`` wraps the boilerplate as a context
  manager. Functions live in a regular importable module (not
  ``__main__``) because pickle can't resolve ``__main__.foo`` on
  the worker side.

- **``functools.partial`` for kwargs.** ``LLM.apply_model(func)``
  takes a single-arg callable ``(model) → result``; extra args must
  be bound. ``TPCanaryHandle.install`` wraps
  ``install_canary_on_worker`` with ``partial(harness_kwargs=...)``;
  same pattern for ``seed_tiers``.

- **11 unit tests** in ``tests/test_canary_tp.py`` — install
  idempotency, hook count, ``seed_tiers``, ``remove`` cleanup,
  snapshot shape, multi-precision per-tier δ̂ readback, and the
  ``functools.partial`` driver-side path. CPU-only mock with a
  module class literally named ``Attention`` so the harness's type
  filter matches. No vLLM dependency in the tests.

### 8B + TP=2 smoke test

``evaluation/serving/smoke_canary_tp_8b.py`` runs Llama-3.1-8B at
TP=2 with the TP-aware install path, generates 10 RULER prompts at
high observability rate, and validates:

- Hook counts ``[32, 32]`` — both ranks hooked all 32 layers.
- 725 canary fires per rank — perfectly balanced (both ranks see
  the same prompts at the same decode steps).
- Tier vectors **diverge between ranks by ~28%** — structural, not
  a bug. TP=2 splits 8 KV heads 4/4 across ranks, so each rank's
  per-layer δ̂ is computed on a different head subset, and the
  per-rank controllers honestly disagree on a fraction of layers.
- ``seed_tiers`` correctly forces a specified tier vector on every
  rank.
- ``remove`` cleans up all hooks.

### Driver: ``run_session14_70b_gsm8k_hard_echokv.py``

3-phase protocol:

1. **Warmup.** TP-aware harness installed with
   ``tier_application=True``, ``freeze_controller=False``,
   ``observability_rate=0.25``. Run 16 GSM8K-Hard prompts.
   Controller updates per-rank tier vectors.
2. **Consolidate.** Snapshot per-rank state. Per-layer max across
   ranks — the rank that wants more precision wins. (Avoids a
   per-rank-per-head precision mosaic at measurement time.)
3. **Measurement.** Install fresh harness with
   ``freeze_controller=True``, seed the consolidated tier vector
   identically on every rank via ``apply_model``. Run the full
   100-prompt batch.

Early-stop: if the consolidated tier vector is uniform-INT4, write
the result and skip the measurement run (controller has converged
to static-KIVI-4; nothing new to measure).

### Calibration probe: ``calibrate_tau_budget_70b_gsm8k_hard.py``

TP-aware variant of the existing 8B+RULER calibration script.
Force-bits at 8, 4, 2; multi-precision canary observes per-tier
δ̂; per-rank vectors averaged across ranks for the per-layer
estimate. Returns per-tier [min, p50, mean, max] distributions and
a τ_budget recommendation by the safe_upper / broken_lower gap
heuristic.

## EchoKV-3C trajectory at n=100

| run | sensor | τ_budget | tier counts | avg bits | acc |
|:--|:--|--:|:--|--:|--:|
| FP16 (reference) | — | — | uniform FP16 | 16.0 | 0.570 |
| static_int4 (reference) | — | — | uniform INT4 | 4.00 | 0.540 |
| EchoKV-3C scalar | scalar canary | 0.01 | INT4:1, INT2:79 | 2.02 | **0.250** |
| EchoKV-3C MP | multi-precision | 0.05 | INT4:77, INT2:3 | 3.92 | **0.520** |
| EchoKV-3C MP (calibrated) | multi-precision | **0.002** | INT4:51, INT2:29 | 3.27 | **0.480** |

### Run 1 — scalar canary collapses to INT2

Tier vector: 79 INT2 + 1 INT4. acc=0.250 — 32pp drop from FP16.

This is exactly the failure mode Session 12 documented: per-step
softmax KL between FP16 K and INT4-quantized K is small even on
workloads where INT2 actually destroys downstream accuracy. The
scalar argmax-with-budget controller sees the small δ̂ and decides
"INT2 is fine here," cascading downward across nearly every layer.
Confirmed on a new model (70B vs 8B) and a new workload (reasoning
vs retrieval).

### Run 2 — multi-precision canary at default τ=0.05

Tier vector: 77 INT4 + 3 INT2 (layers 0, 30, 43). acc=0.520, very
close to static_int4=0.540.

The multi-precision sensor correctly avoids the scalar collapse —
this is the **Phase-3 engineering win**. Per-tier δ̂ isolates
where INT2 is dangerous (layer divergence too high under INT2)
versus where it's safe. Most layers stay at INT4 because their
δ̂_2 exceeds 0.05.

But at τ=0.05, **no layers upgrade to INT8**: every layer's δ̂_4
stays below 0.05, so INT4 always passes the budget and INT8 is
never selected by argmin. Result: tier vector ≈ static_int4 with
a 3-layer INT2 perturbation. Accuracy ≈ static_int4 (not
significantly different at n=100).

### Calibration probe on 70B+GSM8K-Hard

| static probe | acc | δ̂_2 [min, p50, max] | δ̂_4 [min, p50, max] | δ̂_8 max |
|:--|--:|:--|:--|--:|
| INT8 | 0.600 | [0.0004, 0.007, 0.022] | [1e-5, 0.0003, **0.00075**] | 3e-6 |
| INT4 | 0.550 | [0.0016, 0.030, 0.088] | [8e-5, 0.0013, **0.00356**] | 1e-5 |
| INT2 | 0.300 | [0.001, 0.006, 0.026] | [6e-5, 0.0002, 0.00082] | 3e-6 |

**δ̂_4 max under INT4 = 0.00356.** About 10× tighter than 8B+RULER
(~0.04). By the strict critic test (δ̂_4 max < 0.02 → Scenario B),
this is **diffuse damage**.

The 44× max/min spread within δ̂_4 (0.00356 / 0.00008) does exist
— meaningful per-layer variance — but at small absolute scale.
Threshold analysis on the static-probe distribution projected
13 INT8 + 66 INT4 + 1 INT2 at τ=0.002.

### Run 3 — calibration-derived τ=0.002

The static-probe projection didn't predict the closed-loop steady
state. Actual tier vector: 0 INT8 + 51 INT4 + **29 INT2**.

Why the discrepancy: the controller-feedback loop. Once a layer
transitions to INT2 during warmup, its in-state δ̂_2 drops sharply
(the cache becomes INT2-quantized, so subsequent canary fires
measure smaller incremental divergence between FP16 K and INT-N K).
This reinforces the INT2 decision rather than walking it back.
Meanwhile, no layer's δ̂_4 stays high enough to trigger an INT8
upgrade — the upgrade requires δ̂_4 > τ_budget, and once the
neighbouring layers go to INT2 the static reference shifts further
toward what those layers see.

acc = **0.480**. Versus static_int4=0.540: **6pp accuracy loss for
18% memory savings** (3.27 / 4.00 = 0.82). Marginal trade is
worse than the FP16→INT4 hop (75% memory savings, 3pp loss).
Not a Pareto improvement.

## Verdict

**Diffuse damage confirmed empirically on 70B + GSM8K-Hard.**

Three EchoKV-3C runs at three τ_budget settings, none produce a
Pareto-better point than uniform KIVI-4. The 2.2pp INT4 loss
documented in Phase 2 is not localized to a few high-divergence
layers that a per-layer controller can identify and upgrade.
Per-layer δ̂ does not correlate with per-layer responsibility for
diffuse accuracy damage on this workload.

## What's intact

The closed-loop architecture is empirically intact:

- **Hooks install correctly on every TP rank.** TP-aware install
  via ``apply_model`` works.
- **The multi-precision canary correctly avoids the scalar sensor's
  INT2 collapse.** The Session 12 → 13 sensor fix transfers cleanly
  to 70B + reasoning.
- **The controller correctly converges** to its argmin-with-budget
  rule's prediction on the running-tier δ̂ distribution.
- **Tier consolidation across TP ranks** produces a coherent
  measurement-phase configuration.
- **194 tests pass, 1 skipped** (11 new TP-aware tests +
  183 prior).

## What's missing

**The right sensor signal.** Per-step softmax KL on
``softmax(Q · K_FP16)`` vs ``softmax(Q · K_quant)`` is what δ̂
measures. It's the wrong correlate for diffuse damage. A
plausible alternative for Session 15+:

- **Output-distance sensor.** Measure
  ``||attn_output_FP16 − attn_output_quant||₂`` per layer.
  Output is post-softmax, post-projection — closer to what
  downstream layers actually consume. Reuses existing forward-
  pre-hook plumbing; only the measured quantity changes. ~50-100
  LoC + a new τ_budget calibration.

## Reframed paper contribution

The honest story across Sessions 8–14:

1. **Engineering**: a closed-loop quantization control system
   running at +0.09% TPOT measurement overhead, with a TP-aware
   per-worker install path and zero monotonicity violations
   across 1000+ KL computations.
2. **Sensor characterization**: the per-layer canary δ̂ exposes
   real per-layer divergence structure (7.3× spread on RULER at
   128K, 44× spread on GSM8K-Hard at smaller absolute scale).
3. **Empirical limitation**: on the workloads tested
   (Llama-3.1-8B and 70B, retrieval and reasoning, 4K to 128K),
   per-layer δ̂ does not correlate with per-layer responsibility
   for diffuse accuracy damage. The closed-loop controller
   correctly identifies "uniform-INT4 is sufficient" on robust
   workloads (8 of 9 cells) and correctly avoids INT2 collapse
   via the multi-precision sensor (Phase 3 result), but cannot
   recover the small INT4 loss on the one workload where it
   exists.
4. **Negative-result framing**: the paper substantiates that
   per-layer softmax-KL δ̂ is the wrong sensor for diffuse damage
   recovery. A different sensor (output-distance) is required if
   a Pareto win is to be claimed on this regime.

## Files

- ``evaluation/serving/canary_vllm.py`` — ``CanaryHarness.__init__``
  accepts ``model=`` directly.
- ``evaluation/serving/canary_tp.py`` — TP-aware install/snapshot/
  seed/remove helpers + ``TPCanaryHandle``.
- ``evaluation/serving/smoke_canary_tp_8b.py`` — 8B+TP=2 validation.
- ``evaluation/serving/run_session14_70b_gsm8k_hard_echokv.py`` —
  3-phase EchoKV-3C driver.
- ``evaluation/serving/calibrate_tau_budget_70b_gsm8k_hard.py`` —
  TP-aware τ_budget calibration probe.
- ``tests/test_canary_tp.py`` — 11 unit tests.
- ``results/serving/smoke_canary_tp_8b.json``
- ``results/serving/session14_70b/echokv_3c_n100_scalar.json`` —
  scalar canary collapse (acc=0.250).
- ``results/serving/session14_70b/echokv_3c_n100_mp.json`` —
  multi-precision @ τ=0.05 (acc=0.520).
- ``results/serving/session14_70b/echokv_3c_n100_mp_tau002.json``
  — calibration-derived τ=0.002 (acc=0.480).
- ``results/serving/session14_70b/tau_budget_calib_gsm8k_hard.json``
  — three-tier static probe data.
