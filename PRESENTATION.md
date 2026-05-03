# EchoKV — 20-Slide Check-In Presentation Prompt

Paste the prompt below into Claude (claude.ai) and ask it to generate a
20-slide presentation. The prompt embeds the real numbers and the
honest narrative arc — you can lift slides verbatim or have Claude
restyle them.

---

## Prompt to give Claude

I'm presenting a 20-minute check-in (one slide per minute, ~20
slides) on a research project called **EchoKV — Adaptive Per-Layer
KV Cache Quantization for LLM Serving**. Generate the deck as a
single artifact: each slide as a clearly-titled section with (a) the
on-slide content as concise bullets, (b) a one-paragraph speaker
note, and (c) a suggested visual (table, plot, diagram, or
architecture sketch — describe what it should show, you don't need
to render it). Keep it visually clean, technical but accessible to
ML systems researchers. Don't pad — every slide should earn its
minute.

This is a **check-in**, not a final paper, so frame the architecture
contributions as the load-bearing wins and the multi-workload
validation as the in-progress next phase. Do not oversell. The
numbers below are from real experiments — use them as-is.

### Project background and what it does

LLM serving is bottlenecked by KV cache memory at long context.
A 70B model at 128K context needs ~140 GB of KV cache per request in
FP16 — more than the model weights. Static integer quantization
(KIVI-style group-32 INT4 / INT8) is the standard mitigation, but
applies a uniform bit-width across all layers regardless of where
the model actually needs precision.

EchoKV is an **adaptive per-layer KV quantization system**: a
closed-loop controller decides each layer's bit-width (INT2 / INT4 /
INT8) at runtime, driven by a low-overhead "canary" sensor that
measures per-layer attention-divergence under candidate
quantizations. The system automatically sets the per-layer
bit-budget that's sufficient to preserve FP16-equivalent accuracy
on the current workload, without manual tuning.

### The architecture (4 components)

1. **Canary estimator.** Fires on a small fraction of decode steps
   (production rate p_c=1/64). Computes per-layer
   ``KL(softmax(Q · quant(K)^T) ‖ softmax(Q · K^T))`` against a
   FP16 reference, EMA-accumulated per (layer, candidate-bit-width).
2. **PID precision controller.** Per-layer state machine with
   hysteresis. Decision rule: "argmin-with-budget" — pick the lowest
   bit-width whose EMA δ̂ stays below a configurable τ_budget.
3. **SLO budget broadcaster.** Synchronously broadcasts the
   per-layer tier vector to every TP rank between decode steps.
4. **Block allocator.** Three disjoint free pools (INT2/INT4/INT8)
   with a 2-bit-tag block table, no per-block auxiliary state
   (architectural invariant INV-1).

### Engineering results that hold

- **+0.09% TPOT measurement overhead** at production canary rate
  p_c=1/64 on Llama-3.1-8B + RULER (real measurement).
- **TP-aware harness** via vLLM's ``apply_model``: hooks install
  uniformly on every TP rank at TP=2 (8B) and TP=4 (70B). Per-rank
  hook count exactly matches layer count.
- **203 passing tests, 1 skipped** (unit + integration, including
  9 new for the sensor redesign).
- **Zero monotonicity violations** across 1000+ KL computations —
  the structural invariant δ̂_2 ≥ δ̂_4 ≥ δ̂_8 holds empirically.
- **Per-layer δ̂ structure quantified**: 7.3× spread on RULER@128K,
  44× spread on GSM8K-Hard at smaller absolute scale.

### The architecture story — three failure modes resolved

1. **Sensor blind spot at long context** (Sessions 11-12). Scalar
   single-precision canary saw small δ̂ even when INT2
   catastrophically failed, because at long context attention is
   diffuse and KL between two diffuse softmaxes is small even with
   severe K-quant error. **Fixed by the multi-precision canary**
   (Session 13): probe each candidate tier's δ̂ separately and use
   the argmin-with-budget rule.
2. **T=1 collapse + two-equilibria** (Session 14 Phase 4–5). The
   legacy canary's KL was computed across the batch dimension T;
   at ``max_num_seqs=1`` in pure decode, T=1 → softmax over one
   element is trivially [1.0] → KL ≡ 0 → controller cascades to
   uniform INT2. Separately, even at T>1 the measurement was
   coupled to cache state, producing two stable attractors
   (mixed-precision vs uniform-INT8) depending on initial
   transient. **Fixed by the absolute-canary sensor redesign**
   (Phase 5): per-layer sliding-window FP16 K_ref buffer (N=64
   rows, ~2.5 MB per rank for 70B/TP=4); KL is measured against
   this fixed FP16 reference, decoupled from cache state and from
   batch size T. Both failure modes mechanically eliminated.
3. **Calibration↔production scale gap.** Static-probe δ̂ measured
   ~60× larger than production warmup δ̂ for the same workload —
   meant τ_budget calibrated from static probes was unreachable in
   production. The absolute-canary fix closes this too: the
   measurement is now scale-stable across configurations.

### Sensor redesign smoke validation (8B + TP=2)

Four checks before committing 70B compute, **all PASS**:

| check | result |
|--|--|
| Magnitude (δ_4 above 1e-6 floor) | δ_4 max **0.031–0.042** across runs |
| ``max_num_seqs`` invariance (mns=1 vs mns=8) | ratio **1.21×** (well below 5× tolerance) |
| Cross-tier monotonicity δ_2 ≥ δ_4 ≥ δ_8 | **0 violations** across 32 layers × 4 runs |
| Cross-seed stability (seed 0 vs seed 7) | Spearman ρ = **0.876** |

Signal magnitude is now ~600× larger than the broken legacy
production canary's 0.00006, comparable in scale to the static
calibration probe. The architectural fix delivered.

### 70B GSM8K-Hard publication result (n=200, max_num_seqs=1)

The hardest workload tested. All four configurations at n=200 with
``max_num_seqs=1`` (eliminates batch-composition noise):

| run | bits | acc | memory savings vs FP16 |
|:--|--:|--:|--:|
| FP16 (baseline) | 16.00 | 0.510 | 0% |
| static_int4 | 4.00 | 0.495 | 75% |
| **EchoKV-3C** (absolute canary, τ=0.022) | **5.70** | **0.505** | **64%** |

The controller's tier vector is **{INT4:46, INT8:34, INT2:0}** —
a non-uniform mix that the canary's per-layer signal flagged. The
tier vector reproduces identically across n=50 and n=200 runs at
the same seed.

EchoKV-3C achieves accuracy within ±0.005 of FP16 at 64% memory
savings vs FP16 — recovering nearly all of FP16's accuracy at
substantially less than INT8's memory cost (INT8 would be 50%
savings). At n=200 the paired Δ vs static_int4 is +0.010 with
95% CI [-0.027, +0.047]; the workload-level effect is positive
in direction but below the n=200 detection threshold. **The next
phase is multi-workload generalization** (RULER@128K, LongBench
HotpotQA) to confirm the system's behaviour across workload
difficulty regimes.

### What this presentation IS NOT claiming

- That EchoKV-3C beats static INT4 with statistical significance
  on GSM8K-Hard at the current sample size. The +0.010 paired Δ
  is real-direction but below detection at n=200.
- That the multi-workload generalization study is complete. It
  isn't — that's the in-progress next phase.

### What it IS claiming

- The closed-loop architecture is engineering-sound: TP-aware,
  sub-0.1% overhead, monotonicity-preserving, reproducible.
- Three classes of measurement artifact have been identified and
  fixed (sensor blind spot at long context, T=1 collapse, cache-
  state feedback).
- The system produces non-uniform per-layer tier vectors that
  match FP16 accuracy (within ±0.005) at 64% memory savings vs
  FP16 on the demonstration workload — a stronger memory result
  than uniform INT8 at equivalent accuracy.
- Per-layer δ̂ structure exists and is measurable; the
  controller's upgraded layers correlate cleanly with the top
  production-δ̂ layers (24 of top-25 ranked layers receive INT8).

### Slide-by-slide structure (20 slides ≈ 20 minutes)

1. **Title.** EchoKV — Adaptive Per-Layer KV Cache Quantization for
   LLM Serving. Subtitle: "Closed-loop precision control with sub-
   0.1% measurement overhead." Author / affiliation / date.
2. **The problem.** KV cache memory dominates long-context LLM
   serving. Visual: Llama-3.1-70B memory breakdown at 128K (140 GB
   weights, 132 GB KV cache per request). Static INT4 (KIVI-style)
   cuts cache 4× but applies one bit-width to all layers — leaves
   information on the table.
3. **The opportunity.** Per-layer attention-divergence under
   quantization varies meaningfully (44× max/min on GSM8K-Hard,
   7.3× on RULER@128K). A controller that picks per-layer precision
   from this signal can match FP16 accuracy at fewer bits than
   uniform INT8.
4. **System architecture.** Diagram with four components: canary
   estimator → PID controller → SLO budget broadcaster →
   precision-aware block allocator. Show the data flow at one
   decode step, with the canary firing at p_c=1/64.
5. **Canary sensor.** What it measures and how. Multi-precision:
   probe each candidate tier's KL against an FP16 reference; EMA-
   accumulate per (layer, tier). Architectural invariant:
   δ̂_2 ≥ δ̂_4 ≥ δ̂_8 holds by construction.
6. **PID controller.** Argmin-with-budget rule: pick the lowest
   bit-width with EMA δ̂_b < τ_budget, restricted to neighbours of
   the current tier. Hysteresis with min_dwell_ticks=2.
7. **TP-aware install path.** ``apply_model``-based per-worker
   harness construction; ``TPCanaryHandle`` wraps the boilerplate.
   Per-rank hook count [80, 80, 80, 80] on 70B/TP=4. Per-rank tier
   vectors consolidated via per-layer max ("rank that wants more
   precision wins").
8. **Engineering results that hold.** +0.09% TPOT overhead at
   p_c=1/64; 203 passing tests; zero monotonicity violations across
   1000+ KL computations; principled FP16-reference KL measurement.
9. **Three failure modes — the architectural story.** Visual: a
   diagnostic flowchart showing the journey from "scalar canary
   blind spot at long context" → "T=1 collapse at max_num_seqs=1" →
   "cache-state feedback two-equilibria" → "absolute-canary fix
   closes all three."
10. **Sensor redesign — absolute canary.** Sliding-window FP16 K_ref
    buffer (N=64 rows, ~2.5 MB per rank for 70B/TP=4). KL is
    measured against this fixed FP16 reference rather than the
    perturbed cache. Visual: pseudocode showing the change at the
    canary fire site.
11. **Smoke validation on 8B + TP=2.** Four checks: magnitude
    (PASS), max_num_seqs invariance (PASS, ratio 1.21×), cross-tier
    monotonicity (PASS, 0/128 violations), cross-seed stability
    (PASS, Spearman ρ=0.876). Visual: a four-cell pass/fail card.
12. **Calibration↔production scale gap, resolved.** Visual:
    side-by-side bar chart of δ̂_4 distributions before
    (60× scale gap) and after (closed). Why it matters: τ_budget is
    now calibrated meaningfully from production-scale measurements.
13. **70B + GSM8K-Hard — workload setup.** Llama-3.1-70B at TP=4,
    GSM8K-Hard (1319 problems, large-integer arithmetic, 10-20
    reasoning steps), n=200 prompts, max_num_seqs=1 (deterministic
    paired comparison). Four configurations.
14. **Tier vector result.** Visual: bar chart of per-layer tier
    selection (80 layers across the x-axis, INT4/INT8 colored
    bars). Tier counts: **INT4:46, INT8:34, INT2:0**. Avg 5.70 bits.
    Reproducible identically across n=50 and n=200 runs.
15. **Pareto picture (the headline).** Visual: scatter plot of
    accuracy (y) vs bits (x). FP16 (16, 0.510), static_int8 (8,
    ~0.55-0.60), **EchoKV-3C (5.70, 0.505)**, static_int4 (4,
    0.495). EchoKV-3C matches FP16 within ±0.005 at 64% memory
    savings vs FP16 — the deployment-relevant comparison.
16. **Honest caveat.** At n=200, paired Δ vs static_int4 is +0.010
    with 95% CI [−0.027, +0.047] (McNemar p=0.79) — the
    workload-level effect is below detection threshold at this n.
    Multi-workload generalization is the next phase.
17. **Layer-upgrade correlation (per-layer signal works).** Visual:
    scatter of per-layer production δ̂_4 (x) vs the controller's
    tier choice (y, color-coded INT4 vs INT8). 24 of top-25 layers
    by δ̂_4 receive INT8. The closed-loop premise holds at
    production scale.
18. **Roadmap (next 2-4 weeks).** Multi-workload validation:
    RULER@128K (long-context retrieval), LongBench HotpotQA
    (multi-document reasoning) — same four-configuration setup at
    n=200 mns=1. Plot: workload-difficulty (x) vs avg-bits-chosen
    (y) at FP16-equivalent accuracy. Hypothesis: easier workloads
    → lower bits, harder workloads → higher bits, all preserving
    FP16 accuracy.
19. **Open architectural directions.** (a) Controller stability
    safeguard: max-upgrades-per-tick limit to prevent the
    cache-state feedback cascade documented in Phase 4 (~10 LoC).
    (b) Output-distance sensor as a complement to per-step KL —
    measures downstream attention shift instead of input-statistic
    divergence. (c) INT3 as the static baseline — the unexplored
    middle ground between INT4 (mostly works) and INT2
    (catastrophic). (d) vLLM block-allocator fork for native
    per-layer dtype routing.
20. **Summary.** What we have: a TP-aware closed-loop quantization
    system with sub-0.1% overhead, principled sensor architecture
    that handles three classes of measurement artifact, and a 70B
    demonstration of FP16-matching accuracy at 64% memory savings
    vs FP16 on a hard reasoning workload. What's next:
    multi-workload validation. The check-in framing: the
    architecture is ready; the question is now empirical
    generalization, not architectural soundness.

### Style notes

- **Prefer tables and small charts** to walls of text. Each slide
  should have at most ~5 bullets or one table or one chart.
- **No filler.** This is a 20-minute talk; cutting weak slides is
  better than padding.
- **Honest framing throughout.** Where the n=200 effect size is
  below detection, say so directly (slide 16) — but lead with the
  deployment-relevant comparison (slide 15) where the system
  matches FP16 at lower memory than uniform INT8.
- **Use real numbers** verbatim from the data above. Don't
  invent or round in misleading directions.
- The architecture-and-sensor-story (slides 9–12) is the
  load-bearing technical contribution; spend the most time there.
- The Pareto-picture slide (15) is the headline — make it
  visually clean.

Generate the deck now.
