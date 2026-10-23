# EchoKV Architecture

> Ground-truth architecture document. Keep this authoritative. If code and
> this document disagree, the document wins or the document must be updated
> in the same commit as the code change. Every design decision below is
> traceable to either an invariant or an experimental decision rule.

## 1. Thesis

KV quantization error is a runtime property of the current conversation, not
a static property of the model. Every prior KV quantization system (KIVI,
KVQuant, QServe, KVTuner, TurboQuant, PM-KVQ, Oaken) freezes a single
offline calibration decision. EchoKV measures runtime error cheaply and
responds to it — treating quantization as a closed-loop control problem.

## 2. Information-Theoretic Constraint (Architectural Axiom)

**Retroactive INT4 → INT8 re-quantization is vacuous.**

INT4 with group-size 32 produces at most 15 distinct FP16 values per group,
which fills only 15 of 255 INT8 levels. No new information is introduced by
widening the representation of an already-quantized block. Any retroactive
repair therefore requires an auxiliary high-precision shadow (e.g. stored
FP8), which carries ≥50% memory overhead per block and defeats the purpose
of a system motivated by memory reduction.

**Corollary (INV-2 in code form):** precision decisions apply to new block
allocations only. Functions named `refresh`, `repair`, `retroactive`, or
`rewrite` **must not exist** on the three-component core path. Any change
that touches existing blocks must be justified simultaneously against the
memory overhead constraint and the information-theoretic vacuousness
argument above.

## 3. Component Dependency Diagram

```
                        ┌──────────────────────────────────┐
                        │  vLLM V1 Scheduler (single-node) │
                        │                                  │
                        │  kv_precision_budget  ◄──────────┼──┐
                        └────────────────┬─────────────────┘  │
                                         │                    │
                     broadcasts precision updates              │
                     to all TP ranks before next               │
                     decode step  (scheduler-as-barrier)       │
                                         │                    │
                                         ▼                    │
            ┌────────────────────────────────────────────┐    │
            │             Decode Step (per layer ℓ)      │    │
            │                                            │    │
            │  ┌──────────────────────────────────────┐  │    │
            │  │ Canary Attention Kernel (FlashInfer) │  │    │
            │  │  is_canary: bool (rate p_c = 1/64)   │  │    │
            │  │                                      │  │    │
            │  │  if is_canary:                       │  │    │
            │  │    stream0: INT4 paged attn → ȧ_ℓ    │  │    │
            │  │    stream1: dequant via existing     │  │    │
            │  │             scales → FP16 → ā_ℓ      │  │    │
            │  │    δ(ℓ) = KL(ȧ_ℓ ‖ ā_ℓ)              │  │    │
            │  └────────────────┬─────────────────────┘  │    │
            └───────────────────┼────────────────────────┘    │
                                │                             │
                          pinned scratchpad                   │
                          (accumulated δ(ℓ))                  │
                                │                             │
                                ▼                             │
            ┌──────────────────────────────────────────┐      │
            │   PID Precision Controller Thread        │      │
            │   (Python threading.Thread, every N_c)   │      │
            │                                          │      │
            │   δ̂_ℓ^(t) = 0.9·δ̂_ℓ^(t-1) + 0.1·δ_ℓ^(t)  │      │
            │   τ_eff = f(τ, kv_utilization, budget)   │ ◄────┘
            │   hysteresis: INT2 ↔ INT4 ↔ INT8          │
            │                                          │
            │   emits: b_ℓ ∈ {2,4,8}  (NEW blocks only)│
            └──────────────────┬───────────────────────┘
                               │
                               ▼
            ┌──────────────────────────────────────────┐
            │  Precision-Aware Block Allocator         │
            │  allocate_with_precision(layer, b_ℓ)     │
            │                                          │
            │  Separate INT2/INT4/INT8 free lists.     │
            │  No auxiliary per-block buffer (INV-1).  │
            └──────────────────┬───────────────────────┘
                               │
                               ▼
                 existing KIVI-format paged KV pool
                 (per-group FP16 scales + INT4 zero-points
                  are structural metadata, never evicted)


          Experimental path (Path B, ablation-gated inclusion)
          ─────────────────────────────────────────────────────
            ┌──────────────────────────────────────────┐
            │  SinkPredictor (Write-Ahead Precision)   │
            │                                          │
            │  Positional prior: positions 0-3 → INT8  │
            │  Accumulator: after W=256 tokens,        │
            │    if EMA(w_{ℓ,i}) > θ=0.15 → INT8       │
            │                                          │
            │  Decision feeds allocator at FIRST WRITE │
            └──────────────────────────────────────────┘
```

## 4. Component Specifications

### 4.1 Shadow Canary Estimator (`core/canary_estimator.py`, `kernels/canary_attention.py`)

**Purpose:** cheap, unbiased runtime estimator of per-layer KV quantization
error δ(ℓ) on the current conversation.

**Inputs:**
- Current query `q_ℓ` and quantized KV block metadata (paged).
- Boolean `is_canary` (sampled at rate p_c = 1/64 per decode step).

**Outputs:**
- Standard paged attention output `ȧ_ℓ` (always produced, on stream 0).
- When `is_canary=True`: a scalar δ(ℓ) = KL(ȧ_ℓ ‖ ā_ℓ) written to a pinned
  scratchpad ring buffer; `ā_ℓ` is computed on stream 1 by dequantizing the
  same INT4 block using its existing per-group FP16 scales and INT4
  zero-points.

**Invariants:**
- INV-1: the kernel must not allocate any per-block auxiliary buffer. The
  dequantization path reads structural metadata that KIVI-format blocks
  already carry.
- INV-4: when `is_canary=False` for all steps, behaviour must be bit-exact
  with the upstream FlashInfer kernel (the `w/o Shadow Canary` ablation
  must reproduce KIVI-4 output exactly).

**Known limitations:**
- KL divergence requires a softmax-normalized pair of attention vectors.
  For numerical stability near saturated logits, clamp log-probs at
  `log(eps)` with `eps = 1e-9`.
- The two-stream path doubles register pressure in canary steps; p_c=1/64
  is the default to amortize. May tighten if overhead > 2%.
- Initial design logs δ per layer. Whether to log per-head (for heavy-head
  analysis) is an open question (see REPORT.md § Open Questions).

### 4.2 PID Precision Controller (`core/precision_controller.py`)

**Purpose:** map accumulated δ(ℓ) into a discrete per-layer target precision
`b_ℓ ∈ {2, 4, 8}` for subsequent block allocations.

**Inputs:**
- δ_ℓ^(t): the canary estimate at controller tick `t`, pulled from the
  pinned scratchpad.
- Setpoint `τ` (default 0.01, swept over {0.001, 0.005, 0.01, 0.02, 0.05}
  in calibration).
- `kv_utilization` ∈ [0,1] from the allocator.
- `precision_budget` ∈ [0,1] from the SLO scheduler.

**Outputs:**
- Per-layer precision `b_ℓ` emitted to the allocator.
- Controller state: EMA δ̂_ℓ, last update tick, current level.

**Tick cadence:** every N_c = 100 decode steps. This bounds controller
overhead to a Python thread invocation per 100 tokens.

**Update law (sketch — precise form recorded in code):**
```
δ̂_ℓ^(t) = 0.9 · δ̂_ℓ^(t-1) + 0.1 · δ_ℓ^(t)
τ_eff   = τ · g(kv_utilization, precision_budget)
         where g is monotone non-decreasing in utilization (relaxes
         the setpoint under pressure) and monotone non-increasing in
         precision_budget (tight budget → tight setpoint).
```

Hysteresis: two thresholds per level boundary (upgrade threshold >
downgrade threshold) to prevent oscillation. Default: 20% band.

**Invariants:**
- INV-2: controller emits targets for **new** allocations only. It never
  rewrites existing blocks.
- INV-3: emitted `b_ℓ` must pass through the scheduler broadcast path
  before any TP rank uses it in a decode step.

**Known limitations:**
- Pure P control for v1. Full PID (with integral + derivative terms) is a
  follow-up if we observe sustained setpoint tracking error.
- EMA constant 0.9 not yet ablated. Recorded as an open question.

### 4.3 SLO-Coupled Precision Budget (`core/slo_budget.py`)

**Purpose:** couple per-layer precision to end-to-end SLO pressure so that
under burst load the KV pool shrinks and batch size grows.

**Inputs:**
- Rolling p99 TPOT from vLLM metrics.
- Configured SLO (e.g. p99 TPOT ≤ 50 ms for Llama-3.1-8B).

**Outputs:**
- `kv_precision_budget` ∈ [0, 1] exposed to the PID controller.
- Transitions are rate-limited; updates are serialized with scheduler ticks.

**TP correctness argument (single-node only):**
1. The scheduler is the single point of truth for which requests run in the
   next decode step.
2. Before admitting a new decode step, the scheduler broadcasts the current
   `b_ℓ` vector to all TP ranks via the existing global barrier.
3. All ranks either observe the update before the decode step or not at
   all. There is no window in which rank A sees `b_ℓ=4` while rank B sees
   `b_ℓ=8` mid-step.
4. Because precision applies only to new blocks (INV-2), in-flight blocks
   are unaffected and no cross-rank reconciliation is needed.

This argument holds on a single-node TP (≤8 GPUs, NVLink). Multi-node TP
would require cross-node precision broadcasts that may not fit within
scheduler step boundaries; the consequent transient shard-level precision
mismatch requires recovery logic that is **not designed** and is
**explicitly out of scope** for the MLSys 2026 submission (see REPORT.md).

**Invariants:**
- INV-3: no asynchronous update of `b_ℓ` across TP ranks.

### 4.4 Precision-Aware Block Allocator (`core/block_allocator.py`)

**Purpose:** serve paged block allocations at the precision currently
requested by the controller, with three separate free lists (INT2, INT4,
INT8).

**Inputs:**
- `allocate_with_precision(layer_id, precision, num_blocks)`
- `free(block)` — returns block to its native precision pool.

**Outputs:**
- `PhysicalTokenBlock` instances with a `precision` field.

**Data layout decision:** three disjoint physical pools, sized per
configuration. A block allocated at INT8 has the same geometry as an INT4
block but twice the per-element width; pools are pre-sized at startup from
config. Migration between pools is **not supported** (INV-2).

**Invariants:**
- INV-1: the only state attached to a block beyond the block's own data
  is a 2-bit precision tag stored in the block table (not per-block
  metadata). Total auxiliary cost: 2 bits × num_blocks, which rounds to
  zero against the quantized KV pool.
- INV-2: free and re-allocate is the only state transition; no in-place
  rewrite.

### 4.5 Write-Ahead Precision (`experimental/write_ahead_precision.py`)

**Purpose (experimental):** identify heavy-hitter (attention sink) blocks
**before** they are written, so that they can be allocated at INT8 from
first write with zero information loss.

**Signals:**
1. **Positional prior** — positions 0..3 are empirically attention sinks
   across all modern decoder-only models; they are allocated at INT8
   unconditionally. No warm-up needed.
2. **Running accumulator** — for each layer ℓ and position i, track an
   EMA of the attention weight w_{ℓ,i}. After warm-up window W=256
   tokens, if w_{ℓ,i} > θ=0.15, allocate future blocks referencing
   position i at INT8. (Specifically: the block containing the key/value
   for a token that a downstream query repeatedly attends to.)

**Warm-up window W=256:** < 1% of context at 32K+. For short requests
(< 512 tokens), WAP provides little benefit but positional priors (0..3)
still apply.

**Memory overhead:** `f × INT4_pool` where `f` is the measured
heavy-hitter fraction. Bound for inclusion: ≤15% (see decision rule).

**Decision rule:** if EchoKV-4C (full + WAP) improves RULER@128K by
**≥0.5pt** over EchoKV-3C at **≤15% memory overhead**, WAP ships as
Component 4. Otherwise WAP is reported as a **negative result** in
Discussion and the paper ships as the three-component system.

**Invariants:**
- INV-2: WAP's decisions are inputs to the allocator for new blocks. WAP
  never rewrites a block.

## 5. Data Flow — One Decode Step

1. Scheduler admits a batch; broadcasts current `b_ℓ` vector to TP ranks.
2. For each layer ℓ, kernel runs paged attention. With probability 1/64
   the step is marked canary and two streams run; the canary branch writes
   δ(ℓ) to the pinned scratchpad.
3. New KV entries produced by this step request a block from the allocator
   at precision `b_ℓ` (WAP may override to INT8 for positional-prior or
   accumulator-flagged positions).
4. Every N_c=100 decode steps, the PID thread reads δ̂, consults the SLO
   budget, and updates `b_ℓ`.
5. Updated `b_ℓ` is staged; the scheduler broadcasts it before the next
   decode step.

### Canary operating regime (validated by motivation study v8.1)

The canary is a **per-step sensor**, not a per-request classifier — δ(t) at
each decode step is correlated with per-step ΔNLL(t) on the workloads
where INT4 actually causes measurable degradation. v8.1 (n=100/workload,
``max_new_tokens=32``) measured this directly via two teacher-forcing
forward passes per request:

- **In scope**: retrieval-dominated long-context decode (RULER-32K). Per-
  request mean ρ_diff = +0.155 ± 0.019 (8.2σ from zero); on the answer-
  emission window (first 8 decode tokens) ρ_raw = +0.367 ± 0.034. These
  are precisely the steps with peaked attention to specific K positions,
  where small INT4 K errors are amplified by the softmax.
- **Out of scope**: diffuse-attention reasoning (GSM8K-CoT). Per-request
  mean ρ_raw = −0.023 ± 0.019; no window or layer subset recovers signal.
  The mechanism that makes the canary work (peaked attention amplifies K
  quantisation error) is absent here. The canary is documented as not
  designed for this regime; the controller's response on such workloads
  should fall back to a static or workload-class precision policy.

The canary kernel runs **on all 32 layers** at rate ``p_c = 1/64``. Per-
layer ``b_ℓ`` is discovered dynamically by the PID controller from the
EMA δ̂_ℓ vector; the kernel does not hard-code a layer subset. See § 9.5
for the v8.1 per-layer finding and why it does not affect kernel scope.

## 6. Implementation Stack

| Layer                  | Choice                                                  |
|------------------------|---------------------------------------------------------|
| Base system            | vLLM V1 (commit pinned in CHANGELOG)                    |
| Attention engine       | FlashInfer (canary kernel extension)                    |
| Kernel language        | Triton                                                  |
| TP plumbing            | Megatron-LM / NeMo (for any TP-related pieces)          |
| Primary model          | Llama-3.1-8B-Instruct                                   |
| Secondary model        | Nemotron-Nano-V2-12B                                    |
| Hardware target        | 8×H100-SXM5 80GB NVLink (single node)                   |
| Quantization baseline  | KIVI (group-size 32, asymmetric INT4, per-group scales) |

## 7. Open Questions / TODOs (per component)

See REPORT.md § Open Questions for the live list. Summary here:

- **Canary kernel**: per-layer or per-layer-per-head δ? (blocks motivation
  study if per-head is needed).
- **Controller**: EMA coefficient (0.9 default) not ablated.
- **Controller**: whether P control is sufficient or we need PI/PID.
- **Scheduler coupling**: rate-limiting of budget updates — cadence TBD.
- **WAP**: whether per-head signals should gate per-head (not per-block)
  precision; deferred until motivation study.

## 8. Invariants (mirror of main-prompt list)

- **INV-1**: three-component core has zero memory overhead beyond the
  quantized KV pool. No auxiliary per-block buffer.
- **INV-2**: no existing block's contents may be modified. Precision
  decisions apply to new allocations only.
- **INV-3**: under TP, precision updates broadcast through the scheduler
  before the next decode step on any rank.
- **INV-4**: the `w/o Shadow Canary` ablation must produce output
  identical to KIVI-4.
- **INV-5**: all benchmark numbers in REPORT.md are from real experiments
  (TBD until measured).

## 9. Design Decisions

### 9.1 Motivation study: ΔNLL = NLL_INT4 − NLL_FP16 as Spearman ρ target

**Decision (2026-04-25, session 6):** The motivation-study gate correlates
δ against ΔNLL = NLL_INT4(completion) − NLL_FP16(completion), where both
NLL values are computed by teacher-forcing the FP16 reference completion
through the model under each KV configuration.

**Why ΔNLL and not Δacc:** Binary accuracy (Δacc = acc_FP16 − acc_INT4) failed
at gate v5 for two compounding reasons: (1) with ~1% degradation rate, only
~24/420 requests changed outcome — far too sparse for Spearman ρ to converge;
(2) even among changing requests, the mean δ was identical to the stable
majority (0.0077 vs 0.0072), meaning prefill attention KL cannot distinguish
which specific requests will flip. ΔNLL is a continuous metric that provides
a gradient of "how much harder does INT4 find this exact completion" on every
request, not just the ~1% that happen to change the binary decision boundary.

**Two-pass design (same wall-clock as Δacc):**
1. FP16 `model.generate()` → reference completion tokens
2. FP16 teacher-forcing forward on (prompt + completion): → NLL_FP16
3. INT4 teacher-forcing forward (hook installed): → NLL_INT4

The generate call dominates wall time. The two teacher-forcing passes are
single forward passes (O(sequence_length), no autoregressive loop) and add
~10% overhead.

**Gate threshold:** Signed ρ ≥ 0.4 (not `|ρ|`). ΔNLL is unambiguously signed —
higher δ should predict higher ΔNLL (INT4 finds the completion less likely) —
so a negative ρ is a separate failure mode ("inverted signal") rather than a
weaker version of the positive signal. A lower threshold than the previous 0.5
is appropriate because ΔNLL is a continuous, lower-variance signal that should
admit a cleaner correlation even at moderate n.

### 9.2 INT4KVHook: approximate simulation of KIVI KV-cache quantization

**Decision:** The motivation study simulates INT4 KV inference by installing
`register_forward_hook` on every `k_proj` and `v_proj` layer, replacing each
projection output with `dequant(quant(output))` before it enters FlashAttention.

**Known approximation:** KIVI introduces quantization noise at KV *cache-write*
time — the noise is injected between layers, not within a single layer's forward
computation. The hook injects noise at projection-output time, which means the
quantized K/V values also propagate through the current layer's own attention
computation. This slightly overstates the per-layer degradation compared to
actual KIVI behavior. For the purpose of the motivation study — verifying that
δ and Δacc are correlated — the approximation is appropriate: the causal signal
exists and is detectable even if the magnitude is slightly amplified. The canary
kernel and production path are not affected by this approximation.

### 9.3 Pre-validation before full gate run

**Decision:** Before the full 400-request run, `validate_signal()` runs a
20-sample GSM8K batch (seed=99, disjoint from the main seed=0 run) with both
FP16 and INT4 passes. The run aborts if mean Δacc ≤ 0.01 AND n_degraded = 0.

**Rationale:** A broken `INT4KVHook` (wrong layer names, no-op quantization,
float32/bfloat16 mismatch) produces Δacc = 0 across all requests — identical
to the null hypothesis. Without pre-validation this failure mode would consume
400 × 2 × 8 GPU-requests and produce an uninterpretable ρ = undefined result.
The AND condition is intentional: mean alone can cancel a bimodal distribution
where half the requests degrade sharply and half trivially improve; requiring
n_degraded > 0 catches that case.

### 9.4 Gate v8: per-decode-step δ(t) vs ΔNLL(t) within a single request

**Decision:** Replace the per-request scalar gate (v3–v7) with a within-
request time-series gate. For each request, capture per-layer Q/K and per-
position logits over (prompt + FP16 completion) under two teacher-forcing
forward passes (FP16, then K-only INT4 hook). Compute
``δ(t) = mean_layer KL(α_p^INT4 ‖ α_p^FP16)`` from captured causal attention
rows and ``ΔNLL(t) = NLL_INT4(c_{t+1}) − NLL_FP16(c_{t+1})`` from the
captured logits, then take per-request Spearman ρ over decode steps. The
gate fires SELECTIVE PASS if any workload's mean per-request ρ_raw or
ρ_diff (first-differenced, detrended) exceeds 0.15; PASS if all workloads
do; FAIL if every workload is below 0.05.

**Why the reframe was necessary:** v3–v7 gates correlated per-request
prefill δ̄(r) against per-request completion ΔNLL(r). Decomposition
(REPORT.md § Motivation v8): prefill δ averages KL over all prompt
positions and is dominated by prompt-distribution properties (8% CoV
across GSM8K); completion ΔNLL is dominated by per-step decode difficulty
(54% CoV). The two quantities live in different spaces and within-
workload ρ ≈ 0 was a structural fact, not a measurement bug. The
production canary itself does not operate on prefill δ — it computes
δ(t) at each decode step. v8 measures the canary's actual quantity.

**Why two teacher-forcing passes is cleaner than autoregressive decode:**
(1) No EOS / padding contamination — the targets are known FP16
completion tokens, no post-EOS pad averaging that polluted v3–v7.
(2) Same wall-clock as the v5–v7 dual-generate design.
(3) Aligns ΔNLL(t) and δ(t) at the same query position by construction.
(4) The teacher-forcing target is the FP16 completion (not the INT4 path's
own decode), so ΔNLL(t) measures "how much INT4 raises surprise on the
FP16-equivalent output" — directly the canary's design question, without
conflating path differences with quality differences.

**Detrending option B (raw + first-differenced ρ):** Both δ(t) and
ΔNLL(t) trend upward as the KV cache grows, so naive ρ would be
trend-inflated. We report both raw ρ (the production canary's
operational quantity) and ρ on first differences (a clean test of
step-by-step co-variation). v8.1 found ρ_diff = +0.155 > ρ_raw = +0.057
on RULER — the per-step signal is real and was actually being masked by
anti-correlated trend in the raw series.

**First/second-half robustness check (not a gate condition):** The
analyzer also reports ρ on the first and second halves of each
completion. Used to cross-check the accumulation hypothesis
("signal should grow with KV cache length"); not a gate threshold.

### 9.5 Motivation study finding: per-layer signal structure

**Finding (v8.1, RULER-32K, n=100, max_new_tokens=32):** Per-layer mean
within-request ρ̄(δ_ℓ(t), ΔNLL(t)) varies systematically by layer
position in Llama-3.1-8B-Instruct's 32-layer stack. The strongest
layers are 21–27 (mid-late, post-mid block) — layer 24 alone gives
ρ̄ = +0.157 ± 0.020 (7.8σ from zero). Early layers (2, 3, 9, 13, 14)
are weakly negative or near-zero.

**This finding belongs in the paper's Figure 1 (panel C: per-layer ρ̄
bar chart) — not in the kernel design.** The Triton canary kernel
(``kernels/canary_attention.py``) runs on all 32 layers; per-layer
precision is decided dynamically by the PID controller from the
runtime EMA δ̂_ℓ vector (see § 4.2). The closed-loop premise of EchoKV
is that the controller *discovers* which layers need higher precision
on the current conversation — hardcoding a layer subset on the kernel
side breaks that premise and breaks generalisation to other models
where the high-signal layers may differ.

**Why we do not pre-scope the kernel to layers 21–27:**

1. The v8.1 result is for one specific model (Llama-3.1-8B-Instruct)
   and one specific workload (RULER-32K). On Nemotron-Nano-V2-12B (the
   secondary model) the high-signal layers may sit at different
   indices because the attention-pattern profile differs.
2. The mechanism v8.1 uncovered is "peaked-attention layers carry
   signal." Whether a given layer's attention is peaked depends on the
   *current conversation*, not on the layer index. A layer that's
   peaked on RULER's needle-retrieval may be diffuse on a different
   long-context task.
3. The runtime EMA δ̂_ℓ already reflects the per-layer signal — when a
   layer's δ stays low under the current conversation, the controller
   keeps that layer at INT4; when it rises, the controller upgrades.
   This *is* the dynamic equivalent of the per-layer scoping, derived
   from observation rather than calibration.

**What the per-layer finding does justify in the paper:**
- Figure 1 panel C as motivation evidence that per-layer variance
  exists in the underlying signal (not a flat KL across layers).
- A note in § Discussion that the controller's expected behaviour on
  Llama-3.1-8B is to converge on higher precision for layers ~21–27 on
  retrieval-heavy traces — a falsifiable prediction the controller
  ablation can check.

### 9.6 Initial tier as a function of prompt length (cost-asymmetry derivation)

**Decision (2026-04-28, post-session-11 review):** the controller's
initial per-layer tier is **a function of prompt length**, not a
constant. At request start the controller has zero information;
warmup-window blocks (the first ``N_c=100`` decode steps) are
permanent (INV-2). The costs of getting the initial tier wrong are
asymmetric:

```
Cost_low   = E[acc loss | hard request] × P(hard request) × N_warmup
           = constant — independent of T_total

Cost_high  = extra_memory × (T_warmup / T_total)
           → 0 as T_total → ∞
```

At long context the warmup is a vanishing fraction of the request,
so the memory cost of starting high amortises away. The accuracy
penalty from starting low does not. At short context the costs are
symmetric and INT4 is the minimum-regret prior. This gives a
principled three-regime rule:

| prompt length | initial tier | regime                                                                |
|---------------|--------------|-----------------------------------------------------------------------|
| ``< 512``     | INT2         | warmup dominates total request; exploration safe                       |
| ``512–8K``    | INT4         | symmetric costs; minimum-regret prior; matches KIVI-4 baseline         |
| ``≥ 8K``      | INT8         | warmup negligible; protect accuracy through warmup window              |

**Why this matters:** the v1 implementation in ``canary_vllm.py``
(Session 11) used ``initial_level=INT8`` globally because that's what
fixed the 64K accuracy number. The math derives the same INT8 for
long context but for a principled reason. Short and medium contexts
should not start at INT8; the global INT8 default would waste
memory on every short request. Session 12 implements the
function-of-prompt-length rule and replaces the global default.

**Open question:** the thresholds 512 and 8K are first approximations
from ``threshold ≈ N_c × tokens_per_block × constant``. Calibrate on
a workload mix once Path A (adaptive τ) is in place — until then the
threshold sweep would conflate two unprincipled choices.

### 9.7 Sensor locus problem at long context (the canary's blind spot)

**Decision (2026-04-28, post-session-11 review):** The current
canary's locus — fire at decode steps, measure KL on attention
weights — is empirically inadequate at long context. Session 11
showed that on RULER@64K under static INT2, accuracy collapses to
**0.000** but the canary reports *low* δ̂ across all layers, and the
controller observing low δ̂ *downgrades* toward INT2. The sensor is
not detecting the problem it is supposed to detect.

**Diagnosis (mechanism):** the canary's causal chain assumes peaked
attention:

```
K_INT4 ≠ K_FP16 → δ(t) > 0 → attention weight distortion → output error → accuracy loss
```

The chain holds when one query token attends sharply to a small
number of positions. At 64K with RULER, the *decode* phase is 5–20
tokens; during those tokens the model attends diffusely over 64K
positions hunting for the needle. **Diffuse-vs-diffuse softmax
pairs have small KL divergence** even when the underlying K vectors
have accumulated huge quantisation error in the cache. The damage
occurs during **prefill** (each K vector quantised at INT2 as it
enters the cache); the sensor fires only during **decode** (where
attention is diffuse and KL is small). The sensor is at the right
*architectural location* but at the wrong *moment*.

**Implication:** for long-context workloads, the v1 design needs
either a different measurement *moment* or a different measured
*quantity*:

- **Option A — prefill canary.** Fire on a sample of prefill
  positions. Directly measures the K-quantisation error
  accumulating in the cache. Architecturally more complex
  (the production paged kernel writes prefill K into the cache;
  intercepting requires different plumbing than decode-step
  intercept), but addresses the root cause.
- **Option B — different measured quantity.** KL on attention
  weights is insensitive to quantisation when softmax is diffuse.
  Attention-output L2 distance
  ``‖softmax(QK_INT4)·V_INT4 − softmax(QK_FP16)·V_FP16‖`` may be
  more sensitive because V quantisation error propagates linearly
  through diffuse-attention averaging. Earlier session 7 diagnostic
  found δ_OUT had lower per-request variance than δ_KL on prefill
  signal, but at long-context decode the comparison may invert.

**What this does not affect:** the canary's signal at *short* and
*medium* context (where attention is naturally peaked on a few
positions). Sessions 8–11 showed the canary fires correctly there.
The long-context blind spot is the open architectural question
Session 12 must resolve before claiming the canary works at the
context lengths the paper highlights.

**Pre-decision diagnostics (Session 12 task):**

1. Compare δ̂ distributions at 4K / 32K / 64K under matched
   conditions. Monotone drop with context confirms the
   diffuse-attention hypothesis.
2. Compute decode-step attention entropy on a static-INT2 64K RULER
   request. High entropy + acc=0 + low δ̂ confirms the blind spot.

Only after these diagnostics decide between Option A and Option B
should the canary be modified.

### 9.8 Multi-precision canary — neighbors-only (post-session-12 finding)

**Decision (2026-04-28, Session 12, refined post-review):** the
canary probes the **current tier and its one-step neighbours** per
fire — never all three precisions at once. The controller's decision
rule changes from "compare scalar δ̂ to setpoint τ" to "pick the
lowest precision whose δ̂ stays below a divergence budget
``τ_budget``." Tiers measured per state:

| current tier | tiers measured | reason |
|--------------|----------------|--------|
| INT2 | {INT2, INT4}        | only INT4 is reachable in one transition |
| INT4 | {INT2, INT4, INT8}  | both neighbours + current — the all-three case |
| INT8 | {INT4, INT8}        | only INT4 is reachable in one transition |

The argmin-with-budget rule:

```
b* = min { b : EMA(δ̂_b) < τ_budget,  b ∈ measured_tiers(current) }
```

i.e. the most aggressive (lowest-bit) tier whose measured divergence
is still within budget, restricted to tiers we currently measure.

**Why this is the right fix instead of smarter τ tuning.** Session 12
diagnostics produced two key empirical facts:

1. **The sensor's signal IS separating tiers.** At RULER-32K, with
   ``simulate_quantization`` set to bits=4, δ̂ averages **0.0133
   under static INT4** (acc=1.000) and **0.0208 under static INT2**
   (acc=0.125). The 1.6× gap is real, statistically clean, and
   reflects the genuine difference in attention sensitivity between
   the two tiers.
2. **A single setpoint τ cannot reliably exploit the gap.** The
   gap window (0.0133 ≤ δ̂ ≤ 0.0208) is narrow. Naive adaptive τ
   rules — ``p50(δ̂) × scale`` for any scale we tried — landed
   either inside one distribution's deadband (controller holds at
   whatever it started at) or outside both (controller downgrades
   uniformly). Session 11's manually-tuned ``τ=0.005`` happened to
   land just below the gap on the upgrade side, which is why it
   produced the Pareto win — but only because the manual tuner had
   ground-truth accuracy to optimise against.

The structural problem: the canary's ``simulate_quantization``
always quantises at one fixed bit-width (default 4). δ̂ measures
"what would happen if we further INT4-quantised the current K" —
it does *not* directly answer "which precision should the next
block be allocated at?" The controller has to infer that from a
single scalar.

**The architectural fix.** Have the canary fire quantised simulations
at the **current tier and its one-step neighbours** per call — i.e.
two probes when current tier is INT2 or INT8, three when it's INT4.
The estimator maintains per-tier EMAs only for tiers currently being
probed; when the controller transitions, the new neighbour starts
receiving samples immediately.

``τ_budget`` is a **divergence budget** — the maximum acceptable
attention-weight KL between the chosen-tier path and FP16. Because
each tier is probed independently, ``τ_budget`` is a property of the
workload's accuracy tolerance, not of the per-tier δ̂ scale. A single
``τ_budget`` can produce sensible decisions across 4K / 32K / 64K
without recalibration — the per-context τ tuning Session 11 needed
goes away.

#### Why neighbours-only is correct, not just cheaper

The controller's decision space at any tick is **stay / upgrade one
step / downgrade one step**. Hysteresis + ``min_dwell_ticks`` already
forbid two-step transitions. So measurements that aren't neighbours
of the current tier provide information that **cannot be acted upon
in a single tick**:

- INT2 cannot transition directly to INT8; it must first dwell at
  INT4. So measuring δ̂_8 while at INT2 is wasted compute — the
  controller will not move there until it has first moved to INT4.
- During the mandatory INT4 dwell, the canary fires at INT4 and both
  neighbours, which now includes INT8. By the time the controller
  decides whether to move INT4 → INT8, it has fresh EMA(δ̂_8) from
  the INT4 phase. **The "staleness" of non-adjacent EMAs self-heals
  through the mandatory intermediate state.**

INT4 is the symmetric case where neighbours-only equals all-three —
which is also the default starting tier and the most common state.
The savings only materialise when the controller has converged to
an extreme (INT2 or INT8); those are precisely the states where the
savings are also most justified ("a layer firmly at INT2 doesn't
need to keep checking whether INT8 would help").

#### Monotonicity invariant (bug-detection assertion)

By construction, ``δ̂_2 ≥ δ̂_4 ≥ δ̂_8 ≥ 0`` for any given (layer,
step) — more aggressive quantisation produces strictly larger
divergence from FP16. INT4 is a strict refinement of INT2's
representable values; INT8 is a strict refinement of INT4's. If the
ordering is violated on real inference, there is a bug in the
quantisation simulation, the KL computation, or the canary's
random-number reuse across tier probes. Session 13 must add a
runtime assertion that flags any ``(layer, step)`` pair where the
monotone ordering fails. Tolerate small float-noise inversions
(< 1e-6) but treat anything larger as a hard failure.

#### Cost arithmetic — overhead measurement is Session 13's task

Per-fire cost relative to the single-precision canary:

| current tier        | tiers probed | per-fire cost ratio |
|---------------------|--------------|---------------------|
| INT2 or INT8        | 2            | 2× single-precision |
| INT4                | 3            | 3× single-precision |
| **average (uniform)** | **2.33×**  | (2 + 3 + 2) / 3     |

Session 9's single-precision canary measured at **+0.54%** TPOT
overhead at RULER-32K with ``p_c = 1/64`` decode-only. Linear
extrapolation: neighbours-only at ~1.26% average, all-three would
have been ~1.62%. Both nominally under the 2% budget — but the
0.74% headroom for neighbours-only vs 0.38% for all-three is
meaningful. **Measure this directly in Session 13** — do not trust
the linear extrapolation. Three failure modes to watch for:

1. CUDA stream contention if all probes share one canary stream;
   may need one stream per tier or batched probes.
2. Memory-bandwidth saturation reading K twice/thrice (once per
   probe).
3. Per-call Python overhead from the additional branching not
   amortising at low ``p_c``.

If measurement shows >2%, options are: lower ``p_c`` to ``1/128``,
scope to a layer subset (decided by the canary's own δ̂
distribution, not by hardcoding), or accept slightly higher
overhead and update the paper's claim.

#### Setting ``τ_budget`` — calibration result (Session 13)

The calibration probe (``evaluation/serving/calibrate_tau_budget.py``)
runs RULER@4K under static INT2 / INT4 / INT8 with the multi-
precision canary observing. Per-tier EMA distribution under each
static configuration:

| static cache | accuracy | EMA(δ̂_INT2) mean | EMA(δ̂_INT4) mean | EMA(δ̂_INT8) mean |
|--------------|----------|--------------------|--------------------|--------------------|
| INT8 | 1.000 | 0.342 | 0.022 | 0.0001 |
| INT4 | 1.000 | 0.368 | 0.023 | 0.0001 |
| INT2 | **0.050** | 0.483 | 0.030 | 0.0001 |

Two crucial empirical facts:

1. **Each tier's EMA scale is intrinsic to the probed precision**,
   nearly independent of the cache state. INT2 quantisation always
   produces ~0.35 KL divergence; INT4 always ~0.025; INT8 always
   ~0.0001. The cache tier shifts each by ~30% but the order-of-
   magnitude separation between tiers stays.
2. **The gap between safe and broken is wide.** Static INT4 yields
   acc=1.000 with EMA(δ̂_4) max 0.039. Static INT2 yields acc=0.05
   with EMA(δ̂_2) min 0.213. The gap between "safe at INT4" (≤ 0.04)
   and "broken at INT2" (≥ 0.21) is more than half an order of
   magnitude.

**Recommended ``τ_budget = 0.05``.** Lies cleanly in the gap: well
above INT4's noise floor (≈ 0.025) so INT4 always passes the budget
on RULER, well below INT2's typical EMA (≈ 0.36) so INT2 is rejected.
A single static value is sufficient for RULER across context lengths
because the per-tier EMA scales don't shift with context (Session 13
verified at 32K and 64K).

Adaptive ``τ_budget`` (phase-1 FP16 probe) remains an option for
workloads where the per-tier scales drift; not needed at the contexts
the paper currently claims.

#### Empirical validation (Session 13)

End-to-end test: RULER@32K and RULER@64K, n=30, multi-precision
canary with ``initial_tier_for_context()`` and ``τ_budget = 0.05``,
**no manual parameter overrides**.

| context | FP16 | static INT4 | static INT2 | EchoKV-3C (multi-precision) |
|---------|------|-------------|-------------|--------------------------------|
| 32K | 1.000 | 1.000 | 0.133 | **1.000 @ +75.0% mem savings** |
| 64K | 1.000 | 1.000 | 0.000 | **1.000 @ +75.4% mem savings** |

Converged tier vector at 64K: ``[2, 4, 4, 4, ..., 4]`` — 1 INT2 + 31
INT4. The argmin-with-budget rule picks INT4 as the lowest-bit tier
with EMA below 0.05. EchoKV-3C achieves exactly the same memory
profile as static KIVI-4 in this regime — but the choice is
**derived** from the canary signal, not calibrated.

**Overhead at ``p_c = 1/64``, observe-only, RULER@4K, n=50:**
``+0.09%`` throughput overhead (vs ``+1.44%`` for the
single-precision canary on the same run). The CUDA-stream
parallelism absorbs most of the 2.33× per-fire cost; the multi-
precision canary is well within the 2% TPOT budget.

**Monotonicity invariant verified empirically:** zero violations
across 32 × 3 = 96 KL computations on real RULER inference (and
0 violations across 50 random-tensor draws in unit tests). The
``δ̂_2 ≥ δ̂_4 ≥ δ̂_8`` ordering holds by construction; the runtime
assertion catches potential implementation bugs without paying
ongoing cost in production.

#### What this validation does NOT show (post-review caveat)

The Session 13 acceptance run produced converged tier vectors
that are **operationally identical to static KIVI-4** on RULER@32K
+ 64K:

```
32K: 0 INT2 / 32 INT4 / 0 INT8  →  uniform INT4  →  KIVI-4
64K: 1 INT2 / 31 INT4 / 0 INT8  →  ≈ KIVI-4
```

EchoKV-3C and static KIVI-4 produce **the same memory profile and
the same accuracy** on this workload. The controller's runtime
adaptation is contributing nothing beyond the static baseline.
This is a **correctness result** (the controller picks the right
tier and doesn't make things worse) but **not a Pareto-improvement
result** vs KIVI-4. The reported "+75% mem savings vs FP16" is
also what static KIVI-4 saves by definition — there is no gap to
attribute to the closed loop.

The result does not distinguish two scenarios:

- **A.** ``τ_budget = 0.05`` correctly identifies INT4 as the safe
  tier on every layer; INT4 is genuinely the right answer for
  every RULER layer at these context lengths.
- **B.** ``τ_budget = 0.05`` is too permissive at long context;
  INT4's per-tier EMA falls below 0.05 everywhere by accident,
  so the controller never has reason to upgrade specific layers
  to INT8. Tightening τ_budget might expose layer-specific
  structure.

The Session 13 calibration ran at 4K only; Session 14 must run it
at 32K / 64K / 128K to inspect whether ``EMA(δ̂_4)`` varies across
layers at long context. If it doesn't, RULER on Llama-3.1-8B is
simply not a workload where the closed loop can beat KIVI-4 — the
model is robust to INT4 KV at all tested context lengths and the
controller correctly converges on KIVI-4-equivalent.

The Pareto-improvement claim still requires a workload where:

1. Static KIVI-4 actually loses some accuracy (does not happen on
   RULER + Llama-3.1-8B at any context up to 128K).
2. EchoKV-3C recovers that accuracy by upgrading critical layers
   to INT8.
3. Result: less memory than FP16, more accuracy than KIVI-4.

Candidates documented in CHANGELOG session_20260428_13 critic
addendum: LongBench multi-hop QA, multi-needle RULER, 70B model.

The result that **does** stand unconditionally regardless of the
accuracy-Pareto question:

```
Multi-precision canary @ p_c = 1/64 on RULER@4K:  +0.09% TPOT overhead
```

The runtime cost of per-tier quantisation probing is below 0.1%.
That's a clean engineering claim about the canary mechanism's
efficiency, independent of whether the closed-loop control finds
a workload where it can beat KIVI-4.

#### Session 14 calibration sweep — Possibility B confirmed; not actionable on RULER+8B

The post-Session-13 calibration probe at 32K, 64K, 128K resolved
the A-vs-B question. **Possibility B is correct**: τ_budget=0.05
is too loose, and there is real per-layer δ̂_4 variation that a
tighter budget could exploit.

| context | δ̂_4 min | δ̂_4 p50 | δ̂_4 max | max/min |
|---------|----------|----------|----------|---------|
| 4K   | 0.0078 | 0.0228 | 0.0385 | 4.9× |
| 128K | 0.0022 | 0.0044 | **0.0159** | **7.3×** |

The 7.3× per-layer spread at 128K is wider than the 4.9× spread at
4K — the diffuse-attention effect *concentrates* into a few high-
divergence layers at long context rather than spreading uniformly.

**However, exploiting this signal does not produce a Pareto
improvement on RULER+8B**, for two structural reasons:

1. **Static INT4 already produces acc=1.000 at every context.** A
   tighter τ_budget that pushes high-divergence layers to INT8
   raises memory cost without adding any accuracy headroom — the
   opposite of a Pareto win. INT4 saturates the accuracy ceiling.
2. **INT2 EMA is rarely below INT4 EMA.** Across all contexts,
   only a single layer at a time has δ̂_2 low enough to make INT2
   a valid choice. That gives at most ~1 layer per context that
   could safely use INT2 — too few to differentiate the closed-
   loop tier vector from KIVI-4 in any meaningful way.

The single-τ argmin-with-budget rule has a related structural
limit: per-tier EMAs operate at very different scales (INT2 ≈
0.05–0.50, INT4 ≈ 0.005–0.04, INT8 ≈ 1e-5), and **no single
τ_budget value can simultaneously admit some layers to INT2 and
upgrade others to INT8**. To use the full INT2/INT4/INT8 ladder
the controller needs **per-tier τ_budget**:

```
τ_INT2 ≈ p25 of observed δ̂_2  → admits lowest-divergence layers to INT2
τ_INT4 ≈ p50 of observed δ̂_4  → upgrades top-half-divergence layers to INT8
τ_INT8 ≈ ∞                    → trivially safe
```

Per-tier τ_budgets are workload-relative (use runtime percentiles
of observed EMA), so the controller adapts without offline tuning.
This is a future implementation task — **but only after a workload
is found where the closed loop has room to operate**. Implementing
per-tier τ on a workload where INT4 already saturates accuracy
just produces fancier-looking versions of uniform-INT4. The
session priority is finding a workload where KIVI-4 actually
fails (LongBench multi-hop QA, multi-needle RULER, MMLU-Pro
long-context, or 70B) before any further sensor-side
sophistication.

#### Initial tier (§9.6) is necessary but not sufficient

§ 9.6 protects the **warmup window** (first ``N_c=100`` decode
steps) by starting the controller at a context-appropriate tier.
After the first controller tick, the scalar-sensor problem this
section addresses takes over. The two fixes compose:

- § 9.6 — protect blocks written during warmup (cannot be
  retroactively modified, INV-2).
- § 9.8 — protect blocks written *after* warmup by giving the
  controller the right signal to choose tiers from.

Session 12 verified the first half by re-running with principled
initial tier alone — accuracy still collapsed because the scalar
sensor downgraded everything to INT2 after warmup. The
multi-precision canary closes that gap.

#### What this fix does NOT solve

If the diffuse-attention hypothesis is *also* true at very long
context (128K+), all neighbouring-tier δ̂ probes may simultaneously
be small under any quantisation level — KL between two diffuse
softmax distributions is small regardless of which tier the K is
quantised at. In that regime the multi-precision canary inherits
the same blind spot the single-precision canary has, and the
prefill-canary path (§ 9.7 Option A) would still be needed.

However, Session 12's static INT4 vs INT2 comparison at 32K showed
δ̂_INT2 = 0.021 vs δ̂_INT4 = 0.013 (1.6× separation, the exact gap
the new sensor needs), and the gap is preserved at 64K (the
absolute scale shifts but the ratio holds). So the multi-precision
approach is at least worth trying first — it's a smaller change
than restructuring the canary's measurement moment.

#### Session 14 — multi-needle RULER does not break INT4 either

The Session-13 conclusion ("KIVI-4 saturates accuracy on RULER+8B")
predicts that *any* retrieval workload where attention can
concentrate on one position will produce the same uniform-INT4
result. Session 14 tested the cheapest counter-hypothesis: force
attention onto **multiple positions simultaneously** by inserting
five labeled magic numbers at evenly-spread positions in the
haystack and asking for all five at once. If per-key INT4 noise
amortizes only when attention is sharp on one key, multi-key
retrieval should expose it.

| context | FP16 acc | INT4 acc | paired Δ ± SE |
|---:|---:|---:|---:|
| 32K  | 0.947 | 0.940 | −0.007 ± 0.018 |
| 64K  | 0.973 | 0.967 | −0.007 ± 0.018 |
| 128K | 0.833 | 0.860 | +0.027 ± 0.041 |

INT4 tracks FP16 within ±1σ at every context. The workload itself
is genuinely difficult (FP16 drops 0.973 → 0.833 between
64K → 128K), and the 128K partial-credit completions show real
retrieval failures — but the failures are *attention-confusion
between needles* (the model copies one needle's value into another
label), not KV-quantization failures. KIVI-style group-32 INT4
preserves the needle values precisely enough that the closed-loop
controller still has no room to operate.

**Implications for §9.8 design.** The multi-precision canary
machinery is structurally sound — Session 13 proved zero
monotonicity violations across 1161+ KL computations and the
controller correctly identifies when uniform-INT4 is the right
answer. What's missing is a workload where uniform-INT4 is *not*
the right answer. The path forward is workload escalation, not
sensor escalation:

1. **LongBench multi-hop QA** — supporting facts in natural prose
   plus reasoning across them. Different from RULER's pure-copy
   task: a successful answer requires combining multiple retrieved
   facts rather than transcribing them.
2. **Llama-3.1-70B on RULER@128K** — 8× larger KV cache and
   stronger but more KV-dependent representations. Multi-GPU,
   higher cost, but the model the paper's economic argument is
   actually about.

Per-tier τ_budget implementation remains a Session 15+ item,
contingent on a workload where the closed loop has room to operate.

#### Discovered scoring bug (load-bearing for any graded workload)

Until Session 14, ``run_session8._extract_metrics`` cast the per-
request score to ``int`` (the field was typed ``int`` because
single-needle RULER is binary 0/1). For the 5-needle workload this
collapsed every fractional score (0.2, 0.4, 0.6, 0.8) to 0 and
produced an artificially bimodal distribution. Headline accuracies
were systematically biased low. Fixed by changing
``RequestMetrics.correct`` to ``float`` and dropping the cast; the
blob now also carries ``completion_text`` so failure-mode analysis
doesn't require re-running. Binary workloads are unaffected
(``int(0.0) == 0`` and ``int(1.0) == 1`` are the only two values
they ever see). All Session 14 results above are post-fix.

#### Session 14 Phase 2 — 70B + GSM8K-Hard: practical search space exhausted

The critic's prioritized decision tree:

```
8B GSM8K-Hard      → INT4 ≈ FP16 (Δ=+0.05, n=20)
  → 70B RULER@128K → INT4 = FP16 = 1.000 (n=20)
    → 70B GSM8K-Hard → INT4 = FP16 = 0.550, paired 20/20 same
      → genuine null result
```

The 70B path required two pieces of new infrastructure:

1. **TP-aware static-INT4 hook** in ``evaluation/serving/static_int4_tp.py``.
   With TP > 1 and vLLM's ``mp`` distributed executor, the canary
   harness's rank-0-only install path doesn't see the other workers'
   layers. The fix: ship the install function via ``llm.apply_model``
   so it runs on every worker locally. Two pickle issues had to be
   cleared along the way:
   - vLLM 0.16's default serializer rejects ``<class 'function'>``.
     Set ``VLLM_ALLOW_INSECURE_SERIALIZATION=1`` to allow pickle
     fallback (fine for in-process trusted code).
   - Functions defined in ``__main__`` cannot be pickled cleanly
     across the worker boundary. The installer must live in a
     normal importable module — hence ``static_int4_tp.py``, not
     a function in the driver script.

2. **70B drivers** in ``run_session14_70b_ruler.py`` and
   ``run_session14_70b_gsm8k_hard.py``. Both load
   ``NousResearch/Meta-Llama-3.1-70B-Instruct`` with TP=4 + bfloat16
   + ``max_num_seqs=1`` (RULER@128K) or 8 (short-context GSM8K-Hard).
   Both save partial JSON after each cell so a failure during the
   second cell doesn't lose the first cell's data.

The hook installation reported ``[80, 80, 80, 80]`` — every TP
rank hooked all 80 attention modules, exactly as expected for
the 80-layer 70B model.

#### Cumulative INT4-vs-FP16 picture (Sessions 8–14)

| model | workload | context | n | FP16 | INT4 | paired Δ |
|--:|:--|--:|--:|--:|--:|--:|
| 8B  | single-needle RULER     | 32K  | 30 | 1.000 | 1.000 | 0.000 |
| 8B  | single-needle RULER     | 64K  | 30 | 1.000 | 1.000 | 0.000 |
| 8B  | single-needle RULER     | 128K | 30 | 1.000 | 1.000 | 0.000 |
| 8B  | multi-needle (5)        | 32K  | 30 | 0.947 | 0.940 | −0.007 |
| 8B  | multi-needle (5)        | 64K  | 30 | 0.973 | 0.967 | −0.007 |
| 8B  | multi-needle (5)        | 128K | 30 | 0.833 | 0.860 | +0.027 |
| 8B  | GSM8K-Hard              | 4K   | 20 | 0.500 | 0.550 | +0.050 |
| 70B | single-needle RULER     | 128K |  20 | 1.000 | 1.000 | 0.000 |
| 70B | GSM8K-Hard              | 4K   | 500 | 0.488 | 0.466 | **−0.022** (95% CI [−0.042, −0.002]; McNemar p=0.0522) |

Eight of the nine cells show INT4 matching FP16 within ±1σ.
**The 70B GSM8K-Hard cell at n=500 is the first defensible
"INT4 < FP16" signal in the project**, with the asymmetry
(70% of discordant pairs are INT4-worse) holding stably from
n=100 onward but only reaching directional significance at
n=500.

Sample-size trajectory:

| n | FP16 | INT4 | Δ | discord b/c | McNemar p |
|--:|--:|--:|--:|--:|--:|
|  20 | 0.550 | 0.550 | +0.000 | 0/0   | 1.000 |
|  30 | 0.600 | 0.600 | +0.000 | 2/2   | 1.000 |
|  50 | 0.620 | 0.580 | −0.040 | 5/3   | 0.727 |
| 100 | 0.570 | 0.540 | −0.030 | 6/3   | 0.508 |
| 200 | 0.495 | 0.480 | −0.015 | 4/1   | 0.375 |
| 500 | 0.488 | 0.466 | −0.022 | 19/8  | 0.052 |

The signal was invisible at n ≤ 30 (paired 20/20 same was a
small-sample coincidence) and only directionally suggestive at
n ∈ {50, 100, 200} — too few discordant pairs against the
batch-composition noise from ``max_num_seqs=8``. n=500 is where
the signal resolves.

#### Implications for §9.8 design

The closed-loop architecture is sound, and at large n on
70B + GSM8K-Hard we now have a workload where it has room to
operate (Δ = −0.022 vs FP16; the room is small but real). Two
questions need to be answered before EchoKV-3C can claim a
Pareto win on this workload:

- **Per-layer δ̂ correlation with the 19 INT4-worse problems.**
  Does the canary's per-layer EMA flag the layers responsible
  for those failures? If yes, selectively upgrading just those
  layers to INT8 should recover the 2.2pp at modest memory cost
  (Pareto win). If failures are diffuse across all 80 layers,
  only uniform INT8 recovers them at +50% memory cost — no
  Pareto win vs static KIVI-4.
- **TP-aware ``CanaryHarness``.** The existing harness installs
  hooks via the rank-0 model object, which only sees rank 0 in
  TP > 1. Extending to ``llm.apply_model`` (the path used for
  ``static_int4_tp.py`` in this session) is structurally the
  same change: ~50–100 LoC.

Both are blocked on additional compute, but the work is now
actionable rather than speculative.

Other §9.8 follow-ups remain valid future work for further
headroom, in roughly this priority order once the closed-loop
demonstration above lands:

- **Confirm with ``max_num_seqs=1``** that the n=500 signal isn't
  partly a batch-noise artifact. ~5× slower per run, but
  produces a fully deterministic paired comparison.
- **INT3 instead of INT4** as the static baseline. INT2 is known
  to fail catastrophically (Session 11: 0.13/0.00 on RULER); INT3
  might be the per-layer-differentiation regime where some layers
  tolerate it and others don't.
- **Non-Llama models.** Mixtral/Qwen/Gemma may have weaker KV
  robustness than Llama-Instruct.
- **Reasoning chains > 512 tokens.** Agent loops, R1-style
  reflective traces.
- **Natural-prose multi-document retrieval.** HotpotQA over real
  documents.

Per-tier τ_budget refinement, the prefill-canary path, and the
vLLM block-allocator fork are now sequencing dependencies on
the per-layer δ̂ correlation question. If the controller's
per-layer signal does correlate with GSM8K-Hard failures,
per-tier τ_budget is the natural way to exploit it. If not, the
fix has to come from one of the other paths (longer chains,
non-Llama, INT3) before any architectural sophistication is
worth implementing.

#### Session 14 Phase 3 — closed-loop EchoKV-3C tested on 70B GSM8K-Hard; diffuse damage confirmed

Phase 3 built the TP-aware ``CanaryHarness`` install path, the
3-phase warmup→consolidate→measurement protocol, and a τ_budget
calibration probe — and ran EchoKV-3C end-to-end on the
demonstration workload from Phase 2 (70B + GSM8K-Hard).

**TP-aware harness install** (``evaluation/serving/canary_tp.py``).
``CanaryHarness.__init__`` now accepts ``model=`` directly so it
can be constructed inside a vLLM worker. ``TPCanaryHandle`` ships
a picklable installer to each TP rank via ``llm.apply_model``;
it stashes the harness on the worker's local model object
(via attribute, so successive ``apply_model`` calls can find
it). 11 unit tests in ``tests/test_canary_tp.py`` cover install
idempotency, hook count, snapshot shape, ``seed_tiers``,
``remove`` cleanup, and per-tier δ̂ readback (CPU-only mock).
8B+TP=2 smoke test: 32 hooks per rank, 725 canary fires per
rank, per-rank tier vectors diverge by ~28% (structural — TP
shards KV heads, so each rank's δ̂ is on a different head
subset). Driver consolidates via per-layer max across ranks.

**Calibration probe on 70B+GSM8K-Hard.**

| static probe | acc | δ̂_2 [min, p50, max] | δ̂_4 [min, p50, max] |
|:--|--:|:--|:--|
| INT8 | 0.600 | [0.0004, 0.007, 0.022] | [1e-5, 0.0003, **0.00075**] |
| INT4 | 0.550 | [0.0016, 0.030, 0.088] | [8e-5, 0.0013, **0.00356**] |
| INT2 | 0.300 | [0.001, 0.006, 0.026] | [6e-5, 0.0002, 0.00082] |

δ̂_4 max under INT4 = **0.00356** — about 10× tighter than
8B+RULER (~0.04). Strict critic test for diffuse damage
(δ̂_4 max < 0.02): **Scenario B confirmed**.

**EchoKV-3C trajectory at n=100.**

| run | τ_budget | tier counts | avg bits | acc |
|:--|--:|:--|--:|--:|
| static_int4 (baseline) | — | uniform INT4 | 4.00 | 0.540 |
| EchoKV-3C scalar | 0.01 | INT4:1, INT2:79 | 2.02 | 0.250 |
| EchoKV-3C MP | 0.05 | INT4:77, INT2:3 | 3.92 | 0.520 |
| EchoKV-3C MP (calibrated) | 0.002 | INT4:51, INT2:29 | 3.27 | 0.480 |

Three results, three lessons:

1. **Scalar canary collapses to INT2** (acc=0.250). Confirmed
   the Session-12 sensor blind spot on a new model and workload.
2. **Multi-precision canary at τ=0.05 holds at INT4** (acc=0.520).
   The MP fix works — sensor correctly avoids INT2 collapse.
   But no INT8 upgrades, so accuracy ≈ static_int4.
3. **Calibration-derived τ=0.002 actually went MORE aggressive**
   (acc=0.480, 18% memory savings vs uniform INT4 but 6pp
   accuracy loss). Static-probe analysis projected 13 INT8 +
   66 INT4 + 1 INT2; in the closed-loop steady state the actual
   tier vector was 0 INT8 + 51 INT4 + 29 INT2. Once a layer
   transitions to INT2, its in-state δ̂_2 drops (cache becomes
   INT2-quantized, each new fire measures smaller delta),
   reinforcing the INT2 decision via feedback. No layer's δ̂_4
   stayed high enough to trigger an INT8 upgrade.

**Pareto win not achieved on this workload.** The closed-loop
architecture is empirically intact — install, sensor, controller,
consolidation all work — but **per-layer softmax-KL δ̂ does not
correlate with per-layer responsibility for diffuse accuracy
damage** on 70B reasoning. The 2.2pp INT4 loss documented in
Phase 2 is genuinely diffuse: no per-layer differentiation by
δ̂ identifies a recoverable subset of layers.

##### Calibration-to-production EMA scale gap (60×)

Diagnostic check on the τ=0.002 run revealed a load-bearing
methodological issue:

| | Calibration (static INT4 probe) | Production (warmup, multi-prec) | Ratio |
|--|--:|--:|--:|
| δ̂_4 max  | 0.00356 | 0.00006 | **60× smaller** |
| δ̂_4 mean | 0.00143 | 0.00002 | 70× smaller |
| δ̂_2 max  | 0.08752 | 0.00674 | 13× smaller |

**Production δ̂ values are ~60× smaller than calibration δ̂
values measured for the same workload.** With τ=0.002 set from
the calibration-scale distribution, **0 layers ever cross the
INT8-upgrade threshold** in production — explaining why both
τ=0.002 EchoKV-3C runs produced 0 INT8 layers despite the
static-probe projection of ~13 INT8 upgrades.

The cause is a real signal-scale mismatch between two different
measurement regimes, not just calibration drift. Two compounding
factors:

1. **Cache-state dependence.** The calibration probe runs uniform
   INT4 (every layer's cache is INT4-quantized throughout). At
   each canary fire, the per-step KL measures the marginal effect
   of quantizing one new K against a known INT4-quantized cache.
   In production, the controller actively transitions tiers
   during warmup; by end of warmup many layers already have INT2
   cache entries from earlier steps. The marginal effect of
   further quantization on the new K is much smaller against an
   already-perturbed cache.

2. **Per-step vs accumulated timescale mismatch.** The canary
   measures *marginal* per-step KL between FP16 K and INT-quant K
   at one decode step. But GSM8K-Hard accuracy damage accumulates
   over the entire 200–400 token reasoning chain — each
   slightly-corrupted K shifts attention weights slightly, and
   those errors compound. The per-step canary sees each
   individual error as tiny (0.00006 divergence at the upper end
   in production). The calibration probe sees the integrated
   effect of running uniform INT4 throughout (0.00356). These are
   measuring fundamentally different things at different
   timescales.

**Per-layer ranking is correlated but not identical** between
calibration and production. Layer 8 has the calibration max
(0.00356) but is only #5 in production (0.00004); layer 65 is
#1 in production (0.00006) but mid-range in calibration
(0.00153). Which layers look noisy per-step diverges from which
layers accumulate the most error over a full chain.

**Methodological consequence.** τ_budget cannot be calibrated by
static probes; it must be set from production-scale δ̂.
Re-running with production-scale τ ≈ 3e-5 is the correct test
of the per-layer differentiation hypothesis. (Pending; the
calibration-production gap finding is documented before that
run so the sweep is interpretable as a test of the production
distribution rather than a re-test of the static-probe values.)

**Theoretical consequence.** The marginal-vs-accumulated
mismatch strengthens the output-distance sensor hypothesis (§
"What's missing" below). Per-step KL is inherently a marginal
signal; an output-distance measurement integrated over a full
sequence is closer to an accumulated signal, which is what
correlates with downstream accuracy damage.

##### Two-equilibria finding (warmup phase has two stable fixed points)

After recalibrating τ_budget against production-scale δ̂ values
(τ ∈ {6e-5, 4e-5, 3e-5, 2e-5, 1e-5}, n=50 each), the warmup
phase exhibits **two distinct stable equilibria** that the
controller can converge to depending on initial transient
dynamics:

| regime | warmup δ̂_4 max | tier counts | sweep cells |
|:--|--:|:--|:--|
| mixed-precision (stable) | ~6e-5 | ~half INT8, half INT4 | τ=6e-5, 3e-5, 2e-5 |
| uniform-INT8 (cascaded) | ~3e-3 to ~7e-3 | ~all INT8 | τ=4e-5, 1e-5 |

The 50× difference in warmup δ̂_4 magnitude between the two
regimes is the signature of a positive-feedback loop:

1. Some early canary fires push a few layers from INT4 to INT8.
2. With more layers at INT8, the cache stays closer to FP16
   (INT8 is essentially lossless).
3. Subsequent canary fires measure δ̂_4 against this clean cache —
   which is now ~"absolute INT4-vs-FP16 KL" rather than
   ~"marginal INT4-on-top-of-INT2-cache KL". Magnitude jumps 50×.
4. The larger δ̂_4 fails the τ_budget for INT4 across more layers
   → more INT8 upgrades → loop closes back to step 2.

The cascade is *τ-dependent* but not monotonically: the sweep
shows τ=4e-5 cascading while τ=3e-5 (tighter, should be more
aggressive) stays in the mixed regime. The cascade triggers
when the initial transient has a small bias toward upgrades
that the feedback can amplify.

**Implication for stability.** Cells in the cascaded regime are
not measuring EchoKV-3C — they're measuring uniform INT8 with
production-canary overhead. Their accuracy (uniform-INT8-like)
is unrelated to the closed-loop differentiation question. Only
the mixed-regime cells produce valid Pareto data points.

**Future safeguard.** A max-upgrades-per-tick limit on the
controller (e.g., never upgrade more than 10 layers per
``tick_interval``) would prevent the cascade by capping the
positive-feedback amplitude. The other stable regime (mixed
precision) is reached when transitions are gradual; capping
fast transitions forces all runs into that regime regardless of
initial transient. ~10 LoC controller change. Not implemented in
Phase 4 because the τ=2e-5 publication run sat in the stable
regime; the safeguard becomes load-bearing if the n=200
publication run cascades unexpectedly or for any future workload
where the initial transient is harder to control.

##### Layer-upgrade correlation at τ=3e-5

The controller is doing exactly what it should at production
scale: of the **top-25 layers ranked by production warmup δ̂_4**,
24 are upgraded to INT8 (one INT4 exception at rank 23). Of the
bottom-10 layers by production δ̂_4, all 10 stay at INT4. The
INT8 layers cleanly partition from the INT4 layers in
production-δ̂ space:

| group | n | prod_δ̂_4 [min, mean, max] |
|:--|--:|:--|
| INT8 | 37 | [0.00002, 0.00003, 0.00007] |
| INT4 | 43 | [0.00000, 0.00001, 0.00002] |

So the closed-loop premise — "layers with high per-step
divergence are the ones to upgrade" — works at the production
scale, even though the per-step δ̂ ranking differs from the
calibration probe's static-INT4 ranking (correlated but not
identical, because the two measurement regimes pick up different
subsets of accuracy-relevant layers).

#### Sensor architectural implication

The §9.7 + §9.8 sensor analysis was correct that scalar softmax-
KL has blind spots. The multi-precision canary correctly fixes
the *visible* failure mode (scalar-sensor INT2 collapse). What
Phase 3 reveals is that **the underlying sensor signal — softmax
KL on per-step Q · K — is the wrong correlate** for diffuse
damage. The signal predicts per-step divergence but not
per-layer responsibility for accumulated downstream error.

Hypothesis for the next sensor revision (out of scope for current
compute): an **output-distance sensor**, measuring
``||attn_output_FP16 − attn_output_quant||₂`` per layer. The
output is post-softmax, post-projection — closer to what
downstream layers actually consume. Implementation reuses the
existing canary plumbing (forward-pre-hook, EMA accumulator,
controller); only the measured quantity changes. ~50-100 LoC
plus a new τ_budget calibration.

#### What Phase 3 leaves intact

Independent of the Pareto-recovery question:

- The closed-loop framework runs at +0.09% TPOT measurement
  overhead with zero monotonicity violations across 1000+ KL
  computations.
- The TP-aware install via ``apply_model`` works correctly and
  is now reusable for any future TP > 1 experiment.
- The per-layer canary δ̂ exposes real per-layer divergence
  structure (7.3× spread on RULER@128K, 44× on GSM8K-Hard at
  smaller absolute scale).
- The controller correctly identifies "uniform-INT4 is
  sufficient" on robust workloads and correctly avoids INT2
  collapse via the multi-precision sensor.

These are the engineering and characterisation contributions
that don't depend on a Pareto win.

### 9.9 Absolute-canary sensor (post-Phase-4 redesign)

#### What the legacy sensor actually measured

The legacy ``CanaryHarness`` (§§9.5–9.8) computes
``KL(softmax(Q · quant(K)^T) ‖ softmax(Q · K^T))`` where Q and K
are the **just-projected FP16 tensors** for the current attention
call (before ``tier_application``). Q has shape
``(1, H, T, D)``; K has shape ``(1, H_kv, T, D)``. The KL is on
a ``T × T`` attention matrix where each "query" (one of T new
tokens) attends to each "key" (one of T new tokens) within the
same forward call.

T equals:

- **Decode**: the number of decode tokens in this batch
  (``= max_num_seqs`` when all requests are at decode step).
- **Prefill**: the chunk size, capped at
  ``max_decode_tokens=256`` by the harness's prefill filter
  (``decode_only_canary=True``); below the cap, T equals the
  prompt length.

#### Failure mode 1 — the T=1 collapse

When ``max_num_seqs=1`` in pure decode, T=1. Softmax over a
single element is ``[1.0]`` regardless of what's being computed.
KL between any two such distributions is identically 0. The
canary returns no signal, the controller sees ``δ̂_b = 0`` for
every candidate b, and the argmin-with-budget rule picks the
**lowest tier** (INT2) since 0 < τ_budget for every tier.

This caused the Phase-4 publication catastrophe: at n=200
``max_num_seqs=1``, EchoKV-3C cascaded to {INT2:60, INT4:20,
INT8:0} and acc dropped from ~0.50 (FP16/INT4) to **0.365**
(13pp below static_int4). The cascade is mechanical, not
feedback-driven — at T=1 there is no attention distribution to
be perturbed.

#### Failure mode 2 — cache-state feedback (two-equilibria)

Even at T > 1, the legacy sensor's measurement was indirectly
coupled to the running tier via the sequence of K projections
processed so far. Phase 4's τ-sweep documented two distinct
attractors:

| regime | warmup δ̂_4 max | tier counts | sweep cells |
|:--|--:|:--|:--|
| mixed-precision (stable) | ~6e-5 | ~half INT8, half INT4 | τ=6e-5, 3e-5, 2e-5 |
| uniform-INT8 (cascaded) | ~3e-3 to ~7e-3 | ~all INT8 | τ=4e-5, 1e-5 |

The 50× difference in warmup δ̂_4 magnitude between the two
regimes is the signature of positive feedback: a small bias
toward INT8 transitions during warmup keeps the cache cleaner →
the canary's δ̂_4 measurements grow → more INT8 transitions →
loop closes. Conversely, INT2 transitions cause the cache to
become more perturbed → δ̂ measurements shrink → INT2 reinforced.

#### The fix — absolute KL against a sliding-window FP16 reference

Both failure modes share a root cause: the canary's measurement
is coupled to runtime state (T from the batch, K projections
through the cache feedback). The redesign decouples both:

1. Maintain a **per-layer rolling FP16 buffer** of the last N=64
   K-projection rows, captured **before** ``tier_application``.
   The buffer's contents are always FP16 — never quantised, never
   cache-dependent.
2. At canary fires, compute KL using ``K_ref`` (the buffer)
   instead of the current step's K:

   ```
   K_ref ∈ R^(N × num_kv_heads × head_size)   # always FP16
   for b ∈ {2, 4, 8}:
       attn_b = softmax(Q · quant(K_ref, b)^T / √d)
   attn_FP16 = softmax(Q · K_ref^T / √d)
   δ_absolute(b) = KL(attn_b ‖ attn_FP16)
   ```

3. T_key = N (always ≥ 2), so the T=1 collapse mode is
   structurally impossible.
4. Both terms in the KL use the same FP16 reference; the cache's
   current bit-width does not enter the measurement, breaking
   the cache-state feedback loop in both directions.

#### Memory and overhead

Per layer per rank: ``N × num_kv_heads × head_size × 2 bytes``.
For Llama-3.1-70B at TP=4 (``num_kv_heads=8``, sharded to 2 per
rank, ``head_size=128``, N=64): ``64 × 2 × 128 × 2 = 32 KB`` per
layer per rank, **2.5 MB total per rank** for 80 layers — three
orders of magnitude smaller than the KV cache itself. INV-1
(no per-block aux state) is preserved because the buffer is
per-layer, not per-block.

Per canary fire, the marginal cost is one extra
``simulate_quantization`` and one extra small attention
softmax. At ``canary_rate=1/64``, total overhead remains
within the +0.09% TPOT budget measured in Sessions 8-13.

#### Smoke-test validation

Before committing to the 70B re-run, the redesign was validated
on Llama-3.1-8B at TP=2 with four checks:

| check | result |
|:--|:--|
| Magnitude (δ_4 above 1e-6 floor) | δ_4 max 0.031–0.042 across 4 runs |
| ``max_num_seqs`` invariance | mns=1 / mns=8 max-δ_4 ratio = 1.21× |
| Cross-tier monotonicity | 0 violations across 32 layers × 4 runs |
| Cross-seed stability | Spearman ρ = 0.876 between seeds 0 and 7 |

**All four PASS.** The δ_4 magnitude (~0.04) is comparable to
the calibration-probe scale, confirming the previous 60×
calibration-vs-production gap was an artifact of measuring
different things.

#### 70B + GSM8K-Hard re-run

At n=200 ``max_num_seqs=1`` with τ_budget=0.022:

| run | bits | acc | Δ vs INT4 (95% CI) | McNemar p |
|:--|--:|--:|:--|--:|
| FP16        | 16.00 | 0.510 | +0.015 (±0.033) | 0.55 |
| static_int4 |  4.00 | 0.495 | (baseline)      | —    |
| EchoKV-3C abs-canary τ=0.022 | 5.70 | 0.505 | +0.010 (±0.037) | 0.79 |

The tier vector ``{INT4:46, INT8:34, INT2:0}`` is reproducible
from n=50 → n=200 (identical pattern). No INT2 cascade. No INT8
cascade. The closed-loop framework is empirically intact.

The accuracy effect, however, is **below the noise floor at
n=200**: paired Δ vs static_int4 is +0.010 with 95% CI
[−0.027, +0.047]. The static_int4 vs FP16 deficit on this
n=200 mns=1 paired set is also non-significant (Δ=−0.015,
p=0.55) — smaller than the n=500 mns=8 finding of −0.022 with
p=0.052. The honest summary: the workload-level effect is
small enough that confirming a Pareto win on 70B+GSM8K-Hard
requires either a much larger n or a workload with a larger
static-INT4 deficit.

#### What this means for the architecture

The absolute-canary sensor is the right design. It removes
three classes of measurement artifact (T=1 collapse,
cache-state feedback, calibration-vs-production scale gap) and
produces a stable, batch-independent, layer-intrinsic signal.
The closed-loop framework operates correctly with this sensor:
it produces a meaningful non-uniform tier vector that survives
configuration changes and replication. What it does *not* do is
manufacture a large Pareto win on a workload where the
underlying static-tier deficit is itself small. That's the right
behaviour for a measurement system, not a failure of the design.
