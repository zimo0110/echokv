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
