# EchoKV Fork Investigation — Handoff

Self-contained context for continuing this thread with a fresh Claude. Last
updated 2026-05-11 ~09:35 UTC, immediately after the §6.3 multi-seed
reproducibility result.

## How this thread has been working

The user (jasontrinh@berkeley.edu, machine: /home/nazar/echokv on Linux) is
investigating why the EchoKV-3C closed-loop controller's vLLM fork integration
hasn't reproduced the paper's §6.3 GSM8K-Hard Pareto headline. There's an
external "advice author" the user consults: the user pastes the advice author's
recommendations verbatim ("this is what i got back: …") and expects me to act
on them. I run experiments, summarize findings, and propose next steps; the
user takes those back to the advice author and forwards the response.

The user prefers tmux-launchable .sh scripts following the existing
`scripts/NN_*.sh` style, with a Python runner under `evaluation/serving/`.
Reports back from runs come as "i think it's done" or "it's done" — that's the
signal to read result files and analyse, not a new request.

When I schedule autonomous wakeups: I only re-arm when there's a live run to
check; once a result is in or a question is pending, I end the loop. The user
treats `<<autonomous-loop-dynamic>>` notifications as just the harness firing
my prior schedule, not as new input.

The user does **not** want speculative paper edits committed without
confirmation. I halted a §6.8 rewrite mid-session when a diagnosis turned out
to be wrong (the "per-token vs per-batch granularity" story — see below).

## Project: what EchoKV claims and what's at stake

EchoKV is a closed-loop KV-cache-bit-allocation system. The "absolute canary"
fires during attention pre-hooks, measures per-tier δ̂ (KL of
softmax(Q·K) vs softmax(Q·simulate_quant_b(K))), and a PID-like controller
adjusts per-layer precision in {INT2, INT4, INT8}. The paper has results in
**emulation mode** (shadow-quantized K/V via a forward-pre-hook on top of
upstream vLLM at BF16 cache). The **vLLM fork** at
`/mnt/nvme_tmp10/nazar_vllm_fork` (branch `feature/echokv-int4-cache`) adds
real per-token KIVI INT2/4/8 cache support and is the deployment substrate
the project wants to validate.

§6.3 headline (paper draft): Llama-3.1-70B on GSM8K-Hard at n=500 mns=8, the
emulation reports FP16=0.488, static_int4=0.466, **EchoKV-3C (frozen) =
0.488 at avg 5.45 bits**. The "0.620" number that's floated through earlier
discussions is a separate n=50 small-sample cell from session_20260429_14_phase4
(τ=2e-5 scalar canary) — *not* the §6.3 headline. The actual §6.3 headline is
"matches FP16 at 5.45 bits."

## What's been measured this session (everything below is defensible)

| Hypothesis | Magnitude | Verdict | Files |
|---|---|---|---|
| Cache-state feedback loop (controller-frozen vs active, init INT8) at tiers 4/8 | ~5× δ̂ ratio | Real | `results/serving/phaseG_diag_runC_int8_frozen.json` vs `phaseG_pareto_fork_70b_init8_tau3e4.json` |
| Input-distribution contamination (frozen INT4 vs frozen INT8) | ~3.3× | Real, smaller | `phaseG_diag_runA_int4_frozen.json` vs `phaseG_diag_runC_int8_frozen.json` |
| Calibration-corpus shift (RULER@4K vs GSM8K-Hard, frozen INT4) | ≤2.2× | Ruled out as primary | `phaseG_diag_runA_int4_frozen.json` vs `phaseG_diag_runAprime_int4_frozen_ruler.json` |
| Per-rank consolidation (TP-rank disagreement, same warmup) | ~2× | Ruled out | analysis on Run A/B/C snapshots |
| **Seed-to-seed swing in δ̂_4 at real INT4 substrate** (seed=0 vs seed=100 prompts) | **15.06×** at warmup_n=50 | **Real but unstable** | `/tmp/diag_step1_seed0_n50.json` vs `/tmp/diag_step1_seed100_n50.json` |
| Same swing at uniform INT8 substrate | 1.51× | **Compressed** — INT4 substrate amplifies | `/tmp/diag_step3_int8_seed0_n50.json` vs `/tmp/diag_step3_int8_seed100_n50.json` |
| Same swing at fork-shadow-mode (BF16 cache + simulate_quant for canary) | 1.24× | **Compressed** — emulation equivalent | `/tmp/diag_step3_shadow_mode_results.json` |
| Fork static_int4 vs emulation static_int4 GSM8K-Hard n=500 mns=8 | Δ=+0.036 (fork better, n.s.) | Substrates statically equivalent | `results/serving/fork_static_int4_70b_gsm8k_hard_n500.json` |
| Fork FP16 vs fork static_int4 paired McNemar n=500 | Δ=+0.020, p=0.123 | Borderline (Outcome 3) | `results/serving/fork_fp16_vs_int4_70b_gsm8k_hard_n500_mcnemar.json` |
| **Fork FP16 vs fork static_int4 paired McNemar n=1000** | **Δ=+0.010, p=0.32** | **Outcome 1: no meaningful deficit** | `results/serving/fork_fp16_vs_int4_70b_gsm8k_hard_n1000_mcnemar.json` |
| **§6.3 emulation reproducibility (3 seeds × n=500)** | **Doesn't reproduce** | **Outcome A: fragile** | `results/serving/section63_repro/{seed[012]_*.json,aggregate.json}` |

### Diagnoses I tried and retracted

- **"Fork's per-token KIVI is finer-grained than emulation's batch-level
  shadow quant" — WRONG.** Advice author caught this by reading
  `evaluation/motivation/kivi.py`. `simulate_quantization` computes scale at
  shape `(B, T, H_kv, num_groups)` — same per-token-per-head-per-group
  granularity as fork KIVI. I verified by running `quantize` on a real K
  shape and printing `scale.shape`.

### The §6.3 multi-seed result (most recent)

Three seeds × n=500 on emulation, paired McNemar each seed vs FP16 on the
same prompts.

| Seed | FP16 | INT4 | EchoKV-3C | avg bits | tier counts | EchoKV vs FP16 |
|--:|--:|--:|--:|--:|--|--|
| 0 | 0.488 | 0.476 | **0.464** | 5.10 | {4:58, 8:22} | Δ=-0.024, **p=0.045** |
| 1 | 0.494 | 0.496 | **0.506** | 5.55 | {4:49, 8:31} | Δ=+0.012, p=0.286 |
| 2 | 0.456 | 0.440 | **0.456** | **6.95** | {4:21, 8:**59**} | Δ=0.000, p=0.844 |
| mean | 0.479 | 0.471 | 0.475 | 5.87 | — | — |
| spread | 0.038 | 0.056 | 0.050 | **1.85** | — | — |

Key facts:
1. Re-running seed=0 at the published protocol gave 0.464 with tier vector
   {4:58, 8:22} — the original seed=0 cell had 0.488 with {4:51, 8:29}. Same
   seed, same protocol, different warmup outcome. The warmup is nondeterministic
   enough that "seed=0" isn't even reproducible on its own.
2. Tier vectors span 5.10 → 6.95 bits across seeds (1.85-bit spread). Seed 2
   cascaded to 59 INT8 + 21 INT4.
3. Mean EchoKV-3C (0.475) ≈ mean FP16 (0.479) ≈ mean static_int4 (0.471).
   There is no consistent recovery of an INT4 deficit.

Conclusion: §6.3's headline is a favorable-seed-0 outlier. The "mirror-image
recovery" framing (19/8 → 9/20 discordant) is a seed-0 artifact and does not
generalize.

## What's open right now

The user's last question: send the §6.3 reproducibility finding back to the
advice author, draft a paper-rewrite scope, or both?

The advice author's framework called this **Outcome A**: "§6.3 was fragile,
the fork investigation has been chasing a result that doesn't survive multi-seed
scrutiny. Paper rewrites around what does reproduce (HotpotQA, RULER@128K —
both worth re-checking too for the same reason)."

Likely next-step candidates:
1. Send §6.3 result to advice author and wait for their take on what survives
   the rewrite.
2. Run the same multi-seed reproducibility check on §6.4 (HotpotQA) and §6.5
   (RULER@128K) before any rewrite — those headlines are similarly at-risk if
   the warmup nondeterminism reproduces.
3. Hold §6.8 / §7.4 / §7.5 edits I already made in PAPER_DRAFT.md (they're
   based on the older diagnostic chain, which is still partly valid but now
   needs reframing once we know what §6.3 actually claims after the rewrite).

## Files I edited this session

- `core/canary_estimator.py` — added `kl < -1e-6` tolerance for FP-rounding
  negatives at the EMA boundary (line ~160).
- `core/precision_controller.py` — added `upgrade_only: bool` and
  `grace_ticks: int` fields to PIDPrecisionController; gated
  `tick_multi_precision` decisions on grace ticks; suppressed downgrade
  transitions when upgrade_only is True.
- `evaluation/serving/canary_vllm.py` — plumbed `upgrade_only` and
  `grace_ticks` kwargs through CanaryHarness.
- `evaluation/serving/run_phaseG_3phase_fork.py` — added
  `--warmup-only-upgrades` and `--warmup-grace-ticks N` flags; wired into
  harness_kwargs.
- `evaluation/serving/run_fork_fp16_vs_int4_mcnemar.py` — new file. Two-phase
  runner (FP16 + static_int4) + paired McNemar. Parameterized via
  `--n-prompts` (default 500) and `--seed`.
- `PAPER_DRAFT.md` — §6.8 rewrite striking the "~30×" and "~100× signal
  compression" claims (which were back-of-envelope, never measured). New §6.8
  has a measured-effects table. §7.4 rewritten to clarify activation buffer
  addresses only the contamination component. New §7.5 stub for
  "upgrade-only warmup" architectural fix. **These edits were made before the
  per-token-granularity diagnosis turned out to be wrong AND before §6.3
  reproducibility failed. They need another pass given the §6.3 result.**
- `VLLM_CHANGELOG.md` — addendum 10 documenting the active-vs-frozen finding
  (5× at tiers 4/8). **Also predates the §6.3 reproducibility result; needs
  another pass.**

## Scripts I added (all tmux-launchable, match scripts/NN_*.sh style)

- `scripts/20_fork_fp16_vs_int4_mcnemar.sh` — fork FP16 vs static_int4 at
  n=500 (passes `--n-prompts 500` to the runner). Result: Δ=+0.020, p=0.123.
- `scripts/21_fork_fp16_vs_int4_mcnemar_n1000.sh` — same at n=1000. Result:
  Δ=+0.010, p=0.32 → **Outcome 1**.
- `scripts/22_section63_emulation_multiseed.sh` — §6.3 emulation, 3 seeds ×
  n=500 (FP16 + static_int4 baselines + EchoKV-3C abs-canary tau_budget=0.022
  warmup_n=16 kref_window_size=64). Result: see table above → **Outcome A**.

## Tooling tips

- Conda env: `/opt/conda/envs/scalable/bin/python`. The fork lives at
  `/mnt/nvme_tmp10/nazar_vllm_fork`. The `_FORK` path is hardcoded in fork
  scripts and added to `sys.path` + `PYTHONPATH` at script start.
- vLLM env vars: `VLLM_ALLOW_INSECURE_SERIALIZATION=1` and
  `FLASHINFER_DISABLE_VERSION_CHECK=1`.
- 70B requires TP=4. GPUs 0-3 have been free; GPUs 4-7 have been used by
  Jason's caspo training (`scripts.train_caspo` on `Qwen2.5-Math-1.5B`).
  Check with `nvidia-smi --query-gpu=memory.used --format=csv,noheader`
  before launching anything. Override with `CUDA_VISIBLE_DEVICES=…`.
- Model paths: `/mnt/nvme_tmp6/llama-70b` is the 70B used throughout.
  `/mnt/nvme_tmp2/llama-70b` shows up in older results JSON — same model,
  earlier location.
- Disk: /dev/nvme0n1p1 was at 98% earlier in the session; user freed 100GB.
  Currently 80% (96G free). Be careful with large dumps.
- McNemar in this codebase uses Yates' continuity correction and scipy.stats
  if available, else `math.erfc` fallback.
- Per-rank canary δ̂ snapshots are in `warmup_snapshots[rank]["per_tier_delta"]`
  keyed by tier (sometimes as str, sometimes as int — `_normalise_per_tier`
  handles this).

## How to read prior context

Full transcript for this thread is at
`/home/nazar/.claude/projects/-home-nazar-echokv/0d925bc4-44f3-4915-b424-e84418a12a73.jsonl`.
The user's previous earlier conversation (before this one) is at
`/home/nazar/.claude/projects/-home-nazar-echokv/` too.

The earlier Phase-G(e) addenda 1–10 in `VLLM_CHANGELOG.md` cover the original
diagnostic chain (per-rank, contamination, warmup nondeterminism findings)
that all happened *before* the §6.3 reproducibility check exposed that the
target was fragile.

## Important context the advice author has given (in order)

1. "Run upgrade-only warmup as the construction-level fix for the feedback
   loop." → I implemented `--warmup-only-upgrades`. Cascades to uniform INT8.
2. "Add `--warmup-grace-ticks` so EMA stabilises before the controller acts."
   → I implemented it. Same cascade outcome.
3. "Investigate the probe-vs-warmup canary 14× mismatch."
   → I found code paths are identical; 14× is **prompt seed dependent** (the
   `--seed +100` offset between calibration probe and warmup harness).
4. "Run the three-step diagnostic to isolate the 15× seed swing."
   → Steps 1 (warmup_n=50) and 2 (TP head subsetting) ruled out; step 3
   substitute (INT8 substrate) compressed gap 15× → 1.5×.
5. "Run fork shadow-mode (BF16 cache + simulate_quant for canary) at same
   seeds." → gap compressed to 1.24×. Per-token-KIVI-amplifier story
   *seemed* confirmed.
6. "Halt — `simulate_quantization` IS per-token, granularity diagnosis is
   wrong. Run fork static_int4 vs emulation static_int4." → did it, fork is
   actually slightly *better* (0.502 vs 0.466).
7. "Run fork FP16 vs fork static_int4 paired McNemar n=500." → Δ=+0.020,
   p=0.123, borderline.
8. (user asked me to also do n=1000 — Δ=+0.010, p=0.32, Outcome 1)
9. "Run §6.3 multi-seed reproducibility at n=500 × 3 seeds before any paper
   rewrite or workload search." → just done: **does not reproduce.**

## Suggested next prompt to send to the advice author

Tell them: §6.3 multi-seed reproducibility failed. Per-seed table above
(spread 5.10–6.95 bits, mean EchoKV-3C ≈ mean FP16 ≈ mean static_int4 within
0.01, seed-0 self-non-reproducible). Decide between sending HotpotQA/RULER
through the same multi-seed scrutiny before rewriting, or scoping the rewrite
directly from the §6.3 finding.
