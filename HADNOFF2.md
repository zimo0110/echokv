EchoKV — Full Context Dump for Handoff
Project state
EchoKV is a closed-loop control system for KV cache quantization in LLM serving. Two implementation tracks exist:
Emulation (upstream). vLLM with FP16 KV cache and Python simulate_quantization applied as a hook before attention. The "tier application" is shadow-quantization at attention input, not real cache storage. All paper results (§6.3 GSM8K-Hard, §6.4 HotpotQA, §6.5 RULER@128K) were produced on this substrate. Final arxiv preprint v3 (echokv_arxiv_v3.tex) is built around these results.
Fork (production-grade). vLLM fork at /mnt/nvme_tmp10/nazar_vllm_fork, branch feature/echokv-int4-cache. Real KIVI per-token group-asymmetric INT4/INT2/INT8 cache via Triton kernels. Engine init plumbing, per-layer bit-width config, runtime tier swap, TP-aware install all working. 122 fork tests pass. Phases A–F.3.1 complete. Documented in VLLM_ARCHITECTURE.md and VLLM_CHANGELOG.md.
The paper's deployment claim is that EchoKV-3C closed-loop control produces workload-adaptive bit allocation that matches FP16 at substantial memory savings. The emulation results support this for emulation. The fork was supposed to validate it on production-grade KIVI. It hasn't.
Critical implementation fact
simulate_quantization in evaluation/motivation/kivi.py does per-token group-asymmetric quantization at group_size=32. Same granularity as the fork's KIVI kernel. Earlier diagnoses claiming emulation was "coarser-grained" were wrong — the granularities are identical. The two paths produce different numerical outputs from the same input K tensor, but at the same per-token grain.
What has been measured on the fork
Static substrate comparison (no controller, GSM8K-Hard, 70B, TP=4):
Fork FP16Fork static INT4Paired ΔMcNemar pn=5000.5240.504+0.0200.12n=10000.5340.524+0.0100.32
Doubling n halved Δ. Fork's INT4 substrate has no meaningful accuracy deficit vs FP16 on GSM8K-Hard at 70B.
Compare to emulation: FP16 = 0.488, static_int4 = 0.466, Δ = +0.022. Both substrates show the same small (or no) deficit. The emulation §6.3 EchoKV-3C cell at acc=0.620 is +0.132 above emulation FP16 = 0.488, which cannot be "INT4 deficit recovery" since the deficit is at most 0.022.
Closed-loop fork results across many configurations:

EchoKV-3C active (Run B, init INT8, controller-driven): 3-seed mean 0.447, well below static INT4
Upgrade-only @ grace=20/50 various tau: cascades to uniform INT8 (~0.570)
Upgrade-only @ tau=2.87e-4 (fork-calibrated): cascades to uniform INT8 (0.610)
Per-rank tier vectors span uniform-INT4 to uniform-INT8 across seeds in active runs

Diagnostic measurements that ruled things in/out:

Run A (frozen INT4) vs Run C (frozen INT8) at same seed=100: 3.3× contamination ratio at all tiers (real, secondary)
Run B (active INT8) vs Run C (frozen INT8): 5× compression at tiers 4/8 (real, but cause unclear)
Calibration corpus shift (RULER@4K vs GSM8K-Hard production): ≤2.2× (ruled out)
Per-rank consolidation disagreement: ~2× (ruled out)
Seed=0 vs seed=100 GSM8K-Hard prompts on fork, controller-frozen uniform INT4: 15.06× swing in canary δ̂_4 p50 (real, big)
Same comparison at uniform INT8 substrate: 1.51× (gap collapses)
Same comparison at fork-shadow-mode (BF16 cache + simulate_quant at hook): 1.24× (gap also collapses)

Current diagnosis state
The 15× seed-variance is in the canary's measurement path, not the substrate's effect on model output. Two pieces of evidence support this:

Fork static INT4 ≈ fork FP16 statically (Δ=+0.010 n=1000) — substrate doesn't hurt model output meaningfully
Seed-variance collapses to 1.5× when substrate is INT8 or shadow-mode

The current best mechanism candidate (not fully verified): the fork's Triton dequant kernel and Python simulate_quantization produce numerically different outputs from the same input, despite same granularity. Differences are small (don't affect argmax / model output) but compound across 80 layers in a prompt-dependent way, producing 15× swings in the canary's KL signal. The canary measures the gap between two dequant implementations, not the gap between quantized and unquantized model behavior.
Earlier hypotheses that were proposed and discarded:

Activation buffer (input-distribution contamination): would address 3.3× component, not on critical path
Upgrade-only warmup: mechanism works (78/80 rank unanimity) but cascades upward due to early-warmup transients, grace period doesn't fix it
Coarser-grained KIVI: based on wrong granularity diagnosis
Per-token quantization variance amplification: ruled out by static substrate equivalence

Where the project actually is
The fork shows: GSM8K-Hard at 70B has no meaningful FP16-vs-INT4 deficit on this protocol. The closed loop has no signal to recover. Every fork closed-loop experiment has either cascaded to uniform INT8 (paying memory for marginal accuracy) or produced controller-driven accuracy below static INT4 (controller actively harmful).
The §6.3 published result of 0.620 at avg=6.0 bits is +0.132 above emulation FP16 = 0.488. This is not deficit recovery — the deficit is at most 0.022. The number is suspicious in itself. Its reproducibility at multi-seed n=500 has never been checked, even though it's the load-bearing experimental claim of the paper.
What needs to happen next
The §6.3 emulation reproducibility check is the bottleneck. Run §6.3's exact emulation protocol at 3 seeds, n=500 per seed, paired McNemar against emulation FP16 on the same prompts each seed. Report per-seed and aggregate.
Three outcomes:
A. §6.3 reproduces stably (0.620 ± 0.02 across seeds, tier vectors agree). Paper's central result holds; fork is exposing a substrate-specific issue. Then decide: pursue Path 2 (find a workload with real INT4 deficit on the fork to demonstrate closed-loop value) or Path 3 (narrow the paper to claim emulation-only deployment regime, ship at MLSys 2027).
B. §6.3 is fragile (single-seed outlier, doesn't reproduce, or reproduces at very different number). Paper's central GSM8K-Hard claim was never solid. Fork investigation has been chasing a non-reproducible result. Paper rewrites substantially around what does reproduce — HotpotQA and RULER@128K need the same multi-seed re-verification.
C. §6.3 reproduces stably but at a different number than 0.620. Probably some median between FP16 and the outlier. Paper updates with the corrected number and a methodology note about the importance of multi-seed evaluation in this regime.
Cost: 4-6 hours of emulation wall-clock.
Why this experiment matters more than further fork work
Every fork diagnostic has assumed §6.3 is real and the fork is failing to reproduce it. If §6.3 isn't real (Outcome B), most of the fork diagnostic chain has been investigating noise. If §6.3 is real but reproduces at, say, 0.550 instead of 0.620 (Outcome C), the deficit-recovery framing was wrong and the paper rewrites without invalidating the experiment. Either way, the §6.3 reproducibility check determines what the paper is even trying to claim, which determines what the fork is supposed to validate.
Specific commands to run
The driver for §6.3 should be evaluation/serving/run_session14_70b_gsm8k_hard_echokv.py (or whatever the latest emulation EchoKV-3C driver is). Run with three seeds (0, 1, 2 or 0, 7, 42), --n-prompts 500, --max-num-seqs 8, paired McNemar against run_session14_70b_gsm8k_hard.py FP16 baseline on the same prompts. Standard 3-phase protocol: warmup → consolidate → measure with --freeze-controller and --canary-rate 1e-6 during phase 3.
Outputs to report:

Per-seed: tier vector, accuracy, paired Δ vs FP16 on same prompts, McNemar p, discordant counts
Aggregate: mean accuracy across seeds, accuracy std, tier vector stability (per-layer max disagreement across seeds)
Whether any seed lands within ±0.02 of the published 0.620

Key files and paths
Host repo: /home/nazar/echokv

evaluation/motivation/kivi.py — simulate_quantization implementation (per-token group-asymmetric, group_size=32)
evaluation/serving/run_session14_70b_gsm8k_hard_echokv.py — emulation EchoKV-3C driver
evaluation/serving/run_phaseG_3phase_fork.py — fork EchoKV-3C driver
evaluation/serving/fork_tier.py — fork tier vector helpers
evaluation/serving/diagnostic_compare_emas.py — Run A/B/C comparison script
results/serving/gsm8k_hard_n500.json — emulation §6.3 published result (single-seed)
results/serving/phaseG_pareto_fork_70b_init8_tau3e4.json — fork Run B
results/serving/phaseG_diag_runA_int4_frozen.json — fork Run A
results/serving/phaseG_diag_runC_int8_frozen.json — fork Run C
PAPER_DRAFT.md — current draft including §6.8 stub
echokv_arxiv_v3.tex / echokv_arxiv_v3.pdf — final arxiv preprint (compressed structure)

Fork: /mnt/nvme_tmp10/nazar_vllm_fork, branch feature/echokv-int4-cache

VLLM_ARCHITECTURE.md and VLLM_CHANGELOG.md document the full integration
122 unit tests in tests/
Validated at TP=4 on Llama-3.1-70B

Hardware: 8×H100-SXM5 80GB on NVLink. Single node. Fork uses TP=4 for 70B.
Software pins: vLLM 0.16.0, FlashInfer 0.6.3, Triton 3.5.1, PyTorch 2.9.1+cu128, transformers 4.57.6.
Standing instructions for the next Claude
Do not run any further fork-side diagnostic or fix experiments until the §6.3 reproducibility check is complete. The diagnostic chain on the fork has been long and has discovered real phenomena (3.3× contamination, 5× compression, 15× seed variance, kernel numerical drift) but the question "what is the paper actually trying to show" depends on §6.3 holding up, and that hasn't been verified.
After §6.3 results come back, the decision tree is:

Reproduces stably at ~0.620: choose Path 2 (find a workload on the fork with a real deficit) or Path 3 (narrow the paper's deployment claim and ship). Path 2 candidates: RULER@128K multi-needle, LongBench v2 hard, coding benchmarks at 70B.
Reproduces stably at a different number: update §6.3 with the corrected number, soften the deficit-recovery framing, run the same check on §6.4 (HotpotQA) and §6.5 (RULER@128K) before finalizing.
Doesn't reproduce stably: substantial paper rewrite. The paper's contribution becomes the architectural framework (absolute canary, √L scaling, multi-precision controller, TP-aware integration) plus a methodology contribution about multi-seed evaluation being load-bearing in this regime. Fork findings become "the fork integration revealed that single-seed evaluation can produce non-reproducible results in this regime; multi-seed paired McNemar at large n is the right protocol."

Don't speculate beyond what's measured. Don't multiply ratios across independent measurements (an earlier diagnostic chain claimed a 30× feedback effect by inferring it from a 100× total that was never measured — this was caught and corrected, but the pattern is easy to repeat). Don't propose architectural fixes before the measurement question is resolved.
The §6.3 check is half a day. Run it first.