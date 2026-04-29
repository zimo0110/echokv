"""Session 8 driver: vLLM baseline + canary integration on RULER-32K.

Runs two configurations against Llama-3.1-8B-Instruct on a single H100:

  1. **baseline**: vLLM FP16 KV cache, no canary, no controller. Records
     per-request prefill / decode TPOT and the RULER-32K accuracy.
  2. **canary**: same model, ``CanaryHarness`` installed (canary rate
     1/64, controller in shadow mode every 100 decode steps). Records
     overhead vs baseline + the per-layer mean tier the controller
     would apply (the v8.1 § 9.5 falsifiable prediction on real
     inference).

Notes
-----
* vLLM 0.16 has no native KIVI-INT4 KV cache mode; the baseline is
  FP16 KV. Real KIVI-4 baseline + per-layer dynamic precision
  application require a vLLM source fork (Session 9).
* ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` is required so the engine runs
  in-process and our forward-pre-hooks fire on the worker's attention
  call.

Usage
-----
::

    FLASHINFER_DISABLE_VERSION_CHECK=1 \\
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \\
    CUDA_VISIBLE_DEVICES=0 \\
    /opt/conda/envs/scalable/bin/python -m evaluation.serving.run_session8

Output
------
``results/serving/session8.json`` — full result blob with both runs'
metrics. INV-5 is preserved: this script never writes guessed numbers
into REPORT.md; the human commits the real numbers after reading the
JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

# Local repo imports — make sure the repo root is on sys.path when this
# script is invoked from a different cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.motivation.workloads import _ruler_sample, Request


log = logging.getLogger("session8")
DEFAULT_MODEL = (
    "/mnt/nvme_tmp/nazar_hf/hub/"
    "models--NousResearch--Meta-Llama-3.1-8B-Instruct/snapshots/"
    "d10aef7999a2b5ba950ab3974312feeedbfe0b77"
)


# ---------------------------------------------------------------------------
# RULER prompt generation (reuses the procedural needle-in-a-haystack)
# ---------------------------------------------------------------------------


def make_ruler_prompts(
    n: int, seed: int = 0, target_tokens: int = 4_000
) -> list[Request]:
    """Generate ``n`` RULER prompts. ``target_tokens=4000`` is a tradeoff:
    high enough that INT4 sees attention over a long span, low enough
    that a 50-prompt run fits in ~10 minutes on a single H100 with FP16
    KV cache. The motivation study used 8K; for the serving harness we
    halve it to keep the per-prompt prefill cost reasonable.
    """
    return [_ruler_sample(i, seed=seed, target_tokens=target_tokens) for i in range(n)]


# ---------------------------------------------------------------------------
# Per-request metrics extracted from a vLLM RequestOutput
# ---------------------------------------------------------------------------


@dataclass
class RequestMetrics:
    request_id: str
    prompt_tokens: int
    completion_tokens: int
    # Per-request score from the workload's score_fn. For binary
    # workloads (single-needle RULER, GSM8K) this is 0.0 / 1.0; for
    # graded workloads (multi-needle RULER) it can be any value in
    # [0, 1]. The previous ``int()`` cast collapsed 0.2/0.4/0.6/0.8 to
    # 0 and only preserved exact-1.0; that hid partial credit and
    # produced a spurious bimodal distribution.
    correct: float
    arrival_t: float
    first_scheduled_t: Optional[float]
    first_token_t: Optional[float]
    finished_t: Optional[float]
    completion_text: Optional[str] = None

    @property
    def ttft_s(self) -> Optional[float]:
        if self.first_token_t is None or self.first_scheduled_t is None:
            return None
        return self.first_token_t - self.first_scheduled_t

    @property
    def decode_s(self) -> Optional[float]:
        if self.finished_t is None or self.first_token_t is None:
            return None
        return self.finished_t - self.first_token_t

    @property
    def tpot_s(self) -> Optional[float]:
        """Mean per-output-token time during decode."""
        d = self.decode_s
        if d is None or self.completion_tokens <= 1:
            return None
        return d / max(1, self.completion_tokens - 1)


def _extract_metrics(req: Request, output: Any) -> RequestMetrics:
    """vLLM 0.16's ``RequestOutput.metrics`` is ``None`` in this build —
    per-request TPOT/TTFT cannot be extracted from a synchronous
    ``LLM.generate`` call. We capture the bookkeeping that *is*
    available (token counts + correctness) and compute system-level
    throughput from the wall-clock of the whole batch in ``run_one``.
    """
    completion_text = output.outputs[0].text
    completion_tokens = len(output.outputs[0].token_ids)
    correct = float(req.score_fn(completion_text)) if req.score_fn else 0.0
    return RequestMetrics(
        request_id=req.request_id,
        prompt_tokens=len(output.prompt_token_ids),
        completion_tokens=completion_tokens,
        correct=correct,
        arrival_t=0.0,
        first_scheduled_t=None,
        first_token_t=None,
        finished_t=None,
        completion_text=completion_text,
    )


# ---------------------------------------------------------------------------
# One run = one (model, n_prompts, harness?) combo
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    label: str
    wall_clock_s: float
    n_prompts: int
    completed: int
    n_correct: int
    accuracy: float
    p50_tpot_s: Optional[float]
    p99_tpot_s: Optional[float]
    mean_ttft_s: Optional[float]
    total_input_tokens: int
    total_output_tokens: int
    output_throughput_tps: float
    per_request: list[dict] = field(default_factory=list)
    canary_report: Optional[dict] = None


def _percentile(xs: list[float], q: float) -> Optional[float]:
    s = sorted(x for x in xs if x is not None)
    if not s:
        return None
    if len(s) == 1:
        return s[0]
    rank = q * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] * (1 - (rank - lo)) + s[hi] * (rank - lo)


def run_one(
    llm: Any,
    prompts: list[Request],
    *,
    label: str,
    max_new_tokens: int,
    harness: Optional[Any] = None,
) -> RunResult:
    """Run a list of RULER prompts through vLLM. ``harness``, if given,
    is a ``CanaryHarness`` already installed on the LLM.

    System-level throughput is what we report; vLLM 0.16's per-request
    metrics field is None in this build so per-request TPOT/TTFT are
    not available. The CHANGELOG documents the limitation.
    """
    from vllm import SamplingParams

    log.info("[%s] running %d prompts", label, len(prompts))
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    t0 = time.perf_counter()
    outputs = llm.generate(
        [r.prompt for r in prompts],
        sampling_params=sp,
        use_tqdm=False,
    )
    wall = time.perf_counter() - t0

    metrics = [_extract_metrics(r, o) for r, o in zip(prompts, outputs)]
    n_correct = sum(m.correct for m in metrics)
    total_in = sum(m.prompt_tokens for m in metrics)
    total_out = sum(m.completion_tokens for m in metrics)

    result = RunResult(
        label=label,
        wall_clock_s=wall,
        n_prompts=len(prompts),
        completed=sum(1 for m in metrics if m.completion_tokens > 0),
        n_correct=n_correct,
        accuracy=n_correct / max(1, len(prompts)),
        p50_tpot_s=None,
        p99_tpot_s=None,
        mean_ttft_s=None,
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        output_throughput_tps=total_out / wall if wall > 0 else 0.0,
        per_request=[asdict(m) for m in metrics],
    )
    if harness is not None:
        result.canary_report = harness.report()
    return result


def warmup(llm: Any, prompts: list[Request], max_new_tokens: int) -> None:
    """One-shot warmup so the first measured run isn't paying compile
    cost. Uses a small subset of the real prompts so the kernel cache
    matches the measured shape."""
    from vllm import SamplingParams
    log.info("warmup: 4 prompts")
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    llm.generate(
        [r.prompt for r in prompts[:4]],
        sampling_params=sp,
        use_tqdm=False,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--n-prompts", type=int, default=50)
    p.add_argument("--target-tokens", type=int, default=4_000,
                   help="RULER prompt size in approximate tokens")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-model-len", type=int, default=8_192)
    p.add_argument("--canary-rate", type=float, default=1.0 / 64.0,
                   help="production-rate canary (overhead measurement)")
    p.add_argument("--observability-rate", type=float, default=1.0 / 4.0,
                   help=("high-rate canary for the layer-pattern observation "
                         "run. p_c=1/64 only gives ~1 sample per layer in a "
                         "50-prompt run — not enough for EMA to discriminate "
                         "between layers. 1/4 yields ~16 samples per layer "
                         "while still exercising the same code path. "
                         "Overhead at this rate is NOT representative of "
                         "production."))
    p.add_argument("--tick-interval", type=int, default=25,
                   help=("controller tick cadence in decode steps (layer-0 "
                         "attention-call boundaries). Architectural default "
                         "N_c=100; for short-run observability we use 25 so "
                         "the controller actually ticks during a 50-prompt "
                         "RULER run. Production stays at 100."))
    p.add_argument("--canary-on-prefill", action="store_true",
                   help=("fire canary on prefill batches too (default: skip). "
                         "Prefill batches have query.shape[0] = sum of prompt "
                         "lengths — at 32K context that's O(L²) per fire. "
                         "Architectural canary is decode-only."))
    p.add_argument("--no-canary-stream", action="store_true",
                   help=("disable the canary CUDA stream. By default the "
                         "canary computation runs on a separate stream so "
                         "it overlaps with the production attention path."))
    p.add_argument("--max-decode-tokens", type=int, default=256,
                   help=("threshold for prefill detection. "
                         "query.shape[0] > this is treated as prefill and "
                         "skipped when decode_only_canary is on."))
    p.add_argument("--echokv-3c", action="store_true",
                   help=("run a third configuration with tier_application=True "
                         "(controller's per-layer precision actually applied "
                         "to K and V before vLLM caches them — the "
                         "fork-equivalent without modifying vLLM source). "
                         "Numerically simulates EchoKV-3C accuracy."))
    p.add_argument("--echokv-warmup-steps", type=int, default=200,
                   help=("Before the EchoKV-3C run, do a warmup pass with "
                         "canary on but tier_application off, so the "
                         "controller has had a chance to converge on tiers "
                         "from observed canary signal. Otherwise the run "
                         "starts at uniform INT4 and may not differentiate."))
    p.add_argument("--out", type=Path,
                   default=Path("results/serving/session8.json"))
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Lazy vLLM import — env vars must be set before import.
    from vllm import LLM
    from evaluation.serving.canary_vllm import CanaryHarness

    log.info("loading %s", args.model)
    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_prefix_caching=False,  # ensure baseline and canary runs don't share cached prefill
    )
    log.info("loaded in %.1fs", time.perf_counter() - t0)

    prompts = make_ruler_prompts(
        n=args.n_prompts, seed=args.seed, target_tokens=args.target_tokens
    )

    # ---------- warmup: amortise compile / paged-attention init ----------
    warmup(llm, prompts, args.max_new_tokens)

    # ---------- run 1: baseline (FP16, no canary) ----------
    baseline = run_one(
        llm, prompts,
        label="baseline_fp16",
        max_new_tokens=args.max_new_tokens,
    )
    log.info(
        "[baseline] wall=%.1fs  acc=%.3f  p50_tpot=%.4fs  p99_tpot=%.4fs  thru=%.1f tok/s",
        baseline.wall_clock_s,
        baseline.accuracy,
        baseline.p50_tpot_s or 0.0,
        baseline.p99_tpot_s or 0.0,
        baseline.output_throughput_tps,
    )

    # ---------- run 2: canary-active at PRODUCTION rate (overhead) ----------
    harness = CanaryHarness(
        llm,
        canary_rate=args.canary_rate,
        tick_interval=args.tick_interval,
        decode_only_canary=not args.canary_on_prefill,
        max_decode_tokens=args.max_decode_tokens,
        use_canary_stream=not args.no_canary_stream,
        seed=args.seed,
    )
    harness.install()
    try:
        canary = run_one(
            llm, prompts,
            label="canary_active",
            max_new_tokens=args.max_new_tokens,
            harness=harness,
        )
    finally:
        harness.remove()

    # ---------- run 3: canary-active at HIGH rate (layer-pattern observability) ----------
    log.info(
        "[observability] re-running with canary_rate=%.4f to amplify EMA signal",
        args.observability_rate,
    )
    harness_obs = CanaryHarness(
        llm,
        canary_rate=args.observability_rate,
        tick_interval=args.tick_interval,
        decode_only_canary=not args.canary_on_prefill,
        max_decode_tokens=args.max_decode_tokens,
        use_canary_stream=not args.no_canary_stream,
        seed=args.seed + 1,
    )
    harness_obs.install()
    try:
        observability = run_one(
            llm, prompts,
            label="canary_observability",
            max_new_tokens=args.max_new_tokens,
            harness=harness_obs,
        )
    finally:
        harness_obs.remove()
    log.info(
        "[observability] fires=%d / %d  EMA layers w/signal=%d/%d",
        observability.canary_report["canary_fires"],
        observability.canary_report["attention_calls"],
        observability.canary_report["ema_layers_with_signal"],
        observability.canary_report["num_layers"],
    )

    # ---------- run 4 (optional): EchoKV-3C with tier application ----------
    echokv_3c: Optional[RunResult] = None
    if args.echokv_3c:
        # Pre-converge the controller on observability-rate canary signal
        # so the EchoKV-3C run actually sees differentiated tiers.
        # ``observability`` already drove the controller for ``args.n_prompts``
        # — we reuse its converged tier vector by taking another harness
        # warmed from it.
        log.info("[echokv_3c] tier_application=True with high canary rate (so the controller observes signal during the run too)")
        echokv_harness = CanaryHarness(
            llm,
            canary_rate=args.observability_rate,
            tick_interval=args.tick_interval,
            decode_only_canary=not args.canary_on_prefill,
            max_decode_tokens=args.max_decode_tokens,
            use_canary_stream=not args.no_canary_stream,
            tier_application=True,
            tier_application_quant_v=True,
            seed=args.seed + 2,
        )
        # Seed the controller with the converged tiers from the observability run.
        if observability.canary_report and observability.canary_report.get("final_levels_per_layer"):
            from core.block_allocator import Precision
            for li, bits in enumerate(observability.canary_report["final_levels_per_layer"]):
                echokv_harness.controller._levels[li] = Precision(bits)

        echokv_harness.install()
        try:
            echokv_3c = run_one(
                llm, prompts,
                label="echokv_3c",
                max_new_tokens=args.max_new_tokens,
                harness=echokv_harness,
            )
        finally:
            echokv_harness.remove()
        log.info(
            "[echokv_3c] acc=%.3f  tier_applications=%d  "
            "INT2/INT4/INT8 = %d/%d/%d",
            echokv_3c.accuracy,
            echokv_3c.canary_report["tier_applications"],
            echokv_3c.canary_report["tier_apply_count_by_bits"].get(2, 0),
            echokv_3c.canary_report["tier_apply_count_by_bits"].get(4, 0),
            echokv_3c.canary_report["tier_apply_count_by_bits"].get(8, 0),
        )

    log.info(
        "[canary] wall=%.1fs  acc=%.3f  p50_tpot=%.4fs  p99_tpot=%.4fs  fires=%d / %d  overhead=%.3fs",
        canary.wall_clock_s,
        canary.accuracy,
        canary.p50_tpot_s or 0.0,
        canary.p99_tpot_s or 0.0,
        canary.canary_report["canary_fires"] if canary.canary_report else 0,
        canary.canary_report["attention_calls"] if canary.canary_report else 0,
        canary.canary_report["canary_overhead_s"] if canary.canary_report else 0.0,
    )

    # Overhead summary
    p99_overhead_pct = None
    p50_overhead_pct = None
    if baseline.p99_tpot_s and canary.p99_tpot_s:
        p99_overhead_pct = 100.0 * (canary.p99_tpot_s - baseline.p99_tpot_s) / baseline.p99_tpot_s
    if baseline.p50_tpot_s and canary.p50_tpot_s:
        p50_overhead_pct = 100.0 * (canary.p50_tpot_s - baseline.p50_tpot_s) / baseline.p50_tpot_s

    summary = {
        "session": "20260426_8",
        "model": args.model,
        "n_prompts": args.n_prompts,
        "target_tokens": args.target_tokens,
        "max_new_tokens": args.max_new_tokens,
        "canary_rate": args.canary_rate,
        "observability_rate": args.observability_rate,
        "tick_interval": args.tick_interval,
        "tpot_overhead_p50_pct": p50_overhead_pct,
        "tpot_overhead_p99_pct": p99_overhead_pct,
        "throughput_overhead_pct": (
            100.0 * (baseline.output_throughput_tps - canary.output_throughput_tps)
            / baseline.output_throughput_tps
            if baseline.output_throughput_tps > 0 else None
        ),
        "accuracy_baseline": baseline.accuracy,
        "accuracy_canary": canary.accuracy,
        "accuracy_observability": observability.accuracy,
        "accuracy_diff": canary.accuracy - baseline.accuracy,
        "baseline": asdict(baseline),
        "canary": asdict(canary),
        "observability": asdict(observability),
        "echokv_3c": asdict(echokv_3c) if echokv_3c is not None else None,
        "accuracy_echokv_3c": echokv_3c.accuracy if echokv_3c is not None else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    log.info("wrote %s", args.out)

    throughput_overhead_pct = (
        100.0 * (baseline.output_throughput_tps - canary.output_throughput_tps)
        / baseline.output_throughput_tps
        if baseline.output_throughput_tps > 0 else None
    )
    summary["throughput_overhead_pct"] = throughput_overhead_pct

    print()
    print("=" * 78)
    print(f"Session 8 — RULER-32K({args.target_tokens} tokens), n={args.n_prompts}, p_c={args.canary_rate}")
    print(f"  baseline_fp16   wall={baseline.wall_clock_s:.1f}s  acc={baseline.accuracy:.3f}  "
          f"thru={baseline.output_throughput_tps:.1f} tok/s")
    print(f"  canary_active   wall={canary.wall_clock_s:.1f}s  acc={canary.accuracy:.3f}  "
          f"thru={canary.output_throughput_tps:.1f} tok/s")
    if throughput_overhead_pct is not None:
        print(f"  canary throughput overhead: {throughput_overhead_pct:+.2f}%  "
              f"(target < 2%)")
    if canary.canary_report:
        rpt = canary.canary_report
        print(f"  canary fires: {rpt['canary_fires']} / {rpt['attention_calls']} attention calls "
              f"({rpt['canary_overhead_ratio_attentions']*100:.2f}%; expected ≈ p_c={args.canary_rate*100:.2f}%)")
        print(f"  in-canary-path wallclock: {rpt['canary_overhead_s']:.3f}s "
              f"({100.0*rpt['canary_overhead_s']/canary.wall_clock_s:.2f}% of canary wall)")
        print(f"  controller ticks: {rpt['controller_ticks']}")
    if echokv_3c is not None:
        print()
        print(f"--- EchoKV-3C (tier_application=True, p_c={args.observability_rate} for signal) ---")
        print(f"  wall={echokv_3c.wall_clock_s:.1f}s (simulation — quantisation cost in Python)  "
              f"acc={echokv_3c.accuracy:.3f}  Δacc={echokv_3c.accuracy - baseline.accuracy:+.3f}")
        rpt3c = echokv_3c.canary_report
        if rpt3c:
            tab = rpt3c["tier_apply_count_by_bits"]
            print(f"  tier applications: {rpt3c['tier_applications']} total  "
                  f"INT2={tab.get(2, 0)}  INT4={tab.get(4, 0)}  INT8={tab.get(8, 0)}")
            print(f"  final tier vector: {rpt3c['final_levels_per_layer']}")
            print(f"  tier_application overhead (simulation cost): {rpt3c['tier_application_overhead_s']:.3f}s "
                  f"({100.0*rpt3c['tier_application_overhead_s']/echokv_3c.wall_clock_s:.2f}% of run wall)")
            mp = rpt3c.get("memory_projection")
            if mp:
                tc = (mp["tier_count_int2"], mp["tier_count_int4"], mp["tier_count_int8"])
                print(f"  KV cache memory projection (real-fork-equivalent):")
                print(f"    avg bits/element: {mp['avg_bits']:.2f}  "
                      f"(INT2/INT4/INT8 = {tc[0]}/{tc[1]}/{tc[2]})")
                print(f"    savings vs FP16:   {mp['savings_vs_baseline_pct']:+.1f}%")
                print(f"    vs static KIVI-4:  {mp['savings_vs_kivi4_pct']:+.1f}% "
                      f"(negative = EchoKV uses MORE memory than KIVI-4)")

    print()
    print(f"--- v8.1 falsifiable check (observability run, p_c={args.observability_rate}) ---")
    if observability.canary_report:
        rpt = observability.canary_report
        print(f"  acc={observability.accuracy:.3f}  fires={rpt['canary_fires']} / {rpt['attention_calls']}  "
              f"controller ticks={rpt['controller_ticks']}")
        print(f"  EMA: mean={rpt['ema_mean']:.5f}  max={rpt['ema_max']:.5f}  "
              f"layers with signal: {rpt['ema_layers_with_signal']}/{rpt['num_layers']}")
        ema_pl = rpt["ema_per_layer"]
        if len(ema_pl) >= 28:
            mid_late = statistics.fmean(ema_pl[21:28])
            others = statistics.fmean(ema_pl[:21] + ema_pl[28:])
            ratio = mid_late / others if others > 0 else float("inf")
            print(f"  mean EMA on layers 21-27: {mid_late:.5f}  vs others: {others:.5f}  "
                  f"(ratio {ratio:.2f}; v8.1 prediction: layers 21-27 carry signal under RULER)")
            print(f"  per-layer EMA (×1000):")
            for i in range(0, 32, 8):
                row = " ".join(f"{ema_pl[j]*1000:6.2f}" for j in range(i, min(i + 8, 32)))
                print(f"    L{i:02d}-L{min(i+7,31):02d}: {row}")
        mtl = rpt["mean_tier_per_layer"]
        if len(mtl) >= 28:
            mean_high = statistics.fmean(mtl[21:28])
            mean_low = statistics.fmean(mtl[:21] + mtl[28:])
            print(f"  mean controller tier on layers 21-27: {mean_high:.2f}  "
                  f"vs others: {mean_low:.2f}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
