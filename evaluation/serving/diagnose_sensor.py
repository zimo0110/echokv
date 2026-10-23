"""Sensor-failure diagnostic for the long-context blind spot.

Runs RULER under **static INT2** (the configuration that was shown in
Session 11 to have acc=0.000 at 64K — i.e. the model is catastrophically
broken) and collects:

A. **δ̂_ℓ distribution per context length** at 4K, 32K, 64K.
   If δ̂_ℓ drops monotonically with context, the canary is becoming
   weaker at exactly the regime where it should be screaming. That
   confirms the diffuse-attention hypothesis.

B. **Attention entropy at decode steps** under static INT2 at 64K.
   Entropy = −Σ α_t log α_t  (computed in float32 for stability,
   normalised by ``log(L)`` so 1.0 = perfectly uniform, 0.0 = fully
   peaked on one position). If entropy is high (>0.7 or so), the
   model is attending diffusely over the cache — KL divergence
   between two diffuse distributions is small regardless of the
   underlying K-quant error, and the canary cannot fire.

Why these two diagnostics together: (A) shows the canary's signal
disappears at long context; (B) shows *why* — the attention pattern
the canary measures becomes uninformative.

Output: ``results/serving/sensor_diagnosis.json`` with per-(context,
layer) EMA δ̂ values, and per-(decode-step, layer) attention entropy
for one 64K request.

Usage::

    FLASHINFER_DISABLE_VERSION_CHECK=1 \\
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \\
    CUDA_VISIBLE_DEVICES=1 \\
    /opt/conda/envs/scalable/bin/python -m evaluation.serving.diagnose_sensor
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.motivation.workloads import _ruler_sample
from evaluation.serving.canary_vllm import (
    CanaryHarness,
    _to_bhsd,
)
from evaluation.serving.run_session8 import DEFAULT_MODEL


log = logging.getLogger("diagnose_sensor")


# ---------------------------------------------------------------------------
# Diagnostic A: δ̂ distribution at 4K / 32K / 64K under static INT2
# ---------------------------------------------------------------------------


def collect_delta_distribution(
    llm: Any,
    *,
    context: int,
    n_prompts: int = 8,
    max_new_tokens: int = 16,
    canary_rate: float = 0.5,
    seed: int = 0,
) -> dict:
    """Run ``n_prompts`` RULER prompts under static INT2 with full canary
    on, and return the post-run per-layer EMA δ̂_ℓ distribution.
    """
    from vllm import SamplingParams

    prompts = [
        _ruler_sample(i, seed=seed, target_tokens=context)
        for i in range(n_prompts)
    ]

    # Static INT2 baseline: force_bits=2, full canary so EMA gets samples.
    harness = CanaryHarness(
        llm,
        canary_rate=canary_rate,
        tier_application=True,
        tier_application_quant_v=True,
        force_bits=2,
        freeze_controller=True,                # don't let the controller move
        seed=seed + 1000 + context,
    )
    harness.install()
    try:
        sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        outputs = llm.generate(
            [r.prompt for r in prompts], sampling_params=sp, use_tqdm=False,
        )
    finally:
        harness.remove()

    # Score (RULER returns 1 if model emitted the magic number).
    n_correct = 0
    for r, o in zip(prompts, outputs):
        text = o.outputs[0].text
        if r.score_fn is not None:
            try:
                if r.score_fn(text):
                    n_correct += 1
            except Exception:
                pass

    ema = harness.estimator.snapshot()
    return {
        "context": context,
        "n_prompts": n_prompts,
        "accuracy_under_int2": n_correct / max(1, n_prompts),
        "canary_fires": harness.stats.canary_fires,
        "attention_calls": harness.stats.attention_calls,
        "ema_per_layer": list(ema),
        "ema_layers_with_signal": sum(1 for v in ema if v > 0.0),
        "ema_min": min(ema) if ema else None,
        "ema_max": max(ema) if ema else None,
        "ema_mean": (sum(ema) / len(ema)) if ema else None,
        "ema_p50": (
            sorted(ema)[len(ema) // 2] if ema else None
        ),
    }


# ---------------------------------------------------------------------------
# Diagnostic B: attention entropy at decode steps under static INT2 (64K)
# ---------------------------------------------------------------------------
#
# We need access to the per-step attention weights. The canary's pre-hook
# already sees Q and K at every Attention.forward call. For the entropy
# diagnostic we only need *one* request's worth — install a custom hook
# that captures softmax(QK^T) at every layer for every decode step, then
# compute entropy normalised by log(L).


def _entropy_normalised(weights: torch.Tensor) -> torch.Tensor:
    """Normalised attention entropy. weights shape (..., L). Returns
    shape weights.shape[:-1]. 1.0 = uniform attention; 0.0 = fully
    peaked on a single position."""
    eps = 1e-9
    p = weights.clamp_min(eps)
    H = -(p * p.log()).sum(dim=-1)
    L = weights.shape[-1]
    if L <= 1:
        return torch.zeros_like(H)
    return H / math.log(L)


def collect_attention_entropy(
    llm: Any,
    *,
    context: int = 64_000,
    max_new_tokens: int = 16,
    seed: int = 0,
) -> dict:
    """Run one RULER prompt under static INT2 at the requested context;
    record per-(layer, decode-step) attention entropy.
    """
    from vllm import SamplingParams

    # Build the model accessor the same way the harness does.
    model = (
        llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
    )
    attn_modules = []
    for name, mod in model.named_modules():
        if type(mod).__name__ == "Attention":
            import re
            m = re.search(r"layers\.(\d+)", name)
            if m:
                attn_modules.append((int(m.group(1)), mod))
    attn_modules.sort(key=lambda t: t[0])
    L = len(attn_modules)
    if L == 0:
        raise RuntimeError("no Attention modules")

    # Capture (entropy_per_decode_step, prompt_len) per layer.
    capture: dict = {layer_id: [] for layer_id, _ in attn_modules}
    prompt_len_holder = {"v": None}

    def make_hook(layer_id: int, mod: torch.nn.Module):
        num_heads = (
            getattr(mod, "num_heads", None)
            or getattr(getattr(mod, "impl", None), "num_heads", None)
        )
        num_kv_heads = (
            getattr(mod, "num_kv_heads", None)
            or getattr(mod, "num_key_value_heads", None)
            or num_heads
        )

        def pre_hook(module, args):
            try:
                query = args[0]
                key = args[1]
            except Exception:
                return None
            # Decode-only: skip prefill batches (long query).
            if query.shape[0] > 256:
                if prompt_len_holder["v"] is None:
                    prompt_len_holder["v"] = int(query.shape[0])
                return None
            try:
                with torch.no_grad():
                    q = _to_bhsd(query.detach(), num_heads).float()  # (1, H, T, D)
                    k = _to_bhsd(key.detach(), num_kv_heads).float()
                    if k.shape[1] != q.shape[1]:
                        rep = q.shape[1] // k.shape[1]
                        k = k.repeat_interleave(rep, dim=1)
                    d_head = q.shape[-1]
                    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_head ** 0.5)
                    weights = F.softmax(scores, dim=-1)  # (1, H, T, L_kv)
                    H = _entropy_normalised(weights).mean().item()  # mean over heads + tokens
                capture[layer_id].append(float(H))
            except Exception:
                pass
            return None

        return pre_hook

    handles = []
    for layer_id, mod in attn_modules:
        handles.append(mod.register_forward_pre_hook(make_hook(layer_id, mod)))

    try:
        prompt = _ruler_sample(0, seed=seed, target_tokens=context).prompt
        sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        out = llm.generate([prompt], sampling_params=sp, use_tqdm=False)
    finally:
        for h in handles:
            h.remove()

    completion_tokens = len(out[0].outputs[0].token_ids) if out else 0

    # Aggregate: per layer mean entropy across all decode steps; also
    # per-decode-step entropy averaged over layers (for time-series
    # plotting later).
    per_layer_mean_entropy = {
        li: (sum(vals) / len(vals)) if vals else None for li, vals in capture.items()
    }
    # length of each layer's series should be the same (= num decode steps)
    series_len = max((len(v) for v in capture.values()), default=0)
    per_step_mean_entropy = []
    for t in range(series_len):
        vals = [capture[li][t] for li in capture if t < len(capture[li])]
        per_step_mean_entropy.append(sum(vals) / len(vals) if vals else None)

    return {
        "context": context,
        "completion_tokens": completion_tokens,
        "decode_steps_captured": series_len,
        "per_layer_mean_entropy": per_layer_mean_entropy,
        "per_step_mean_entropy": per_step_mean_entropy,
        "overall_mean_entropy": (
            sum(v for v in per_step_mean_entropy if v is not None)
            / max(1, sum(1 for v in per_step_mean_entropy if v is not None))
        ),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-model-len", type=int, default=65_536)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--out", type=Path,
                   default=Path("results/serving/sensor_diagnosis.json"))
    p.add_argument("--n-prompts-A", type=int, default=8,
                   help="prompts per context for diagnostic A")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--contexts-A", type=int, nargs="+",
                   default=[4_000, 32_000, 64_000])
    p.add_argument("--context-B", type=int, default=64_000,
                   help="context for diagnostic B (entropy)")
    a = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    from vllm import LLM

    log.info("loading %s (max_model_len=%d)", a.model, a.max_model_len)
    t0 = time.perf_counter()
    llm = LLM(
        model=a.model,
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=a.gpu_memory_utilization,
        max_model_len=a.max_model_len,
        enable_prefix_caching=False,
    )
    log.info("loaded in %.1fs", time.perf_counter() - t0)

    summary = {"model": a.model, "diagnostics_A": [], "diagnostic_B": None}

    # Diagnostic A
    for ctx in a.contexts_A:
        if ctx > a.max_model_len:
            log.info("skipping context=%d (> max_model_len)", ctx)
            continue
        log.info("=== Diagnostic A: context=%d, n=%d, static INT2 ===",
                 ctx, a.n_prompts_A)
        cell = collect_delta_distribution(
            llm,
            context=ctx,
            n_prompts=a.n_prompts_A,
            max_new_tokens=a.max_new_tokens,
            seed=a.seed,
        )
        summary["diagnostics_A"].append(cell)
        log.info(
            "    acc(INT2)=%.3f  EMA mean=%.5f  p50=%.5f  max=%.5f  layers w/signal=%d/32",
            cell["accuracy_under_int2"],
            cell["ema_mean"] or 0.0,
            cell["ema_p50"] or 0.0,
            cell["ema_max"] or 0.0,
            cell["ema_layers_with_signal"],
        )

    # Diagnostic B
    if a.context_B <= a.max_model_len:
        log.info("=== Diagnostic B: context=%d attention entropy under INT2 ===",
                 a.context_B)
        cell = collect_attention_entropy(
            llm, context=a.context_B,
            max_new_tokens=a.max_new_tokens, seed=a.seed,
        )
        summary["diagnostic_B"] = cell
        log.info(
            "    decode steps captured=%d  overall mean entropy=%.4f",
            cell["decode_steps_captured"],
            cell["overall_mean_entropy"],
        )

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=2))
    log.info("wrote %s", a.out)

    print()
    print("=" * 80)
    print("Sensor-failure diagnostic — summary")
    print("=" * 80)
    print(f"{'context':>10}  {'INT2 acc':>10}  {'EMA p50':>12}  {'EMA mean':>12}  {'EMA max':>12}")
    for cell in summary["diagnostics_A"]:
        print(f"{cell['context']:>10}  "
              f"{cell['accuracy_under_int2']:>10.3f}  "
              f"{(cell['ema_p50'] or 0):>12.5f}  "
              f"{(cell['ema_mean'] or 0):>12.5f}  "
              f"{(cell['ema_max'] or 0):>12.5f}")
    if summary["diagnostic_B"] is not None:
        b = summary["diagnostic_B"]
        print()
        print(f"Diagnostic B: 64K static-INT2 attention entropy")
        print(f"  decode steps captured: {b['decode_steps_captured']}")
        print(f"  overall mean normalised entropy: {b['overall_mean_entropy']:.4f}  "
              f"(1.0 = uniform, 0.0 = peaked-on-one-position)")
        # Per-layer summary (top/bottom 3)
        ple = b["per_layer_mean_entropy"]
        ranked = sorted(((li, h) for li, h in ple.items() if h is not None),
                        key=lambda t: t[1])
        if ranked:
            print(f"  most-peaked layers: " +
                  ", ".join(f"L{li:02d}={h:.3f}" for li, h in ranked[:3]))
            print(f"  most-diffuse layers: " +
                  ", ".join(f"L{li:02d}={h:.3f}" for li, h in ranked[-3:]))

    print()
    if summary["diagnostics_A"]:
        contexts = [c["context"] for c in summary["diagnostics_A"]]
        emas = [c["ema_mean"] or 0.0 for c in summary["diagnostics_A"]]
        # Decreasing with context = diffuse-attention hypothesis confirmed
        if len(emas) >= 2 and emas[-1] < emas[0] * 0.6:
            print("VERDICT: δ̂ drops with context length under static INT2 — "
                  "diffuse-attention hypothesis CONFIRMED.")
            print("         Sensor architecture must change (Step 4 Option A "
                  "or B) before 128K experiments.")
        else:
            print("VERDICT: δ̂ does NOT drop monotonically with context.")
            print("         Sensor failure has a different cause — investigate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
