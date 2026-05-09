"""Phase F.3.0: run the EchoKV CanaryHarness against the vLLM fork.

The fork's KV cache is real KIVI INTb (Phase F.2 split-buffer layout)
rather than upstream vLLM's FP16. The canary measures Q/K **before**
attention via a forward-pre-hook, so the harness operates the same way
either side of attention; the only difference is what the kernel does
downstream of the hook (real INTb pack vs FP16 splice).

This driver does ONE pass:
  1. Boot the fork at uniform INT4 (``kv_cache_dtype="int4_kivi"``).
  2. Install the CanaryHarness in *observation-only* mode
     (``tier_application=False``) — it computes deltas and ticks the
     controller, but does NOT mutate K/V before attention. The fork
     already does the real KIVI quant at write time.
  3. Run a batch of warmup-style prompts to drive the controller.
  4. Snapshot the per-rank state: converged tier vector, per-layer
     EMA delta, and harness stats.

Why observation-only on the fork
--------------------------------
``tier_application=True`` was the upstream-vLLM workaround for
"controller decisions actually affect outputs without a source fork".
With the fork, the cache IS quantized at the controller's chosen bits;
shadow-quantizing K/V before attention would double-apply the noise.

Why this driver doesn't apply tier changes at runtime
-----------------------------------------------------
Phase F.2 sizes each layer's paged data buffer for the bits its spec
was constructed with. ``apply_per_layer_kivi_precision`` mutates
``impl.kivi_bits`` in place but the next ``slice_kivi_cache`` asserts
the buffer matches ``page_bytes(active_bits)`` — so runtime tier
*upgrades* would fail (buffer too small) and *downgrades* fail too
(slice asserts exact match, not "at least"). Resolving this requires
either (a) allocating each layer at the max bits with a tolerant
slice, or (b) per-tier dedicated buffers. Both are Phase F.3.1.

The deliverable for F.3.0 is therefore a "what would the controller
recommend?" pass: the resulting tier vector is the input you'd pass
as ``LLM(kivi_bits_per_layer=...)`` to a *second* engine boot — the
warmup-then-freeze pattern matches the host's existing 70B harness
(``run_session14_70b_*_echokv.py``), now exercised end-to-end on the
fork.

Usage
-----
::

    VLLM_ALLOW_INSECURE_SERIALIZATION=1 \\
    FLASHINFER_DISABLE_VERSION_CHECK=1 \\
    python evaluation/serving/run_phaseF3_canary_fork.py --tp 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Force the fork onto sys.path BEFORE any vllm import.
_FORK = "/mnt/nvme_tmp10/nazar_vllm_fork"
if _FORK not in sys.path:
    sys.path.insert(0, _FORK)
_REPO = str(Path(__file__).resolve().parents[2])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


MODEL = (
    "/mnt/nvme_tmp/nazar_hf/hub/"
    "models--NousResearch--Meta-Llama-3.1-8B-Instruct/"
    "snapshots/d10aef7999a2b5ba950ab3974312feeedbfe0b77"
)

WARMUP_PROMPTS = [
    "Q: What is 2 + 3?  A:",
    "Q: What is 7 - 4?  A:",
    "Q: What is 6 * 9?  A:",
    "The capital of France is",
    "The capital of Germany is",
    "Once upon a time, in a kingdom far away,",
    "Explain photosynthesis in one sentence:",
    "List three primary colors:",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument(
        "--cache-dtype", default="int4_kivi",
        choices=["int2_kivi", "int4_kivi", "int8_kivi"],
        help="Engine-wide kv_cache_dtype for the warmup pass.",
    )
    ap.add_argument(
        "--canary-rate", type=float, default=1.0 / 8.0,
        help="Bernoulli rate at which the canary fires per attention "
             "call. Default 1/8 for the smoke (production usually 1/64).",
    )
    ap.add_argument(
        "--tau", type=float, default=0.01,
        help="PID setpoint passed to the controller.",
    )
    ap.add_argument(
        "--tick-interval", type=int, default=10,
        help="Controller ticks every N attention calls (low for the "
             "smoke so we observe at least a few ticks on short prompts).",
    )
    ap.add_argument(
        "--max-tokens", type=int, default=32,
        help="SamplingParams.max_tokens for the warmup batch.",
    )
    ap.add_argument(
        "--out", type=Path,
        default=Path("results/serving/phaseF3_canary_fork.json"),
        help="Path to write the per-rank snapshot.",
    )
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from evaluation.serving.canary_tp import (
        TPCanaryHandle,
    )

    print("=" * 72)
    print(f"Phase F.3.0: canary against fork at {args.cache_dtype!r} (tp={args.tp})")
    print("=" * 72)

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        kv_cache_dtype=args.cache_dtype,
        block_size=16,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        enable_prefix_caching=False,
        max_num_seqs=4,
    )

    # Install the canary in observation-only mode. The fork's KIVI cache
    # already applies the bit-width-correct quantization at write time;
    # tier_application would shadow-double-quantize and is therefore off.
    handle = TPCanaryHandle(
        llm,
        harness_kwargs={
            "canary_rate": args.canary_rate,
            "tau": args.tau,
            "tick_interval": args.tick_interval,
            "tier_application": False,   # observation-only on the fork
            "force_bits": None,
            "freeze_controller": False,
            "decode_only_canary": True,
            "use_canary_stream": True,
            "seed": 0,
        },
    )
    handle.install()
    try:
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
        outputs = llm.generate(WARMUP_PROMPTS, sampling_params=sp)
    finally:
        # Snapshot before remove (snapshot reads from harness state).
        snapshots = handle.snapshot()
        handle.remove()

    print("\nWarmup outputs:")
    for i, o in enumerate(outputs):
        prompt = o.prompt[:60] + ("..." if len(o.prompt) > 60 else "")
        text = o.outputs[0].text.replace("\n", "\\n")
        print(f"  [{i}] {prompt!r} → {text[:80]!r}")

    print("\nPer-rank canary snapshot:")
    summary = []
    for rank_idx, snap in enumerate(snapshots):
        if not snap:
            print(f"  rank {rank_idx}: empty snapshot")
            continue
        tier = snap.get("tier_vector", [])
        deltas = snap.get("per_layer_delta", [])
        stats = snap.get("stats", {})
        nonzero_deltas = [d for d in deltas if d and d != 0.0]
        tier_counts = {b: tier.count(b) for b in sorted(set(tier))}
        print(
            f"  rank {rank_idx}: "
            f"layers={snap.get('num_layers')} "
            f"tier_counts={tier_counts} "
            f"canary_fires={stats.get('canary_fires', 0)} "
            f"attention_calls={stats.get('attention_calls', 0)} "
            f"non-zero per-layer δ̂={len(nonzero_deltas)}/{len(deltas)} "
            f"max δ̂={max(deltas) if deltas else 0:.4e}"
        )
        summary.append({
            "rank": rank_idx,
            "tier_vector": tier,
            "tier_counts": tier_counts,
            "stats": stats,
            "max_per_layer_delta": (
                max(deltas) if deltas else 0.0
            ),
            "num_nonzero_deltas": len(nonzero_deltas),
            "num_layers": snap.get("num_layers"),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": {
            "cache_dtype": args.cache_dtype,
            "tp": args.tp,
            "canary_rate": args.canary_rate,
            "tau": args.tau,
            "tick_interval": args.tick_interval,
            "max_tokens": args.max_tokens,
            "model": MODEL,
        },
        "per_rank": summary,
        "warmup_outputs": [
            {"prompt": o.prompt, "output": o.outputs[0].text}
            for o in outputs
        ],
    }, indent=2))
    print(f"\nWrote snapshot to {args.out}")
    print("\nDONE")


if __name__ == "__main__":
    main()
