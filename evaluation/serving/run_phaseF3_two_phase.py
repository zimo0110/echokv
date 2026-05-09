"""Phase F.3.0: warmup-then-freeze demo on the vLLM fork.

Two-step driver:

  Phase 1 — warmup (subprocess): boot fork at uniform bits, install
            CanaryHarness in observation-only mode, run warmup prompts,
            emit converged tier vector to JSON.
  Phase 2 — measurement (subprocess): boot fresh fork engine with
            ``LLM(kivi_bits_per_layer=<warmup tier>)`` so each layer's
            cache is allocated at the controller-chosen bits from the
            first token; run measurement prompts.

Why two subprocesses
--------------------
Running both phases in one Python process leaves GPU memory dirty
between the LLM teardown and the second LLM init (NCCL groups, vLLM's
own pool, plus our per-layer scale/zp side buffers from the first
engine all linger past ``del llm``). With a heterogeneous tier vector
in phase 2, the side-buffer cost is multi-GiB and the second engine
OOMs. Subprocess isolation gives each engine a clean slate.

Why observation-only on the fork (phase 1)
------------------------------------------
``CanaryHarness.tier_application=True`` was the upstream-vLLM
workaround for "controller decisions actually affect outputs without a
source fork". With the fork, the cache IS quantised at the controller's
chosen bits at write time; shadow-quantising K/V before attention would
double-apply the noise.

What this driver does NOT do
----------------------------
Tier vectors are pinned at engine-init time (Phase F.2's
``kivi_bits_per_layer``). True per-layer dynamic tier swapping inside
a *single* engine still requires Phase F.3.1 (allocate-at-max-bits
+ tolerant slice, or per-tier dedicated buffers).

Usage
-----
::

    CUDA_VISIBLE_DEVICES=4 \\
    VLLM_ALLOW_INSECURE_SERIALIZATION=1 \\
    FLASHINFER_DISABLE_VERSION_CHECK=1 \\
    python evaluation/serving/run_phaseF3_two_phase.py --tp 1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_FORK = "/mnt/nvme_tmp10/nazar_vllm_fork"
_REPO = str(Path(__file__).resolve().parents[2])
if _FORK not in sys.path:
    sys.path.insert(0, _FORK)
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

MEASUREMENT_PROMPTS = [
    "Q: What is 12 + 15?  A:",
    "Q: What is 100 - 37?  A:",
    "The capital of Japan is",
    "Who wrote Romeo and Juliet?",
    "Name the largest ocean on Earth:",
    "List three states of matter:",
]


PHASE1_SCRIPT = '''
import json, sys
sys.path.insert(0, "/mnt/nvme_tmp10/nazar_vllm_fork")
sys.path.insert(0, "{repo}")

PROMPTS = {prompts!r}

from vllm import LLM, SamplingParams
from evaluation.serving.canary_tp import TPCanaryHandle

llm = LLM(
    model="{model}",
    tensor_parallel_size={tp},
    dtype="bfloat16",
    kv_cache_dtype="{cache_dtype}",
    block_size=16,
    enforce_eager=True,
    gpu_memory_utilization=0.85,
    max_model_len=4096,
    enable_prefix_caching=False,
    max_num_seqs=4,
)
from core.block_allocator import Precision
handle = TPCanaryHandle(
    llm,
    harness_kwargs={{
        "canary_rate": {canary_rate},
        "tau": {tau},
        "tick_interval": {tick_interval},
        "tier_application": False,
        "decode_only_canary": True,
        "use_canary_stream": True,
        "seed": 0,
        # Pin initial level to INT4 so a short warmup converges to a
        # balanced (often uniform-INT4) tier vector, sidestepping the
        # Phase F.2 caveat that an INT2-heavy heterogeneous mix
        # explodes the per-layer scale/zp side-buffer total.
        "initial_level": Precision.INT4,
    }},
)
handle.install()
try:
    sp = SamplingParams(temperature=0.0, max_tokens={max_tokens})
    outputs = llm.generate(PROMPTS, sampling_params=sp)
finally:
    snapshots = handle.snapshot()
    handle.remove()

# Emit machine-readable record on a unique sentinel line.
record = {{
    "tier_vector": snapshots[0]["tier_vector"],
    "stats": snapshots[0]["stats"],
    "num_layers": snapshots[0]["num_layers"],
    "outputs": [{{"prompt": o.prompt, "output": o.outputs[0].text}} for o in outputs],
}}
print("__PHASE1_JSON__" + json.dumps(record))
'''

PHASE2_SCRIPT = '''
import json, sys
sys.path.insert(0, "/mnt/nvme_tmp10/nazar_vllm_fork")
sys.path.insert(0, "{repo}")

PROMPTS = {prompts!r}
TIER = {tier!r}

# Build the per-layer dict; engine cache_dtype must be at least the max
# bit-width in the tier vector so each spec's data buffer fits.
unique = sorted(set(TIER))
engine_dtype = "int{{}}_kivi".format(max(unique))
# Phase F.2 caveat: vLLM's profiler doesn't subtract per-layer scale/zp
# side buffer cost from the cache budget. For Llama-3.1-8B (32 layers)
# the side buffer total per block is 32 * 3072 = 98 KB while the data
# page per block is 32-128 KB depending on bits. The sweet-spot
# ``gpu_memory_utilization`` is bounded below by model + activation
# memory (about 0.30 of an H100 80GB) and above by available
# memory minus side-buffer overhead. Empirically this lands at 0.30-0.35
# for full Llama-8B; production drivers should wait for the profile-
# accounting follow-up to use 0.85 like the upstream serving harnesses.
gpu_util = 0.30
max_model_len = 1024

bits_per_layer = {{
    "model.layers.{{}}.self_attn.attn".format(i): int(b)
    for i, b in enumerate(TIER)
}}

from vllm import LLM, SamplingParams
from evaluation.serving.fork_tier import read_tier_vector_from_model

llm = LLM(
    model="{model}",
    tensor_parallel_size={tp},
    dtype="bfloat16",
    kv_cache_dtype=engine_dtype,
    kivi_bits_per_layer=bits_per_layer,
    block_size=16,
    enforce_eager=True,
    gpu_memory_utilization=gpu_util,
    max_model_len=max_model_len,
    enable_prefix_caching=False,
    max_num_seqs=4,
)

rank_vectors = llm.apply_model(read_tier_vector_from_model)
match = all(v == TIER for v in rank_vectors)
sp = SamplingParams(temperature=0.0, max_tokens={max_tokens})
outputs = llm.generate(PROMPTS, sampling_params=sp)

record = {{
    "engine_dtype": engine_dtype,
    "gpu_memory_utilization": gpu_util,
    "rank_read_back_matches": match,
    "rank_vectors": rank_vectors,
    "outputs": [{{"prompt": o.prompt, "output": o.outputs[0].text}} for o in outputs],
}}
print("__PHASE2_JSON__" + json.dumps(record))
'''


def _run_subprocess(script: str, *, label: str) -> dict:
    sentinel = f"__{label}_JSON__"
    env = {
        **os.environ,
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "FLASHINFER_DISABLE_VERSION_CHECK": "1",
    }
    print(f"\n[{label}] launching subprocess ...")
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"{label} subprocess failed (rc={proc.returncode})")
    record_line = next(
        (l for l in proc.stdout.splitlines() if l.startswith(sentinel)), None,
    )
    if record_line is None:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"{label}: no sentinel line in stdout")
    return json.loads(record_line[len(sentinel):])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--cache-dtype", default="int4_kivi",
                    choices=["int2_kivi", "int4_kivi", "int8_kivi"])
    ap.add_argument("--canary-rate", type=float, default=1.0 / 8.0)
    ap.add_argument(
        "--tau", type=float, default=1.0,
        help="Controller setpoint. The default ``1.0`` is intentionally "
             "conservative for the demo so the warmup observes signal "
             "without aggressively retiring layers down to INT2 (which "
             "blows up the per-layer scale/zp side-buffer total in phase "
             "2). Production harnesses use 1e-2 or 1e-3.",
    )
    ap.add_argument("--tick-interval", type=int, default=10)
    ap.add_argument("--warmup-max-tokens", type=int, default=32)
    ap.add_argument("--measure-max-tokens", type=int, default=24)
    ap.add_argument(
        "--out", type=Path,
        default=Path("results/serving/phaseF3_two_phase.json"),
    )
    args = ap.parse_args()

    print("=" * 72)
    print(f"Phase F.3.0 two-phase: model=Llama-3.1-8B, tp={args.tp}, "
          f"warmup at {args.cache_dtype!r}")
    print("=" * 72)

    # ---- Phase 1 ----
    p1 = _run_subprocess(
        PHASE1_SCRIPT.format(
            repo=_REPO, model=MODEL, tp=args.tp,
            cache_dtype=args.cache_dtype,
            prompts=WARMUP_PROMPTS,
            canary_rate=args.canary_rate, tau=args.tau,
            tick_interval=args.tick_interval,
            max_tokens=args.warmup_max_tokens,
        ),
        label="PHASE1",
    )
    tier_vector = p1["tier_vector"]
    counts = {b: tier_vector.count(b) for b in sorted(set(tier_vector))}
    print(f"\n[PHASE1] converged tier_vector: counts={counts}, "
          f"canary_fires={p1['stats']['canary_fires']}, "
          f"attention_calls={p1['stats']['attention_calls']}")
    print("[PHASE1] outputs:")
    for i, o in enumerate(p1["outputs"]):
        prompt = o["prompt"][:60] + ("..." if len(o["prompt"]) > 60 else "")
        text = o["output"].replace("\n", "\\n")
        print(f"  [{i}] {prompt!r} → {text[:80]!r}")

    # ---- Phase 2 ----
    p2 = _run_subprocess(
        PHASE2_SCRIPT.format(
            repo=_REPO, model=MODEL, tp=args.tp,
            tier=tier_vector,
            prompts=MEASUREMENT_PROMPTS,
            max_tokens=args.measure_max_tokens,
        ),
        label="PHASE2",
    )
    print(f"\n[PHASE2] engine_dtype={p2['engine_dtype']!r} "
          f"gpu_util={p2['gpu_memory_utilization']} "
          f"rank_read_back_matches={p2['rank_read_back_matches']}")
    print("[PHASE2] outputs:")
    for i, o in enumerate(p2["outputs"]):
        prompt = o["prompt"][:60] + ("..." if len(o["prompt"]) > 60 else "")
        text = o["output"].replace("\n", "\\n")
        print(f"  [{i}] {prompt!r} → {text[:80]!r}")

    # ---- Persist combined record ----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": {
            "cache_dtype_warmup": args.cache_dtype,
            "tp": args.tp,
            "canary_rate": args.canary_rate,
            "tau": args.tau,
            "tick_interval": args.tick_interval,
            "warmup_max_tokens": args.warmup_max_tokens,
            "measure_max_tokens": args.measure_max_tokens,
            "model": MODEL,
        },
        "tier_vector_after_warmup": tier_vector,
        "tier_counts_after_warmup": counts,
        "phase1": p1,
        "phase2": p2,
    }, indent=2))
    print(f"\nWrote combined record to {args.out}")
    print("\nDONE")


if __name__ == "__main__":
    main()
