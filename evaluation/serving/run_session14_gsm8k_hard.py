"""Session 14 / Phase 1 — GSM8K-Hard diagnostic.

The Step-2 hypothesis from the critic's prioritized plan: multi-step
reasoning with precise large-integer arithmetic stresses the KV cache
in a way pure retrieval doesn't, because the model has to carry 7-9
digit intermediate values across 10-20 reasoning steps. If
KIVI-style group-32 INT4 introduces even small per-key noise that
the model latches onto during arithmetic, INT4 should drop below
FP16 even on Llama-3.1-8B at modest context.

This is a 20-prompt, 2-config diagnostic — confirm signal exists
before running the full 100-prompt sweep.

Usage
-----
::

    FLASHINFER_DISABLE_VERSION_CHECK=1 \\
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \\
    CUDA_VISIBLE_DEVICES=0 \\
    /opt/conda/envs/scalable/bin/python \\
        -m evaluation.serving.run_session14_gsm8k_hard

Output
------
``results/serving/session14_gsm8k_hard/diag.json`` — full per-cell
blob, including completion_text per request (load-bearing for
failure-mode analysis since these traces are long and partial-credit
analysis is interesting).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.motivation.workloads import _gsm8k_hard_stream, Request
from evaluation.serving.run_session8 import (
    DEFAULT_MODEL,
    RunResult,
    run_one,
    warmup,
)
from evaluation.serving.run_session11_sweep import make_harness


log = logging.getLogger("session14_gsm8k_hard")


def materialize_chat_prompts(
    llm: Any, requests: list[Request],
) -> list[Request]:
    """Apply the tokenizer's chat template in place.

    GSM8K-Hard requests carry both ``prompt`` (a logging-only label)
    and ``messages`` (the actual chat). ``run_one`` calls
    ``llm.generate([r.prompt for r in prompts])``, so we must replace
    ``r.prompt`` with the templated string before handing requests
    off.
    """
    tok = llm.get_tokenizer()
    out: list[Request] = []
    for r in requests:
        if r.messages is None:
            out.append(r)
            continue
        templated = tok.apply_chat_template(
            r.messages, tokenize=False, add_generation_prompt=True,
        )
        # Build a shallow copy with prompt replaced; keep the rest.
        out.append(
            Request(
                workload=r.workload,
                request_id=r.request_id,
                prompt=templated,
                reference=r.reference,
                score_fn=r.score_fn,
                messages=r.messages,
            )
        )
    return out


def run_cell(
    llm: Any,
    *,
    n_prompts: int,
    max_new_tokens: int,
    config: str,
    seed: int,
    tick_interval: int,
    observability_rate: float,
) -> dict:
    raw = list(_gsm8k_hard_stream(n=n_prompts, seed=seed))
    prompts = materialize_chat_prompts(llm, raw)

    harness = make_harness(
        llm, config,
        seed=seed,
        tick_interval=tick_interval,
        observability_rate=observability_rate,
    )
    label = f"{config}_gsm8k_hard"
    if harness is not None:
        harness.install()
    try:
        result: RunResult = run_one(
            llm, prompts,
            label=label,
            max_new_tokens=max_new_tokens,
            harness=harness,
        )
    finally:
        if harness is not None:
            harness.remove()
    blob = asdict(result)
    blob["config"] = config
    blob["n_prompts"] = n_prompts
    blob["workload"] = "gsm8k_hard"
    return blob


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-model-len", type=int, default=4_096)
    p.add_argument(
        "--max-new-tokens", type=int, default=512,
        help="GSM8K-Hard CoT traces can be long; 512 covers a 15-step chain",
    )
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--n-prompts", type=int, default=20)
    p.add_argument("--tick-interval", type=int, default=10)
    p.add_argument("--observability-rate", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out", type=Path,
        default=Path("results/serving/session14_gsm8k_hard/diag.json"),
    )
    p.add_argument(
        "--configs", nargs="+",
        default=["fp16", "static_int4"],
        choices=["fp16", "static_int4", "static_int2", "echokv_3c"],
    )
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

    # Cheap warmup — pull 4 prompts and run them through.
    raw_warmup = list(_gsm8k_hard_stream(n=4, seed=a.seed + 1000))
    warmup_prompts = materialize_chat_prompts(llm, raw_warmup)
    warmup(llm, warmup_prompts, a.max_new_tokens)

    cells: list[dict] = []
    for config in a.configs:
        log.info("=== config=%s n=%d ===", config, a.n_prompts)
        cell = run_cell(
            llm,
            n_prompts=a.n_prompts,
            max_new_tokens=a.max_new_tokens,
            config=config,
            seed=a.seed,
            tick_interval=a.tick_interval,
            observability_rate=a.observability_rate,
        )
        cells.append(cell)
        log.info("    acc=%.3f wall=%.1fs", cell["accuracy"], cell["wall_clock_s"])

    summary = {
        "session": "20260428_14_gsm8k_hard",
        "model": a.model,
        "max_new_tokens": a.max_new_tokens,
        "n_prompts": a.n_prompts,
        "cells": cells,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=2))
    log.info("wrote %s", a.out)

    print()
    print("=" * 72)
    print(f"Session 14 / GSM8K-Hard diagnostic ({a.n_prompts} prompts)")
    print("=" * 72)
    print(f"{'config':>16}  {'accuracy':>9}  {'wall (s)':>9}")
    for cell in cells:
        print(f"{cell['config']:>16}  {cell['accuracy']:>9.3f}  "
              f"{cell['wall_clock_s']:>9.1f}")
    print("=" * 72)

    fp16 = next((c for c in cells if c["config"] == "fp16"), None)
    int4 = next((c for c in cells if c["config"] == "static_int4"), None)
    if fp16 and int4:
        delta = int4["accuracy"] - fp16["accuracy"]
        if delta < -0.05:
            print(f"VERDICT: INT4 degrades by {-delta:.3f} vs FP16 — "
                  "candidate workload found. Run full sweep.")
        elif delta < -0.01:
            print(f"VERDICT: INT4 lower than FP16 by {-delta:.3f} but within "
                  "noise at n=20. Consider larger n.")
        else:
            print(f"VERDICT: INT4 ≈ FP16 (Δ={delta:+.3f}). 8B GSM8K-Hard does "
                  "not break INT4. Move to 70B.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
