"""Session 14 / Phase 2 — 70B RULER@128K diagnostic.

Builds on the critic's prioritized plan: after 8B failed to break INT4
on every workload tested (single-needle RULER, multi-needle RULER,
GSM8K-Hard), the next step is Llama-3.1-70B at long context. The 70B
model's KV cache is 8× larger per-request and its representations are
both stronger and more KV-dependent — exactly the regime where
group-32 INT4 noise is most likely to compound.

Phases
------
``--phase fp16_confirm``
    Run ``--n-prompts`` RULER@128K prompts under FP16 only. Cheap
    confirmation that 70B + TP loads, that the long-context path
    survives without OOM, and that accuracy is sane. No harness, no
    INT4. Use this as the gating step before building the TP-aware
    INT4 hook.

``--phase fp16_int4``
    Full diagnostic: ``--n-prompts`` prompts under FP16 + static_int4.
    Static INT4 is implemented via a per-worker forward-pre-hook
    installed through ``llm.apply_model`` so it fires on every TP
    rank. (See ``_install_static_int4_hooks`` below.)

Usage
-----
::

    FLASHINFER_DISABLE_VERSION_CHECK=1 \\
    CUDA_VISIBLE_DEVICES=0,1,2,3 \\
    /opt/conda/envs/scalable/bin/python \\
        -m evaluation.serving.run_session14_70b_ruler \\
        --phase fp16_confirm --n-prompts 10

Notes on TP and the canary harness
----------------------------------
The existing ``CanaryHarness`` registers forward-pre-hooks on the
driver-rank-0 model (``llm_engine.model_executor.driver_worker.worker.model_runner.model``).
With TP > 1 and the ``mp`` distributed executor, that path only sees
rank 0's slice of the layers — the other ranks would still cache
FP16. So for TP runs the static-INT4 path uses ``llm.apply_model``
to install the same KIVI-style group-32 quantization hook on every
worker's local model. This is intentionally a **standalone** hook,
not a port of the full canary harness; we don't need observability
for the diagnostic, we need uniform-INT4 behaviour across all ranks.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# Ensure subprocess workers spawned by vLLM can import our modules.
os.environ["PYTHONPATH"] = (
    f"{_REPO_ROOT}:" + os.environ.get("PYTHONPATH", "")
)

from evaluation.serving.run_session8 import (
    RunResult,
    make_ruler_prompts,
    run_one,
    warmup,
)
from evaluation.serving.static_int4_tp import install_static_int4_hooks


log = logging.getLogger("session14_70b")
DEFAULT_70B = "/mnt/nvme_tmp6/llama-70b"


# ---------------------------------------------------------------------------
# TP-aware static-INT4 hook installer
# ---------------------------------------------------------------------------
#
# The actual hook function lives in ``static_int4_tp.py`` so it is a
# normal module-level reference (not ``__main__.foo``); pickle only
# resolves the worker-side function correctly when the source module is
# importable, not when the function was defined under ``__main__``.


def install_static_int4_via_apply_model(llm: Any) -> list[int]:
    """Install the per-worker INT4 hook. Returns per-worker hook
    counts (one int per TP rank); they should all match."""
    counts = llm.apply_model(install_static_int4_hooks)
    log.info("static_int4: installed hooks per rank: %s", counts)
    return counts


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _save_summary(out: Path, summary: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    log.info("wrote %s", out)


def run_phase(
    llm: Any,
    *,
    phase: str,
    n_prompts: int,
    context: int,
    seed: int,
    max_new_tokens: int,
    out_path: Path,
    summary_base: dict,
) -> list[dict]:
    prompts = make_ruler_prompts(n=n_prompts, seed=seed, target_tokens=context)
    cells: list[dict] = []

    # FP16 cell — no harness, no hooks.
    log.info("=== fp16  n=%d  context=%d ===", n_prompts, context)
    res: RunResult = run_one(
        llm, prompts,
        label=f"fp16_70b_{context}",
        max_new_tokens=max_new_tokens,
        harness=None,
    )
    cells.append({**asdict(res), "config": "fp16", "context": context})
    log.info("    fp16: acc=%.3f wall=%.1fs", res.accuracy, res.wall_clock_s)
    # Persist incrementally so an INT4 crash doesn't lose the FP16 result.
    _save_summary(out_path, {**summary_base, "cells": cells, "partial": True})

    if phase == "fp16_confirm":
        return cells

    # static_int4 cell — install per-rank hooks via apply_model.
    log.info("=== static_int4  n=%d  context=%d ===", n_prompts, context)
    install_static_int4_via_apply_model(llm)
    res4: RunResult = run_one(
        llm, prompts,
        label=f"static_int4_70b_{context}",
        max_new_tokens=max_new_tokens,
        harness=None,
    )
    cells.append(
        {**asdict(res4), "config": "static_int4", "context": context}
    )
    log.info("    static_int4: acc=%.3f wall=%.1fs",
             res4.accuracy, res4.wall_clock_s)
    return cells


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_70B)
    p.add_argument("--tensor-parallel-size", type=int, default=4)
    p.add_argument("--max-model-len", type=int, default=131_072)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-num-seqs", type=int, default=1)
    p.add_argument("--n-prompts", type=int, default=10)
    p.add_argument("--context", type=int, default=128_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--phase",
        choices=["fp16_confirm", "fp16_int4"],
        default="fp16_confirm",
    )
    p.add_argument(
        "--out", type=Path,
        default=Path("results/serving/session14_70b/diag.json"),
    )
    a = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    from vllm import LLM

    log.info(
        "loading %s tp=%d max_model_len=%d",
        a.model, a.tensor_parallel_size, a.max_model_len,
    )
    t0 = time.perf_counter()
    llm = LLM(
        model=a.model,
        tensor_parallel_size=a.tensor_parallel_size,
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=a.gpu_memory_utilization,
        max_model_len=a.max_model_len,
        enable_prefix_caching=False,
        max_num_seqs=a.max_num_seqs,
    )
    log.info("loaded in %.1fs", time.perf_counter() - t0)

    # Tiny warmup at the target context so we pay compile cost once.
    warmup_prompts = make_ruler_prompts(
        n=2, seed=a.seed, target_tokens=a.context,
    )
    warmup(llm, warmup_prompts, a.max_new_tokens)

    summary_base = {
        "session": "20260428_14_70b",
        "model": a.model,
        "tensor_parallel_size": a.tensor_parallel_size,
        "phase": a.phase,
        "context": a.context,
        "n_prompts": a.n_prompts,
        "max_num_seqs": a.max_num_seqs,
    }
    cells = run_phase(
        llm,
        phase=a.phase,
        n_prompts=a.n_prompts,
        context=a.context,
        seed=a.seed,
        max_new_tokens=a.max_new_tokens,
        out_path=a.out,
        summary_base=summary_base,
    )
    _save_summary(a.out, {**summary_base, "cells": cells})

    print()
    print("=" * 72)
    print(f"Session 14 / 70B RULER@{a.context} (phase={a.phase})")
    print("=" * 72)
    print(f"{'config':>16}  {'accuracy':>9}  {'wall (s)':>9}")
    for cell in cells:
        print(f"{cell['config']:>16}  {cell['accuracy']:>9.3f}  "
              f"{cell['wall_clock_s']:>9.1f}")
    print("=" * 72)
    if a.phase == "fp16_int4":
        fp = next((c for c in cells if c["config"] == "fp16"), None)
        i4 = next((c for c in cells if c["config"] == "static_int4"), None)
        if fp and i4:
            d = i4["accuracy"] - fp["accuracy"]
            if d < -0.05:
                print(f"VERDICT: INT4 degrades by {-d:.3f} vs FP16 — "
                      "candidate workload found. Run full sweep.")
            elif d < -0.01:
                print(f"VERDICT: INT4 lower than FP16 by {-d:.3f} but "
                      f"within noise at n={a.n_prompts}. Consider larger n.")
            else:
                print(f"VERDICT: INT4 ≈ FP16 (Δ={d:+.3f}). 70B RULER@128K "
                      "does not break INT4 on this hardware/workload.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
