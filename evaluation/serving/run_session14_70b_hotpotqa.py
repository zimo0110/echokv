"""Session 14 / Phase 2.5 — 70B HotpotQA diagnostic.

After 70B + RULER@128K showed FP16 = INT4 = 1.000, the critic's
decision tree calls for one more workload at 70B before declaring
a genuine null result: HotpotQA. The hypothesis is that
multi-step large-integer reasoning is structurally different from
retrieval — the model has to track 7-9 digit intermediate values
across 10-20 reasoning steps, which is the regime where any
KV-quantization noise on intermediate states would compound.

This is the same workload as Phase 1's 8B HotpotQA run, but on
the 70B model where representations are stronger and more
KV-dependent.

Usage
-----
::

    VLLM_ALLOW_INSECURE_SERIALIZATION=1 \\
    FLASHINFER_DISABLE_VERSION_CHECK=1 \\
    CUDA_VISIBLE_DEVICES=0,1,2,3 \\
    /opt/conda/envs/scalable/bin/python \\
        -m evaluation.serving.run_session14_70b_hotpotqa \\
        --n-prompts 20 --configs fp16 static_int4
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
os.environ["PYTHONPATH"] = (
    f"{_REPO_ROOT}:" + os.environ.get("PYTHONPATH", "")
)

from evaluation.motivation.workloads import _hotpotqa_stream, Request
from evaluation.serving.run_session8 import RunResult, run_one, warmup
from evaluation.serving.run_session14_gsm8k_hard import materialize_chat_prompts
from evaluation.serving.static_int4_tp import install_static_int4_hooks
from evaluation.serving.static_int8_tp import install_static_int8_hooks
from evaluation.serving.nvfp4_tp import install_nvfp4_hooks


log = logging.getLogger("session14_70b_hotpotqa")
DEFAULT_70B = "/mnt/nvme_tmp6/llama-70b"


def _save_summary(out: Path, summary: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    log.info("wrote %s", out)


def install_static_int4_via_apply_model(llm: Any) -> list[int]:
    counts = llm.apply_model(install_static_int4_hooks)
    log.info("static_int4: installed hooks per rank: %s", counts)
    return counts


def install_static_int8_via_apply_model(llm: Any) -> list[int]:
    counts = llm.apply_model(install_static_int8_hooks)
    log.info("static_int8: installed hooks per rank: %s", counts)
    return counts


def install_nvfp4_via_apply_model(llm: Any) -> list[int]:
    counts = llm.apply_model(install_nvfp4_hooks)
    log.info("nvfp4: installed hooks per rank: %s", counts)
    return counts


def run_phase(
    llm: Any,
    *,
    n_prompts: int,
    seed: int,
    max_new_tokens: int,
    configs: list[str],
    out_path: Path,
    summary_base: dict,
) -> list[dict]:
    raw = list(_hotpotqa_stream(n=n_prompts, seed=seed))
    prompts = materialize_chat_prompts(llm, raw)
    cells: list[dict] = []

    int4_installed = False
    int8_installed = False
    nvfp4_installed = False
    for cfg in configs:
        if cfg == "fp16":
            log.info("=== fp16  n=%d ===", n_prompts)
            res: RunResult = run_one(
                llm, prompts,
                label=f"fp16_70b_hotpotqa",
                max_new_tokens=max_new_tokens,
                harness=None,
            )
            cells.append({**asdict(res), "config": "fp16"})
            log.info("    fp16: acc=%.3f wall=%.1fs",
                     res.accuracy, res.wall_clock_s)
        elif cfg == "static_int4":
            log.info("=== static_int4  n=%d ===", n_prompts)
            if not int4_installed:
                install_static_int4_via_apply_model(llm)
                int4_installed = True
            res = run_one(
                llm, prompts,
                label=f"static_int4_70b_hotpotqa",
                max_new_tokens=max_new_tokens,
                harness=None,
            )
            cells.append({**asdict(res), "config": "static_int4"})
            log.info("    static_int4: acc=%.3f wall=%.1fs",
                     res.accuracy, res.wall_clock_s)
        elif cfg == "static_int8":
            log.info("=== static_int8  n=%d ===", n_prompts)
            if not int8_installed:
                install_static_int8_via_apply_model(llm)
                int8_installed = True
            res = run_one(
                llm, prompts,
                label=f"static_int8_70b_hotpotqa",
                max_new_tokens=max_new_tokens,
                harness=None,
            )
            cells.append({**asdict(res), "config": "static_int8"})
            log.info("    static_int8: acc=%.3f wall=%.1fs",
                     res.accuracy, res.wall_clock_s)
        elif cfg == "nvfp4":
            log.info("=== nvfp4  n=%d ===", n_prompts)
            if not nvfp4_installed:
                install_nvfp4_via_apply_model(llm)
                nvfp4_installed = True
            res = run_one(
                llm, prompts,
                label=f"nvfp4_70b_hotpotqa",
                max_new_tokens=max_new_tokens,
                harness=None,
            )
            cells.append({**asdict(res), "config": "nvfp4"})
            log.info("    nvfp4: acc=%.3f wall=%.1fs",
                     res.accuracy, res.wall_clock_s)
        else:
            raise ValueError(f"unknown config: {cfg}")
        _save_summary(
            out_path,
            {**summary_base, "cells": cells,
             "partial": cfg != configs[-1]},
        )
    return cells


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_70B)
    p.add_argument("--tensor-parallel-size", type=int, default=4)
    p.add_argument("--max-model-len", type=int, default=4_096)
    p.add_argument(
        "--max-new-tokens", type=int, default=64,
        help="HotpotQA CoT traces — 512 covers a 15-step chain",
    )
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-num-seqs", type=int, default=8)
    p.add_argument("--n-prompts", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--configs", nargs="+",
        default=["fp16", "static_int4"],
        choices=["fp16", "static_int4", "static_int8", "nvfp4"],
        help="order matters; quantization hooks are sticky once installed. "
             "Note: cannot mix int4/int8/nvfp4 hooks in the same process; "
             "run them in separate invocations.",
    )
    p.add_argument(
        "--out", type=Path,
        default=Path("results/serving/session14_70b/hotpotqa.json"),
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

    raw_warmup = list(_hotpotqa_stream(n=4, seed=a.seed + 1000))
    warmup_prompts = materialize_chat_prompts(llm, raw_warmup)
    warmup(llm, warmup_prompts, a.max_new_tokens)

    summary_base = {
        "session": "20260428_14_70b_hotpotqa",
        "model": a.model,
        "tensor_parallel_size": a.tensor_parallel_size,
        "max_model_len": a.max_model_len,
        "max_new_tokens": a.max_new_tokens,
        "n_prompts": a.n_prompts,
        "max_num_seqs": a.max_num_seqs,
        "configs": a.configs,
    }
    cells = run_phase(
        llm,
        n_prompts=a.n_prompts,
        seed=a.seed,
        max_new_tokens=a.max_new_tokens,
        configs=a.configs,
        out_path=a.out,
        summary_base=summary_base,
    )
    _save_summary(a.out, {**summary_base, "cells": cells})

    print()
    print("=" * 72)
    print(f"Session 14 / 70B HotpotQA ({a.n_prompts} prompts, "
          f"max_new_tokens={a.max_new_tokens})")
    print("=" * 72)
    print(f"{'config':>16}  {'accuracy':>9}  {'wall (s)':>9}")
    for cell in cells:
        print(f"{cell['config']:>16}  {cell['accuracy']:>9.3f}  "
              f"{cell['wall_clock_s']:>9.1f}")
    print("=" * 72)
    fp = next((c for c in cells if c["config"] == "fp16"), None)
    i4 = next((c for c in cells if c["config"] == "static_int4"), None)
    if fp and i4:
        d = i4["accuracy"] - fp["accuracy"]
        if d < -0.05:
            print(f"VERDICT: INT4 degrades by {-d:.3f} vs FP16 — "
                  "candidate workload found.")
        elif d < -0.01:
            print(f"VERDICT: INT4 lower than FP16 by {-d:.3f} but "
                  f"within noise at n={a.n_prompts}.")
        else:
            print(f"VERDICT: INT4 ≈ FP16 (Δ={d:+.3f}). 70B HotpotQA "
                  "does not break INT4. Reframe paper.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
