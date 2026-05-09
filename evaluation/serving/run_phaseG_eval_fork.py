"""Phase G (Lane 1(a)): RULER n=10 on the EchoKV vLLM fork — accuracy
parity check between FP16 and INT4_kivi caches.

Validates that the fork's *real* KIVI INT4 KV cache produces the same
accuracy on a real eval as the host's existing shadow-mode INT4 path
(which the paper's prior numbers were generated against). If the
fork's INT4 matches the upstream FP16 baseline (within RULER's per-task
noise), the fork is ready to host the paper's mixed-precision Pareto
runs.

Two-engine pattern, subprocess-isolated (each engine gets a clean
GPU). For each cache_dtype the subprocess:
  1. Boots Llama-3.1-8B at the requested dtype.
  2. Generates ``n_prompts`` RULER samples at the requested context.
  3. Scores each via the workload's needle-recall ``score_fn``.
  4. Emits a per-prompt + summary record on a sentinel JSON line.

Usage
-----
::

    CUDA_VISIBLE_DEVICES=4 \\
    VLLM_ALLOW_INSECURE_SERIALIZATION=1 \\
    FLASHINFER_DISABLE_VERSION_CHECK=1 \\
    python evaluation/serving/run_phaseG_eval_fork.py \\
        --n-prompts 10 --context 4000

The default context (4 000 tokens) matches the EchoKV motivation
study's RULER@4K. Production paper runs use 8K-32K; we use 4K for the
parity check because (a) it fits comfortably in a single H100 80GB
without OOM at FP16, (b) it's enough to exercise multi-block attention
in the cache, and (c) any kernel correctness issue surfaces here just
as well as at longer context.
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


SUBPROCESS_SCRIPT = '''
import json, sys, time
sys.path.insert(0, "/mnt/nvme_tmp10/nazar_vllm_fork")
sys.path.insert(0, "{repo}")

from vllm import LLM, SamplingParams
from evaluation.motivation.workloads import _ruler_sample, _hotpotqa_stream

WORKLOAD = "{workload}"
if WORKLOAD == "ruler":
    prompts = [
        _ruler_sample(i, seed={seed}, target_tokens={target_tokens})
        for i in range({n})
    ]
elif WORKLOAD == "hotpotqa":
    prompts = list(_hotpotqa_stream(n={n}, seed={seed}, level="hard"))
else:
    raise ValueError("unknown workload: " + WORKLOAD)

llm_kwargs = dict(
    model="{model}",
    tensor_parallel_size={tp},
    dtype="bfloat16",
    kv_cache_dtype="{cache_dtype}",
    block_size=16,
    enforce_eager=True,
    gpu_memory_utilization=0.85,
    max_model_len={max_model_len},
    enable_prefix_caching=False,
    max_num_seqs=4,
)

t0 = time.perf_counter()
llm = LLM(**llm_kwargs)
load_s = time.perf_counter() - t0

sp = SamplingParams(temperature=0.0, max_tokens={max_tokens})
t1 = time.perf_counter()
# RULER prompts use raw text; HotpotQA uses chat messages.
if WORKLOAD == "ruler":
    inputs = [r.prompt for r in prompts]
    outputs = llm.generate(inputs, sampling_params=sp, use_tqdm=False)
else:
    inputs = [r.messages for r in prompts]
    outputs = llm.chat(inputs, sampling_params=sp, use_tqdm=False)
gen_s = time.perf_counter() - t1

per_prompt = []
n_correct = 0.0
total_in = 0
total_out = 0
for r, o in zip(prompts, outputs):
    completion = o.outputs[0].text
    correct = float(r.score_fn(completion))
    n_correct += correct
    total_in += len(o.prompt_token_ids)
    total_out += len(o.outputs[0].token_ids)
    per_prompt.append({{
        "request_id": r.request_id,
        "reference": r.reference,
        "prompt_tokens": len(o.prompt_token_ids),
        "completion_tokens": len(o.outputs[0].token_ids),
        "completion": completion,
        "correct": correct,
    }})

record = {{
    "workload": WORKLOAD,
    "cache_dtype": "{cache_dtype}",
    "n_prompts": len(prompts),
    "target_tokens": {target_tokens},
    "n_correct": n_correct,
    "accuracy": n_correct / max(1, len(prompts)),
    "load_s": load_s,
    "gen_s": gen_s,
    "total_in_tokens": total_in,
    "total_out_tokens": total_out,
    "tps_in": total_in / gen_s if gen_s > 0 else 0.0,
    "tps_out": total_out / gen_s if gen_s > 0 else 0.0,
    "per_prompt": per_prompt,
}}
print("__PHASEG_JSON__" + json.dumps(record))
'''


def _run_one(cache_dtype: str, args) -> dict:
    sentinel = "__PHASEG_JSON__"
    env = {
        **os.environ,
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "FLASHINFER_DISABLE_VERSION_CHECK": "1",
    }
    print(f"\n[{cache_dtype}] launching subprocess ...")
    proc = subprocess.run(
        [sys.executable, "-c", SUBPROCESS_SCRIPT.format(
            repo=_REPO, model=MODEL, tp=args.tp,
            workload=args.workload,
            cache_dtype=cache_dtype, n=args.n_prompts, seed=args.seed,
            target_tokens=args.context,
            max_model_len=args.max_model_len,
            max_tokens=args.max_tokens,
        )],
        env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"{cache_dtype} subprocess failed (rc={proc.returncode})")
    line = next((l for l in proc.stdout.splitlines() if l.startswith(sentinel)), None)
    if line is None:
        sys.stdout.write(proc.stdout)
        raise RuntimeError(f"{cache_dtype}: no sentinel line in stdout")
    return json.loads(line[len(sentinel):])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument(
        "--workload", default="ruler", choices=["ruler", "hotpotqa"],
        help="ruler = synthetic needle-in-haystack at --context tokens; "
             "hotpotqa = real multi-hop QA from the HotpotQA distractor "
             "split (level=hard).",
    )
    ap.add_argument("--n-prompts", type=int, default=10)
    ap.add_argument("--context", type=int, default=4_000,
                    help="RULER target_tokens for the haystack body "
                         "(ignored for hotpotqa).")
    ap.add_argument("--max-model-len", type=int, default=8_192)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out", type=Path,
        default=Path("results/serving/phaseG_eval_fork.json"),
    )
    args = ap.parse_args()

    print("=" * 72)
    print(f"Phase G eval fork: {args.workload} n={args.n_prompts} "
          f"target={args.context}, max_model_len={args.max_model_len}")
    print(f"  model: {MODEL.split('/')[-1]}, tp={args.tp}")
    print("=" * 72)

    fp16 = _run_one("auto", args)
    int4 = _run_one("int4_kivi", args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": {
            "n_prompts": args.n_prompts,
            "target_tokens": args.context,
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            "tp": args.tp,
            "seed": args.seed,
            "model": MODEL,
        },
        "fp16": fp16,
        "int4_kivi": int4,
    }, indent=2))

    print()
    print("=" * 72)
    print(f"{'config':>14}  {'accuracy':>9}  {'gen_s':>7}  {'tps_in':>8}  {'tps_out':>8}")
    print("=" * 72)
    for label, rec in (("fp16", fp16), ("int4_kivi", int4)):
        print(f"{label:>14}  {rec['accuracy']:>9.3f}  "
              f"{rec['gen_s']:>7.1f}  {rec['tps_in']:>8.1f}  "
              f"{rec['tps_out']:>8.1f}")
    print("=" * 72)

    # Verdict
    delta = int4["accuracy"] - fp16["accuracy"]
    if abs(delta) <= 0.05:
        print(f"VERDICT: INT4_kivi matches FP16 within RULER noise (Δ={delta:+.3f})")
    elif delta < 0:
        print(f"VERDICT: INT4_kivi degrades by {-delta:.3f} vs FP16 — "
              "investigate before larger n.")
    else:
        print(f"VERDICT: INT4_kivi nominally higher than FP16 by {delta:.3f} — "
              "small-sample noise; rerun with larger n.")
    print(f"\nWrote record to {args.out}")


if __name__ == "__main__":
    main()
