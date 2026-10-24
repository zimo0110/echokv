"""End-to-end check of the EchoKV vLLM plugin (CPU or CUDA).

Boots vLLM with the EchoKV backend, generates, and reports the live tier
vector, stored-bit mix and per-layer canary EMAs. With ``--fp-baseline`` it
also runs stock vLLM on the same prompts and reports the mean absolute
prompt-logprob difference.

    ECHOKV_ENABLE=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
      python evaluation/serving/echokv_plugin_smoke.py --tau0 0.6 --interval 8

Settings are passed to the backend through ECHOKV_* environment variables
(see echokv_vllm/config.py); the flags below just set them. On macOS, TP > 1
needs VLLM_CPU_OMP_THREADS_BIND, e.g. "0-4|5-9".
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

PROMPTS = [
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain step by step how to compute the derivative of x^3 * sin(x).",
    "List ten facts about the planet Mars.",
    "Describe the water cycle to a ten year old.",
]


def _engine_kwargs(args) -> dict:
    kw = dict(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        tensor_parallel_size=args.tp,
        kv_cache_memory_bytes=args.kv_cache_mib * 2**20,
        enable_prefix_caching=False,
    )
    import torch

    if torch.cuda.is_available() and args.mode == "echokv":
        kw["attention_config"] = {"backend": "CUSTOM"}
    return kw


def run_child(args) -> None:
    from vllm import LLM, SamplingParams

    llm = LLM(**_engine_kwargs(args))
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, prompt_logprobs=0)
    t = time.time()
    outs = llm.generate(PROMPTS, sp)
    result = {
        "elapsed_s": time.time() - t,
        "texts": [o.outputs[0].text for o in outs],
        "prompt_logprobs": [
            [None if d is None else next(iter(d.values())).logprob for d in o.prompt_logprobs]
            for o in outs
        ],
    }
    if args.mode == "echokv":
        from echokv_vllm import control

        states = llm.apply_model(control.get_state)
        result["state"] = states[0]
        result["ranks_agree"] = all(s["tier_vector"] == states[0]["tier_vector"] for s in states)
    with open(args.child_out, "w") as f:
        json.dump(result, f)


def spawn(args, mode: str, out: str) -> dict:
    env = dict(os.environ, VLLM_ALLOW_INSECURE_SERIALIZATION="1")
    if mode == "echokv":
        env["ECHOKV_ENABLE"] = "1"
        for flag, var in (
            ("tau0", "ECHOKV_TAU0"),
            ("interval", "ECHOKV_CANARY_INTERVAL"),
            ("mem_bits", "ECHOKV_MEM_BITS"),
            ("init_bits", "ECHOKV_INIT_BITS"),
            ("fixed_bits", "ECHOKV_FIXED_BITS"),
        ):
            v = getattr(args, flag)
            if v is not None:
                env[var] = str(v)
    else:
        env["ECHOKV_ENABLE"] = "0"
    cmd = [sys.executable, __file__, "--child", mode, "--child-out", out] + args.passthrough
    subprocess.run(cmd, env=env, check=True)
    with open(out) as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--kv-cache-mib", type=int, default=256)
    p.add_argument("--tau0", type=float)
    p.add_argument("--interval", type=int)
    p.add_argument("--mem-bits", type=int)
    p.add_argument("--init-bits", type=int)
    p.add_argument("--fixed-bits", type=int)
    p.add_argument("--fp-baseline", action="store_true")
    p.add_argument("--out", default="echokv_plugin_smoke.json")
    p.add_argument("--child", choices=["echokv", "stock"])
    p.add_argument("--child-out")
    args = p.parse_args()

    if args.child:
        args.mode = args.child
        run_child(args)
        return

    engine_flags = ["--model", args.model, "--dtype", args.dtype, "--max-model-len",
                    str(args.max_model_len), "--max-tokens", str(args.max_tokens), "--tp",
                    str(args.tp), "--kv-cache-mib", str(args.kv_cache_mib)]
    args.passthrough = engine_flags
    report = {"echokv": spawn(args, "echokv", args.out + ".echokv.tmp")}
    if args.fp_baseline:
        report["stock"] = spawn(args, "stock", args.out + ".stock.tmp")
        diffs = [
            abs(a - b)
            for pe, ps in zip(report["echokv"]["prompt_logprobs"], report["stock"]["prompt_logprobs"])
            for a, b in zip(pe, ps)
            if a is not None and b is not None
        ]
        report["mean_abs_prompt_logprob_diff"] = sum(diffs) / len(diffs)
    for suffix in (".echokv.tmp", ".stock.tmp"):
        if os.path.exists(args.out + suffix):
            os.remove(args.out + suffix)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)

    e = report["echokv"]
    st = e["state"]
    print("tier vector:", st["tier_vector"], "| ranks agree:", e["ranks_agree"])
    print("avg stored bits: %.2f" % st["avg_stored_bits"], st["tokens_written"])
    mixed = [l["layer"] for l in st["layers"] if sum(1 for v in l["blocks_opened"].values() if v) > 1]
    print("layers holding blocks at more than one tier:", len(mixed))
    if "mean_abs_prompt_logprob_diff" in report:
        print("mean |Δ prompt logprob| vs stock: %.4f" % report["mean_abs_prompt_logprob_diff"])
    for text in e["texts"]:
        print(" ", repr(text[:100]))


if __name__ == "__main__":
    main()
