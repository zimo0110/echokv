"""Throughput and KV capacity: stock vLLM vs EchoKV, same memory budget.

Each configuration runs in its own engine (subprocess). Reported:
KV capacity in tokens (from vLLM's allocator), wall time, output tokens/s,
and how many requests vLLM had to preempt for lack of KV space.

    python evaluation/serving/echokv_throughput.py --model meta-llama/Llama-3.1-8B-Instruct \
        --num-prompts 256 --input-len 1024 --output-len 256 --kv-cache-gib 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time

CONFIGS = {
    "stock": {"ECHOKV_ENABLE": "0"},
    "echokv-int8": {"ECHOKV_ENABLE": "1", "ECHOKV_MEM_BITS": "8", "ECHOKV_FIXED_BITS": "8"},
    "echokv-int4": {"ECHOKV_ENABLE": "1", "ECHOKV_MEM_BITS": "4", "ECHOKV_FIXED_BITS": "4"},
    "echokv-live-mem6": {"ECHOKV_ENABLE": "1", "ECHOKV_MEM_BITS": "6"},
    "echokv-live-mem4": {"ECHOKV_ENABLE": "1", "ECHOKV_MEM_BITS": "4"},
}


def child(args) -> None:
    import torch
    from vllm import LLM, SamplingParams

    kw = dict(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.input_len + args.output_len + 64,
        kv_cache_memory_bytes=int(args.kv_cache_gib * 2**30),
        tensor_parallel_size=args.tp,
        block_size=args.block_size,
        enable_prefix_caching=False,
        enforce_eager=args.enforce_eager,
    )
    if torch.cuda.is_available():
        # vLLM refuses to start unless free memory covers this fraction of the
        # device, even with an explicit KV cache size. Its 0.92 default fails on
        # shared/unified-memory machines (DGX Spark) and busy GPUs.
        kw["gpu_memory_utilization"] = float(os.environ.get("ECHOKV_GPU_UTIL", "0.7"))
    if os.environ.get("ECHOKV_ENABLE") == "1" and torch.cuda.is_available():
        kw["attention_config"] = {"backend": "CUSTOM"}
    llm = LLM(**kw)
    cfg = llm.llm_engine.vllm_config.cache_config
    rng = random.Random(0)
    vocab = llm.get_tokenizer().vocab_size
    prompts = [
        {"prompt_token_ids": [rng.randrange(100, vocab - 100) for _ in range(args.input_len)]}
        for _ in range(args.num_prompts)
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=args.output_len, ignore_eos=True)
    llm.generate(prompts[: min(4, len(prompts))], sp)  # warm up
    t = time.perf_counter()
    outs = llm.generate(prompts, sp)
    wall = time.perf_counter() - t
    out_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    res = {
        "kv_capacity_tokens": (cfg.num_gpu_blocks or 0) * cfg.block_size,
        "wall_s": wall,
        "output_tok_s": out_tokens / wall,
        "total_tok_s": (out_tokens + args.input_len * len(prompts)) / wall,
    }
    if os.environ.get("ECHOKV_ENABLE") == "1":
        from echokv_vllm import control

        st = llm.apply_model(control.get_state)[0]
        res["avg_stored_bits"] = st["avg_stored_bits"]
        res["tier_vector"] = st["tier_vector"]
    with open(args.child_out, "w") as f:
        json.dump(res, f)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--kv-cache-gib", type=float, default=1.0)
    p.add_argument("--num-prompts", type=int, default=64)
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--configs", nargs="*", default=list(CONFIGS))
    p.add_argument("--out", default="echokv_throughput.json")
    p.add_argument("--child-out")
    args = p.parse_args()
    if args.child_out:
        child(args)
        return
    results = {}
    for name in args.configs:
        tmp = f"{args.out}.{name}.tmp"
        env = dict(os.environ, VLLM_ALLOW_INSECURE_SERIALIZATION="1", **CONFIGS[name])
        proc = subprocess.run([sys.executable, __file__, "--child-out", tmp] + sys.argv[1:], env=env)
        # vLLM can abort during teardown after finishing: the result file is the success signal.
        if not os.path.exists(tmp):
            print(f"[{name}] FAILED (exit {proc.returncode})")
            results[name] = {"error": f"exit {proc.returncode}"}
            continue
        with open(tmp) as f:
            results[name] = json.load(f)
        os.remove(tmp)
    base = results.get("stock")
    if base is not None and "error" in base:
        base = None
    print(f"\n{args.model}  KV budget {args.kv_cache_gib} GiB, {args.num_prompts} x ({args.input_len} in, {args.output_len} out)")
    print(f"{'config':<18} {'KV tokens':>10} {'capacity':>9} {'out tok/s':>10} {'speed':>7} {'bits':>5}")
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<18} FAILED ({r['error']})")
            continue
        cap = r["kv_capacity_tokens"] / base["kv_capacity_tokens"] if base else float("nan")
        spd = r["output_tok_s"] / base["output_tok_s"] if base else float("nan")
        bits = r.get("avg_stored_bits")
        print(f"{name:<18} {r['kv_capacity_tokens']:>10,} {cap:>8.2f}x {r['output_tok_s']:>10.1f} {spd:>6.2f}x "
              f"{'' if bits is None else f'{bits:.2f}':>5}")
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=1)
    if any("error" in r for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
