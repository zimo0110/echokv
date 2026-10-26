"""Profile stock vLLM vs EchoKV on the same workload and summarize where time goes.

Each configuration runs in its own engine with vLLM's torch profiler on.
After the run the worker trace is parsed into a short summary:

* wall time of the profiled generate() call
* GPU busy time, and the top GPU kernels by total time
* CPU and GPU time inside each ``echokv.*`` scope (see backend.py):
  build (host planning + batch layout), write, prefill_attn, decode_attn,
  canary

    python evaluation/serving/echokv_profile.py --model unsloth/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import random
import shutil
import subprocess
import sys
import time

CONFIGS = {
    "stock": {"ECHOKV_ENABLE": "0"},
    "echokv-int4": {"ECHOKV_ENABLE": "1", "ECHOKV_MEM_BITS": "4", "ECHOKV_FIXED_BITS": "4"},
}


def child(args) -> None:
    import torch
    from vllm import LLM, SamplingParams

    trace_dir = os.path.abspath(args.trace_dir)
    shutil.rmtree(trace_dir, ignore_errors=True)
    os.makedirs(trace_dir)
    kw = dict(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.input_len + args.output_len + 64,
        kv_cache_memory_bytes=int(args.kv_cache_gib * 2**30),
        block_size=64,
        enable_prefix_caching=False,
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": trace_dir,
            "torch_profiler_with_stack": False,
            "torch_profiler_use_gzip": True,
        },
    )
    if torch.cuda.is_available():
        kw["gpu_memory_utilization"] = float(os.environ.get("ECHOKV_GPU_UTIL", "0.7"))
        if os.environ.get("ECHOKV_ENABLE") == "1":
            kw["attention_config"] = {"backend": "CUSTOM"}
    llm = LLM(**kw)
    rng = random.Random(0)
    vocab = llm.get_tokenizer().vocab_size
    prompts = [
        {"prompt_token_ids": [rng.randrange(100, vocab - 100) for _ in range(args.input_len)]}
        for _ in range(args.num_prompts)
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=args.output_len, ignore_eos=True)
    llm.generate(prompts[:4], sp)  # warm up
    llm.start_profile()
    t = time.perf_counter()
    llm.generate(prompts, sp)
    wall = time.perf_counter() - t
    llm.stop_profile()
    time.sleep(5)  # let the worker flush its trace
    with open(args.child_out, "w") as f:
        json.dump({"wall_s": wall, "trace_dir": trace_dir}, f)


def summarize(trace_dir: str, top: int = 25) -> dict:
    """Aggregate a worker's torch-profiler trace (Chrome JSON)."""
    files = sorted(glob.glob(os.path.join(trace_dir, "**", "*.json*"), recursive=True))
    files = [f for f in files if "trace" in os.path.basename(f) or f.endswith(".pt.trace.json.gz")]
    if not files:
        return {"error": f"no trace files under {trace_dir}"}
    # The largest file is the GPU worker's trace (the frontend's is CPU-only and small).
    path = max(files, key=os.path.getsize)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        events = json.load(f).get("traceEvents", [])
    kernels = collections.Counter()
    kernel_calls = collections.Counter()
    scope_cpu = collections.Counter()
    scope_gpu = collections.Counter()
    scope_calls = collections.Counter()
    gpu_busy = 0.0
    for e in events:
        if e.get("ph") != "X":
            continue
        cat, name, dur = e.get("cat", ""), e.get("name", ""), float(e.get("dur", 0.0))
        if cat == "kernel":
            kernels[name[:90]] += dur
            kernel_calls[name[:90]] += 1
            gpu_busy += dur
        elif name.startswith("echokv."):
            if cat == "gpu_user_annotation":
                scope_gpu[name] += dur
            else:
                scope_cpu[name] += dur
                scope_calls[name] += 1
    ms = lambda us: round(us / 1000.0, 1)  # noqa: E731
    return {
        "trace": os.path.basename(path),
        "gpu_busy_ms": ms(gpu_busy),
        "scopes": {
            k: {"cpu_ms": ms(scope_cpu[k]), "gpu_ms": ms(scope_gpu[k]), "calls": scope_calls[k]}
            for k in sorted(set(scope_cpu) | set(scope_gpu))
        },
        "top_kernels": [
            {"name": k, "ms": ms(v), "calls": kernel_calls[k]} for k, v in kernels.most_common(top)
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/Llama-3.1-8B-Instruct")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--kv-cache-gib", type=float, default=8.0)
    p.add_argument("--num-prompts", type=int, default=64)
    p.add_argument("--input-len", type=int, default=1024)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--configs", nargs="*", default=list(CONFIGS))
    p.add_argument("--out", default="echokv_profile.json")
    p.add_argument("--child-out")
    p.add_argument("--trace-dir")
    args = p.parse_args()
    if args.child_out:
        child(args)
        return
    report = {}
    for name in args.configs:
        tmp = f"{args.out}.{name}.tmp"
        trace_dir = f"{os.path.splitext(args.out)[0]}_traces/{name}"
        env = dict(os.environ, VLLM_ALLOW_INSECURE_SERIALIZATION="1", **CONFIGS[name])
        subprocess.run(
            [sys.executable, __file__, "--child-out", tmp, "--trace-dir", trace_dir] + sys.argv[1:],
            env=env,
        )
        if not os.path.exists(tmp):
            report[name] = {"error": "run failed"}
            continue
        with open(tmp) as f:
            res = json.load(f)
        os.remove(tmp)
        res.update(summarize(res["trace_dir"]))
        report[name] = res
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    for name, r in report.items():
        print(f"\n=== {name}: wall {r.get('wall_s', 0):.1f}s, GPU busy {r.get('gpu_busy_ms', 0) / 1000:.1f}s")
        for k, v in r.get("scopes", {}).items():
            print(f"  {k:<22} cpu {v['cpu_ms']:>9.1f} ms  gpu {v['gpu_ms']:>9.1f} ms  calls {v['calls']}")
        for kd in r.get("top_kernels", [])[:12]:
            print(f"  {kd['ms']:>9.1f} ms  x{kd['calls']:<6} {kd['name']}")


if __name__ == "__main__":
    main()
