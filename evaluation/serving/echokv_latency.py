"""Serving latency: stock vLLM vs EchoKV behind `vllm serve`, same KV memory.

For each configuration, starts an OpenAI-compatible server and drives it
with `vllm bench serve` (random prompts, ignore_eos) at each concurrency
level. A closed loop at fixed concurrency is used: past the point where
stock vLLM runs out of KV memory, its requests queue and TTFT grows.
Reports TTFT / TPOT / ITL / end-to-end percentiles and throughput.

    python evaluation/serving/echokv_latency.py --model unsloth/Llama-3.1-8B-Instruct \
        --input-len 8192 --output-len 256 --concurrency 4 16 32
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

CONFIGS = {
    "stock": {"ECHOKV_ENABLE": "0"},
    "echokv-int4": {"ECHOKV_ENABLE": "1", "ECHOKV_MEM_BITS": "4", "ECHOKV_FIXED_BITS": "4"},
    "echokv-int8": {"ECHOKV_ENABLE": "1", "ECHOKV_MEM_BITS": "8", "ECHOKV_FIXED_BITS": "8"},
    # Live controller (dynamic tiers) under a memory budget, default tau.
    "echokv-live-mem4": {"ECHOKV_ENABLE": "1", "ECHOKV_MEM_BITS": "4"},
    "echokv-live-mem6": {"ECHOKV_ENABLE": "1", "ECHOKV_MEM_BITS": "6"},
}
METRICS = ["ttft", "tpot", "itl", "e2el"]


def vllm_bin() -> str:
    return os.path.join(os.path.dirname(sys.executable), "vllm")


def wait_ready(port: int, proc: subprocess.Popen, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/Llama-3.1-8B-Instruct")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--kv-cache-gib", type=float, default=8.0)
    p.add_argument("--input-len", type=int, default=8192)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--concurrency", type=int, nargs="*", default=[4, 16, 32])
    p.add_argument("--prompts-per-level", type=int, default=2, help="prompts = this x concurrency (min 16)")
    p.add_argument("--configs", nargs="*", default=["stock", "echokv-int4"])
    p.add_argument("--port", type=int, default=8321)
    p.add_argument("--out", default="echokv_latency.json")
    args = p.parse_args()

    out_dir = os.path.splitext(args.out)[0] + "_runs"
    os.makedirs(out_dir, exist_ok=True)
    results: dict = {}
    for name in args.configs:
        env = dict(os.environ, VLLM_ALLOW_INSECURE_SERIALIZATION="1", **CONFIGS[name])
        cmd = [
            vllm_bin(), "serve", args.model,
            "--port", str(args.port),
            "--tensor-parallel-size", str(args.tp),
            "--max-model-len", str(args.input_len + args.output_len + 64),
            "--kv-cache-memory-bytes", str(int(args.kv_cache_gib * 2**30)),
            "--gpu-memory-utilization", os.environ.get("ECHOKV_GPU_UTIL", "0.7"),
            "--block-size", "64",
            "--no-enable-prefix-caching",
        ]
        if CONFIGS[name]["ECHOKV_ENABLE"] == "1":
            cmd += ["--attention-config", json.dumps({"backend": "CUSTOM"})]
        log_path = os.path.join(out_dir, f"{name}.server.log")
        print(f"== {name}: starting server (log {log_path})", flush=True)
        with open(log_path, "w") as log:
            server = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            if not wait_ready(args.port, server, timeout=1800):
                print(f"[{name}] server failed to start; see {log_path}", flush=True)
                results[name] = {"error": "server failed to start"}
                continue
            results[name] = {}
            for c in args.concurrency:
                fname = f"{name}_c{c}.json"
                bench = [
                    vllm_bin(), "bench", "serve",
                    "--backend", "vllm",
                    "--base-url", f"http://127.0.0.1:{args.port}",
                    "--model", args.model,
                    "--dataset-name", "random",
                    "--random-input-len", str(args.input_len),
                    "--random-output-len", str(args.output_len),
                    "--num-prompts", str(max(16, args.prompts_per_level * c)),
                    "--max-concurrency", str(c),
                    "--request-rate", "inf",
                    "--ignore-eos",
                    "--percentile-metrics", ",".join(METRICS),
                    "--metric-percentiles", "50,90,99",
                    "--save-result",
                    "--result-dir", out_dir,
                    "--result-filename", fname,
                ]
                print(f"   concurrency {c} ...", flush=True)
                with open(os.path.join(out_dir, f"{name}_c{c}.bench.log"), "w") as log:
                    subprocess.run(bench, env=env, stdout=log, stderr=subprocess.STDOUT)
                path = os.path.join(out_dir, fname)
                if not os.path.exists(path):
                    results[name][c] = {"error": "bench failed"}
                    continue
                with open(path) as f:
                    r = json.load(f)
                results[name][c] = {
                    k: r.get(k)
                    for k in (
                        "request_throughput", "output_throughput", "total_token_throughput",
                        "median_ttft_ms", "p90_ttft_ms", "p99_ttft_ms",
                        "median_tpot_ms", "p90_tpot_ms", "p99_tpot_ms",
                        "median_itl_ms", "p99_itl_ms", "median_e2el_ms", "p99_e2el_ms",
                    )
                }
        finally:
            stop(server)

    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=1)
    print(f"\n{args.model}: {args.input_len} in / {args.output_len} out, KV {args.kv_cache_gib} GiB")
    print(f"{'config':<13} {'conc':>4} {'out tok/s':>9} {'TTFT p50':>9} {'TTFT p99':>9} "
          f"{'TPOT p50':>9} {'TPOT p99':>9} {'E2E p50':>9}  (ms)")
    for name, levels in results.items():
        if "error" in levels:
            print(f"{name:<13} FAILED ({levels['error']})")
            continue
        for c, r in levels.items():
            if "error" in r:
                print(f"{name:<13} {c:>4} FAILED")
                continue
            f = lambda k: "-" if r.get(k) is None else f"{r[k]:.0f}"  # noqa: E731
            print(f"{name:<13} {c:>4} {f('output_throughput'):>9} {f('median_ttft_ms'):>9} {f('p99_ttft_ms'):>9} "
                  f"{f('median_tpot_ms'):>9} {f('p99_tpot_ms'):>9} {f('median_e2el_ms'):>9}")
    if any("error" in v or any("error" in x for x in v.values()) for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
