"""Quality vs. memory sweep for the EchoKV vLLM plugin (CPU or CUDA).

Each configuration runs in its own engine (subprocess) and is scored by
prompt NLL on WikiText-103 test passages, against stock vLLM:

  stock        unmodified vLLM attention
  int{8,4,2}   EchoKV store pinned to one tier (ECHOKV_FIXED_BITS)
  tau=X        live controller: warm up on a generation workload so the
               tiers converge, reset counters, then score the passages with
               the controller still running

Reported per config: ΔNLL vs stock, average bits of the K/V tokens stored
during scoring, and the tier vector at the end.

    python evaluation/serving/echokv_quality_sweep.py \
        --model unsloth/Llama-3.2-1B-Instruct --taus 0.005 0.01 0.02 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

WARMUP_PROMPTS = [
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain step by step how to compute the derivative of x^3 * sin(x).",
    "List ten facts about the planet Mars.",
    "Describe the water cycle to a ten year old.",
    "Summarize the causes of the First World War.",
    "Write a Python function that checks whether a string is a palindrome, and explain it.",
    "What are the main differences between TCP and UDP?",
    "Plan a three-day trip to Kyoto.",
]


def load_passages(n: int, tokens: int, tokenizer) -> list[list[int]]:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    articles = "".join(ds["text"]).split("\n = ")
    out = []
    for a in articles:
        ids = tokenizer(a).input_ids
        if len(ids) >= tokens:
            out.append(ids[:tokens])
        if len(out) == n:
            break
    return out


def child(args) -> None:
    import torch
    from vllm import LLM, SamplingParams

    kw = dict(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.passage_tokens + 512,
        enforce_eager=True,
        kv_cache_memory_bytes=args.kv_cache_mib * 2**20,
        enable_prefix_caching=False,
        block_size=args.block_size,
        tensor_parallel_size=args.tp,
    )
    echokv = os.environ.get("ECHOKV_ENABLE") == "1"
    if echokv and torch.cuda.is_available():
        kw["attention_config"] = {"backend": "CUSTOM"}
    llm = LLM(**kw)
    result = {}
    if echokv:
        from echokv_vllm import control

        if args.warmup_tokens:
            llm.generate(WARMUP_PROMPTS, SamplingParams(temperature=0.0, max_tokens=args.warmup_tokens))
        llm.apply_model(control.reset_counters)
    passages = load_passages(args.passages, args.passage_tokens, llm.get_tokenizer())
    outs = llm.generate(
        [{"prompt_token_ids": p} for p in passages],
        SamplingParams(max_tokens=1, prompt_logprobs=0),
    )
    lps = [lp for o in outs for d in o.prompt_logprobs if d is not None for lp in [next(iter(d.values())).logprob]]
    result["nll"] = -sum(lps) / len(lps)
    if echokv:
        st = llm.apply_model(control.get_state)[0]
        result["avg_stored_bits"] = st["avg_stored_bits"]
        result["tier_vector"] = st["tier_vector"]
    with open(args.child_out, "w") as f:
        json.dump(result, f)


def run(args, name: str, env_extra: dict) -> dict:
    out = f"{args.out}.{name}.tmp"
    env = dict(os.environ, VLLM_ALLOW_INSECURE_SERIALIZATION="1", **env_extra)
    cmd = [sys.executable, __file__, "--child-out", out] + [
        a for a in sys.argv[1:] if a != "--child-out"
    ]
    subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL if args.quiet else None,
                   stderr=subprocess.DEVNULL if args.quiet else None)
    with open(out) as f:
        res = json.load(f)
    os.remove(out)
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--kv-cache-mib", type=int, default=512)
    p.add_argument("--passages", type=int, default=8)
    p.add_argument("--passage-tokens", type=int, default=768)
    p.add_argument("--warmup-tokens", type=int, default=128)
    p.add_argument("--static", type=int, nargs="*", default=[8, 4, 2])
    p.add_argument("--taus", type=float, nargs="*", default=[0.005, 0.01, 0.02, 0.05])
    p.add_argument("--canary-interval", type=int, default=8)
    p.add_argument("--out", default="echokv_quality_sweep.json")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--child-out")
    args = p.parse_args()
    if args.child_out:
        child(args)
        return

    on = {"ECHOKV_ENABLE": "1", "ECHOKV_CANARY_INTERVAL": str(args.canary_interval)}
    results = {"stock": run(args, "stock", {"ECHOKV_ENABLE": "0"})}
    for b in args.static:
        results[f"int{b}"] = run(args, f"int{b}", {**on, "ECHOKV_FIXED_BITS": str(b)})
    for t in args.taus:
        results[f"tau={t}"] = run(args, f"tau{t}", {**on, "ECHOKV_TAU0": str(t)})
    base = results["stock"]["nll"]
    print(f"\n{args.model}   stock NLL {base:.4f}")
    print(f"{'config':<12} {'ΔNLL':>8} {'bits':>6}  tiers")
    for name, r in results.items():
        if name == "stock":
            continue
        r["delta_nll"] = r["nll"] - base
        tiers = r.get("tier_vector") or []
        mix = " ".join(f"{b}:{tiers.count(b)}" for b in (2, 4, 8) if tiers.count(b))
        bits = r.get("avg_stored_bits")
        print(f"{name:<12} {r['delta_nll']:>+8.4f} {bits if bits is None else round(bits, 2):>6}  {mix}")
    with open(args.out, "w") as f:
        json.dump({"model": args.model, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
