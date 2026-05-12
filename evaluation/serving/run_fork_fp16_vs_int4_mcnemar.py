"""Fork FP16 vs Fork static INT4 paired-McNemar on Llama-70B GSM8K-Hard, n=500.

The diagnostic chain has compared fork INT4 against emulation INT4 and
emulation FP16, but never against fork FP16 on the same prompts. This
prerequisite should have run before any of the canary diagnostics.
Running it now decides the path forward:

  Outcome 1 (paired Δ within ±0.02): no meaningful INT4 deficit on the
    fork. Closed loop has no signal to recover — paper claim narrows.
  Outcome 2 (paired Δ ≥ 0.03, p < 0.05): real INT4 deficit. Closed
    loop's failure to recover is a real bug. Proceed to differential
    dequant test, self-consistent canary, multi-seed McNemar.
  Outcome 3 (Δ in 0.02–0.03 band): ambiguous; warrants the three-prong.

Same prompts (GSM8K-Hard, seed=0, n=500), same protocol (TP=4, mns=8,
max_new_tokens=512, max_model_len=4096), same harness, same engine —
the only thing that changes between phases is whether the cache is
booted at kv_cache_dtype="int8_kivi" + swap-to-uniform-INT4 or at
default BF16.

Launch in tmux:
  tmux new -s mcnemar \
    'CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/conda/envs/scalable/bin/python \
       /tmp/run_fork_fp16_vs_int4_mcnemar.py'
  # Detach: Ctrl-b then d.  Reattach: tmux attach -t mcnemar
"""
from __future__ import annotations

import functools
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

_FORK = "/mnt/nvme_tmp10/nazar_vllm_fork"
_REPO = "/home/nazar/echokv"
if _FORK not in sys.path:
    sys.path.insert(0, _FORK)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.environ["PYTHONPATH"] = f"{_REPO}:{_FORK}:" + os.environ.get("PYTHONPATH", "")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("paired_mcnemar")

import argparse

MODEL = "/mnt/nvme_tmp6/llama-70b"
TP = 4
MAX_NEW_TOKENS = 512
MAX_MODEL_LEN = 4096
MAX_NUM_SEQS = 8

OUT_DIR = Path("/home/nazar/echokv/results/serving")


def materialize_chat_prompts(llm, requests):
    tok = llm.get_tokenizer()
    out = []
    for r in requests:
        msgs = [{"role": "user", "content": r.prompt}]
        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        out.append((text, r))
    return out


def run_phase(label: str, kv_dtype: str | None, swap_to_int4: bool,
              n_prompts: int, seed: int):
    """Boot the engine, optionally swap cache to uniform INT4, generate
    on the n GSM8K-Hard prompts at the given seed, return per-prompt scores.

    Each phase runs in the same process; the LLM is torn down between
    phases via ``del`` + ``gc.collect()`` + ``torch.cuda.empty_cache()``
    + a small sleep to let worker processes exit cleanly.
    """
    import torch
    from vllm import LLM, SamplingParams
    from evaluation.motivation.workloads import _gsm8k_hard_stream
    from evaluation.serving.fork_tier import (
        build_uniform_tier_vector,
        read_buffer_bits_from_model,
        read_tier_vector_from_model,
        swap_tier_vector_to_model,
    )

    log.info("=" * 70)
    log.info("PHASE %s  | kv_cache_dtype=%s  swap_to_int4=%s",
             label, kv_dtype or "auto (bfloat16)", swap_to_int4)
    log.info("=" * 70)
    t0 = time.perf_counter()
    kwargs = dict(
        model=MODEL,
        tensor_parallel_size=TP,
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        max_model_len=MAX_MODEL_LEN,
        enable_prefix_caching=False,
        max_num_seqs=MAX_NUM_SEQS,
    )
    if kv_dtype:
        kwargs["kv_cache_dtype"] = kv_dtype
    llm = LLM(**kwargs)
    log.info("loaded in %.1fs", time.perf_counter() - t0)

    if swap_to_int4:
        rank_buf_bits = llm.apply_model(read_buffer_bits_from_model)
        num_layers = len(rank_buf_bits[0])
        log.info("buffer_bits across %d layers; swapping all to INT4", num_layers)
        initial_vec = build_uniform_tier_vector(num_layers, 4)
        llm.apply_model(
            functools.partial(swap_tier_vector_to_model,
                              bits_per_layer=initial_vec),
        )
        confirmed = llm.apply_model(read_tier_vector_from_model)
        for ri, vec in enumerate(confirmed):
            if any(b != 4 for b in vec):
                raise RuntimeError(
                    f"rank {ri} not at uniform INT4: {sorted(set(vec))}"
                )
        log.info("all layers swapped to uniform INT4 (cache is real INT4 KIVI)")

    raw = list(_gsm8k_hard_stream(n=n_prompts, seed=seed))
    chat_prompts = materialize_chat_prompts(llm, raw)
    log.info("generating: n=%d, mns=%d, max_new_tokens=%d",
             n_prompts, MAX_NUM_SEQS, MAX_NEW_TOKENS)

    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)
    t0 = time.perf_counter()
    outputs = llm.generate(
        [t for t, _ in chat_prompts],
        sampling_params=sp,
        use_tqdm=False,
    )
    wall = time.perf_counter() - t0

    scores = []
    for idx, ((txt, req), out) in enumerate(zip(chat_prompts, outputs)):
        gen = out.outputs[0].text
        score = float(req.score_fn(gen))
        scores.append({"idx": idx, "score": score, "q_head": req.prompt[:80]})
    acc = sum(s["score"] for s in scores) / len(scores)
    log.info("phase %s  | acc=%.3f  wall=%.1fs  n=%d", label, acc, wall, n_prompts)

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)

    return scores, acc, wall


def mcnemar_test(scores_a: list[dict], scores_b: list[dict],
                 label_a: str, label_b: str) -> dict:
    """Paired-McNemar (with Yates' continuity correction) on binary
    per-prompt outcomes. Returns a dict with raw counts, accuracies,
    discordant-pair counts, χ², p-value.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(f"length mismatch: {len(scores_a)} vs {len(scores_b)}")
    n = len(scores_a)

    acc_a = sum(s["score"] for s in scores_a) / n
    acc_b = sum(s["score"] for s in scores_b) / n
    delta = acc_a - acc_b

    # Discordant pairs
    a_wins = 0  # A correct, B wrong
    b_wins = 0  # A wrong, B correct
    both_right = 0
    both_wrong = 0
    for sa, sb in zip(scores_a, scores_b):
        a_ok = sa["score"] >= 0.5
        b_ok = sb["score"] >= 0.5
        if a_ok and b_ok:
            both_right += 1
        elif a_ok and not b_ok:
            a_wins += 1
        elif b_ok and not a_ok:
            b_wins += 1
        else:
            both_wrong += 1

    n_disc = a_wins + b_wins
    if n_disc == 0:
        chi2 = 0.0
        p_value = 1.0
    else:
        chi2 = (abs(a_wins - b_wins) - 1) ** 2 / n_disc
        try:
            from scipy.stats import chi2 as _chi2_dist
            p_value = float(_chi2_dist.sf(chi2, df=1))
        except ImportError:
            from math import erfc, sqrt
            # χ²(1) is the square of a std normal; sf(z²) = erfc(|z|/√2)
            z = chi2 ** 0.5
            p_value = erfc(z / sqrt(2.0))

    return {
        "label_a": label_a,
        "label_b": label_b,
        "n": n,
        "acc_a": acc_a,
        "acc_b": acc_b,
        "delta_acc": delta,
        "a_wins": a_wins,                # A correct, B wrong
        "b_wins": b_wins,                # A wrong, B correct
        "both_right": both_right,
        "both_wrong": both_wrong,
        "concordant": both_right + both_wrong,
        "chi2_yates": chi2,
        "p_value": p_value,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-prompts", type=int, default=500,
                    help="Number of GSM8K-Hard prompts (default 500). "
                         "Filenames and the McNemar JSON encode this.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Prompt-stream seed (default 0).")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fp16_out = OUT_DIR / f"fork_fp16_70b_gsm8k_hard_n{args.n_prompts}_per_prompt.json"
    int4_out = OUT_DIR / f"fork_static_int4_70b_gsm8k_hard_n{args.n_prompts}_per_prompt.json"
    mcnemar_out = OUT_DIR / f"fork_fp16_vs_int4_70b_gsm8k_hard_n{args.n_prompts}_mcnemar.json"

    # ---- Phase 1: Fork FP16 (BF16 cache, no swap) ----
    fp16_scores, fp16_acc, fp16_wall = run_phase(
        label="FP16", kv_dtype=None, swap_to_int4=False,
        n_prompts=args.n_prompts, seed=args.seed,
    )
    fp16_out.write_text(json.dumps({
        "phase": "fork_fp16",
        "model": MODEL, "tp": TP, "n_prompts": args.n_prompts,
        "max_num_seqs": MAX_NUM_SEQS, "max_new_tokens": MAX_NEW_TOKENS,
        "max_model_len": MAX_MODEL_LEN, "seed": args.seed,
        "kv_cache_dtype": "auto (bfloat16)",
        "accuracy": fp16_acc, "wall_clock_s": fp16_wall,
        "per_prompt": fp16_scores,
    }, indent=2))
    log.info("wrote %s", fp16_out)

    # ---- Phase 2: Fork static INT4 (int8_kivi cache + swap to INT4) ----
    int4_scores, int4_acc, int4_wall = run_phase(
        label="INT4", kv_dtype="int8_kivi", swap_to_int4=True,
        n_prompts=args.n_prompts, seed=args.seed,
    )
    int4_out.write_text(json.dumps({
        "phase": "fork_static_int4",
        "model": MODEL, "tp": TP, "n_prompts": args.n_prompts,
        "max_num_seqs": MAX_NUM_SEQS, "max_new_tokens": MAX_NEW_TOKENS,
        "max_model_len": MAX_MODEL_LEN, "seed": args.seed,
        "kv_cache_dtype": "int8_kivi (swapped to uniform INT4)",
        "accuracy": int4_acc, "wall_clock_s": int4_wall,
        "per_prompt": int4_scores,
    }, indent=2))
    log.info("wrote %s", int4_out)

    # ---- Phase 3: paired McNemar ----
    log.info("=" * 70)
    log.info("PAIRED McNEMAR  |  fork FP16 vs fork static INT4  (n=%d)", args.n_prompts)
    log.info("=" * 70)
    result = mcnemar_test(fp16_scores, int4_scores, "fork_fp16", "fork_int4")
    log.info("  Acc(FP16) = %.4f", result["acc_a"])
    log.info("  Acc(INT4) = %.4f", result["acc_b"])
    log.info("  Δ (FP16 − INT4) = %+.4f", result["delta_acc"])
    log.info("  Both right: %d   |  Both wrong: %d", result["both_right"], result["both_wrong"])
    log.info("  Discordant: FP16 wins=%d   INT4 wins=%d", result["a_wins"], result["b_wins"])
    log.info("  McNemar χ² (Yates) = %.3f", result["chi2_yates"])
    log.info("  p-value = %.4g", result["p_value"])
    mcnemar_out.write_text(json.dumps(result, indent=2))
    log.info("wrote %s", mcnemar_out)

    log.info("=" * 70)
    delta = result["delta_acc"]
    p = result["p_value"]
    if abs(delta) < 0.02:
        verdict = (f"Outcome 1 — no meaningful INT4 deficit on the fork "
                   f"(Δ={delta:+.4f}, p={p:.4g}). Closed loop has no signal "
                   f"to recover; the paper claim should narrow.")
    elif delta >= 0.03 and p < 0.05:
        verdict = (f"Outcome 2 — real INT4 deficit on the fork "
                   f"(Δ={delta:+.4f}, p={p:.4g}). Closed loop's failure to "
                   f"recover is a real bug; proceed to differential dequant "
                   f"test + self-consistent canary + multi-seed McNemar.")
    else:
        verdict = (f"Outcome 3 — ambiguous (Δ={delta:+.4f}, p={p:.4g}). "
                   f"Warrants the three-pronged experiment to clarify.")
    log.info("VERDICT: %s", verdict)
    log.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
