"""Session 15 — EchoKV-3C closed-loop on τ-Bench airline (companion to
``run_session15_70b_taubench.py``).

Three-phase protocol (mirrors the existing 70B echokv drivers):

Phase 1 (warmup):
    Run the first ``--warmup-n`` τ-Bench airline tasks with the canary
    observing at ``--observability-rate`` and ``tier_application=True``
    so the controller's K/V quant decisions take effect during warmup.
    Multi-turn agentic decoding accumulates many EMA updates per task
    (~1-3K decode tokens × N turns × warmup_n tasks).

Phase 2 (consolidate):
    Take the per-layer max across TP ranks. Most-conservative tier per
    layer wins.

Phase 3 (measurement):
    Run the remaining tasks under either:
    * frozen-tier protocol (default): ``canary_rate=1e-6`` and
      ``freeze_controller=True``; the consolidated tier vector is
      seeded onto every rank.
    * continuous-dynamic (``--continuous-dynamic``): canary at p_c=1/64,
      controller unfrozen, tier vector continues to adapt during
      measurement (the headline EchoKV protocol per CHANGELOG Phase-8).

Output JSON shape matches the GSM8K-Hard / HotpotQA / RULER echokv
drivers, with the τ-Bench-specific per-task fields (reward, n_turns,
decode_tokens, finished_reason, agent actions) carried in the
``measurement`` block.
"""
from __future__ import annotations

import argparse
import builtins
import json
import logging
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

# Suppress tau-bench's stdin-blocking user simulator before importing
builtins.input = lambda *a, **k: ""

from tau_bench.envs import get_env

from evaluation.serving.canary_tp import TPCanaryHandle
from evaluation.serving.run_session8 import RunResult
from evaluation.serving.run_session15_70b_taubench import (
    SYSTEM_PROMPT_TEMPLATE,        # noqa: F401  (kept for cross-driver consistency)
    parse_agent_action,
    run_one_task,
    TaskResult,
)


log = logging.getLogger("session15_70b_taubench_echokv")
DEFAULT_70B = "/mnt/nvme_tmp6/llama-70b"


# ---------------------------------------------------------------------------
# Tier consolidation / memory projection — shared with GSM8K-Hard
# ---------------------------------------------------------------------------

def consolidate_tiers(per_rank_snapshots: list[dict]) -> list[int]:
    n = per_rank_snapshots[0]["num_layers"]
    out: list[int] = []
    for li in range(n):
        out.append(max(s["tier_vector"][li] for s in per_rank_snapshots))
    return out


def consolidate_deltas(per_rank_snapshots: list[dict]) -> list[float]:
    n = per_rank_snapshots[0]["num_layers"]
    out: list[float] = []
    for li in range(n):
        vals = [
            s["per_layer_delta"][li] for s in per_rank_snapshots
            if s["per_layer_delta"][li] > 0.0
        ]
        out.append(sum(vals) / len(vals) if vals else 0.0)
    return out


def memory_projection(tiers: list[int]) -> dict:
    n = len(tiers)
    avg_bits = sum(tiers) / n
    return {
        "num_layers": n,
        "avg_bits": avg_bits,
        "savings_vs_fp16_pct": 100.0 * (1.0 - avg_bits / 16.0),
        "savings_vs_kivi4_pct": 100.0 * (1.0 - avg_bits / 4.0),
        "tier_counts": dict(Counter(tiers)),
    }


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def _build_env(env_name: str = "airline") -> Any:
    """Construct the test env. user_strategy='human' is stubbed via our
    input() override so env init doesn't block. ``env_name`` is one of
    "airline" (50 tasks) or "retail" (115 tasks)."""
    return get_env(
        env_name=env_name,
        user_strategy="human",
        user_model="dummy",
        task_split="test",
        user_provider="openai",
        task_index=0,
    )


def run_tasks_in_range(
    llm: Any,
    *,
    label: str,
    task_indices: list[int],
    max_turns: int,
    max_new_tokens: int,
    seed: int,
    env_name: str = "airline",
) -> RunResult:
    """Run the agentic loop across the given task indices, returning
    a RunResult with per-task records. Per-task ``request_id`` is
    namespaced by env so airline + retail blobs can be merged for
    cross-env paired McNemar without ID collisions."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        seed=seed,
    )
    tokenizer = llm.get_tokenizer()
    env = _build_env(env_name=env_name)

    t0 = time.perf_counter()
    per_request: list[dict[str, Any]] = []
    n_correct = 0
    finished_counts: Counter[str] = Counter()
    total_decode = total_turns = 0

    for ti in task_indices:
        try:
            tr: TaskResult = run_one_task(
                llm, env,
                task_idx=ti,
                max_turns=max_turns,
                max_new_tokens=max_new_tokens,
                sampling_params=sampling_params,
                tokenizer=tokenizer,
            )
        except Exception as exc:
            log.exception("task %d raised: %s", ti, exc)
            tr = TaskResult(task_id=ti, reward=0.0, n_turns=0,
                            decode_tokens=0, correct=0,
                            finished_reason="exception")
        per_request.append({
            "request_id": f"taubench_{env_name}:{ti}",
            "correct": tr.correct,
            "reward": tr.reward,
            "n_turns": tr.n_turns,
            "decode_tokens": tr.decode_tokens,
            "finished_reason": tr.finished_reason,
            "actions": tr.agent_actions,
        })
        n_correct += tr.correct
        total_decode += tr.decode_tokens
        total_turns += tr.n_turns
        finished_counts[tr.finished_reason] += 1
        log.info(
            "    [%s] task %d: reward=%.0f turns=%d decode_tok=%d (%s)",
            label, ti, tr.reward, tr.n_turns, tr.decode_tokens,
            tr.finished_reason,
        )

    wall = time.perf_counter() - t0
    n = len(task_indices)
    accuracy = n_correct / max(n, 1)
    log.info(
        "    [%s] done: n=%d acc=%.3f total_decode=%d "
        "avg_turns=%.1f finished=%s wall=%.1fs",
        label, n, accuracy, total_decode,
        total_turns / max(n, 1), dict(finished_counts), wall,
    )
    return RunResult(
        label=label,
        wall_clock_s=wall,
        n_prompts=n,
        completed=n,
        n_correct=n_correct,
        accuracy=accuracy,
        p50_tpot_s=None,
        p99_tpot_s=None,
        mean_ttft_s=None,
        total_input_tokens=0,
        total_output_tokens=total_decode,
        output_throughput_tps=total_decode / max(wall, 1e-6),
        per_request=per_request,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_70B)
    p.add_argument("--tensor-parallel-size", type=int, default=8)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-turns", type=int, default=15)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-num-seqs", type=int, default=8)
    p.add_argument("--n-tasks", type=int, default=50,
                   help="airline=50, retail=115 test tasks; warmup uses "
                        "the first --warmup-n, measurement uses the rest")
    p.add_argument("--env", default="airline",
                   choices=["airline", "retail"],
                   help="τ-Bench env (airline test = 50 tasks; "
                        "retail test = 115 tasks)")
    p.add_argument("--warmup-n", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--observability-rate", type=float, default=0.25)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--tick-interval", type=int, default=10)
    p.add_argument("--multi-precision", action="store_true")
    p.add_argument("--tau-budget", type=float, default=0.022)
    p.add_argument("--absolute-canary", action="store_true")
    p.add_argument("--kref-window-size", type=int, default=64)
    p.add_argument("--continuous-dynamic", action="store_true")
    p.add_argument("--measurement-canary-rate", type=float,
                   default=1.0 / 64.0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if a.warmup_n >= a.n_tasks:
        raise ValueError(
            f"warmup_n ({a.warmup_n}) must be < n_tasks ({a.n_tasks})"
        )
    warmup_indices = list(range(a.warmup_n))
    measurement_indices = list(range(a.warmup_n, a.n_tasks))

    from vllm import LLM
    log.info("loading %s tp=%d", a.model, a.tensor_parallel_size)
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

    # Phase 1: warmup with canary observing + tier_application on
    log.info(
        "=== phase 1: warmup (n=%d tasks, observability=%.2f) ===",
        len(warmup_indices), a.observability_rate,
    )
    warmup_handle = TPCanaryHandle(
        llm,
        harness_kwargs={
            "canary_rate": a.observability_rate,
            "tau": a.tau,
            "tick_interval": a.tick_interval,
            "tier_application": True,
            "tier_application_quant_v": True,
            "freeze_controller": False,
            "multi_precision": a.multi_precision,
            "tau_budget": a.tau_budget,
            "absolute_canary": a.absolute_canary,
            "kref_window_size": a.kref_window_size,
            "seed": a.seed,
        },
    )
    warmup_handle.install()
    try:
        warmup_result = run_tasks_in_range(
            llm,
            label="echokv_warmup_taubench",
            task_indices=warmup_indices,
            max_turns=a.max_turns,
            max_new_tokens=a.max_new_tokens,
            seed=a.seed,
            env_name=a.env,
        )
        warmup_snapshots = warmup_handle.snapshot()
    finally:
        warmup_handle.remove()

    # Phase 2: consolidate
    consolidated = consolidate_tiers(warmup_snapshots)
    consolidated_delta = consolidate_deltas(warmup_snapshots)
    proj = memory_projection(consolidated)
    log.info("    consolidated tier vector: %s", consolidated)
    log.info("    memory projection: avg_bits=%.2f tier_counts=%s",
             proj["avg_bits"], proj["tier_counts"])

    # Phase 3: measurement
    if a.continuous_dynamic:
        meas_canary_rate = a.measurement_canary_rate
        meas_freeze = False
        protocol_label = "continuous-dynamic"
    else:
        meas_canary_rate = 1e-6
        meas_freeze = True
        protocol_label = "frozen-tiers"

    log.info(
        "=== phase 3: measurement (n=%d, %s; canary_rate=%.4g freeze=%s) ===",
        len(measurement_indices), protocol_label, meas_canary_rate, meas_freeze,
    )
    measure_handle = TPCanaryHandle(
        llm,
        harness_kwargs={
            "canary_rate": meas_canary_rate,
            "tau": a.tau,
            "tick_interval": a.tick_interval,
            "tier_application": True,
            "tier_application_quant_v": True,
            "freeze_controller": meas_freeze,
            "multi_precision": a.multi_precision,
            "tau_budget": a.tau_budget,
            "absolute_canary": a.absolute_canary,
            "kref_window_size": a.kref_window_size,
            "seed": a.seed + 1,
        },
    )
    measure_handle.install()
    try:
        measure_handle.seed_tiers(consolidated)
        seeded_snaps = measure_handle.snapshot()
        for ri, s in enumerate(seeded_snaps):
            if s["tier_vector"] != consolidated:
                raise RuntimeError(
                    f"rank {ri} did not pick up consolidated tiers"
                )
        log.info("    seeded tiers verified on all %d ranks",
                 len(seeded_snaps))
        measure_result = run_tasks_in_range(
            llm,
            label="echokv_3c_taubench",
            task_indices=measurement_indices,
            max_turns=a.max_turns,
            max_new_tokens=a.max_new_tokens,
            seed=a.seed + 1,
            env_name=a.env,
        )
        measure_snapshots = measure_handle.snapshot()
    finally:
        measure_handle.remove()

    summary = {
        "session": "20260506_15_70b_taubench_echokv_3c",
        "model": a.model,
        "tensor_parallel_size": a.tensor_parallel_size,
        "n_tasks": a.n_tasks,
        "warmup_n": a.warmup_n,
        "max_turns": a.max_turns,
        "max_new_tokens": a.max_new_tokens,
        "max_num_seqs": a.max_num_seqs,
        "phase": "measurement_complete",
        "measurement_protocol": protocol_label,
        "measurement_canary_rate": meas_canary_rate,
        "measurement_freeze_controller": meas_freeze,
        "tau_budget": a.tau_budget,
        "warmup_tiers_per_rank": [
            s["tier_vector"] for s in warmup_snapshots
        ],
        "warmup_deltas_per_rank": [
            s["per_layer_delta"] for s in warmup_snapshots
        ],
        "warmup_per_tier_delta_per_rank": [
            s.get("per_tier_delta") for s in warmup_snapshots
        ],
        "consolidated_tiers": consolidated,
        "consolidated_delta": consolidated_delta,
        "final_tiers_per_rank": [
            s["tier_vector"] for s in measure_snapshots
        ],
        "memory_projection": proj,
        "warmup_run": asdict(warmup_result),
        "measurement": {
            **asdict(measure_result),
            "config": "echokv_3c",
        },
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=2))
    log.info("wrote %s", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
