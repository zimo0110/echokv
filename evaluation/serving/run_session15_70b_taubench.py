"""Session 15 / Phase 9 — Llama-3.1-70B + TP=8 on τ-Bench (airline).

Agentic / decode-heavy benchmark (PAPER_DRAFT §7.2). Each task is a
multi-turn airline customer-service interaction: the agent receives
the task instruction, calls airline tools (book_reservation,
cancel_reservation, ...) iteratively, and is scored on whether the
resulting database state matches the gold trajectory's database
state (env.calculate_reward).

Why "decode-heavy": each task produces 1-3K agent decode tokens
across multiple turns of thought + tool use. Compared to GSM8K-Hard
(~278 tokens) or HotpotQA (~5 tokens), τ-Bench exercises the
EchoKV closed-loop controller far more aggressively because the EMAs
get many more updates per task.

Usage:
    /opt/conda/envs/scalable/bin/python -m evaluation.serving.run_session15_70b_taubench \\
        --tensor-parallel-size 8 \\
        --n-tasks 50 \\
        --max-turns 15 \\
        --configs fp16 static_int4 static_int8 nvfp4 \\
        --out results/serving/session15_70b/taubench_airline.json

For the EchoKV-3C cell use --configs echokv_3c (also accepts
--continuous-dynamic for the live-controller protocol).

Notes
-----

*   We use the tau-bench airline env (50 test tasks). The user
    simulator is replaced with a deterministic stub (the agent
    always sees the task.instruction as the initial user turn; if
    the agent issues a "respond" action we treat it as terminal).
    The reward function (env.calculate_reward) is unchanged — it
    hashes the database state after the agent's actions and
    compares to the gold trajectory's hash, plus checks that any
    required output strings appear in the agent's respond actions.

*   Tool calling format: Llama-3.1's native tool-calling chat
    template works but is finicky to pickle through TP. We use a
    simpler ReAct-style format: the agent emits a JSON block with
    {name, kwargs} or {respond: text}. Parsing is regex + JSON.

*   For the EchoKV cell we run a 3-phase warmup → consolidate →
    measure protocol identical to the existing echokv drivers,
    sharing the same TPCanaryHandle plumbing.
"""
from __future__ import annotations

import argparse
import builtins
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

# Replace input() globally so tau-bench's HumanUserSimulationEnv
# doesn't block on stdin when we construct the env.
builtins.input = lambda *args, **kwargs: ""

# tau-bench imports — local clone install
from tau_bench.envs import get_env
from tau_bench.types import Action, RESPOND_ACTION_NAME

from evaluation.serving.run_session8 import RunResult


log = logging.getLogger("session15_70b_taubench")
DEFAULT_70B = "/mnt/nvme_tmp6/llama-70b"


SYSTEM_PROMPT_TEMPLATE = """You are a helpful airline customer-service agent.

# Critical operating rules

1. **Use IATA airport codes** (JFK, SEA, EWR, LAX, ORD, ...) in every
   tool call — never city names like "New York" or "Seattle". If the
   user gives a city, look it up first via ``list_all_airports``.
2. **Be persistent.** If a search returns no results, try different
   dates, alternative airports for the same city, or a one-stop
   flight. Do not give up after one failed search.
3. **Read the user's request carefully** for time / cabin / passenger
   constraints and apply them; many tasks fail simply by ignoring
   "before 11 AM" or "economy" or "one stopover OK".
4. **Confirm before charging** — when booking or modifying a
   reservation, summarise the proposed change and the cost in your
   intermediate respond so the user can confirm.
5. **Respond exactly once at the end**, after all tool calls are
   complete, with the final answer / confirmation. The final
   respond MUST contain any specific strings the user asked for
   (prices, flight numbers, etc.) verbatim.

# Knowledge base (policies and rules):
{wiki}

# Available tools:
{tools_summary}

# Output format

Each turn: produce one short Thought line, then EXACTLY ONE JSON
object in a fenced ```json``` block. The JSON must be one of:

1. To call a tool (preferred — most turns):
```json
{{"name": "<tool_name>", "kwargs": {{ ... }}}}
```

2. To deliver the final answer (this ENDS the conversation):
```json
{{"respond": "<final answer text containing all required details>"}}
```

The JSON block must be the LAST thing in your response — no text
after the closing ```. Be concise.

# Worked example (one tool call followed by the final respond)

Thought: I need to find direct flights from JFK to LAX on May 20 in economy.
```json
{{"name": "search_direct_flight", "kwargs": {{"origin": "JFK", "destination": "LAX", "date": "2024-05-20"}}}}
```
[tool returns flight HAT123 economy $249]

Thought: One option found; the user asked for the cheapest. I'll book it.
```json
{{"name": "book_reservation", "kwargs": {{"user_id": "...", "origin": "JFK", "destination": "LAX", "flight_type": "one_way", "cabin": "economy", "flights": [{{"flight_number": "HAT123", "date": "2024-05-20"}}], "passengers": [...], "payment_methods": [...], "total_baggages": 0, "nonfree_baggages": 0, "insurance": "no"}}}}
```
[tool returns reservation_id R12345]

Thought: Booking confirmed.
```json
{{"respond": "Booking confirmed: reservation R12345, flight HAT123 JFK→LAX on May 20 economy, $249."}}
```
"""


# ---------------------------------------------------------------------------
# JSON-action parser
# ---------------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON = re.compile(r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})", re.DOTALL)


def parse_agent_action(text: str) -> Optional[Action]:
    """Extract a single JSON action object from the agent's reply.

    Supports either a fenced ``json`` block (preferred) or a bare
    JSON object as a fallback. Returns None if no parseable JSON.
    """
    candidates: list[str] = []
    m = _JSON_BLOCK.search(text)
    if m:
        candidates.append(m.group(1))
    # Fall back to last bare JSON object in the text
    bares = list(_BARE_JSON.finditer(text))
    if bares:
        candidates.append(bares[-1].group(1))
    for raw in candidates:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if "respond" in obj and isinstance(obj["respond"], str):
            return Action(name=RESPOND_ACTION_NAME,
                          kwargs={"content": obj["respond"]})
        if "name" in obj and "kwargs" in obj and isinstance(obj["kwargs"], dict):
            return Action(name=str(obj["name"]), kwargs=dict(obj["kwargs"]))
    return None


# ---------------------------------------------------------------------------
# Per-task runner
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    task_id: int
    reward: float                          # 0.0 or 1.0
    n_turns: int
    decode_tokens: int                     # total agent-side output tokens
    correct: int                           # 1 if reward >= 1.0 else 0
    agent_actions: list[dict[str, Any]] = field(default_factory=list)
    finished_reason: str = "ok"            # ok / max_turns / parse_fail / step_error


def run_one_task(
    llm: Any,
    env: Any,
    *,
    task_idx: int,
    max_turns: int,
    max_new_tokens: int,
    sampling_params: Any,
    tokenizer: Any,
) -> TaskResult:
    """Run a single τ-Bench task end to end.

    Constructs the chat history, loops up to ``max_turns`` agent
    turns, executes parsed tool calls via ``env.step``, and computes
    the final reward via ``env.calculate_reward()``.
    """
    env.reset(task_index=task_idx)

    # Build system prompt with the task's wiki + a short tool summary
    tools_summary_lines = []
    for ti in env.tools_info:
        fn = ti["function"]
        tools_summary_lines.append(
            f"- {fn['name']}: {fn.get('description', '').strip()}"
        )
    sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        wiki=env.wiki,
        tools_summary="\n".join(tools_summary_lines),
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": env.task.instruction},
    ]

    decode_tokens = 0
    n_turns = 0
    actions_log: list[dict[str, Any]] = []
    finished_reason = "ok"

    for _ in range(max_turns):
        n_turns += 1
        try:
            out = llm.chat([messages], sampling_params=sampling_params)[0]
        except Exception as exc:
            # Context overflow, scheduler error, etc. — fail this task
            # and keep the sweep going.
            log.warning("llm.chat raised on task %d turn %d: %s",
                        task_idx, n_turns, exc)
            finished_reason = "chat_error"
            break
        agent_text = out.outputs[0].text
        decode_tokens += len(out.outputs[0].token_ids)
        messages.append({"role": "assistant", "content": agent_text})

        action = parse_agent_action(agent_text)
        if action is None:
            finished_reason = "parse_fail"
            break
        actions_log.append({"name": action.name, "kwargs": action.kwargs})

        try:
            resp = env.step(action)
        except Exception as exc:
            log.warning("env.step raised on task %d: %s", task_idx, exc)
            finished_reason = "step_error"
            break

        if resp.done:
            break
        # Feed the tool result (or user response) back as the next user turn
        messages.append({
            "role": "user",
            "content": f"Tool result: {resp.observation}",
        })

        if action.name == RESPOND_ACTION_NAME:
            # In our simplified harness we treat the first respond action
            # as terminal — no second user turn from the simulator.
            break
    else:
        finished_reason = "max_turns"

    # Compute reward via env.calculate_reward (compares post-state hash)
    try:
        reward_res = env.calculate_reward()
        reward = float(reward_res.reward)
    except Exception as exc:
        log.warning("calculate_reward raised on task %d: %s", task_idx, exc)
        reward = 0.0

    return TaskResult(
        task_id=task_idx,
        reward=reward,
        n_turns=n_turns,
        decode_tokens=decode_tokens,
        correct=int(reward >= 1.0),
        agent_actions=actions_log,
        finished_reason=finished_reason,
    )


# ---------------------------------------------------------------------------
# Cell runner — runs all tasks for one quantization config
# ---------------------------------------------------------------------------

def run_cell(
    llm: Any,
    *,
    config_name: str,
    n_tasks: int,
    max_turns: int,
    max_new_tokens: int,
    seed: int,
    env_name: str = "airline",
) -> RunResult:
    """Run all ``n_tasks`` τ-Bench tasks under the currently installed
    quantization config. ``env_name`` is one of "airline" (50 tasks)
    or "retail" (115 tasks). Returns a RunResult shaped like the other
    workloads' cells so paired McNemar can compose blobs across cells.
    Per-task ``request_id`` is namespaced with the env so airline and
    retail blobs can be merged without ID collisions."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        seed=seed,
    )
    tokenizer = llm.get_tokenizer()

    env = get_env(
        env_name=env_name,
        user_strategy="human",      # stubbed via our input() override
        user_model="dummy",
        task_split="test",
        user_provider="openai",
        task_index=0,
    )
    n_tasks = min(n_tasks, len(env.tasks))

    t0 = time.perf_counter()
    per_request = []
    total_decode = 0
    total_turns = 0
    n_correct = 0
    finished_counts: dict[str, int] = {}

    for ti in range(n_tasks):
        try:
            tr = run_one_task(
                llm, env,
                task_idx=ti,
                max_turns=max_turns,
                max_new_tokens=max_new_tokens,
                sampling_params=sampling_params,
                tokenizer=tokenizer,
            )
        except Exception as exc:
            log.exception("task %d failed unexpectedly: %s", ti, exc)
            tr = TaskResult(
                task_id=ti, reward=0.0, n_turns=0, decode_tokens=0,
                correct=0, finished_reason="exception",
            )
        per_request.append({
            "request_id": f"taubench_{env_name}:{ti}",
            "correct": tr.correct,
            "reward": tr.reward,
            "n_turns": tr.n_turns,
            "decode_tokens": tr.decode_tokens,
            "finished_reason": tr.finished_reason,
            "actions": tr.agent_actions,
        })
        total_decode += tr.decode_tokens
        total_turns += tr.n_turns
        n_correct += tr.correct
        finished_counts[tr.finished_reason] = (
            finished_counts.get(tr.finished_reason, 0) + 1
        )
        log.info(
            "    [%s] task %d: reward=%.0f turns=%d decode_tok=%d (%s)",
            config_name, ti, tr.reward, tr.n_turns, tr.decode_tokens,
            tr.finished_reason,
        )

    wall = time.perf_counter() - t0
    accuracy = n_correct / max(n_tasks, 1)
    log.info(
        "    [%s] done: n=%d acc=%.3f total_decode=%d "
        "avg_turns=%.1f avg_decode=%.0f finished=%s wall=%.1fs",
        config_name, n_tasks, accuracy, total_decode,
        total_turns / max(n_tasks, 1),
        total_decode / max(n_tasks, 1),
        finished_counts, wall,
    )
    return RunResult(
        label=f"taubench_airline_{config_name}",
        wall_clock_s=wall,
        n_prompts=n_tasks,
        completed=n_tasks,
        n_correct=n_correct,
        accuracy=accuracy,
        p50_tpot_s=None,
        p99_tpot_s=None,
        mean_ttft_s=None,
        total_input_tokens=0,                     # not tracked per-turn here
        total_output_tokens=total_decode,
        output_throughput_tps=total_decode / max(wall, 1e-6),
        per_request=per_request,
    )


# ---------------------------------------------------------------------------
# Static-baseline hook installers
# ---------------------------------------------------------------------------

def install_static_int4(llm: Any) -> None:
    from evaluation.serving.static_int4_tp import install_static_int4_hooks
    counts = llm.apply_model(install_static_int4_hooks)
    log.info("static_int4: hooks per rank: %s", counts)


def install_static_int8(llm: Any) -> None:
    from evaluation.serving.static_int8_tp import install_static_int8_hooks
    counts = llm.apply_model(install_static_int8_hooks)
    log.info("static_int8: hooks per rank: %s", counts)


def install_nvfp4(llm: Any) -> None:
    from evaluation.serving.nvfp4_tp import install_nvfp4_hooks
    counts = llm.apply_model(install_nvfp4_hooks)
    log.info("nvfp4: hooks per rank: %s", counts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_70B)
    p.add_argument("--tensor-parallel-size", type=int, default=8)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-new-tokens", type=int, default=512,
                   help="per-turn cap; total decode is sum across turns")
    p.add_argument("--max-turns", type=int, default=15)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-num-seqs", type=int, default=8)
    p.add_argument("--n-tasks", type=int, default=50,
                   help="airline=50, retail=115 test tasks; "
                        "default 50 runs all of airline")
    p.add_argument("--env", default="airline",
                   choices=["airline", "retail"],
                   help="τ-Bench env (airline test = 50 tasks; "
                        "retail test = 115 tasks)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--configs", nargs="+",
        default=["fp16", "static_int4", "nvfp4"],
        choices=["fp16", "static_int4", "static_int8", "nvfp4"],
        help="hooks are sticky once installed; cannot mix in one process. "
             "EchoKV cells live in run_session15_70b_taubench_echokv.py.",
    )
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    from vllm import LLM
    log.info("loading %s tp=%d max_model_len=%d",
             a.model, a.tensor_parallel_size, a.max_model_len)
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

    cells: list[dict[str, Any]] = []
    int4_done = int8_done = nvfp4_done = False
    for cfg in a.configs:
        log.info("=== %s  n=%d ===", cfg, a.n_tasks)
        if cfg == "static_int4" and not int4_done:
            install_static_int4(llm); int4_done = True
        elif cfg == "static_int8" and not int8_done:
            install_static_int8(llm); int8_done = True
        elif cfg == "nvfp4" and not nvfp4_done:
            install_nvfp4(llm); nvfp4_done = True
        elif cfg == "fp16":
            pass  # no hooks
        elif cfg in ("static_int4", "static_int8", "nvfp4"):
            log.info("    %s hooks already installed", cfg)
        else:
            raise ValueError(f"unknown config: {cfg}")

        res = run_cell(
            llm,
            config_name=cfg,
            n_tasks=a.n_tasks,
            max_turns=a.max_turns,
            max_new_tokens=a.max_new_tokens,
            seed=a.seed,
            env_name=a.env,
        )
        cells.append({**asdict(res), "config": cfg})

        # Save partial after every cell
        a.out.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "session": "20260506_15_70b_taubench",
            "model": a.model,
            "tensor_parallel_size": a.tensor_parallel_size,
            "n_tasks": a.n_tasks,
            "max_turns": a.max_turns,
            "max_new_tokens": a.max_new_tokens,
            "max_num_seqs": a.max_num_seqs,
            "configs": a.configs,
            "cells": cells,
            "partial": cfg != a.configs[-1],
        }
        a.out.write_text(json.dumps(summary, indent=2))
        log.info("wrote %s", a.out)

    log.info("all done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
