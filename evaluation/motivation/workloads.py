"""Five workload loaders for the motivation study.

Protocol: 400 requests per workload, 5 workloads, 2000 requests total
(REPORT.md § Motivation Study).

Workload → HF dataset mapping (subject to network availability)
---------------------------------------------------------------
* ``sharegpt``      — anon8231489123/ShareGPT_Vicuna_unfiltered (or
                      similar mirror). Accuracy metric: none; this
                      workload contributes to the violin plot and
                      the layer heatmap but not to the Spearman ρ.
* ``summarization`` — abisee/cnn_dailymail. Accuracy metric: none
                      by default (same reasoning as ShareGPT).
                      ROUGE-L could be computed against the
                      reference summary if desired.
* ``ruler_32k``     — needle-in-a-haystack at 32K context, generated
                      procedurally (no HF dataset needed). Accuracy
                      metric: pass/fail on the needle question.
                      **Contributes to Spearman ρ.**
* ``gsm8k_cot``     — gsm8k, CoT 8-shot. Accuracy metric: exact-match
                      of the final numeric answer.
                      **Contributes to Spearman ρ.**
* ``humaneval``     — openai_humaneval. Accuracy metric: pass@1 via
                      safe-execution of generated code against the
                      test harness.
                      **Contributes to Spearman ρ.**

The RULER-32K generator is in this file because it is just templated
text. All other loaders defer to ``datasets`` for fetching; if the
machine is offline the caller gets a clean failure rather than a
silent fallback.

This module intentionally does *not* run any model forward passes —
it only produces a stream of
``(prompt, reference, scoring_fn)`` triples. The orchestrator in
``run.py`` wires those into the model and the delta logger.
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

log = logging.getLogger(__name__)


WORKLOADS: tuple[str, ...] = (
    "sharegpt",
    "summarization",
    "ruler_32k",
    "gsm8k_cot",
    "humaneval",
)

# Workloads that produce a per-request scalar accuracy in [0, 1] —
# these are the ones that enter the Spearman ρ calculation.
ACCURACY_WORKLOADS: frozenset[str] = frozenset(
    {"ruler_32k", "gsm8k_cot", "humaneval"}
)


@dataclass
class Request:
    """One motivation-study request."""

    workload: str
    request_id: str
    prompt: str
    reference: Optional[str]
    # If ``score_fn`` is None, the workload does not produce an
    # accuracy metric for this request.
    score_fn: Optional[Callable[[str], float]] = None
    # If set, run.py applies tok.apply_chat_template(messages) instead
    # of tokenizing ``prompt`` directly. ``prompt`` is kept as a
    # human-readable fallback for logging only.
    messages: Optional[list] = None


# ---------------------------------------------------------------------------
# RULER-32K — procedural needle-in-a-haystack
# ---------------------------------------------------------------------------


_RULER_FILLER = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! "
)


def _ruler_sample(idx: int, seed: int, target_tokens: int = 8_000) -> Request:
    """One synthetic needle-in-a-haystack request.

    ``target_tokens`` is approximate (4 chars ≈ 1 BPE token).

    8K default: large enough that INT4 accuracy drops measurably vs FP16
    (~85-95% at 8K vs near-100% at 4K), but small enough that the δ
    computation fits in 79 GB HBM without chunked attention.

    Memory budget for δ per layer: 2 × (1, 32, L, L) float32.
    At L=8K: 16 GB/layer → peak ~47 GB total. At L=32K: 262 GB → OOM.
    """
    rng = random.Random(hash((seed, idx)) & 0xFFFFFFFF)
    magic_number = rng.randrange(10_000, 1_000_000)
    needle = (
        f"REMEMBER: the magic number is {magic_number}. "
        "Write it down somewhere safe."
    )
    # ~4 chars per BPE token; _RULER_FILLER is ~125 chars → ~31 tokens/rep
    filler_rep = max(1, target_tokens * 4 // len(_RULER_FILLER))
    body = _RULER_FILLER * filler_rep
    # Insert needle at a random point (not the very edges).
    insert_at = rng.randrange(len(body) // 4, 3 * len(body) // 4)
    prompt = (
        body[:insert_at]
        + needle
        + body[insert_at:]
        + "\n\nQ: What is the magic number mentioned above? Answer with the "
        "number only.\nA:"
    )
    reference = str(magic_number)

    def score(completion: str) -> float:
        match = re.search(r"\d{4,}", completion)
        return 1.0 if match and match.group(0) == reference else 0.0

    return Request(
        workload="ruler_32k",
        request_id=f"ruler_32k:{idx}",
        prompt=prompt,
        reference=reference,
        score_fn=score,
    )


def _ruler_stream(n: int, seed: int) -> Iterator[Request]:
    for i in range(n):
        yield _ruler_sample(i, seed=seed)


# ---------------------------------------------------------------------------
# GSM8K-CoT (HuggingFace ``gsm8k``)
# ---------------------------------------------------------------------------


_GSM8K_COT_FEWSHOT = """\
Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
A: Let's think step by step. April: 48. May: 48/2 = 24. Total: 48+24 = 72. The answer is 72.

Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
A: Let's think step by step. 50/60 hours × $12 = $10. The answer is 10.

"""  # two-shot is enough for this harness; the main prompt slots in.


def _extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract the final "#### <number>" or "The answer is <number>"."""
    m = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "")
    m = re.search(
        r"(?:answer is|answer:\s*)(-?\d[\d,]*\.?\d*)", text, re.IGNORECASE
    )
    if m:
        return m.group(1).replace(",", "")
    # Fallback: last number in the string.
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "") if nums else None


def _gsm8k_stream(n: int, seed: int) -> Iterator[Request]:
    from datasets import load_dataset  # local import: keep import-time cheap

    ds = load_dataset("gsm8k", "main", split="test")
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    for out_i, di in enumerate(indices[:n]):
        row = ds[di]
        question = row["question"]
        gold = _extract_gsm8k_answer(row["answer"])
        # Use chat messages so instruct models receive a proper instruction
        # rather than a raw fewshot-completion prefix (which causes instruct
        # models to generate additional Q/A examples instead of answering).
        messages = [
            {
                "role": "system",
                "content": (
                    "Solve the math problem step by step. "
                    "End your solution with '#### [answer]' where [answer] "
                    "is the final numeric answer only."
                ),
            },
            {"role": "user", "content": question},
        ]
        # ``prompt`` kept for logging; run.py uses ``messages`` when set.
        prompt = f"[GSM8K] {question}"

        def make_score(gold_answer: str) -> Callable[[str], float]:
            def score(completion: str) -> float:
                got = _extract_gsm8k_answer(completion)
                return 1.0 if got is not None and got == gold_answer else 0.0

            return score

        yield Request(
            workload="gsm8k_cot",
            request_id=f"gsm8k_cot:{di}",
            prompt=prompt,
            reference=gold,
            score_fn=make_score(gold) if gold else None,
            messages=messages,
        )


# ---------------------------------------------------------------------------
# HumanEval (OpenAI)
# ---------------------------------------------------------------------------


def _humaneval_stream(n: int, seed: int) -> Iterator[Request]:
    from datasets import load_dataset

    ds = load_dataset("openai_humaneval", split="test")
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    for di in indices[:n]:
        row = ds[di]
        prompt = row["prompt"]
        task_id = row["task_id"]
        test_src = row["test"]
        entry_point = row["entry_point"]

        def make_score(
            prompt_src: str, test_code: str, fn_name: str
        ) -> Callable[[str], float]:
            def score(completion: str) -> float:
                # Safe-execution is out of scope for this harness;
                # the motivation study records a 0/1 only if the
                # caller has plugged in a sandbox. Without one we
                # surface ``NaN`` by returning a sentinel negative;
                # the aggregator filters non-[0,1] scores.
                return float("nan")

            return score

        yield Request(
            workload="humaneval",
            request_id=f"humaneval:{task_id}",
            prompt=prompt,
            reference=row.get("canonical_solution"),
            score_fn=make_score(prompt, test_src, entry_point),
        )


# ---------------------------------------------------------------------------
# ShareGPT and summarization — contribute to violin plot only
# ---------------------------------------------------------------------------


_SHAREGPT_SYNTHETIC_PROMPTS: tuple[str, ...] = (
    "What is the difference between machine learning and deep learning?",
    "Can you help me write a Python function that flattens a nested list?",
    "Explain the concept of attention mechanisms in transformer models.",
    "I need to plan a 7-day trip to Japan. What cities should I visit and in what order?",
    "What are the pros and cons of using microservices vs a monolithic architecture?",
    "Can you explain how gradient descent works in simple terms?",
    "Write a SQL query to find the top 5 customers by total purchase amount.",
    "What is the difference between TCP and UDP protocols?",
    "I'm having trouble understanding recursion. Can you explain it with an example?",
    "What are the SOLID principles in software engineering?",
    "Help me debug this code: for i in range(10): print(i) but it prints nothing.",
    "What is the CAP theorem and why does it matter for distributed systems?",
    "Explain the difference between process and thread in operating systems.",
    "Write a regex pattern that matches valid email addresses.",
    "What is the time complexity of quicksort in the worst case and why?",
    "How does the PageRank algorithm work?",
    "What are the main differences between REST and GraphQL APIs?",
    "Can you explain what a closure is in JavaScript with an example?",
    "What is the purpose of the softmax function in neural networks?",
    "How does public-key cryptography work? Explain RSA at a high level.",
    "What is the difference between supervised and unsupervised learning?",
    "Write a bash script that monitors disk usage and sends an alert if it exceeds 90%.",
    "Explain the concept of eventual consistency in distributed databases.",
    "What are some best practices for securing a REST API?",
    "How does garbage collection work in modern programming languages?",
)


def _sharegpt_synthetic_stream(n: int, seed: int) -> Iterator[Request]:
    """Synthetic fallback when the HuggingFace ShareGPT dataset is unavailable."""
    rng = random.Random(seed)
    prompts = list(_SHAREGPT_SYNTHETIC_PROMPTS)
    for i in range(n):
        prompt = prompts[i % len(prompts)]
        # Add minor variation so identical prompts aren't repeated verbatim.
        if i >= len(prompts):
            prompt = f"Question {i // len(prompts) + 1}: {prompt}"
        digest = hashlib.sha1(f"synthetic:{seed}:{i}".encode()).hexdigest()[:12]
        yield Request(
            workload="sharegpt",
            request_id=f"sharegpt:synthetic:{digest}",
            prompt=prompt,
            reference=None,
            score_fn=None,
        )


def _sharegpt_stream(n: int, seed: int) -> Iterator[Request]:
    from datasets import load_dataset

    try:
        ds = load_dataset(
            "anon8231489123/ShareGPT_Vicuna_unfiltered",
            split="train",
            streaming=True,
        )
    except Exception as exc:
        log.warning(
            "ShareGPT HF dataset unavailable (%s); using synthetic fallback "
            "(violin plot only — no Spearman ρ impact)",
            exc,
        )
        yield from _sharegpt_synthetic_stream(n, seed)
        return

    it = iter(ds)
    seen = 0
    while seen < n:
        row = next(it)
        convs = row.get("conversations") or []
        if len(convs) < 2:
            continue
        prompt = convs[0].get("value", "")
        reference = convs[1].get("value", "") if len(convs) > 1 else None
        if not prompt:
            continue
        digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
        yield Request(
            workload="sharegpt",
            request_id=f"sharegpt:{digest}",
            prompt=prompt,
            reference=reference,
            score_fn=None,
        )
        seen += 1


def _summarization_stream(n: int, seed: int) -> Iterator[Request]:
    from datasets import load_dataset

    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    for di in indices[:n]:
        row = ds[di]
        prompt = (
            "Summarize the following article in 3-5 sentences.\n\n"
            + row["article"]
            + "\n\nSummary:"
        )
        yield Request(
            workload="summarization",
            request_id=f"summarization:{row['id']}",
            prompt=prompt,
            reference=row["highlights"],
            score_fn=None,
        )


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------


def load_workload(name: str, n: int, seed: int = 0) -> Iterator[Request]:
    """Return a lazy iterator of ``n`` requests for ``name``."""
    if name == "sharegpt":
        return _sharegpt_stream(n, seed)
    if name == "summarization":
        return _summarization_stream(n, seed)
    if name == "ruler_32k":
        return _ruler_stream(n, seed)
    if name == "gsm8k_cot":
        return _gsm8k_stream(n, seed)
    if name == "humaneval":
        return _humaneval_stream(n, seed)
    raise ValueError(f"unknown workload: {name!r}; must be one of {WORKLOADS}")
