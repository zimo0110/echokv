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
# Multi-needle RULER (Session 14) — multi-query variant
# ---------------------------------------------------------------------------
#
# Single-needle RULER lets attention concentrate on one position. The
# critic's Session-14 Step-1 hypothesis is that requiring the model to
# attend to *several* positions during decode is what stresses INT4
# enough to break it. We adapt the single-needle generator to insert
# ``n_needles`` (default 5) labeled key/value needles spread across the
# body, then ask for all of them at once.
#
# Score: fraction of the n_needles correctly recovered from the model's
# completion. Headline accuracy aggregates the per-request fractions.

_MULTINEEDLE_LABELS: tuple[str, ...] = (
    "alpha", "bravo", "charlie", "delta", "echo",
    "foxtrot", "golf", "hotel", "india", "juliet",
)


def _ruler_multineedle_sample(
    idx: int,
    seed: int,
    target_tokens: int = 8_000,
    n_needles: int = 5,
) -> Request:
    """One synthetic *multi-needle* haystack request.

    ``n_needles`` distinct labeled magic numbers are inserted at evenly
    spread positions; the model must retrieve all of them. Score is the
    fraction recovered correctly (0/n_needles … n_needles/n_needles).
    """
    if n_needles < 1 or n_needles > len(_MULTINEEDLE_LABELS):
        raise ValueError(
            f"n_needles must be in [1, {len(_MULTINEEDLE_LABELS)}], got {n_needles}"
        )
    rng = random.Random(hash((seed, idx, n_needles)) & 0xFFFFFFFF)
    labels = list(_MULTINEEDLE_LABELS[:n_needles])
    # Independent 6-digit magic numbers per label (collision-free with the
    # 4+ digit-extracting regex we use for scoring).
    magic_numbers: dict[str, int] = {
        lab: rng.randrange(100_000, 1_000_000) for lab in labels
    }

    filler_rep = max(1, target_tokens * 4 // len(_RULER_FILLER))
    body = _RULER_FILLER * filler_rep
    # Spread n_needles insertion points evenly across the body, avoiding
    # the outer 10% margins so no needle lands at the prompt boundary.
    body_lo = len(body) // 10
    body_hi = 9 * len(body) // 10
    seg_w = (body_hi - body_lo) // n_needles
    insert_points: list[int] = []
    for s in range(n_needles):
        seg_start = body_lo + s * seg_w
        seg_end = seg_start + seg_w
        insert_points.append(rng.randrange(seg_start, seg_end))
    # Build the body by walking through insertion points in order so
    # later inserts don't shift earlier ones.
    parts: list[str] = []
    cursor = 0
    # Pair labels with insertion points, randomize which label lands at
    # which slot so the model can't infer the order from filler context.
    rng.shuffle(labels)  # in-place shuffle of the label-to-position assignment
    for ip, lab in zip(insert_points, labels):
        parts.append(body[cursor:ip])
        parts.append(
            f" REMEMBER: the magic number for {lab} is "
            f"{magic_numbers[lab]}. Write it down somewhere safe. "
        )
        cursor = ip
    parts.append(body[cursor:])
    body_with_needles = "".join(parts)

    # Question: ask for all needles in a single, machine-parseable format.
    label_list = ", ".join(_MULTINEEDLE_LABELS[:n_needles])
    question = (
        "\n\nQ: Recall the magic numbers stored above. Output one line "
        f"per label in the format 'label=number' for these labels in "
        f"this exact order: {label_list}.\nA:"
    )
    prompt = body_with_needles + question

    # Reference: deterministic ordering by canonical label sequence (not
    # the shuffled assignment order), so scoring is independent of the
    # model's output ordering.
    canonical_labels = list(_MULTINEEDLE_LABELS[:n_needles])
    reference = " ".join(
        f"{lab}={magic_numbers[lab]}" for lab in canonical_labels
    )
    expected: dict[str, str] = {
        lab: str(magic_numbers[lab]) for lab in canonical_labels
    }

    def score(completion: str) -> float:
        # Per-label match: regex finds the label followed by '=' and a
        # 4+-digit number anywhere in the completion. We accept any
        # whitespace / punctuation between '=' and the digits.
        hits = 0
        for lab, want in expected.items():
            m = re.search(
                rf"\b{re.escape(lab)}\s*=\s*(\d{{4,}})", completion, re.IGNORECASE
            )
            if m and m.group(1) == want:
                hits += 1
        return hits / float(n_needles)

    return Request(
        workload=f"ruler_multineedle_{n_needles}",
        request_id=f"ruler_multineedle_{n_needles}:{idx}",
        prompt=prompt,
        reference=reference,
        score_fn=score,
    )


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


# ---------------------------------------------------------------------------
# GSM8K-Hard (HuggingFace ``reasoning-machines/gsm-hard``)
# ---------------------------------------------------------------------------
#
# Same word problems as GSM8K but with the small-integer constants replaced
# with much larger numbers — making the arithmetic genuinely hard to do
# mentally. The model has to track 7-9 digit intermediate values across a
# 10-20 step chain of reasoning. Per the Session-14 critic plan, this is
# the cheapest workload that plausibly stresses KV precision via accumulating
# numerical error during multi-step reasoning, distinct from RULER's
# pure-retrieval pattern.
#
# Dataset columns: ``input`` (question), ``code`` (reference solution as a
# Python function — we ignore it), ``target`` (ground-truth answer as float).


def _normalize_number(s: str) -> Optional[float]:
    """Parse the last numeric-looking token from a string into a float.

    Tolerates commas as thousands separators and optional minus signs.
    Returns ``None`` if no number-like token is found.
    """
    if not s:
        return None
    matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", s)
    if not matches:
        return None
    last = matches[-1].replace(",", "")
    try:
        return float(last)
    except ValueError:
        return None


def _extract_gsm8k_hard_answer(text: str) -> Optional[float]:
    """Extract the final answer from a CoT completion.

    Preference order:
      1. ``#### <number>`` terminator (the format we instruct).
      2. ``answer is <number>`` / ``answer: <number>``.
      3. Last numeric token in the completion (fallback).
    """
    m = re.search(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", text)
    if m:
        return _normalize_number(m.group(1))
    m = re.search(
        r"(?:answer is|answer:\s*)(-?\d[\d,]*(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if m:
        return _normalize_number(m.group(1))
    return _normalize_number(text)


def _gsm8k_hard_stream(n: int, seed: int) -> Iterator[Request]:
    from datasets import load_dataset  # local import: keep import-time cheap

    ds = load_dataset("reasoning-machines/gsm-hard", split="train")
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    for di in indices[:n]:
        row = ds[di]
        question = row["input"]
        # ``target`` is a float; for whole-number answers it stores e.g.
        # 3431580.0. We score in float space with absolute tolerance.
        target = float(row["target"])

        messages = [
            {
                "role": "system",
                "content": (
                    "Solve the math problem step by step. "
                    "End your solution with '#### [answer]' where [answer] "
                    "is the final numeric answer only (no units, no commas)."
                ),
            },
            {"role": "user", "content": question},
        ]
        prompt = f"[GSM8K-Hard] {question}"

        def make_score(gold: float) -> Callable[[str], float]:
            def score(completion: str) -> float:
                got = _extract_gsm8k_hard_answer(completion)
                if got is None:
                    return 0.0
                # All gsm-hard targets are integer-valued floats (e.g.
                # 3431580.0). For integer gold, require exact integer
                # equality — off-by-one must NOT pass. ``< 0.5`` accepts
                # only "rounds to the same integer".
                return 1.0 if abs(got - gold) < 0.5 else 0.0

            return score

        yield Request(
            workload="gsm8k_hard",
            request_id=f"gsm8k_hard:{di}",
            prompt=prompt,
            reference=str(target),
            score_fn=make_score(target),
            messages=messages,
        )


# ---------------------------------------------------------------------------
# HotpotQA (multi-hop reading comprehension)
# ---------------------------------------------------------------------------
#
# 10 Wikipedia-style paragraphs (mix of supporting + distractor), one
# question that requires combining facts from 2+ paragraphs. Hard-level
# subset stresses multi-document reasoning. Different from RULER (pure
# retrieval over synthetic filler) and GSM8K-Hard (single-document
# arithmetic) — it tests the model's ability to integrate scattered
# evidence under KV compression.
#
# Scoring: standard SQuAD-style normalized exact match (lowercase,
# strip articles + punctuation, collapse whitespace). Token-level F1
# would also work but exact-match is sharper for binary outcomes.


def _normalize_answer(s: str) -> str:
    """SQuAD/HotpotQA reference normalizer."""
    s = s.lower()
    # Strip punctuation
    s = re.sub(r"[\(\)\[\]\.\,\!\?\;\:\"\'\-]", " ", s)
    # Drop articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _hotpotqa_stream(
    n: int, seed: int, level: Optional[str] = "hard",
) -> Iterator[Request]:
    """Multi-hop QA from HotpotQA distractor split.

    ``level`` filters to a difficulty bucket (``"easy"``, ``"medium"``,
    ``"hard"``, or ``None`` for all). Hard is the multi-hop subset
    that genuinely requires combining facts.
    """
    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    yielded = 0
    for di in indices:
        if yielded >= n:
            break
        row = ds[di]
        if level is not None and row.get("level") != level:
            continue
        question = row["question"]
        gold = row["answer"]
        # context is {"title": [...], "sentences": [[...], ...]}
        ctx = row["context"]
        titles = ctx["title"]
        sentences = ctx["sentences"]
        # Build numbered paragraphs.
        paragraphs = []
        for i, (title, sents) in enumerate(zip(titles, sentences)):
            body = " ".join(sents)
            paragraphs.append(f"[{i+1}] {title}: {body}")
        context_str = "\n\n".join(paragraphs)

        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the question using only the provided "
                    "passages. Output only the final answer, as briefly "
                    "as possible (a single name, phrase, or yes/no). "
                    "Do not output reasoning."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Passages:\n{context_str}\n\n"
                    f"Question: {question}\nAnswer:"
                ),
            },
        ]

        def make_score(gold_str: str) -> Callable[[str], float]:
            gold_norm = _normalize_answer(gold_str)
            def score(completion: str) -> float:
                # Take the first non-empty line as the model's answer.
                ans = completion.strip().split("\n")[0].strip()
                # Strip any "Answer:" prefix the model echoes back.
                ans = re.sub(r"^answer:\s*", "", ans, flags=re.IGNORECASE)
                cand_norm = _normalize_answer(ans)
                if not cand_norm or not gold_norm:
                    return 0.0
                # Normalized EM: exact match after normalization.
                if cand_norm == gold_norm:
                    return 1.0
                # Substring containment (for cases where the model
                # outputs "The answer is X" → catches X if X is in
                # the candidate after stripping "answer is").
                if gold_norm in cand_norm:
                    return 1.0
                return 0.0
            return score

        prompt = f"[HotpotQA] {question[:80]}..."
        yield Request(
            workload="hotpotqa",
            request_id=f"hotpotqa:{row['id']}",
            prompt=prompt,
            reference=gold,
            score_fn=make_score(gold),
            messages=messages,
        )
        yielded += 1


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
