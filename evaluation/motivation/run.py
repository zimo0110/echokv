"""Motivation-study orchestration.

Ties the pieces together:
    workloads.load_workload  ->  model forward  ->  delta.compute_request_delta
                                                ->  JSONL record

Two modes:
    ``--smoke`` — uses a small HF model that fits in <4 GB VRAM (or
        CPU). Verifies pipeline mechanics without consuming model
        weight bandwidth. Not suitable for the research gate.
    default   — target model (Llama-3.1-8B-Instruct by default).

Gate v5 design: dual-inference per request
------------------------------------------
Each request runs two ``model.generate()`` calls:

1. FP16 KV path (baseline) → ``task_score`` (acc_FP16)
2. INT4 KV path (quantized K/V via hook) → ``task_score_int4`` (acc_INT4)

The gate then correlates δ (KL divergence of attention weights) against
Δacc = acc_FP16 − acc_INT4. This is the principled measure the study
requires; correlating δ against acc_FP16 (v4 design) is structurally
flawed because the two quantities are independent.

Pre-validation
--------------
Before the full run we execute ``validate_signal`` on a small batch
(``--validate-batch-size``, default 20 GSM8K requests) to confirm that
INT4 KV inference actually degrades accuracy. If Δacc is zero across the
batch the full run is skipped and an error is returned — this catches
wrong hook placement, accidental no-op quantization, etc.

Data parallelism
----------------
``--num-gpus N`` forks N worker processes, each loading the model onto
``cuda:{rank}``. Requests are distributed round-robin. Workers write per-
rank JSONL shards which are merged in the main process.

Uses ``multiprocessing.get_context("fork")`` which is safe on Linux
provided CUDA is not initialised in the main process before the fork.
``_collect_requests`` materialises all workload Requests (dataset reads
only, no CUDA) before forking; workers then receive their shard list
directly.

Hook strategy (trace)
---------------------
For decoder-only Llama-family models we wrap each attention module's
``forward`` at the Python level. The wrapper captures Q/K/V from inside
the call so ``delta.compute_request_delta`` can run afterwards.

Hook strategy (INT4 KV simulation)
-----------------------------------
``INT4KVHook`` installs ``register_forward_hook`` on every ``k_proj``
and ``v_proj`` layer. The hook returns
``dequant(quant(output))`` — the quantisation error is injected at the
projection output, so FlashAttention and the KV-cache accumulate
quantised-dequantised values throughout the generation pass. This
simulates KIVI's INT4 KV cache without modifying the model weights.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import torch

from .delta import (
    AttentionTrace,
    DeltaRecord,
    DeltaStepRecord,
    compute_per_step_delta,
    compute_request_delta,
)
from .kivi import DEFAULT_BITS, DEFAULT_GROUP_SIZE, simulate_quantization
from .workloads import Request, WORKLOADS, load_workload


log = logging.getLogger("motivation.run")

_VALIDATION_SIGNAL_THRESHOLD = 0.01  # min mean Δacc to consider signal present


# ---------------------------------------------------------------------------
# Attention-trace hook (captures Q/K/V for δ computation)
# ---------------------------------------------------------------------------


@dataclass
class _HookState:
    """Holds one request's captured Q/K/V traces, one entry per layer."""

    traces: list[AttentionTrace] = field(default_factory=list)
    enabled: bool = False


class AttentionTraceHook:
    """Installs wrappers on every decoder layer's self-attention.

    Call :meth:`begin_request` before a model forward to clear the
    trace buffer, :meth:`end_request` to retrieve it. Install and
    remove with :meth:`install` / :meth:`remove`.
    """

    def __init__(self) -> None:
        self.state = _HookState()
        self._originals: list[tuple[torch.nn.Module, str, object]] = []

    def install(self, model: torch.nn.Module) -> int:
        """Install the wrappers. Returns the number of layers hooked."""
        hooked = 0
        for mod in _iter_attention_modules(model):
            q_proj = getattr(mod, "q_proj", None)
            k_proj = getattr(mod, "k_proj", None)
            v_proj = getattr(mod, "v_proj", None)
            if q_proj is None or k_proj is None or v_proj is None:
                continue
            hooked += self._install_on(mod, q_proj, k_proj, v_proj)
        return hooked

    def _install_on(
        self,
        attn_module: torch.nn.Module,
        q_proj: torch.nn.Module,
        k_proj: torch.nn.Module,
        v_proj: torch.nn.Module,
    ) -> int:
        trace_buf: dict[str, torch.Tensor] = {}
        state = self.state

        def snap(name: str):
            def hook(module: torch.nn.Module, inputs, output: torch.Tensor):
                if not state.enabled:
                    return
                trace_buf[name] = output.detach()
            return hook

        handles = [
            q_proj.register_forward_hook(snap("q")),
            k_proj.register_forward_hook(snap("k")),
            v_proj.register_forward_hook(snap("v")),
        ]

        cfg = getattr(attn_module, "config", None)
        num_heads = (
            getattr(attn_module, "num_heads", None)
            or getattr(cfg, "num_attention_heads", None)
        )
        num_kv_heads = (
            getattr(attn_module, "num_key_value_heads", None)
            or getattr(cfg, "num_key_value_heads", None)
            or num_heads
        )
        head_dim = (
            getattr(attn_module, "head_dim", None)
            or getattr(cfg, "head_dim", None)
        )
        if num_heads is None or head_dim is None:
            for h in handles:
                h.remove()
            return 0

        original_forward = attn_module.forward

        def wrapped_forward(*args, **kwargs):
            out = original_forward(*args, **kwargs)
            if state.enabled and {"q", "k", "v"} <= trace_buf.keys():
                q = _reshape_heads(trace_buf.pop("q"), num_heads, head_dim)
                k = _reshape_heads(
                    trace_buf.pop("k"), num_kv_heads or num_heads, head_dim
                )
                v = _reshape_heads(
                    trace_buf.pop("v"), num_kv_heads or num_heads, head_dim
                )
                if k.shape[1] != q.shape[1]:
                    rep = q.shape[1] // k.shape[1]
                    k = k.repeat_interleave(rep, dim=1)
                    v = v.repeat_interleave(rep, dim=1)
                state.traces.append(AttentionTrace(q=q, k=k, v=v))
            trace_buf.clear()
            return out

        attn_module.forward = wrapped_forward
        self._originals.append((attn_module, "forward", original_forward))
        self._originals.extend(
            (proj, "__hook_handle__", h)
            for proj, h in zip((q_proj, k_proj, v_proj), handles)
        )
        return 1

    def remove(self) -> None:
        for target, name, payload in self._originals:
            if name == "__hook_handle__":
                payload.remove()  # type: ignore[attr-defined]
            else:
                setattr(target, name, payload)
        self._originals.clear()

    def begin_request(self) -> None:
        self.state.traces.clear()
        self.state.enabled = True

    def end_request(self) -> list[AttentionTrace]:
        self.state.enabled = False
        out = list(self.state.traces)
        self.state.traces.clear()
        return out


# ---------------------------------------------------------------------------
# INT4 KV simulation hook (injects quantisation error into k_proj/v_proj)
# ---------------------------------------------------------------------------


class INT4KVHook:
    """Injects INT4 quantisation error into k_proj and v_proj outputs.

    Installs ``register_forward_hook`` on every k_proj and v_proj in the
    model. Each hook replaces the projection output with
    ``dequant(quant(output))`` so that FlashAttention and the KV-cache
    accumulate quantised-dequantised values throughout the generation pass.

    Usage::

        hook = INT4KVHook()
        hook.install(model)
        try:
            output = model.generate(...)
        finally:
            hook.remove()
    """

    def __init__(
        self,
        bits: int = DEFAULT_BITS,
        group_size: int = DEFAULT_GROUP_SIZE,
        k_only: bool = False,
    ) -> None:
        self.bits = bits
        self.group_size = group_size
        self.k_only = k_only
        self._handles: list = []

    def install(self, model: torch.nn.Module) -> int:
        """Install hooks. Returns number of attention modules hooked.

        If ``k_only`` is True, only k_proj is hooked — V is left unquantised.
        This matches δ's K-only scope (δ = KL(softmax(QK_int4) ‖ softmax(QK_fp16))
        is independent of V) and is used by the v7 gate to test whether δ
        predicts NLL degradation when both quantities measure the same thing.
        """
        hooked = 0
        for mod in _iter_attention_modules(model):
            k_proj = getattr(mod, "k_proj", None)
            v_proj = getattr(mod, "v_proj", None)
            if k_proj is None or v_proj is None:
                continue
            self._handles.append(k_proj.register_forward_hook(self._quant_hook))
            if not self.k_only:
                self._handles.append(v_proj.register_forward_hook(self._quant_hook))
            hooked += 1
        return hooked

    def _quant_hook(
        self, module: torch.nn.Module, inputs, output: torch.Tensor
    ) -> torch.Tensor:
        return simulate_quantization(
            output,
            bits=self.bits,
            group_size=self.group_size,
            group_axis=-1,
        )

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iter_attention_modules(model: torch.nn.Module) -> Iterable[torch.nn.Module]:
    for mod in model.modules():
        if type(mod).__name__.lower().endswith("attention"):
            yield mod


def _reshape_heads(
    projected: torch.Tensor, num_heads: int, head_dim: int
) -> torch.Tensor:
    """``(B, L, num_heads*head_dim) -> (B, num_heads, L, head_dim)``."""
    b, l, _ = projected.shape
    return projected.view(b, l, num_heads, head_dim).transpose(1, 2)


def _tokenize(tokenizer, request: Request):
    if request.messages is not None:
        formatted = tokenizer.apply_chat_template(
            request.messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return tokenizer(formatted, return_tensors="pt", truncation=False)
    return tokenizer(request.prompt, return_tensors="pt", truncation=False)


def _prompt_text(tokenizer, request: Request) -> str:
    """Return the raw prompt string (post chat-template if applicable)."""
    if request.messages is not None:
        return tokenizer.apply_chat_template(
            request.messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return request.prompt


def _compute_nll(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,      # (1, prompt_len)
    completion_ids: torch.Tensor,  # (1, comp_len)
    eos_token_id: Optional[int] = None,
) -> float:
    """Mean NLL of completion given prompt under the current model state.

    Uses teacher-forcing (single forward pass). Prompt positions are masked
    with label=-100 so the loss is computed only over completion tokens.
    Returns NaN if completion is empty.

    If ``eos_token_id`` is provided, completion positions strictly after the
    first EOS token are masked too. This matters for batched generation: when
    a sequence finishes early, the batched ``generate`` call pads its
    completion with repeated EOS/pad tokens up to the batch's longest output.
    Including those positions in the NLL average dominates the loss with
    "predict EOS again" entries, which drift differently under FP16 vs INT4
    and inflate apparent ΔNLL — the v3–v7 measurement contamination.
    """
    if completion_ids.shape[-1] == 0:
        return float("nan")
    full_ids = torch.cat([prompt_ids, completion_ids], dim=-1)
    labels = full_ids.clone()
    labels[:, : prompt_ids.shape[-1]] = -100  # mask prompt
    if eos_token_id is not None:
        comp_start = prompt_ids.shape[-1]
        for b in range(completion_ids.shape[0]):
            eos_positions = (completion_ids[b] == eos_token_id).nonzero(as_tuple=True)[0]
            if eos_positions.numel() > 0:
                first_eos = int(eos_positions[0].item())
                # Keep label at first EOS (predicting EOS is meaningful); mask after.
                labels[b, comp_start + first_eos + 1 :] = -100
    with torch.no_grad():
        out = model(input_ids=full_ids, labels=labels)
    return float(out.loss)


# ---------------------------------------------------------------------------
# Per-request run (dual-inference: FP16 + INT4 KV)
# ---------------------------------------------------------------------------


def run_request(
    model: torch.nn.Module,
    tokenizer,
    request: Request,
    hook: AttentionTraceHook,
    *,
    device: torch.device,
    max_new_tokens: int,
    quant_bits: int,
    quant_group_size: int,
    int4_k_only: bool = False,
) -> DeltaRecord:
    inputs = _tokenize(tokenizer, request)
    input_ids = inputs["input_ids"].to(device)

    # --- δ computation (single FP16 prefill forward) ---
    hook.begin_request()
    with torch.no_grad():
        _ = model(input_ids=input_ids, use_cache=False)
    traces = hook.end_request()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    deltas = compute_request_delta(
        traces,
        bits=quant_bits,
        group_size=quant_group_size,
    ).tolist()

    task_score_fp16: Optional[float] = None
    nll_fp16: Optional[float] = None
    nll_int4: Optional[float] = None
    completion_tokens = 0

    if max_new_tokens > 0:
        # Pass 1: FP16 generate — reference completion
        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        completion_ids = gen[0, input_ids.shape[-1]:].unsqueeze(0)
        completion_tokens = int(completion_ids.shape[-1])
        if request.score_fn is not None:
            completion_text = tokenizer.decode(
                completion_ids[0], skip_special_tokens=True
            )
            try:
                raw = request.score_fn(completion_text)
                task_score_fp16 = float(raw) if raw == raw else None
            except Exception as e:
                log.warning("score_fn failed for %s: %s", request.request_id, e)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Pass 2: FP16 NLL (teacher-forcing, no hook)
        nll_fp16 = _compute_nll(
            model, input_ids, completion_ids, eos_token_id=tokenizer.eos_token_id
        )

        # Pass 3: INT4 NLL (teacher-forcing with INT4KVHook)
        int4_hook = INT4KVHook(
            bits=quant_bits, group_size=quant_group_size, k_only=int4_k_only
        )
        int4_hook.install(model)
        try:
            nll_int4 = _compute_nll(
                model, input_ids, completion_ids, eos_token_id=tokenizer.eos_token_id
            )
        finally:
            int4_hook.remove()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return DeltaRecord(
        request_id=request.request_id,
        workload=request.workload,
        num_layers=len(deltas),
        delta=deltas,
        task_score=task_score_fp16,
        nll_fp16=nll_fp16,
        nll_int4=nll_int4,
        prompt_tokens=int(input_ids.shape[-1]),
        completion_tokens=completion_tokens,
    )


# ---------------------------------------------------------------------------
# Batched run — amortises model-weight reads across N sequences
# ---------------------------------------------------------------------------


def run_batch(
    model: torch.nn.Module,
    tokenizer,
    requests: List[Request],
    hook: AttentionTraceHook,
    *,
    device: torch.device,
    max_new_tokens: int,
    quant_bits: int,
    quant_group_size: int,
    int4_k_only: bool = False,
) -> List[DeltaRecord]:
    """Process a batch of requests together.

    Delta computation runs sequentially (one prefill per request) so that
    each request's attention traces stay clean. The two expensive
    ``model.generate()`` calls (FP16 and INT4) are batched together using
    left-padded inputs, which amortises the model-weight memory reads and
    dramatically improves GPU utilisation vs batch_size=1.
    """
    # --- per-request delta (sequential prefills, no generate) ---
    deltas_list: list[list[float]] = []
    input_lengths: list[int] = []
    for req in requests:
        enc = _tokenize(tokenizer, req)
        ids = enc["input_ids"].to(device)
        input_lengths.append(int(ids.shape[-1]))
        hook.begin_request()
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
        traces = hook.end_request()
        deltas_list.append(
            compute_request_delta(
                traces, bits=quant_bits, group_size=quant_group_size
            ).tolist()
        )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    task_scores_fp16: list[Optional[float]] = [None] * len(requests)
    nll_fp16_list: list[Optional[float]] = [None] * len(requests)
    nll_int4_list: list[Optional[float]] = [None] * len(requests)
    completion_tokens_list = [0] * len(requests)
    # completion_ids_list: unpadded (1, comp_len) tensors, one per request
    completion_ids_list: list[Optional[torch.Tensor]] = [None] * len(requests)

    if max_new_tokens > 0:
        # Left-padded batch tokenisation for generate
        texts = [_prompt_text(tokenizer, req) for req in requests]
        enc_batch = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=False
        ).to(device)
        ids_batch = enc_batch["input_ids"]
        mask_batch = enc_batch["attention_mask"]
        padded_len = ids_batch.shape[-1]

        # Pass 1: FP16 batch generate — reference completions
        with torch.no_grad():
            gen_fp16 = model.generate(
                input_ids=ids_batch,
                attention_mask=mask_batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        for i, req in enumerate(requests):
            comp_ids = gen_fp16[i, padded_len:].unsqueeze(0)  # (1, comp_len)
            completion_tokens_list[i] = int(comp_ids.shape[-1])
            completion_ids_list[i] = comp_ids
            if req.score_fn is not None:
                comp_text = tokenizer.decode(comp_ids[0], skip_special_tokens=True)
                try:
                    raw = req.score_fn(comp_text)
                    task_scores_fp16[i] = float(raw) if raw == raw else None
                except Exception as e:
                    log.warning("score_fn failed for %s: %s", req.request_id, e)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        eos_id = tokenizer.eos_token_id

        # Pass 2: FP16 NLL — teacher-forcing, no hook
        for i in range(len(requests)):
            if completion_ids_list[i] is None:
                continue
            prompt_ids = _tokenize(tokenizer, requests[i])["input_ids"].to(device)
            nll_fp16_list[i] = _compute_nll(
                model, prompt_ids, completion_ids_list[i], eos_token_id=eos_id
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Pass 3: INT4 NLL — teacher-forcing with INT4KVHook
        int4_hook = INT4KVHook(
            bits=quant_bits, group_size=quant_group_size, k_only=int4_k_only
        )
        int4_hook.install(model)
        try:
            for i in range(len(requests)):
                if completion_ids_list[i] is None:
                    continue
                prompt_ids = _tokenize(tokenizer, requests[i])["input_ids"].to(device)
                nll_int4_list[i] = _compute_nll(
                    model, prompt_ids, completion_ids_list[i], eos_token_id=eos_id
                )
        finally:
            int4_hook.remove()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return [
        DeltaRecord(
            request_id=req.request_id,
            workload=req.workload,
            num_layers=len(deltas_list[i]),
            delta=deltas_list[i],
            task_score=task_scores_fp16[i],
            nll_fp16=nll_fp16_list[i],
            nll_int4=nll_int4_list[i],
            prompt_tokens=input_lengths[i],
            completion_tokens=completion_tokens_list[i],
        )
        for i, req in enumerate(requests)
    ]


# ---------------------------------------------------------------------------
# Gate v8 — per-decode-step δ(t) and ΔNLL(t) time series via two teacher-
# forcing forward passes. See ARCHITECTURE.md § 9.4.
# ---------------------------------------------------------------------------


def _per_token_nll_from_logits(
    logits: torch.Tensor,        # (1, T_total, vocab)
    target_ids: torch.Tensor,    # (1, T_total)
    prompt_len: int,
    completion_len: int,
) -> list[float]:
    """Per-token NLL on the completion tokens.

    logits at position p predicts token at position p+1. For completion token
    c_t (absolute position prompt_len + t), use logits at position
    prompt_len + t - 1. Returns a list of length ``completion_len``.
    """
    if completion_len == 0:
        return []
    start = max(0, prompt_len - 1)
    end = prompt_len + completion_len - 1  # exclusive
    if end > logits.shape[-2]:
        end = logits.shape[-2]
    n = end - start
    if n <= 0:
        return [float("nan")] * completion_len
    lg = logits[0, start:end, :].float()
    tgt = target_ids[0, prompt_len : prompt_len + n]
    log_probs = torch.log_softmax(lg, dim=-1)
    nll = -log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # (n,)
    out = nll.detach().cpu().tolist()
    if prompt_len == 0 and completion_len > n:
        # Couldn't compute NLL for the very first token without any context
        out = [float("nan")] + out
    return out


def run_request_v8(
    model: torch.nn.Module,
    tokenizer,
    request: Request,
    hook: AttentionTraceHook,
    *,
    device: torch.device,
    max_new_tokens: int,
    quant_bits: int,
    quant_group_size: int,
) -> DeltaStepRecord:
    """Two teacher-forcing passes; per-decode-step δ(t) and per-token ΔNLL(t).

    Pass 1 (FP16): forward over [prompt; FP16 completion] with no INT4 hook,
    captures per-layer Q/K (via ``hook``) and per-position logits.
    Pass 2 (INT4 K-only): same input, with k_proj-only INT4KVHook installed.
    Same captures.

    From the captured (Q^FP16, K^FP16) and (Q^INT4, K^INT4) we compute the
    causal attention row at each completion query position and take
        δ(t) = mean_layer KL(α^INT4 ‖ α^FP16).
    From the per-position logits we extract NLL on each completion token under
    each path. Both are time series of length completion_tokens.
    """
    enc = _tokenize(tokenizer, request)
    prompt_ids = enc["input_ids"].to(device)
    T_prompt = int(prompt_ids.shape[-1])

    # --- 1. Generate FP16 completion (sequential, single request) ---
    completion_ids: Optional[torch.Tensor] = None
    task_score: Optional[float] = None
    if max_new_tokens > 0:
        with torch.no_grad():
            gen = model.generate(
                input_ids=prompt_ids, max_new_tokens=max_new_tokens, do_sample=False
            )
        completion_ids = gen[0, T_prompt:].unsqueeze(0)  # (1, T_comp_raw)
        # Truncate at first EOS (inclusive) for clean teacher-forcing target
        eos_id = tokenizer.eos_token_id
        if eos_id is not None and completion_ids.numel() > 0:
            eos_pos = (completion_ids[0] == eos_id).nonzero(as_tuple=True)[0]
            if eos_pos.numel() > 0:
                completion_ids = completion_ids[:, : int(eos_pos[0].item()) + 1]
        if request.score_fn is not None and completion_ids.shape[-1] > 0:
            comp_text = tokenizer.decode(completion_ids[0], skip_special_tokens=True)
            try:
                raw = request.score_fn(comp_text)
                task_score = float(raw) if raw == raw else None
            except Exception as e:
                log.warning("score_fn failed for %s: %s", request.request_id, e)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if completion_ids is None or completion_ids.shape[-1] == 0:
        return DeltaStepRecord(
            request_id=request.request_id,
            workload=request.workload,
            num_layers=0,
            prompt_tokens=T_prompt,
            completion_tokens=0,
            delta_series=[],
            nll_fp16_series=[],
            nll_int4_series=[],
            task_score=task_score,
        )

    T_comp = int(completion_ids.shape[-1])
    full_ids = torch.cat([prompt_ids, completion_ids], dim=-1)

    # --- 2. Pass 1: FP16, capture Q/K + logits ---
    hook.begin_request()
    with torch.no_grad():
        out = model(input_ids=full_ids, use_cache=False)
    traces_fp16 = hook.end_request()
    nll_fp16_series = _per_token_nll_from_logits(
        out.logits.detach(), full_ids, T_prompt, T_comp
    )
    del out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- 3. Pass 2: INT4-K-only, capture Q/K + logits ---
    int4_hook = INT4KVHook(bits=quant_bits, group_size=quant_group_size, k_only=True)
    int4_hook.install(model)
    try:
        hook.begin_request()
        with torch.no_grad():
            out = model(input_ids=full_ids, use_cache=False)
        traces_int4 = hook.end_request()
        nll_int4_series = _per_token_nll_from_logits(
            out.logits.detach(), full_ids, T_prompt, T_comp
        )
        del out
    finally:
        int4_hook.remove()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- 4. Per-step δ(t), with per-layer breakdown for downstream analysis ---
    delta_series, per_layer_delta = compute_per_step_delta(
        traces_fp16, traces_int4,
        prompt_len=T_prompt, completion_len=T_comp,
        causal=True,
        return_per_layer=True,
    )
    num_layers = len(per_layer_delta) if per_layer_delta else len(traces_fp16)
    del traces_fp16, traces_int4
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return DeltaStepRecord(
        request_id=request.request_id,
        workload=request.workload,
        num_layers=num_layers,
        prompt_tokens=T_prompt,
        completion_tokens=T_comp,
        delta_series=delta_series,
        nll_fp16_series=nll_fp16_series,
        nll_int4_series=nll_int4_series,
        task_score=task_score,
        per_layer_delta=per_layer_delta,
    )


# ---------------------------------------------------------------------------
# Pre-validation: confirm INT4 KV actually degrades accuracy
# ---------------------------------------------------------------------------


def validate_signal(
    model: torch.nn.Module,
    tokenizer,
    hook: AttentionTraceHook,
    *,
    device: torch.device,
    n: int,
    quant_bits: int,
    quant_group_size: int,
    max_new_tokens: int,
    int4_k_only: bool = False,
) -> tuple[bool, list[DeltaRecord]]:
    """Run n GSM8K requests (seed=99) with dual inference.

    Returns ``(signal_found, records)``. ``signal_found`` is True if
    mean Δacc > threshold OR any individual request shows degradation.
    Records are returned so they can be prepended to the main output.
    """
    log.info("pre-validation: %d GSM8K requests (seed=99) ...", n)
    records: list[DeltaRecord] = []
    for req in load_workload("gsm8k_cot", n, seed=99):
        try:
            rec = run_request(
                model,
                tokenizer,
                req,
                hook,
                device=device,
                max_new_tokens=max_new_tokens,
                quant_bits=quant_bits,
                quant_group_size=quant_group_size,
                int4_k_only=int4_k_only,
            )
            records.append(rec)
        except Exception as e:
            log.exception("validate_signal: request %s failed: %s", req.request_id, e)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    scoreable = [
        r for r in records
        if r.nll_fp16 is not None
        and r.nll_int4 is not None
        and math.isfinite(r.nll_fp16)
        and math.isfinite(r.nll_int4)
    ]
    if not scoreable:
        log.warning("validate_signal: no NLL records — treating as no-signal")
        return False, records

    delta_nlls = [r.nll_int4 - r.nll_fp16 for r in scoreable]  # type: ignore[operator]
    mean_dnll = sum(delta_nlls) / len(delta_nlls)
    n_degraded = sum(1 for d in delta_nlls if d > 0)

    log.info(
        "validate_signal: n=%d  mean_ΔNLL=%.4f  n_degraded=%d",
        len(scoreable),
        mean_dnll,
        n_degraded,
    )

    signal_found = mean_dnll > _VALIDATION_SIGNAL_THRESHOLD or n_degraded > 0
    return signal_found, records


# ---------------------------------------------------------------------------
# Orchestration config
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    model: str
    workloads: list[str]
    num_requests_per_workload: int
    output_path: Path
    device: str = "cuda:0"
    max_new_tokens: int = 64
    quant_bits: int = 4
    quant_group_size: int = 32
    smoke: bool = False
    num_gpus: int = 1
    validate_batch_size: int = 20
    batch_size: int = 16  # requests processed together per generate call
    int4_k_only: bool = False  # if True, only k_proj is quantised (matches δ scope)
    gate_version: str = "v7"  # "v7" (per-request scalar) or "v8" (per-step time series)


def load_model(model_name: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if "cuda" in device else torch.float32
    log.info("loading %s on %s (%s)", model_name, device, dtype)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # Left-padding required for batched generation with decoder-only models.
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
    )
    model.to(device)
    model.eval()
    return model, tok


# ---------------------------------------------------------------------------
# Single-GPU orchestration
# ---------------------------------------------------------------------------


def _run_single_gpu(
    cfg: RunConfig,
    requests: List[Request],
    out_path: Path,
    *,
    run_validation: bool = True,
) -> int:
    """Load model on cfg.device, run requests, write JSONL to out_path.

    Returns number of records written.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model, tok = load_model(cfg.model, str(device))
    hook = AttentionTraceHook()
    hooked = hook.install(model)
    if hooked == 0:
        raise RuntimeError(
            "no attention modules were hooked — check model family / "
            "Q/K/V projection names."
        )
    log.info("installed attention hooks on %d layers", hooked)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    n_written = 0

    if cfg.gate_version == "v8":
        # v8 path: per-request, two teacher-forcing passes, time-series records.
        with out_path.open("w") as out:
            for i, req in enumerate(requests):
                try:
                    rec = run_request_v8(
                        model, tok, req, hook,
                        device=device,
                        max_new_tokens=cfg.max_new_tokens,
                        quant_bits=cfg.quant_bits,
                        quant_group_size=cfg.quant_group_size,
                    )
                except Exception as e:
                    log.exception("v8 request %s failed: %s", req.request_id, e)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                out.write(json.dumps(dataclasses.asdict(rec)) + "\n")
                out.flush()
                n_written += 1
                if n_written % 10 == 0 or n_written == 1:
                    log.info("  v8 %d/%d records in %.1fs", n_written, len(requests), time.time() - t_start)
        hook.remove()
        log.info("done: %d records in %.1fs -> %s",
                 n_written, time.time() - t_start, out_path)
        return n_written

    with out_path.open("w") as out:
        # Pre-validation (only on primary worker / single-GPU runs)
        if run_validation and cfg.validate_batch_size > 0:
            signal_found, val_records = validate_signal(
                model,
                tok,
                hook,
                device=device,
                n=cfg.validate_batch_size,
                quant_bits=cfg.quant_bits,
                quant_group_size=cfg.quant_group_size,
                max_new_tokens=cfg.max_new_tokens,
                int4_k_only=cfg.int4_k_only,
            )
            if not signal_found:
                log.error(
                    "SIGNAL NOT FOUND in validation batch — "
                    "INT4 KV hook may be a no-op. Aborting full run."
                )
                hook.remove()
                return 0

            log.info("signal confirmed — writing %d validation records", len(val_records))
            for rec in val_records:
                out.write(json.dumps(dataclasses.asdict(rec)) + "\n")
                out.flush()
                n_written += 1

        bs = max(1, cfg.batch_size)
        for batch_start in range(0, len(requests), bs):
            batch = requests[batch_start : batch_start + bs]
            try:
                recs = run_batch(
                    model,
                    tok,
                    batch,
                    hook,
                    device=device,
                    max_new_tokens=cfg.max_new_tokens,
                    quant_bits=cfg.quant_bits,
                    quant_group_size=cfg.quant_group_size,
                    int4_k_only=cfg.int4_k_only,
                )
            except Exception as e:
                log.exception(
                    "batch starting at %d failed: %s",
                    batch_start,
                    e,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            for rec in recs:
                out.write(json.dumps(dataclasses.asdict(rec)) + "\n")
            out.flush()
            n_written += len(recs)
            if n_written % 50 == 0 or n_written <= bs:
                log.info("  %d records in %.1fs", n_written, time.time() - t_start)

    hook.remove()
    log.info(
        "done: %d records in %.1fs -> %s",
        n_written,
        time.time() - t_start,
        out_path,
    )
    return n_written


# ---------------------------------------------------------------------------
# Multi-GPU data-parallel orchestration
# ---------------------------------------------------------------------------


def _collect_requests(cfg: RunConfig) -> List[Request]:
    """Materialise all workload requests without touching CUDA.

    Must be called before forking so that dataset reads happen in the
    main process (avoids multiprocessing-unsafe file handles inside
    HuggingFace datasets).
    """
    all_reqs: List[Request] = []
    for wk in cfg.workloads:
        all_reqs.extend(load_workload(wk, cfg.num_requests_per_workload, seed=0))
    return all_reqs


def _worker_main(
    rank: int,
    cfg: RunConfig,
    requests: List[Request],
    out_path: Path,
    log_level: int,
) -> None:
    """Entry point for each forked worker process."""
    logging.basicConfig(
        level=log_level,
        format=f"%(asctime)s [gpu{rank}] %(levelname)s %(message)s",
    )
    # Override device to this rank's GPU.
    worker_cfg = dataclasses.replace(cfg, device=f"cuda:{rank}")
    # Only rank 0 runs validation (to avoid redundant checks).
    run_validation = rank == 0
    _run_single_gpu(worker_cfg, requests, out_path, run_validation=run_validation)


def _run_multi_gpu(cfg: RunConfig) -> Path:
    """Fork N worker processes, merge their JSONL shards."""
    num_gpus = cfg.num_gpus
    all_requests = _collect_requests(cfg)
    log.info(
        "multi-GPU run: %d requests across %d GPUs", len(all_requests), num_gpus
    )

    # Round-robin sharding
    shards: List[List[Request]] = [[] for _ in range(num_gpus)]
    for i, req in enumerate(all_requests):
        shards[i % num_gpus].append(req)

    shard_paths = [
        cfg.output_path.parent / f"_shard_{r}.jsonl" for r in range(num_gpus)
    ]

    log_level = logging.getLogger().level or logging.INFO

    ctx = multiprocessing.get_context("fork")
    procs = []
    for rank in range(num_gpus):
        p = ctx.Process(
            target=_worker_main,
            args=(rank, cfg, shards[rank], shard_paths[rank], log_level),
        )
        p.start()
        procs.append(p)

    failed: list[int] = []
    for rank, p in enumerate(procs):
        p.join()
        if p.exitcode != 0:
            log.error("worker rank %d exited with code %d", rank, p.exitcode)
            failed.append(rank)

    if failed:
        raise RuntimeError(f"workers failed on ranks: {failed}")

    # Merge shards interleaved (preserves approximate time order)
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    handles = [sp.open() for sp in shard_paths if sp.exists()]
    n_merged = 0
    with cfg.output_path.open("w") as out:
        exhausted = [False] * len(handles)
        while not all(exhausted):
            for i, fh in enumerate(handles):
                if exhausted[i]:
                    continue
                line = fh.readline()
                if not line:
                    exhausted[i] = True
                    continue
                out.write(line)
                n_merged += 1
    for fh in handles:
        fh.close()
    for sp in shard_paths:
        if sp.exists():
            sp.unlink()

    log.info("merged %d records -> %s", n_merged, cfg.output_path)
    return cfg.output_path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run(cfg: RunConfig) -> Path:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if cfg.smoke and cfg.model != "sshleifer/tiny-gpt2":
        log.info("smoke mode: overriding model to sshleifer/tiny-gpt2")
        cfg = dataclasses.replace(cfg, model="sshleifer/tiny-gpt2")

    if cfg.num_gpus > 1:
        return _run_multi_gpu(cfg)

    # Single-GPU path
    all_requests = _collect_requests(cfg)
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_single_gpu(cfg, all_requests, cfg.output_path, run_validation=True)
    return cfg.output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> RunConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HF model id",
    )
    p.add_argument(
        "--workloads",
        nargs="+",
        default=list(WORKLOADS),
        choices=list(WORKLOADS),
    )
    p.add_argument(
        "-n", "--num-per-workload", type=int, default=400,
        help="requests per workload",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("motivation_out/delta_records.jsonl"),
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--quant-bits", type=int, default=4)
    p.add_argument("--quant-group-size", type=int, default=32)
    p.add_argument(
        "--smoke", action="store_true",
        help="use a tiny model to verify pipeline mechanics only",
    )
    p.add_argument(
        "--num-gpus", type=int, default=1,
        help="number of GPUs for data parallelism (fork-based)",
    )
    p.add_argument(
        "--validate-batch-size", type=int, default=20,
        help="GSM8K requests to run before the full run to confirm INT4 signal",
    )
    p.add_argument(
        "--batch-size", type=int, default=16,
        help="requests per batched generate call (higher = better GPU utilisation)",
    )
    p.add_argument(
        "--int4-k-only", action="store_true",
        help=(
            "quantise only k_proj (not v_proj) — matches δ's K-only scope. "
            "Use this when correlating δ against ΔNLL."
        ),
    )
    p.add_argument(
        "--gate-version", choices=["v7", "v8"], default="v7",
        help=(
            "v7: per-request scalar δ + scalar ΔNLL. "
            "v8: per-decode-step δ(t) + per-token ΔNLL(t) time series via "
            "two teacher-forcing passes (forces K-only INT4)."
        ),
    )
    a = p.parse_args(argv)
    return RunConfig(
        model=a.model,
        workloads=a.workloads,
        num_requests_per_workload=a.num_per_workload,
        output_path=a.out,
        device=a.device,
        max_new_tokens=a.max_new_tokens,
        quant_bits=a.quant_bits,
        quant_group_size=a.quant_group_size,
        smoke=a.smoke,
        num_gpus=a.num_gpus,
        validate_batch_size=a.validate_batch_size,
        batch_size=a.batch_size,
        int4_k_only=a.int4_k_only,
        gate_version=a.gate_version,
    )


def main(argv: Optional[list[str]] = None) -> int:
    cfg = _parse_args(argv)
    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
