"""Attention-weight δ(ℓ, r) logger.

For each decode step (or prefill position — the motivation study can
use both), we compute two attention-weight distributions:

  w_fp16  = softmax(Q · K_fp16 / √d_k)          # reference
  w_int4  = softmax(Q · dequant(quant(K)) / √d_k)   # simulated INT4

and record

  δ(ℓ, r) = mean over positions of  KL(w_int4 ‖ w_fp16)

(consistent with the production canary which computes
``KL(ȧ_ℓ ‖ ā_ℓ)`` where ȧ is the INT4 path and ā is the FP16
reference — see ARCHITECTURE.md § 4.1).

Design choices
--------------
* **KL on attention weights, not on attention outputs.** The
  production kernel spec calls out "KL divergence". Attention
  outputs are linear combinations of V — they are vectors, not
  distributions — so KL only makes sense on the softmax weights. JS
  is listed as a candidate in REPORT.md § Open Questions; swap here
  if we switch.
* **Only K is quantized here.** δ = KL(w_int4 ‖ w_fp16) depends on
  Q and K but not on V. To capture V-quantization error propagation
  across layers we would need to re-run the model with a quantized
  KV cache affecting subsequent layers — a separate experiment.
  The motivation study measures the per-layer *local* sensitivity,
  which is what the Spearman-ρ gate cares about.
* **Group axis.** KIVI groups K along the head-dim axis. We follow
  that convention here (``group_axis=-1`` with
  shape ``(B, H, L, D)``).

This module is deliberately framework-light: given a stack of
``(Q, K, V)`` tensors it returns a per-layer δ vector. It does *not*
know about transformers; the hook that extracts Q/K/V lives in
``run.py`` so this module stays unit-testable with synthetic
tensors.

Invariants
----------
INV-1  This module allocates no shadow buffer. Quantization produces
       (q_int, scale, zero_point) and dequantization is computed on
       demand; the result is a transient FP16 tensor that goes out
       of scope at the end of the call.
INV-4  In the full system, the no-canary path must be bit-exact with
       KIVI-4. The motivation study does not exercise INV-4
       directly; it is a separate measurement and is not on the
       canary's hot path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F

from .kivi import DEFAULT_BITS, DEFAULT_GROUP_SIZE, simulate_quantization


_KL_EPS = 1e-9


@dataclass
class AttentionTrace:
    """One layer's (Q, K, V) from one forward pass of one request.

    Shapes follow the usual convention
    ``(batch, num_heads, seq_len, head_dim)``. The motivation study
    typically sets ``batch=1``.
    """

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor  # kept for completeness; δ only needs Q, K

    def __post_init__(self) -> None:
        if self.q.shape[-1] != self.k.shape[-1]:
            raise ValueError(
                "Q and K must share head_dim; got "
                f"{self.q.shape} vs {self.k.shape}"
            )


def _attention_weights(
    q: torch.Tensor,
    k: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """Return softmax(QK^T / √d_k) with optional causal mask.

    Shapes: ``q (B, H, Lq, D)``, ``k (B, H, Lk, D)`` ->
    ``(B, H, Lq, Lk)``.
    """
    d_k = q.shape[-1]
    # compute in float32 for stability regardless of input dtype
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / (d_k ** 0.5)
    if causal:
        lq, lk = scores.shape[-2], scores.shape[-1]
        # Only apply a causal mask if it would actually mask anything.
        # Prefill uses lq == lk; decode often uses lq=1 against full lk.
        if lq == lk:
            mask = torch.ones(lq, lk, dtype=torch.bool, device=scores.device)
            mask = torch.triu(mask, diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
    return F.softmax(scores, dim=-1)


def _kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL(p ‖ q), elementwise over the last dim.

    Returns a tensor of shape ``p.shape[:-1]``.
    """
    p_safe = p.clamp_min(_KL_EPS)
    q_safe = q.clamp_min(_KL_EPS)
    return (p_safe * (p_safe.log() - q_safe.log())).sum(dim=-1)


def compute_layer_delta(
    trace: AttentionTrace,
    *,
    bits: int = DEFAULT_BITS,
    group_size: int = DEFAULT_GROUP_SIZE,
    k_group_axis: int = -1,  # group K along head_dim (KIVI)
    causal: bool = True,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """Compute δ for one layer given its (Q, K, V) trace.

    Returns
    -------
    torch.Tensor
        If ``reduction`` is ``"mean"`` or ``"sum"``, a scalar
        tensor. If ``"none"``, shape ``(B, H, Lq)`` — per-head,
        per-query-position KL.
    """
    w_fp16 = _attention_weights(trace.q, trace.k, causal=causal)

    k_quantized = simulate_quantization(
        trace.k, bits=bits, group_size=group_size, group_axis=k_group_axis
    )
    w_int4 = _attention_weights(trace.q, k_quantized, causal=causal)

    kl_per_pos = _kl_divergence(w_int4, w_fp16)  # (B, H, Lq)

    if reduction == "none":
        return kl_per_pos
    if reduction == "sum":
        return kl_per_pos.sum()
    return kl_per_pos.mean()


def compute_request_delta(
    traces: list[AttentionTrace],
    *,
    bits: int = DEFAULT_BITS,
    group_size: int = DEFAULT_GROUP_SIZE,
    causal: bool = True,
) -> torch.Tensor:
    """Compute δ(ℓ) for every layer in ``traces``.

    Returns a 1-D tensor of length ``len(traces)`` — one scalar per
    layer, mean KL over batch / heads / query positions.
    """
    out = torch.empty(len(traces), dtype=torch.float32)
    for ell, trace in enumerate(traces):
        out[ell] = compute_layer_delta(
            trace,
            bits=bits,
            group_size=group_size,
            causal=causal,
            reduction="mean",
        ).detach().cpu()
    return out


@dataclass
class DeltaRecord:
    """One row in the motivation-study output."""

    request_id: str
    workload: str
    num_layers: int
    delta: list[float]           # per-layer δ (KL of attention weights)
    task_score: Optional[float]  # FP16 accuracy (optional, for monitoring)
    nll_fp16: Optional[float]    # mean NLL of FP16 completion under FP16 KV
    nll_int4: Optional[float]    # mean NLL of FP16 completion under INT4 KV
    prompt_tokens: int
    completion_tokens: int       # tokens in the FP16 completion used for NLL


# ---------------------------------------------------------------------------
# Gate v8 — per-decode-step time series
# ---------------------------------------------------------------------------
#
# v8 reframes the gate from "does prefill δ predict completion ΔNLL?" (which
# v3-v7 showed empirically is unanswerable because the two quantities live in
# different spaces — prefill δ averages over prompt positions, ΔNLL is dominated
# by per-step decode difficulty) to "does δ(t) at decode step t predict ΔNLL(t)
# at decode step t, within a single request?" — the question the production
# canary actually answers when it triggers per-step precision changes.
#
# The implementation uses two teacher-forcing forward passes over the same
# (prompt + FP16-completion) input sequence:
#   Pass 1: FP16 K. Capture per-layer Q, K (via AttentionTraceHook) and per-
#       position logits.
#   Pass 2: K-only INT4 hook. Same captures.
# From the captured Q, K we compute the causal attention row α_p =
# softmax(q_p K_{0:p+1}^T / √d) at each completion query position p and take
# δ(p) = mean_ℓ KL(α_p^INT4 ‖ α_p^FP16). From the captured logits we compute
# per-token NLL on the FP16 completion target.


@dataclass
class DeltaStepRecord:
    """v8 record: per-decode-step δ(t) and ΔNLL(t) time series for one request."""

    request_id: str
    workload: str
    num_layers: int
    prompt_tokens: int
    completion_tokens: int       # length of delta_series / nll_*_series
    delta_series: list[float]    # δ(t) for t in [0, completion_tokens) — layer-mean
    nll_fp16_series: list[float]  # NLL of completion token t under FP16 KV
    nll_int4_series: list[float]  # NLL of completion token t under INT4 (K-only) KV
    task_score: Optional[float]  # FP16 task score (for monitoring; not on gate path)
    per_layer_delta: Optional[list[list[float]]] = None
    """Optional per-layer δ_ℓ(t) — shape (num_layers, completion_tokens). When
    captured, the layer-mean ``delta_series`` is mean over the first axis. Used
    for the per-layer ρ analysis that scopes the canary to the highest-signal
    layer subset."""


def compute_per_step_delta(
    traces_fp16: list,        # list[AttentionTrace], one per layer
    traces_int4: list,
    *,
    prompt_len: int,
    completion_len: int,
    causal: bool = True,
    return_per_layer: bool = False,
) -> list[float] | tuple[list[float], list[list[float]]]:
    """Per-decode-step δ(t) averaged over layers and heads.

    For each completion step t in [0, completion_len), compute
        δ(t) = mean_ℓ KL(α_{p,ℓ}^INT4 ‖ α_{p,ℓ}^FP16)
    where p = prompt_len + t - 1 is the absolute query position whose
    attention row is used to predict completion token c_t. ``α_{p,ℓ}`` is the
    causal attention row from layer ℓ at position p:
        α_{p,ℓ} = softmax(q_p^ℓ K_{0:p+1}^ℓ⊤ / √d_head).

    Implementation goes layer-by-layer to keep peak memory bounded by one
    layer's (T_comp, T_total) attention matrix rather than all layers'.
    Computed in float32 for numerical stability.
    """
    assert len(traces_fp16) == len(traces_int4), "trace length mismatch"
    L = len(traces_fp16)
    if L == 0 or completion_len == 0:
        return []

    # Per-completion-step query positions. p = prompt_len - 1 + t for t in [0, T_comp).
    # If prompt_len == 0, t=0 has no prior tokens to attend to — handle that case.
    start = max(0, prompt_len - 1)
    # If prompt_len >= 1: positions [prompt_len-1 .. prompt_len + T_comp - 2]
    # If prompt_len == 0: positions [0 .. T_comp - 2] (skip t=0 which has nothing to attend to)
    if prompt_len >= 1:
        n_steps = completion_len
        p_offset = prompt_len - 1
    else:
        n_steps = max(0, completion_len - 1)
        p_offset = 0
    if n_steps == 0:
        return [float("nan")] * completion_len

    delta_sum = torch.zeros(n_steps, dtype=torch.float64)
    per_layer: list[list[float]] = [] if return_per_layer else []
    T_total_min = traces_fp16[0].q.shape[-2]

    for ell in range(L):
        qf = traces_fp16[ell].q.float()  # (1, H, T_total, D)
        kf = traces_fp16[ell].k.float()
        qi = traces_int4[ell].q.float()
        ki = traces_int4[ell].k.float()
        d_head = qf.shape[-1]
        T_total = qf.shape[-2]
        if T_total != T_total_min:
            raise ValueError(
                f"layer {ell} trace length {T_total} != layer 0 length {T_total_min}"
            )

        # Slice queries at completion positions
        qf_c = qf[:, :, p_offset : p_offset + n_steps, :]   # (1, H, n_steps, D)
        qi_c = qi[:, :, p_offset : p_offset + n_steps, :]

        # Scaled dot products (1, H, n_steps, T_total)
        scale = 1.0 / (d_head ** 0.5)
        sf = torch.matmul(qf_c, kf.transpose(-1, -2)) * scale
        si = torch.matmul(qi_c, ki.transpose(-1, -2)) * scale

        # Causal mask: row r corresponds to query position p_offset + r.
        # Allowed key positions: [0, p_offset + r] inclusive → mask k > p_offset + r.
        k_idx = torch.arange(T_total, device=sf.device).view(1, 1, 1, T_total)
        q_pos = torch.arange(p_offset, p_offset + n_steps, device=sf.device).view(1, 1, n_steps, 1)
        if causal:
            mask = k_idx > q_pos
            sf = sf.masked_fill(mask, float("-inf"))
            si = si.masked_fill(mask, float("-inf"))

        af = torch.softmax(sf, dim=-1)
        ai = torch.softmax(si, dim=-1)

        eps = 1e-9
        af_safe = af.clamp_min(eps)
        ai_safe = ai.clamp_min(eps)
        kl = (ai_safe * (ai_safe.log() - af_safe.log())).sum(dim=-1)  # (1, H, n_steps)
        kl_mean_heads = kl.mean(dim=1)  # (1, n_steps)
        kl_layer_cpu = kl_mean_heads.squeeze(0).double().detach().cpu()
        delta_sum = delta_sum + kl_layer_cpu
        if return_per_layer:
            per_layer.append(kl_layer_cpu.tolist())

        # free per-layer scratch
        del qf, kf, qi, ki, qf_c, qi_c, sf, si, af, ai, af_safe, ai_safe, kl, kl_mean_heads
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    delta_per_step = (delta_sum / L).tolist()  # length n_steps

    # Pad: if we skipped t=0 because prompt_len==0, prepend a NaN.
    if prompt_len == 0 and completion_len > 0:
        delta_per_step = [float("nan")] + delta_per_step
        if return_per_layer:
            per_layer = [[float("nan")] + row for row in per_layer]
    if return_per_layer:
        return delta_per_step, per_layer
    return delta_per_step
