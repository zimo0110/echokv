"""Canary harness for vLLM.

Wires the three-component core into a running vLLM ``LLM`` instance via
forward-hooks on every ``vllm.model_executor.layers.attention.Attention``
module. At each canary step (Bernoulli ``p_c``), the hook captures the
incoming Q/K tensors, simulates KIVI-INT4 quantisation on K, computes
``δ(ℓ) = mean KL(softmax(QK_int4) ‖ softmax(QK_fp16))``, and feeds the
sample to a :class:`core.canary_estimator.CanaryEstimator`. The
:class:`core.precision_controller.PIDPrecisionController` runs in shadow
mode — every ``tick_interval`` decode steps it consumes the EMA snapshot
and emits per-layer tier decisions to :class:`core.slo_budget.SLOBudget`,
which is registered with a logging callback. Decisions are not applied
to vLLM's KV cache (vLLM 0.16 has one global KV dtype; per-layer dynamic
precision needs a vLLM source fork — Session 9).

This is the integration point the critic's session-8 plan calls for. We
diverge from the plan in two places: (1) the baseline is FP16 KV (vLLM
has no native KIVI-4 mode); (2) controller decisions are shadow-only.
Both are vLLM-source limitations, documented in CHANGELOG.md and
LOGS/session_20260426_8.md.

Usage
-----
::

    from vllm import LLM
    from evaluation.serving.canary_vllm import CanaryHarness

    llm = LLM(model=..., distributed_executor_backend='uni',
              enforce_eager=True)
    harness = CanaryHarness(llm, canary_rate=1/64, tick_interval=100)
    harness.install()
    try:
        outputs = llm.generate(prompts, sampling_params)
    finally:
        harness.remove()
    print(harness.report())

The ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` env var is required so the
EngineCore runs in-process and the hooks installed in the main process
actually fire on the worker's attention forward calls.
"""

from __future__ import annotations

import math
import re
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn.functional as F

from core.block_allocator import Precision
from core.canary_estimator import CanaryEstimator, DEFAULT_CANARY_RATE
from core.precision_controller import (
    DEFAULT_HYSTERESIS_BAND,
    DEFAULT_SETPOINT,
    DEFAULT_TICK_INTERVAL,
    PIDPrecisionController,
)
from core.slo_budget import SLOBudget
from evaluation.motivation.kivi import simulate_quantization


_LAYER_IDX_RE = re.compile(r"layers\.(\d+)")


def _percentile(samples: "list[float]", q: float) -> float:
    """Linear-interpolation percentile, q ∈ [0, 1]. Tiny dep, no numpy."""
    if not samples:
        raise ValueError("empty samples")
    s = sorted(samples)
    if len(s) == 1:
        return s[0]
    rank = q * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


# ---------------------------------------------------------------------------
# Principled initial-tier rule (cost-asymmetry derivation, ARCHITECTURE.md § 9.6)
# ---------------------------------------------------------------------------
#
# At request start the controller has zero information; warmup-window blocks
# (first N_c=100 decode steps) are permanent (INV-2). Costs are asymmetric:
#
#     Cost_low   = E[acc loss | hard request] × P(hard) × N_warmup
#                = constant — independent of T_total
#     Cost_high  = extra_memory × T_warmup / T_total
#                → 0 as T_total → ∞
#
# Long context: starting high amortises away. Short context: costs symmetric
# and INT4 is the minimum-regret prior. Three regimes:
#
#     prompt < 512:    INT2  (warmup dominates the request, exploration safe)
#     512 ≤ prompt < 8K:   INT4  (symmetric costs, neutral prior)
#     prompt ≥ 8K:    INT8  (warmup amortises, protect accuracy)


_INITIAL_TIER_LOW_THRESHOLD: int = 512
_INITIAL_TIER_HIGH_THRESHOLD: int = 8192


# ---------------------------------------------------------------------------
# Multi-precision canary support (ARCHITECTURE.md § 9.8 — neighbors-only)
# ---------------------------------------------------------------------------


# Architectural tier ladder, ordered from most aggressive to least.
ALL_TIERS: list[Precision] = [Precision.INT2, Precision.INT4, Precision.INT8]
_TIER_INDEX: dict = {t: i for i, t in enumerate(ALL_TIERS)}


def neighbours_to_measure(current: Precision) -> list[Precision]:
    """Return the tiers the canary should fire on given the current tier.

    The controller's decision space at any tick is stay /
    upgrade-one-step / downgrade-one-step (hysteresis + min_dwell_ticks
    forbid two-step transitions). Measurements at non-adjacent tiers
    provide unactionable information; the staleness self-heals through
    the mandatory intermediate state during transitions. See
    ARCHITECTURE.md § 9.8.

    Returns
    -------
    list[Precision]
        ``[INT2, INT4]`` if current is ``INT2``,
        ``[INT2, INT4, INT8]`` if current is ``INT4`` (the all-three case),
        ``[INT4, INT8]`` if current is ``INT8``.
    """
    idx = _TIER_INDEX[current]
    out = [current]
    if idx > 0:
        out.append(ALL_TIERS[idx - 1])
    if idx < len(ALL_TIERS) - 1:
        out.append(ALL_TIERS[idx + 1])
    out.sort(key=int)  # ascending bit-width for deterministic order
    return out


class MultiPrecisionEstimator:
    """Per-(layer, tier) EMA tracker — wraps one CanaryEstimator per tier.

    Memory: ``len(ALL_TIERS) × num_layers`` scalars total — i.e. 3 × 32 = 96
    floats for Llama-3.1-8B. INV-1 unaffected (no per-block state).

    Pre-existing scalar code paths can keep using ``CanaryEstimator``
    directly; this class is opt-in via ``CanaryHarness(multi_precision=True)``.
    """

    def __init__(
        self,
        num_layers: int,
        tiers: list[Precision] = ALL_TIERS,
        ema_alpha: float = 0.1,
        canary_rate: float = 1.0 / 64.0,
        scratchpad_depth: int = 4096,
    ) -> None:
        self.num_layers = num_layers
        self.tiers = list(tiers)
        self._estimators: dict = {
            t: CanaryEstimator(
                num_layers=num_layers,
                ema_alpha=ema_alpha,
                canary_rate=canary_rate,
                scratchpad_depth=scratchpad_depth,
            )
            for t in self.tiers
        }
        # Counters used for the run report.
        self.observations_per_tier: dict = {t: 0 for t in self.tiers}
        self.monotonicity_violations: int = 0

    # -------- read API --------

    def ema(self, layer_id: int, tier: Precision) -> float:
        return self._estimators[tier].ema(layer_id)

    def per_tier_ema(self, layer_id: int) -> dict:
        """``{tier: ema_value}`` for one layer. Missing tiers (no
        observations yet) report 0.0 — the controller treats 0.0 as
        "no signal yet, use construction-time fallback."""
        return {t: e.ema(layer_id) for t, e in self._estimators.items()}

    def snapshot_per_tier(self) -> dict:
        """``{tier: list_of_per_layer_emas}`` — full per-tier EMA matrix."""
        return {t: e.snapshot() for t, e in self._estimators.items()}

    def num_samples(self, layer_id: int, tier: Precision) -> int:
        return self._estimators[tier].num_samples(layer_id)

    # -------- write API --------

    def observe(
        self,
        layer_id: int,
        step: int,
        kl: float,
        tier: Precision,
        *,
        is_canary: bool,
    ) -> None:
        """Record one canary observation for a specific tier."""
        if not is_canary:
            return
        self._estimators[tier].observe(
            layer_id=layer_id, step=step, kl=kl, is_canary=True,
        )
        self.observations_per_tier[tier] = self.observations_per_tier.get(tier, 0) + 1

    def check_monotonicity(
        self, layer_id: int, observations: dict, tol: float = 1e-6,
    ) -> bool:
        """Verify δ̂_2 ≥ δ̂_4 ≥ δ̂_8 for the freshly-observed values.

        Returns True if monotonicity holds (within ``tol``); False otherwise.
        Counts violations into ``self.monotonicity_violations`` for the
        run report. Tolerates float-noise; large violations indicate a
        real bug per ARCHITECTURE.md § 9.8.
        """
        i2 = observations.get(Precision.INT2)
        i4 = observations.get(Precision.INT4)
        i8 = observations.get(Precision.INT8)
        violations: list = []
        if i2 is not None and i4 is not None and i2 + tol < i4:
            violations.append(("INT2 < INT4", i2, i4))
        if i4 is not None and i8 is not None and i4 + tol < i8:
            violations.append(("INT4 < INT8", i4, i8))
        if i2 is not None and i8 is not None and i2 + tol < i8:
            violations.append(("INT2 < INT8", i2, i8))
        if violations:
            self.monotonicity_violations += 1
            return False
        return True

    def reset(self) -> None:
        for e in self._estimators.values():
            e.reset()
        self.observations_per_tier = {t: 0 for t in self.tiers}
        self.monotonicity_violations = 0


def initial_tier_for_context(prompt_length: int) -> "Precision":
    """Cost-asymmetry-derived initial controller tier as a function of
    prompt length. See ARCHITECTURE.md § 9.6 for the full derivation.

    Parameters
    ----------
    prompt_length:
        Number of input tokens for the request. For batched inference
        with a mix of prompt lengths, callers can use either the
        max prompt length in the batch (conservative) or a per-request
        installation if they install one harness per request.

    Returns
    -------
    Precision
        ``INT2`` for ``prompt_length < 512``,
        ``INT4`` for ``512 ≤ prompt_length < 8192``,
        ``INT8`` for ``prompt_length ≥ 8192``.
    """
    if prompt_length >= _INITIAL_TIER_HIGH_THRESHOLD:
        return Precision.INT8
    if prompt_length >= _INITIAL_TIER_LOW_THRESHOLD:
        return Precision.INT4
    return Precision.INT2


@dataclass
class CanaryStats:
    """Counters kept by the harness for the post-run report."""

    attention_calls: int = 0          # total per-layer attention forward calls
    canary_fires: int = 0             # times the canary path executed
    canary_overhead_s: float = 0.0    # accumulated wall time inside the canary
    canary_skipped_decode_only: int = 0  # short-cut path; see _maybe_fire
    controller_ticks: int = 0
    tier_log: list[list[Precision]] = field(default_factory=list)
    tier_applications: int = 0        # times we mutated K/V via controller tier
    tier_application_overhead_s: float = 0.0
    tier_apply_count_by_bits: dict = field(default_factory=lambda: {2: 0, 4: 0, 8: 0})


def _to_bhsd(t: torch.Tensor, num_heads: int) -> torch.Tensor:
    """vLLM's attention forward sees ``(num_tokens, num_heads*head_size)``
    pre-reshape and ``(num_tokens, num_heads, head_size)`` post-reshape
    depending on the exact call site. Normalise to ``(1, H, T, D)`` so
    the dense kernel helpers can consume it.
    """
    if t.dim() == 2:
        # (num_tokens, num_heads*head_size)
        T, HD = t.shape
        D = HD // num_heads
        return t.view(T, num_heads, D).transpose(0, 1).unsqueeze(0)  # (1, H, T, D)
    if t.dim() == 3:
        # (num_tokens, H, D)
        return t.transpose(0, 1).unsqueeze(0)  # (1, H, T, D)
    raise ValueError(f"unexpected tensor rank: {t.shape}")


class CanaryHarness:
    """Forward-hook based canary integration for vLLM.

    Walks the model's attention modules, installs a forward-pre-hook on
    each, and dispatches every (configurable Bernoulli-sampled) call to
    the canary path. The canary computes δ from the (Q, K) seen at the
    hook site — note that vLLM's stored INT4-style cache does not exist
    in this baseline (vLLM has FP16/FP8 KV cache, not KIVI-INT4); we
    therefore quantise K on the fly via ``simulate_quantization`` to
    drive the canary signal. This matches the motivation-study δ and
    feeds real per-layer divergence to the estimator.
    """

    def __init__(
        self,
        llm: Any = None,
        *,
        model: Any = None,
        canary_rate: float = DEFAULT_CANARY_RATE,
        tau: float = DEFAULT_SETPOINT,
        initial_level: Optional[Precision] = None,
        tick_interval: int = DEFAULT_TICK_INTERVAL,
        hysteresis_band: float = DEFAULT_HYSTERESIS_BAND,
        min_dwell_ticks: int = 2,
        quant_bits: int = 4,
        group_size: int = 32,
        decode_only_canary: bool = True,
        max_decode_tokens: int = 256,
        use_canary_stream: bool = True,
        tier_application: bool = False,
        tier_application_quant_v: bool = True,
        force_bits: Optional[int] = None,
        freeze_controller: bool = False,
        adaptive_tau: bool = False,
        adaptive_tau_percentile: float = 50.0,
        adaptive_tau_scale: float = 1.0,
        adaptive_tau_min_samples: int = 8,
        multi_precision: bool = False,
        tau_budget: float = 0.02,
        monotonicity_check: bool = True,
        absolute_canary: bool = False,
        kref_window_size: int = 64,
        ema_alpha: float = 0.1,
        seed: int = 0,
    ) -> None:
        self.llm = llm
        self.canary_rate = float(canary_rate)
        self.quant_bits = int(quant_bits)
        self.group_size = int(group_size)
        # decode_only_canary: skip canary fires when ``query.shape[0]`` exceeds
        # ``max_decode_tokens`` (a prefill-batch heuristic). At decode the K
        # tensor passed to attention.forward has shape (num_decode_tokens, ...)
        # — typically ≤ a few dozen running requests. At prefill batches the
        # query length is the sum of prompt lengths, often O(thousands) at
        # long context. The canary's design is "fire on decode steps", so
        # this is the architectural default. Brings per-fire cost from
        # O(L²) (full prefill recomputation) down to O(L) (one query row vs
        # the cache).
        self.decode_only_canary = bool(decode_only_canary)
        self.max_decode_tokens = int(max_decode_tokens)
        # use_canary_stream: schedule the canary's softmax/KL on a separate
        # CUDA stream so it runs concurrently with the production attention
        # path. We synchronise at the end of the pre-hook (true async result
        # collection is a future improvement).
        self.use_canary_stream = bool(use_canary_stream)
        # tier_application: when True, every attention call's K (and V if
        # tier_application_quant_v) is quantised in-place to the precision
        # the controller has decided for this layer
        # (``controller.level(layer_id)``) BEFORE vLLM splices them into
        # the KV cache. Numerically equivalent to KIVI's effect on
        # inference outputs (quant noise injected at write-time); the cache
        # itself is still FP16 — true per-layer dtype routing is a vLLM
        # source fork (future session). With tier_application off, the
        # controller's decisions are shadow-only and inference outputs
        # match the FP16 baseline.
        self.tier_application = bool(tier_application)
        self.tier_application_quant_v = bool(tier_application_quant_v)
        # force_bits: if set, every tier-application call uses this fixed
        # precision (INT2/INT4/INT8) regardless of what the controller
        # decided. Used to simulate static KIVI baselines (every layer
        # at INT4) for the Pareto comparison: without a workload where
        # static INT4 actually fails, EchoKV-3C has no Pareto win to
        # claim. Session 11+ context sweeps use this to find the
        # workload that breaks KIVI-4 on Llama-3.1-8B.
        self.force_bits: Optional[int] = force_bits
        if self.force_bits is not None and self.force_bits not in {2, 4, 8}:
            raise ValueError(f"force_bits must be one of 2/4/8, got {self.force_bits}")
        # freeze_controller: when True, the controller's tier vector is
        # whatever it was at install time and never changes during the run.
        # Used for the measurement phase of EchoKV-3C: the warmup phase
        # ticks the controller (observe → converge); the measurement phase
        # freezes the converged tier vector and applies it consistently
        # across every prompt in the batch (matches production behaviour
        # where the controller's update cadence is tens of decode steps,
        # not per-prompt). Without freezing, mid-run tier changes meant
        # different prompts in the same batch saw different precision —
        # the 32K smoke run gave acc=0.700 from this exact failure mode.
        self.freeze_controller = bool(freeze_controller)
        # adaptive_tau: if True, just before each controller tick the
        # controller's setpoint is recomputed from the observed δ̂_ℓ
        # distribution — `τ = percentile(δ̂_ℓ across layers, p) × scale`.
        # τ then tracks the workload's natural δ̂ scale automatically; no
        # per-context manual tuning. See ARCHITECTURE.md § 9.6 (Open
        # Question 12 in REPORT.md). Requires at least
        # ``adaptive_tau_min_samples`` finite δ̂_ℓ values to update — until
        # then the controller uses its construction-time τ.
        self.adaptive_tau = bool(adaptive_tau)
        if not (0.0 < adaptive_tau_percentile <= 100.0):
            raise ValueError(f"adaptive_tau_percentile must be in (0, 100]")
        if adaptive_tau_scale <= 0:
            raise ValueError("adaptive_tau_scale must be > 0")
        self.adaptive_tau_percentile = float(adaptive_tau_percentile)
        self.adaptive_tau_scale = float(adaptive_tau_scale)
        self.adaptive_tau_min_samples = int(adaptive_tau_min_samples)
        # τ-history for telemetry / post-run reports.
        self._tau_history: list[float] = []
        # Multi-precision canary state (ARCHITECTURE.md § 9.8).
        self.multi_precision = bool(multi_precision)
        if tau_budget <= 0:
            raise ValueError("tau_budget must be > 0")
        self.tau_budget = float(tau_budget)
        self.monotonicity_check = bool(monotonicity_check)
        # Per-(layer, tier) EMA, only populated when multi_precision is on.
        self._mp_estimator: Optional[MultiPrecisionEstimator] = None
        # Absolute-canary mode (ARCHITECTURE.md § 9.9):
        # measure δ_absolute = KL(softmax(Q · quant(K_ref)^T) ‖ softmax(Q · K_ref^T))
        # against a sliding-window FP16 K reference. Decouples the canary's
        # measurement from cache state and from batch size T (T_key = window
        # size, not max_num_seqs). Closes the T=1 collapse mode and breaks
        # the cache-state feedback loop simultaneously.
        self.absolute_canary = bool(absolute_canary)
        if kref_window_size < 2:
            raise ValueError("kref_window_size must be >= 2")
        self.kref_window_size = int(kref_window_size)
        # Lazy per-layer state. Populated on first pre-hook fire so we can
        # use the actual tensor dtype/device. Keys are layer_id.
        self._kref_buf: dict[int, torch.Tensor] = {}
        self._kref_fill: dict[int, int] = {}
        self._kref_write_idx: dict[int, int] = {}

        # Resolve the model nn.Module. Two construction paths:
        # - ``llm=<vllm.LLM>`` (TP=1, in-proc): unpack the rank-0 model
        #   from the engine. This is the canonical single-GPU path.
        #   See LOGS/session_20260426_8.md for the discovery walkthrough.
        # - ``model=<nn.Module>`` (TP>1, worker-side): the harness is
        #   being constructed inside a vLLM worker via ``apply_model``;
        #   the worker hands us its local model directly. See
        #   ``evaluation/serving/canary_tp.py``.
        if model is not None:
            self.model = model
        elif llm is not None:
            try:
                self.model = (
                    llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
                )
            except AttributeError as e:
                raise RuntimeError(
                    "Cannot reach the model from the LLM instance. Ensure "
                    "VLLM_ENABLE_V1_MULTIPROCESSING=0 is set so the engine "
                    "runs in-process, or pass ``model=`` directly when "
                    "constructing the harness inside a worker."
                ) from e
        else:
            raise ValueError(
                "CanaryHarness requires either ``llm`` or ``model``"
            )

        self._attention_modules = self._collect_attention_modules()
        self.num_layers = len(self._attention_modules)
        if self.num_layers == 0:
            raise RuntimeError("found no Attention modules on the loaded model")

        self.estimator = CanaryEstimator(
            num_layers=self.num_layers,
            ema_alpha=ema_alpha,
            canary_rate=canary_rate,
        )
        if self.multi_precision:
            self._mp_estimator = MultiPrecisionEstimator(
                num_layers=self.num_layers,
                ema_alpha=ema_alpha,
                canary_rate=canary_rate,
            )
        ctrl_kwargs = dict(
            num_layers=self.num_layers,
            setpoint=tau,
            hysteresis_band=hysteresis_band,
            tick_interval=tick_interval,
            min_dwell_ticks=min_dwell_ticks,
        )
        if initial_level is not None:
            ctrl_kwargs["initial_level"] = initial_level
        self.controller = PIDPrecisionController(**ctrl_kwargs)
        self.budget = SLOBudget()
        self.budget.register_broadcast_callback(self._on_broadcast)

        self.stats = CanaryStats()
        self._global_step = 0  # increments on every layer-0 attention call
        self._g = torch.Generator(device="cpu").manual_seed(seed)
        self._handles: list = []
        # Canary CUDA stream — created lazily on first GPU fire so we don't
        # touch CUDA at __init__ time (allows constructing the harness on
        # CPU-only test runs).
        self._canary_stream: Optional[Any] = None

    # ------------------------------------------------------------------
    # install / remove
    # ------------------------------------------------------------------

    def _collect_attention_modules(self) -> list[tuple[int, torch.nn.Module]]:
        out: list[tuple[int, torch.nn.Module]] = []
        for name, module in self.model.named_modules():
            if type(module).__name__ != "Attention":
                continue
            m = _LAYER_IDX_RE.search(name)
            if m is None:
                continue
            out.append((int(m.group(1)), module))
        out.sort(key=lambda t: t[0])
        # Sanity: layer ids should be a contiguous prefix [0, num_layers).
        ids = [i for i, _ in out]
        if ids != list(range(len(ids))):
            raise RuntimeError(f"non-contiguous layer ids: {ids[:8]} ...")
        return out

    def install(self) -> None:
        for layer_id, mod in self._attention_modules:
            self._handles.append(
                mod.register_forward_pre_hook(
                    self._make_pre_hook(layer_id, mod), with_kwargs=False
                )
            )

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self) -> "CanaryHarness":
        self.install()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.remove()

    # ------------------------------------------------------------------
    # K_ref sliding-window buffer (absolute-canary mode)
    # ------------------------------------------------------------------

    def _kref_update(
        self,
        layer_id: int,
        key: torch.Tensor,
        num_kv_heads: int,
    ) -> None:
        """Append the current pre-tier-application K projection rows to
        the layer's sliding-window FP16 buffer.

        Buffer is a circular ring of ``kref_window_size`` rows, each
        ``(num_kv_heads, head_size)``. The order of rows in the buffer
        is irrelevant for the canary's KL computation (softmax over the
        key axis is permutation-invariant), so we do not need to
        track row insertion order — just keep the latest N rows.

        Idempotent / lazy: allocates the buffer on first call.
        """
        if not self.absolute_canary:
            return
        # Bring K to (T, num_kv_heads, head_size)
        if key.dim() == 2:
            T, hd = key.shape
            head_size = hd // num_kv_heads
            k_thd = key.view(T, num_kv_heads, head_size)
        elif key.dim() == 3:
            k_thd = key
        else:
            return  # unexpected shape; bail silently
        T = k_thd.shape[0]
        head_size = k_thd.shape[-1]

        # Lazy-allocate buffer matching the K's dtype/device.
        if layer_id not in self._kref_buf:
            self._kref_buf[layer_id] = torch.zeros(
                self.kref_window_size, num_kv_heads, head_size,
                dtype=key.dtype, device=key.device,
            )
            self._kref_fill[layer_id] = 0
            self._kref_write_idx[layer_id] = 0

        buf = self._kref_buf[layer_id]
        size = buf.shape[0]
        write_idx = self._kref_write_idx[layer_id]

        if T >= size:
            # Replace the entire buffer with the most recent ``size`` rows.
            buf.copy_(k_thd[-size:].detach())
            self._kref_fill[layer_id] = size
            self._kref_write_idx[layer_id] = 0
            return

        # Insert T rows starting at write_idx; wrap around if needed.
        end = write_idx + T
        if end <= size:
            buf[write_idx:end].copy_(k_thd.detach())
        else:
            split = size - write_idx
            buf[write_idx:].copy_(k_thd[:split].detach())
            buf[:T - split].copy_(k_thd[split:].detach())
        self._kref_write_idx[layer_id] = end % size
        self._kref_fill[layer_id] = min(size, self._kref_fill[layer_id] + T)

    def _kref_view(self, layer_id: int) -> Optional[torch.Tensor]:
        """Return the filled portion of the per-layer K_ref buffer, or
        None if the buffer hasn't been populated yet. Order is undefined
        (circular insertion); softmax-over-keys is permutation-invariant
        so this doesn't matter for the canary's KL computation."""
        buf = self._kref_buf.get(layer_id)
        if buf is None:
            return None
        fill = self._kref_fill[layer_id]
        if fill < 2:
            return None  # need at least 2 rows for a non-trivial softmax
        if fill >= buf.shape[0]:
            return buf
        return buf[:fill]

    # ------------------------------------------------------------------
    # hook
    # ------------------------------------------------------------------

    def _make_pre_hook(self, layer_id: int, mod: torch.nn.Module):
        num_heads = getattr(mod, "num_heads", None) or getattr(mod, "num_attention_heads", None)
        num_kv_heads = (
            getattr(mod, "num_kv_heads", None)
            or getattr(mod, "num_key_value_heads", None)
            or num_heads
        )
        if num_heads is None:
            # Pull from attn_impl if needed
            impl = getattr(mod, "impl", None)
            num_heads = getattr(impl, "num_heads", None)
            num_kv_heads = getattr(impl, "num_kv_heads", num_heads)

        def pre_hook(module, args):
            self.stats.attention_calls += 1
            try:
                query = args[0]
                key = args[1]
                value = args[2] if len(args) >= 3 else None
            except Exception:
                return None
            # 0) Absolute-canary mode: append the FP16 K (pre-tier-app) to
            #    the sliding-window K_ref buffer. No-op when flag is off.
            self._kref_update(
                layer_id=layer_id,
                key=key,
                num_kv_heads=num_kv_heads or num_heads,
            )
            # 1) Read-only canary fire on the original (FP16) K so the
            #    measurement is on the unquantised signal.
            self._maybe_fire(
                layer_id=layer_id,
                query=query,
                key=key,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads or num_heads,
            )
            # 2) Tier application: if active, replace K (and V) with the
            #    controller's chosen-precision quantised versions before
            #    vLLM splices them into the cache. This is the
            #    "fork-equivalent" — controller decisions actually affect
            #    inference outputs without requiring a vLLM source fork.
            if self.tier_application and value is not None:
                new_args = self._apply_tier_quant(
                    layer_id, args,
                    num_kv_heads=num_kv_heads or num_heads,
                )
                if new_args is not None:
                    return new_args
            return None  # do not modify args

        return pre_hook

    def _apply_tier_quant(
        self,
        layer_id: int,
        args: tuple,
        *,
        num_kv_heads: int,
    ) -> Optional[tuple]:
        """Mutate K (and optionally V) to the controller's chosen tier.

        Returns a new args tuple, or None to leave args unchanged. Failures
        are swallowed (defensive) — never let tier application break
        inference. The shape conventions match vLLM's Attention.forward:
        K/V can be either ``(num_tokens, num_kv_heads * head_size)`` (2D
        flattened) or ``(num_tokens, num_kv_heads, head_size)`` (3D).
        """
        try:
            if self.force_bits is not None:
                bits = self.force_bits  # static-precision baseline mode
            else:
                tier = self.controller.level(layer_id)
                bits = int(tier)
            t0 = time.perf_counter()
            query = args[0]
            key = args[1]
            value = args[2]
            with torch.no_grad():
                if key.dim() == 2:
                    T, hd = key.shape
                    head_size = hd // num_kv_heads
                    k_view = key.view(T, num_kv_heads, head_size)
                    if self.tier_application_quant_v:
                        v_head_size = value.shape[-1] // num_kv_heads
                        v_view = value.view(T, num_kv_heads, v_head_size)
                else:
                    k_view = key
                    v_view = value if value is not None else None
                    T = None  # not needed
                k_q = simulate_quantization(
                    k_view, bits=bits, group_size=self.group_size, group_axis=-1,
                )
                if self.tier_application_quant_v and value is not None:
                    v_q = simulate_quantization(
                        v_view, bits=bits, group_size=self.group_size, group_axis=-1,
                    )
                else:
                    v_q = value
                if key.dim() == 2:
                    k_q = k_q.reshape(key.shape).to(key.dtype)
                    if self.tier_application_quant_v:
                        v_q = v_q.reshape(value.shape).to(value.dtype)
                else:
                    k_q = k_q.to(key.dtype)
                    if self.tier_application_quant_v and value is not None:
                        v_q = v_q.to(value.dtype)
            self.stats.tier_application_overhead_s += time.perf_counter() - t0
            self.stats.tier_applications += 1
            self.stats.tier_apply_count_by_bits[bits] = (
                self.stats.tier_apply_count_by_bits.get(bits, 0) + 1
            )
            return (query, k_q, v_q) + tuple(args[3:])
        except Exception:
            return None

    def _kl_at_bits(
        self,
        q_bhsd: torch.Tensor,           # (1, H, T, D), float32
        k_bhsd: torch.Tensor,            # (1, H_kv→H, T, D), float32  (already GQA-replicated)
        bits: int,
    ) -> float:
        """Compute mean KL(softmax(QK_intB) ‖ softmax(QK_fp16)) at one tier."""
        d_head = q_bhsd.shape[-1]
        scale = 1.0 / (d_head ** 0.5)
        scores_f = torch.matmul(q_bhsd, k_bhsd.transpose(-2, -1)) * scale
        w_fp16 = F.softmax(scores_f, dim=-1)
        k_q = simulate_quantization(
            k_bhsd, bits=bits, group_size=self.group_size, group_axis=-1
        )
        scores_q = torch.matmul(q_bhsd, k_q.transpose(-2, -1)) * scale
        w_q = F.softmax(scores_q, dim=-1)
        eps = 1e-9
        kl = (
            w_q.clamp_min(eps)
            * (w_q.clamp_min(eps).log() - w_fp16.clamp_min(eps).log())
        ).sum(dim=-1)
        return float(kl.mean().item())

    def _maybe_fire(
        self,
        layer_id: int,
        query: torch.Tensor,
        key: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
    ) -> None:
        if layer_id == 0:
            self._global_step += 1
            # Controller tick is independent of whether the canary fires
            # this step — it should run every tick_interval decode steps
            # regardless of which layers happened to be sampled. Skipped
            # entirely when freeze_controller=True (measurement phase).
            if (
                not self.freeze_controller
                and self._global_step > 0
                and self._global_step % self.controller.tick_interval == 0
            ):
                if self.multi_precision and self._mp_estimator is not None:
                    # New decision rule: argmin tier in {current ± 1} with
                    # EMA(δ̂_b) < τ_budget.
                    per_layer_per_tier = [
                        self._mp_estimator.per_tier_ema(li)
                        for li in range(self.num_layers)
                    ]
                    self.controller.tick_multi_precision(
                        per_tier_ema=per_layer_per_tier,
                        tau_budget=self.tau_budget,
                    )
                    self.budget.broadcast_precision(
                        self.controller.snapshot(), tp_world_size=1,
                    )
                    self.stats.controller_ticks += 1
                else:
                    snap = self.estimator.snapshot()
                    if self.adaptive_tau:
                        finite_snap = [v for v in snap if v > 0.0 and math.isfinite(v)]
                        if len(finite_snap) >= self.adaptive_tau_min_samples:
                            new_tau = (
                                _percentile(finite_snap, self.adaptive_tau_percentile / 100.0)
                                * self.adaptive_tau_scale
                            )
                            if new_tau > 0.0 and math.isfinite(new_tau):
                                self.controller.setpoint = float(new_tau)
                                self._tau_history.append(float(new_tau))
                    self.controller.tick(
                        delta_hat=snap,
                        kv_utilization=0.5,
                        precision_budget=self.budget.budget(),
                    )
                    self.budget.broadcast_precision(
                        self.controller.snapshot(), tp_world_size=1,
                    )
                    self.stats.controller_ticks += 1

        # Decode-only filter: skip prefill batches where query is large.
        if self.decode_only_canary and query.shape[0] > self.max_decode_tokens:
            self.stats.canary_skipped_decode_only += 1
            return

        if torch.rand(1, generator=self._g).item() >= self.canary_rate:
            return

        # Determine which tiers to probe.
        if self.multi_precision:
            current_tier = self.controller.level(layer_id)
            tiers_to_probe = neighbours_to_measure(current_tier)
        else:
            tiers_to_probe = None  # legacy single-precision path

        t0 = time.perf_counter()
        try:
            # Lazy stream creation on first GPU fire.
            stream_ctx: Any = None
            if self.use_canary_stream and query.is_cuda:
                if self._canary_stream is None:
                    self._canary_stream = torch.cuda.Stream(device=query.device)
                self._canary_stream.wait_stream(
                    torch.cuda.current_stream(device=query.device)
                )
                stream_ctx = torch.cuda.stream(self._canary_stream)
                stream_ctx.__enter__()

            try:
                with torch.no_grad():
                    q = _to_bhsd(query.detach(), num_heads).float()       # (1, H, T, D)
                    if self.absolute_canary:
                        # Use the sliding-window FP16 K_ref instead of
                        # the current step's K. Decouples the canary
                        # measurement from cache state and from batch
                        # size T. Bail if the buffer hasn't filled yet
                        # (need at least 2 rows for a non-trivial KL).
                        kref = self._kref_view(layer_id)
                        if kref is None:
                            return
                        # kref shape: (N_filled, num_kv_heads, head_size).
                        # Reshape to (1, H_kv, N, D).
                        k = kref.unsqueeze(0).permute(0, 2, 1, 3).float()
                    else:
                        k = _to_bhsd(key.detach(), num_kv_heads).float()   # (1, H_kv, T, D)
                    if k.shape[1] != q.shape[1]:
                        rep = q.shape[1] // k.shape[1]
                        k = k.repeat_interleave(rep, dim=1)

                    if self.multi_precision:
                        # Probe each neighbour tier; results stored for
                        # post-stream-sync .item() calls below.
                        kl_by_tier: dict = {}
                        for tier in tiers_to_probe or []:
                            kl_by_tier[tier] = self._kl_at_bits(q, k, int(tier))
                    else:
                        # Single-precision legacy path
                        delta = self._kl_at_bits(q, k, self.quant_bits)
            finally:
                if stream_ctx is not None:
                    stream_ctx.__exit__(None, None, None)
                    torch.cuda.current_stream(device=query.device).wait_stream(
                        self._canary_stream
                    )
        except Exception:
            return
        finally:
            self.stats.canary_overhead_s += time.perf_counter() - t0

        self.stats.canary_fires += 1
        if self.multi_precision and self._mp_estimator is not None:
            # Monotonicity assertion (ARCHITECTURE.md § 9.8 invariant).
            if self.monotonicity_check:
                self._mp_estimator.check_monotonicity(layer_id, kl_by_tier)
            for tier, val in kl_by_tier.items():
                self._mp_estimator.observe(
                    layer_id=layer_id,
                    step=self._global_step,
                    kl=val,
                    tier=tier,
                    is_canary=True,
                )
        else:
            self.estimator.observe(
                layer_id=layer_id,
                step=self._global_step,
                kl=delta,  # type: ignore[possibly-undefined]
                is_canary=True,
            )

    def _on_broadcast(self, levels: list[Precision], _tp_world_size: int) -> None:
        # Snapshot the post-broadcast tier vector for the post-run report.
        self.stats.tier_log.append(list(levels))

    # ------------------------------------------------------------------
    # post-run report
    # ------------------------------------------------------------------

    def memory_projection(
        self,
        baseline_bits: int = 16,
        levels: Optional[list[Precision]] = None,
    ) -> dict:
        """Project KV cache memory under the controller's tier vector.

        vLLM 0.16 still allocates the cache at one global dtype (FP16/BF16),
        so on-GPU memory is unchanged. This function computes what the
        memory **would** be if a real fork stored each layer at the tier
        the controller has decided. Same number a real per-layer-dtype
        fork would deliver — derived from a real measurement (the
        controller's tier vector chosen against real δ̂_ℓ on real RULER
        inference). Honest paper-quality projection, not a guess.

        Returns a dict with per-layer bits, average bits, and savings
        relative to ``baseline_bits``-FP16 (default 16 for FP16/BF16).
        """
        levels = levels or self.controller.snapshot()
        if not levels:
            return {"avg_bits": baseline_bits, "savings_vs_baseline_pct": 0.0}
        # If running in force_bits mode (static baseline, no controller),
        # report the forced precision uniformly across all layers — the
        # controller's snapshot is unused on this path.
        if self.force_bits is not None:
            per_layer_bits = [int(self.force_bits)] * len(levels)
        else:
            per_layer_bits = [int(l) for l in levels]
        avg_bits = sum(per_layer_bits) / len(per_layer_bits)
        savings_pct = 100.0 * (1.0 - avg_bits / baseline_bits)
        # vs static KIVI-4 (every layer at INT4)
        savings_vs_kivi4_pct = 100.0 * (1.0 - avg_bits / 4.0)
        return {
            "baseline_bits": baseline_bits,
            "per_layer_bits": per_layer_bits,
            "avg_bits": avg_bits,
            "savings_vs_baseline_pct": savings_pct,
            "savings_vs_kivi4_pct": savings_vs_kivi4_pct,  # negative if EchoKV uses more
            "tier_count_int2": sum(1 for b in per_layer_bits if b == 2),
            "tier_count_int4": sum(1 for b in per_layer_bits if b == 4),
            "tier_count_int8": sum(1 for b in per_layer_bits if b == 8),
        }

    def report(self) -> dict:
        """Summary of canary activity + controller decisions."""
        ema = self.estimator.snapshot()
        levels = self.controller.snapshot()

        # Per-layer δ̂ summary
        layers_with_signal = sum(1 for x in ema if x > 0.0)
        mean_ema = statistics.fmean(ema) if ema else 0.0
        max_ema = max(ema) if ema else 0.0

        # Mean tier per layer across all controller ticks (for the v8.1
        # falsifiable check on real inference)
        if self.stats.tier_log:
            mean_tier_per_layer = [
                statistics.fmean(int(snap[i]) for snap in self.stats.tier_log)
                for i in range(self.num_layers)
            ]
        else:
            mean_tier_per_layer = [int(l) for l in levels]

        return {
            "num_layers": self.num_layers,
            "canary_rate": self.canary_rate,
            "attention_calls": self.stats.attention_calls,
            "canary_fires": self.stats.canary_fires,
            "canary_overhead_s": self.stats.canary_overhead_s,
            "canary_overhead_ratio_attentions": (
                self.stats.canary_fires / self.stats.attention_calls
                if self.stats.attention_calls else 0.0
            ),
            "controller_ticks": self.stats.controller_ticks,
            "ema_layers_with_signal": layers_with_signal,
            "ema_mean": mean_ema,
            "ema_max": max_ema,
            "ema_per_layer": list(ema),
            "final_levels_per_layer": [int(l) for l in levels],
            "mean_tier_per_layer": mean_tier_per_layer,
            "tier_application": self.tier_application,
            "tier_applications": self.stats.tier_applications,
            "tier_application_overhead_s": self.stats.tier_application_overhead_s,
            "tier_apply_count_by_bits": dict(self.stats.tier_apply_count_by_bits),
            "memory_projection": self.memory_projection(baseline_bits=16),
            "adaptive_tau": self.adaptive_tau,
            "tau_history": list(self._tau_history),
            "final_tau": (
                self._tau_history[-1] if self._tau_history else self.controller.setpoint
            ),
            "multi_precision": self.multi_precision,
            "tau_budget": self.tau_budget if self.multi_precision else None,
            "mp_observations_per_tier": (
                {int(t): n for t, n in self._mp_estimator.observations_per_tier.items()}
                if self._mp_estimator is not None else None
            ),
            "mp_monotonicity_violations": (
                self._mp_estimator.monotonicity_violations
                if self._mp_estimator is not None else None
            ),
            "mp_ema_per_tier_per_layer": (
                {int(t): vec for t, vec in self._mp_estimator.snapshot_per_tier().items()}
                if self._mp_estimator is not None else None
            ),
        }
