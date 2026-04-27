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
        llm: Any,
        canary_rate: float = DEFAULT_CANARY_RATE,
        tau: float = DEFAULT_SETPOINT,
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

        # Resolve the path to the model nn.Module (in-proc only).
        # See LOGS/session_20260426_8.md for the discovery walkthrough.
        try:
            self.model = (
                llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
            )
        except AttributeError as e:
            raise RuntimeError(
                "Cannot reach the model from the LLM instance. Ensure "
                "VLLM_ENABLE_V1_MULTIPROCESSING=0 is set so the engine "
                "runs in-process."
            ) from e

        self._attention_modules = self._collect_attention_modules()
        self.num_layers = len(self._attention_modules)
        if self.num_layers == 0:
            raise RuntimeError("found no Attention modules on the loaded model")

        self.estimator = CanaryEstimator(
            num_layers=self.num_layers,
            canary_rate=canary_rate,
        )
        self.controller = PIDPrecisionController(
            num_layers=self.num_layers,
            setpoint=tau,
            hysteresis_band=hysteresis_band,
            tick_interval=tick_interval,
            min_dwell_ticks=min_dwell_ticks,
        )
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
            # regardless of which layers happened to be sampled.
            if (
                self._global_step > 0
                and self._global_step % self.controller.tick_interval == 0
            ):
                self.controller.tick(
                    delta_hat=self.estimator.snapshot(),
                    kv_utilization=0.5,           # placeholder; vLLM's allocator info
                                                  # isn't trivially exposed at this level
                    precision_budget=self.budget.budget(),
                )
                self.budget.broadcast_precision(
                    self.controller.snapshot(), tp_world_size=1,
                )
                self.stats.controller_ticks += 1

        # Decode-only filter: skip prefill batches where query is large.
        # This is the architectural default — the canary fires at decode
        # steps with rate p_c, not at prefill batches whose attention
        # cost is already O(L²) inside the production kernel.
        if self.decode_only_canary and query.shape[0] > self.max_decode_tokens:
            self.stats.canary_skipped_decode_only += 1
            return

        if torch.rand(1, generator=self._g).item() >= self.canary_rate:
            return

        t0 = time.perf_counter()
        try:
            # Lazy stream creation on first GPU fire.
            stream_ctx: Any = None
            if self.use_canary_stream and query.is_cuda:
                if self._canary_stream is None:
                    self._canary_stream = torch.cuda.Stream(device=query.device)
                # Have the canary stream wait for the production stream so
                # it sees a consistent view of (Q, K).
                self._canary_stream.wait_stream(
                    torch.cuda.current_stream(device=query.device)
                )
                stream_ctx = torch.cuda.stream(self._canary_stream)
                stream_ctx.__enter__()

            try:
                with torch.no_grad():
                    q = _to_bhsd(query.detach(), num_heads).float()       # (1, H, T, D)
                    k = _to_bhsd(key.detach(), num_kv_heads).float()       # (1, H_kv, T, D)
                    if k.shape[1] != q.shape[1]:
                        rep = q.shape[1] // k.shape[1]
                        k = k.repeat_interleave(rep, dim=1)

                    # FP16 reference attention weights
                    d_head = q.shape[-1]
                    scale = 1.0 / (d_head ** 0.5)
                    scores_f = torch.matmul(q, k.transpose(-2, -1)) * scale
                    w_fp16 = F.softmax(scores_f, dim=-1)

                    # INT4 (KIVI sim): quantise K along head_dim group axis
                    k_int4 = simulate_quantization(
                        k, bits=self.quant_bits, group_size=self.group_size, group_axis=-1
                    )
                    scores_i = torch.matmul(q, k_int4.transpose(-2, -1)) * scale
                    w_int4 = F.softmax(scores_i, dim=-1)

                    eps = 1e-9
                    kl = (
                        w_int4.clamp_min(eps)
                        * (w_int4.clamp_min(eps).log() - w_fp16.clamp_min(eps).log())
                    ).sum(dim=-1)
                    delta_t = kl.mean()
            finally:
                if stream_ctx is not None:
                    stream_ctx.__exit__(None, None, None)
                    # Make the production stream wait for the canary
                    # stream before this pre-hook returns; this ensures
                    # the kl tensor is fully computed before .item().
                    torch.cuda.current_stream(device=query.device).wait_stream(
                        self._canary_stream
                    )
            delta = float(delta_t.item())
        except Exception:
            # Defensive: never let the canary path break inference.
            return
        finally:
            self.stats.canary_overhead_s += time.perf_counter() - t0

        self.stats.canary_fires += 1
        self.estimator.observe(
            layer_id=layer_id,
            step=self._global_step,
            kl=delta,
            is_canary=True,
        )

    def _on_broadcast(self, levels: list[Precision], _tp_world_size: int) -> None:
        # Snapshot the post-broadcast tier vector for the post-run report.
        self.stats.tier_log.append(list(levels))

    # ------------------------------------------------------------------
    # post-run report
    # ------------------------------------------------------------------

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
        }
