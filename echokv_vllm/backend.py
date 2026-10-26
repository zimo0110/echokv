"""vLLM attention backend for EchoKV's closed-loop mixed-precision KV cache.

No vLLM fork: the backend changes only what vLLM asks every backend for.

* ``customize_spec`` declares the page as one opaque byte state per token
  (``num_head_slots=1``, ``state_content_bytes`` sized for ``mem_bits``),
  so vLLM's cache manager, memory profiler and block tables budget exactly
  the bytes ``TieredKVCache`` uses. Nothing is allocated outside vLLM's pool
  except the per-layer chunk free list (8 bytes per chunk).
* ``do_kv_cache_update`` quantizes incoming K/V into the tiered store at the
  layer's current tier and pushes the full-precision keys into K_ref.
* ``forward`` reads each block at the precision it was stored at, runs
  attention, and on canary steps measures δ̂ for every tier, averages it
  across TP ranks (so all ranks reach the same decision) and ticks the
  layer's controller.

Attention here is the pure-PyTorch reference path (gather, dequantize,
SDPA per request). It runs on CPU and CUDA and is the oracle for the Triton
kernels.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right
from torch.profiler import record_function

from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.forward_context import get_forward_context
from vllm.v1.kv_cache_interface import AttentionSpec

from echokv_vllm.canary import KRefRing, canary_kl
from echokv_vllm.config import load_config
from echokv_vllm.controller import LayerController
from echokv_vllm.kernels.decode import decode_attention
from echokv_vllm.quant import SUPPORTED_BITS
from echokv_vllm.tiered_cache import TieredKVCache, TieredLayout
from echokv_vllm.write_plan import WritePlan, WritePlanner


class EchoKVAttentionBackend(AttentionBackend):
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["auto", "float16", "bfloat16"]

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        cfg = load_config()
        if spec.head_size_v != spec.head_size:
            raise NotImplementedError("EchoKV requires head_size_v == head_size")
        per_token = TieredLayout.per_token_bytes(
            spec.num_kv_heads, spec.head_size, cfg.mem_bits, cfg.k_group, cfg.v_group
        )
        return replace(spec, num_head_slots=1, state_content_bytes=per_token)

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [MultipleOf(_block_multiple())]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        return block_size is None or block_size % _block_multiple() == 0

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size % load_config().v_group == 0 and head_size % 4 == 0

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["EchoKVAttentionImpl"]:
        return EchoKVAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["EchoKVMetadataBuilder"]:
        return EchoKVMetadataBuilder


@dataclass
class EchoKVMetadata:
    num_actual_tokens: int
    num_reqs: int
    query_start_loc_cpu: torch.Tensor
    seq_lens_cpu: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    write_plan: WritePlan
    # Per-step batch layout, computed once by the builder for all layers.
    last_rows: list[int]  # each request's newest query row (canary samples)
    last_ctx: list[int]
    decode_rows: torch.Tensor | None  # [nd] query rows of single-token requests
    decode_reqs: torch.Tensor | None  # [nd] their request indices
    decode_lens: torch.Tensor | None  # [nd] their cache lengths
    prefill: list[tuple[int, int, int, int]]  # (q_start, q_end, ctx, gather_start)
    prefill_blk: torch.Tensor | None  # gather index over every prefill request's context
    prefill_off: torch.Tensor | None


class EchoKVMetadataBuilder(AttentionMetadataBuilder[EchoKVMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        cfg = load_config()
        # One planner per KV cache group: every layer applies the same plan, so
        # host-side planning (and its one device-to-host copy) happens once per step.
        self.planner = WritePlanner(kv_cache_spec.block_size, cfg.k_group, _residual_slots(cfg))

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> EchoKVMetadata:
        with record_function("echokv.build"):
            return self._build(common_attn_metadata)

    def _build(self, common_attn_metadata: CommonAttentionMetadata) -> EchoKVMetadata:
        cm = common_attn_metadata
        if cm.causal is not True:
            raise NotImplementedError("EchoKV supports causal decoder attention only")
        n = cm.num_reqs
        plan = self.planner.plan(cm.slot_mapping[: cm.num_actual_tokens].cpu(), self.device)
        qsl = cm.query_start_loc_cpu[: n + 1].tolist()
        seq_lens_cpu = cm.seq_lens[:n].cpu()
        lens = seq_lens_cpu.tolist()
        bs = self.kv_cache_spec.block_size
        last_rows, last_ctx, d_rows, d_reqs, prefill = [], [], [], [], []
        gathers = []
        g_start = 0
        for r in range(n):
            qs, qe, ctx = qsl[r], qsl[r + 1], lens[r]
            if qe == qs or ctx == 0:
                continue
            last_rows.append(qe - 1)
            last_ctx.append(ctx)
            if qe - qs == 1:
                d_rows.append(qs)
                d_reqs.append(r)
            else:
                prefill.append((qs, qe, ctx, g_start))
                gathers.append((r, ctx))
                g_start += ctx
        dev = self.device
        decode_rows = decode_reqs = decode_lens = None
        if d_rows:
            decode_rows = torch.tensor(d_rows, device=dev)
            decode_reqs = torch.tensor(d_reqs, device=dev)
            decode_lens = seq_lens_cpu[d_reqs].to(dev)
        prefill_blk = prefill_off = None
        if gathers:
            blks, offs = [], []
            for r, ctx in gathers:
                pos = torch.arange(ctx, device=dev)
                blks.append(cm.block_table_tensor[r, pos // bs].long())
                offs.append(pos % bs)
            prefill_blk, prefill_off = torch.cat(blks), torch.cat(offs)
        return EchoKVMetadata(
            num_actual_tokens=cm.num_actual_tokens,
            num_reqs=n,
            query_start_loc_cpu=cm.query_start_loc_cpu[: n + 1].cpu(),
            seq_lens_cpu=cm.seq_lens[:n].cpu(),
            block_table=cm.block_table_tensor,
            slot_mapping=cm.slot_mapping,
            write_plan=plan,
            last_rows=last_rows,
            last_ctx=last_ctx,
            decode_rows=decode_rows,
            decode_reqs=decode_reqs,
            decode_lens=decode_lens,
            prefill=prefill,
            prefill_blk=prefill_blk,
            prefill_off=prefill_off,
        )


class EchoKVAttentionImpl(AttentionImpl[EchoKVMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        unsupported = {
            "alibi": alibi_slopes is not None,
            "sliding window": sliding_window is not None,
            "logits soft cap": bool(logits_soft_cap),
            "attention sinks": sinks is not None,
            "KV sharing": kv_sharing_target_layer_name is not None,
            f"attn_type={attn_type}": attn_type != AttentionType.DECODER,
        }
        bad = [k for k, v in unsupported.items() if v]
        if bad:
            raise NotImplementedError(f"EchoKV does not support: {', '.join(bad)}")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads or num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.cfg = load_config()
        self.controller = LayerController(self.cfg.controller)
        self.kref = KRefRing(self.cfg.kref_window)
        # Full-precision K for each block's open group (+1 trash row). Allocated
        # here, during model construction, so vLLM's memory profiler counts it.
        self.residual = torch.zeros(
            _residual_slots(self.cfg) + 1,
            self.cfg.k_group,
            self.num_kv_heads,
            head_size,
            dtype=_model_dtype(),
        )
        self.store: TieredKVCache | None = None
        self._store_key: tuple | None = None
        self._step = 0
        self._fire_pending = False

    # -- cache binding ----------------------------------------------------

    def _get_store(self, kv_cache: torch.Tensor) -> TieredKVCache:
        key = (kv_cache.data_ptr(), tuple(kv_cache.shape), tuple(kv_cache.stride()))
        if self.store is not None and key == self._store_key:
            return self.store
        raw = kv_cache.view(torch.uint8)  # [B, 1, bs, C] bytes
        num_blocks, slots, block_size, row = raw.shape
        if slots != 1 or raw.stride(3) != 1 or raw.stride(2) != row:
            raise RuntimeError(
                f"unexpected EchoKV cache view shape={tuple(raw.shape)} stride={raw.stride()}"
            )
        pages = torch.as_strided(raw, (num_blocks, block_size * row), (raw.stride(0), 1))
        layout = TieredLayout(
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            block_size=block_size,
            mem_bits=self.cfg.mem_bits,
            k_group=self.cfg.k_group,
            v_group=self.cfg.v_group,
        )
        if self.residual.device != pages.device:
            self.residual = self.residual.to(pages.device)
        self.store = TieredKVCache(layout, pages, residual=self.residual)
        self._store_key = key
        return self.store

    # -- write path -------------------------------------------------------

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        store = self._get_store(kv_cache)
        md = get_forward_context().attn_metadata
        if isinstance(md, list):  # speculative decoding: [0] is the base model
            md = md[0]
        if isinstance(md, dict):
            md = md[layer.layer_name]
        plan: WritePlan = md.write_plan
        if (plan.block_size, plan.k_group) != (store.layout.block_size, store.layout.k_group):
            raise RuntimeError(
                f"EchoKV write plan ({plan.block_size}, {plan.k_group}) does not match the cache "
                f"({store.layout.block_size}, {store.layout.k_group})"
            )
        with record_function("echokv.write"):
            store.apply_plan(plan, key, value, self.controller.bits)
            if plan.rows.numel():
                self.kref.push(key[plan.rows[-self.cfg.kref_window :]])

    # -- read path --------------------------------------------------------

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: EchoKVMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not supported by EchoKV")
        if attn_metadata is None:  # profiling / warmup run
            return output
        store = self._get_store(kv_cache)
        md = attn_metadata
        if md.prefill:
            # One gather for every prefill request's context, then per-request
            # attention in the model dtype with a bottom-right causal mask
            # (lets PyTorch pick its fused SDPA kernels).
            with record_function("echokv.prefill_attn"):
                k_all, v_all = store.gather(md.prefill_blk, md.prefill_off)
                k_all, v_all = k_all.to(query.dtype), v_all.to(query.dtype)
                for qs, qe, ctx, g in md.prefill:
                    output[qs:qe] = self._attend(
                        query[qs:qe], k_all[g : g + ctx], v_all[g : g + ctx]
                    ).to(output.dtype)
        if md.decode_rows is not None:
            with record_function("echokv.decode_attn"):
                out = decode_attention(
                    store,
                    query[md.decode_rows],
                    md.block_table[md.decode_reqs],
                    md.decode_lens,
                    self.scale,
                )
                output[md.decode_rows] = out.to(output.dtype)

        self._step += 1
        if self._step % self.cfg.canary_interval == 0:
            self._fire_pending = True
        if self._fire_pending and md.last_rows:
            self._fire_pending = False
            with record_function("echokv.canary"):
                self._fire(query, md.last_rows, md.last_ctx)
        return output

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Causal attention of ``q [nq, Hq, D]`` over a request's whole context ``k, v [ctx, H, D]``."""
        nq, ctx = q.shape[0], k.shape[0]
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            attn_mask=causal_lower_right(nq, ctx),
            scale=self.scale,
            enable_gqa=True,
        )
        return out.squeeze(0).transpose(0, 1)

    # -- canary + controller ----------------------------------------------

    def _fire(self, query: torch.Tensor, rows: list[int], ctx: list[int]) -> None:
        kref = self.kref.view()
        if kref is None:
            return
        rows = rows[: self.cfg.canary_max_q]
        ctx = ctx[: self.cfg.canary_max_q]
        q = query[torch.tensor(rows, device=query.device)]
        if kref.shape[0] < self.cfg.k_group:
            return
        deltas = canary_kl(q, kref, SUPPORTED_BITS, self.cfg.k_group, self.scale)
        deltas = _tp_mean(deltas)
        values = deltas.tolist()
        self.controller.observe(dict(zip(SUPPORTED_BITS, values)), sum(ctx) / len(ctx))


def _block_multiple() -> int:
    return max(16, load_config().k_group)


def _residual_slots(cfg) -> int:
    if cfg.residual_slots is not None:
        return cfg.residual_slots
    from vllm.config import get_current_vllm_config_or_none

    vc = get_current_vllm_config_or_none()
    max_seqs = vc.scheduler_config.max_num_seqs if vc is not None else 256
    return max_seqs + 16


def _model_dtype() -> torch.dtype:
    from vllm.config import get_current_vllm_config_or_none

    vc = get_current_vllm_config_or_none()
    return vc.model_config.dtype if vc is not None else torch.float32


def _tp_mean(x: torch.Tensor) -> torch.Tensor:
    """Average across TP ranks; each rank's canary sees only its own KV heads."""
    from vllm.distributed import (
        get_tensor_model_parallel_world_size,
        tensor_model_parallel_all_reduce,
    )

    world = get_tensor_model_parallel_world_size()
    if world == 1:
        return x
    return tensor_model_parallel_all_reduce(x) / world
