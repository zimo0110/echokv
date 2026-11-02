"""vLLM attention backend for EchoKV's closed-loop mixed-precision KV cache.

No vLLM fork: the backend changes only what vLLM asks every backend for.

* ``customize_spec`` declares the page as one opaque byte state per token
  (``num_head_slots=1``, ``state_content_bytes`` sized for ``mem_bits``),
  so vLLM's cache manager, memory profiler and block tables budget exactly
  the bytes ``TieredKVCache`` uses. Nothing is allocated outside vLLM's pool
  except the per-layer chunk free list (8 bytes per chunk).
* The metadata builder does every host-side step once per step, for all
  layers: it plans the step's writes (write_plan.py), opens blocks at each
  layer's current tier, uploads all metadata in one copy, and runs the
  canary for every layer at once on the step after a fire (δ̂ for every
  tier, averaged across TP ranks so all ranks reach the same decision,
  then each layer's controller ticks).
* ``do_kv_cache_update`` and ``forward`` only launch kernels on device
  metadata: quantize incoming K/V into the tiered store, read each block at
  the precision it was stored at, and stash the canary's query rows. So
  decode-only batches run as full CUDA graphs, like stock attention.

Without Triton (CPU, or head sizes that are not powers of 2) the kernels
fall back to the pure-PyTorch reference path (gather, dequantize, SDPA per
request), which is also the oracle the Triton kernels are tested against.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np
import torch
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
from vllm.v1.attention.backends.utils import get_num_attention_heads_from_layers
from vllm.v1.kv_cache_interface import AttentionSpec

from echokv_vllm._util import to_device
from echokv_vllm.canary import canary_kl, ring_view
from echokv_vllm.config import load_config
from echokv_vllm.controller import LayerController
from echokv_vllm.host_slots import host_lens_and_slots, is_dummy_step
from echokv_vllm.kernels.decode import BLOCKS_PER_SPLIT, _use_triton, decode_attention
from echokv_vllm.kernels.prefill import (
    FUSED_MAX_CHUNK,
    build_dequant_work,
    build_tiles,
    prefill_attention,
)
from echokv_vllm.quant import SUPPORTED_BITS
from echokv_vllm.tiered_cache import TieredKVCache, TieredLayout
from echokv_vllm.write_plan import SharedState, WriteMeta

# Upper bound on decode splits per sequence: the split grid is fixed so a
# CUDA graph can replay it, and longer sequences use longer splits.
MAX_DECODE_SPLITS = 32


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
    write: WriteMeta | None  # None: no KV cache bound yet (memory profiling); attention is skipped
    canary_rows: torch.Tensor | None  # [canary_max_q] query rows to stash for the canary
    fire: bool  # this step's query rows feed the canary (computed at the next build)
    decode_rows: torch.Tensor | None  # [nd] query rows of single-token requests
    decode_reqs: torch.Tensor | None  # [nd] their request indices
    decode_lens: torch.Tensor | None  # [nd] their cache lengths (0 = CUDA-graph padding)
    num_splits: int  # decode split grid (fixed, so CUDA graphs can replay it)
    prefill: list[tuple[int, int, int, int]]  # (q_start, q_end, ctx, gather_start)
    prefill_reqs: list[int]  # request index (block-table row) of each prefill entry
    prefill_tiles: torch.Tensor | None  # fused Triton prefill work list (tiny chunks)
    prefill_dense: tuple | None  # dequantize + flash-attention inputs (see kernels/prefill.py)
    prefill_blk: torch.Tensor | None  # PyTorch path: gather index over all prefill contexts
    prefill_off: torch.Tensor | None


class _Staging:
    """One step's int64 metadata in one device buffer, filled by one host-to-device copy.

    Layout: ``[n_fin, 0 | canary rows | decode (3 x cd) | tokens (4 x ct) |
    completed groups (cf x w) | tail]``, regions 16-byte aligned (even
    element offsets) so every step specializes the kernels alike. Steps that fit ``cap`` use a
    persistent buffer whose graph-visible parts (everything before the
    tail) sit at fixed offsets, so a captured CUDA graph reads the current
    step's values; the tail (residual map, page headers) is consumed by
    the builder's own ops. Larger steps (big prefills, never graph-replayed)
    use a fresh buffer of exactly their size.
    """

    RING = 4  # pinned host buffers in flight

    def __init__(self, cap: int, nq: int, w: int, tail: int, device) -> None:
        cap += cap % 2
        self.cap, self.nq, self.w, self.tail = cap, nq, w, tail
        self.o_d = 2 + nq + nq % 2
        self.device = torch.device(device)
        size = self.o_d + 7 * cap + cap * w + tail
        self.dev = torch.zeros(size, dtype=torch.long, device=self.device)
        cuda = self.device.type == "cuda"
        self.host = [torch.zeros(size, dtype=torch.long, pin_memory=cuda) for _ in range(self.RING if cuda else 1)]
        self.events: list = [None] * len(self.host)
        self.i = 0

    def stage(self, n_fin, canary, decode, tok, fin, tail):
        """Upload; returns device views ``(fin_count, canary, decode, tok, fin, tail)``."""
        nq, w = self.nq, self.w
        s, t = decode[0].size, tok[0].size
        sizes = [a.size for a in tail]
        persistent = max(s, t, n_fin) <= self.cap and sum(sizes) <= self.tail
        cd = ct = self.cap if persistent else None
        if persistent:
            cf = self.cap
            buf = self.host[self.i]
            if self.events[self.i] is not None:
                self.events[self.i].synchronize()  # its previous copy has long finished
            h = buf.numpy()
        else:
            cd, ct, cf = s + s % 2, t + t % 2, n_fin
            h = np.zeros(self.o_d + 3 * cd + 4 * ct + cf * w + sum(sizes), dtype=np.int64)
        o_d = self.o_d
        o_t = o_d + 3 * cd
        o_f = o_t + 4 * ct
        o_tail = o_f + n_fin * w
        h[0] = n_fin
        h[2 : 2 + nq] = canary
        for k, a in enumerate(decode):
            h[o_d + k * cd : o_d + k * cd + s] = a
        for k, a in enumerate(tok):
            h[o_t + k * ct : o_t + k * ct + t] = a
        h[o_f:o_tail] = fin.reshape(-1)
        o = o_tail
        for a in tail:
            h[o : o + a.size] = a.reshape(-1)
            o += a.size
        if persistent:
            dev = self.dev
            dev[:o].copy_(buf[:o], non_blocking=True)
            if self.device.type == "cuda":
                ev = self.events[self.i] or torch.cuda.Event()
                ev.record()
                self.events[self.i] = ev
            self.i = (self.i + 1) % len(self.host)
        else:
            dev = to_device(torch.from_numpy(h), self.device)
        views, o = [], o_tail
        for n in sizes:
            views.append(dev[o : o + n])
            o += n
        return (
            dev[0:1],
            dev[2 : 2 + nq],
            [dev[o_d + k * cd : o_d + k * cd + s] for k in range(3)],
            dev[o_t : o_t + 4 * ct].view(4, ct)[:, :t],
            dev[o_f : o_f + cf * w].view(cf, w),
            views,
        )


class EchoKVMetadataBuilder(AttentionMetadataBuilder[EchoKVMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec) -> AttentionCGSupport:
        # Decode-only batches can replay a full CUDA graph: build() makes every
        # host-side decision and forward() only launches Triton kernels.
        # ECHOKV_CUDAGRAPH=0 turns that off (attention then runs eagerly).
        d, bs = kv_cache_spec.head_size, kv_cache_spec.block_size
        if (
            os.environ.get("ECHOKV_CUDAGRAPH", "1") != "1"
            or not _use_triton(torch.device("cuda"))
            or d & (d - 1)
            or bs & (bs - 1)
        ):
            return AttentionCGSupport.NEVER
        return cls._cudagraph_support

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.cfg = cfg = load_config()
        num_heads = get_num_attention_heads_from_layers(vllm_config, layer_names) or (
            vllm_config.model_config.get_num_attention_heads(vllm_config.parallel_config)
        )
        self.gqa = num_heads // kv_cache_spec.num_kv_heads
        self.use_triton = _use_triton(torch.device(device))
        # Host slot mapping needs exact CPU sequence lengths, which async
        # speculative decoding does not provide (see host_slots.py).
        self.host_slots = vllm_config.speculative_config is None
        self.check_host_slots = os.environ.get("ECHOKV_CHECK_HOST_SLOTS", "0") == "1"
        self.host_slot_steps = 0  # steps planned from the host mapping (tests read this)
        bs = kv_cache_spec.block_size
        max_blocks = -(-vllm_config.model_config.max_model_len // bs)
        self.num_splits = max(1, min(-(-max_blocks // BLOCKS_PER_SPLIT), MAX_DECODE_SPLITS))
        self.residual_slots = _residual_slots(cfg, vllm_config)
        self.cap = getattr(vllm_config.compilation_config, "max_cudagraph_capture_size", None) or 512
        # Bound on attach: the layers, their impls and the state they share.
        self._layers: list | None = None
        self._impls: list[EchoKVAttentionImpl] = []
        self._cache_ptr: int | None = None
        self.shared: SharedState | None = None
        self._staging: _Staging | None = None
        # Canary schedule (one for all layers).
        self._step = 0
        self._fire_pending = False
        self._measure: tuple[int, float] | None = None  # (rows, mean ctx) stashed last step
        self._pending: tuple[torch.Tensor, float] | None = None  # last δ̂ [layers, tiers], not yet consumed

    # -- binding ----------------------------------------------------------

    def _attach(self) -> bool:
        """Bind every layer of this group to its KV cache and the shared state."""
        if self._layers is None:
            from vllm.config import get_layers_from_vllm_config
            from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

            self._layers = list(
                get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase, self.layer_names).values()
            )
        first = getattr(self._layers[0], "kv_cache", None) if self._layers else None
        if first is not None and self._cache_ptr is not None and first.data_ptr() == self._cache_ptr:
            return True  # every step after the first
        caches = [getattr(layer, "kv_cache", None) for layer in self._layers]
        if not caches or any(c is None or c.numel() == 0 for c in caches):
            return False  # memory profiling: no KV cache yet
        num_blocks = caches[0].shape[0]
        cfg = self.cfg
        self.shared = SharedState(
            num_blocks, self.kv_cache_spec.block_size, cfg.k_group, self.residual_slots,
            cfg.kref_window, self.device,
        )
        self._impls = [layer.impl for layer in self._layers]
        for impl, cache in zip(self._impls, caches):
            impl.bind(cache, self.shared)
        tail = self.cap * (4 + 3 * len(self._impls)) + 64
        self._staging = _Staging(self.cap, cfg.canary_max_q, 3 + cfg.k_group, tail, self.device)
        self._cache_ptr = caches[0].data_ptr()
        self._measure = self._pending = None
        return True

    # -- canary -----------------------------------------------------------

    def _run_canary(self) -> None:
        """δ̂ for every layer from the query rows stashed on the fire step."""
        nq, ctx_mean = self._measure
        self._measure = None
        # The controllers consume the previous measurement (``canary_interval``
        # steps old, long finished on the GPU), so reading it does not stall the
        # stream the way reading this one immediately would.
        if self._pending is not None:
            deltas, prev_ctx = self._pending
            self._pending = None
            for impl, d in zip(self._impls, deltas.tolist()):
                impl.controller.observe(dict(zip(SUPPORTED_BITS, d)), prev_ctx)
        pl = self.shared.planner
        if pl.kref_fill < self.cfg.k_group:
            return
        with record_function("echokv.canary"):
            kref = ring_view(torch.stack([i.kref_buf for i in self._impls]), pl.kref_pos, pl.kref_fill, dim=1)
            q = torch.stack([i.stash[:nq] for i in self._impls])
            deltas = _tp_mean(canary_kl(q, kref, SUPPORTED_BITS, self.cfg.k_group, self._impls[0].scale))
        self._pending = (deltas, ctx_mean)

    # -- build ------------------------------------------------------------

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
        t = cm.num_actual_tokens
        bs = self.kv_cache_spec.block_size
        dummy = is_dummy_step(cm)
        host = host_lens_and_slots(cm, bs) if self.host_slots else None
        if host is not None:
            seq_lens_cpu, slots_cpu = host
            self.host_slot_steps += 1
        if host is None and self.check_host_slots and self.host_slots and not dummy:
            raise AssertionError("ECHOKV_CHECK_HOST_SLOTS=1 but the host slot mapping is unavailable")
        if host is None or self.check_host_slots:
            # Device-to-host copies: each waits for the GPU to drain. Normal
            # steps avoid them; dummy steps (all-PAD slots) always take them.
            dev_lens, dev_slots = cm.seq_lens[:n].cpu(), cm.slot_mapping[:t].cpu()
            if host is not None and not (torch.equal(seq_lens_cpu, dev_lens.long()) and torch.equal(slots_cpu, dev_slots.long())):
                raise AssertionError("EchoKV host slot mapping differs from vLLM's")
            seq_lens_cpu, slots_cpu = dev_lens, dev_slots

        # Batch layout, vectorized over requests.
        qsl = cm.query_start_loc_cpu[: n + 1].numpy().astype(np.int64)
        lens = seq_lens_cpu.numpy().astype(np.int64)
        qlen = np.diff(qsl)
        live = (qlen > 0) & (lens > 0)
        last_rows = (qsl[1:] - 1)[live]
        last_ctx = lens[live]
        if (qlen <= 1).all():
            # Decode-only (as CUDA graphs replay it): one decode entry per request,
            # padding requests included with length 0.
            d_idx = np.arange(n)
        else:
            d_idx = np.flatnonzero(live & (qlen == 1))
        d_lens = np.where(qlen[d_idx] == 1, lens[d_idx], 0)
        d_rows = np.minimum(qsl[:-1][d_idx], max(t - 1, 0))
        prefill, gathers, g_start = [], [], 0
        for r in np.flatnonzero(live & (qlen > 1)).tolist():
            qs, qe, ctx = int(qsl[r]), int(qsl[r + 1]), int(lens[r])
            prefill.append((qs, qe, ctx, g_start))
            gathers.append((r, ctx))
            g_start += ctx

        attached = self._attach()
        write = canary_rows = None
        fire = False
        decode_rows = decode_reqs = decode_lens = None
        if attached:
            if self._measure is not None:
                self._run_canary()  # before this step's plan moves the K_ref ring
            if not dummy:
                self._step += 1
                if self._step % self.cfg.canary_interval == 0:
                    self._fire_pending = True
            nq = self.cfg.canary_max_q
            canary = np.zeros(nq, dtype=np.int64)
            if self._fire_pending and last_rows.size:
                self._fire_pending = False
                fire = True
                k = min(nq, last_rows.size)
                canary[:k] = last_rows[:k]
                self._measure = (k, float(last_ctx[:k].mean()))
            write, canary_rows, decode = self._plan(slots_cpu, canary, (d_rows, d_idx, d_lens))
            if d_idx.size:
                decode_rows, decode_reqs, decode_lens = decode

        dev = self.device
        prefill_reqs = [r for r, _ in gathers]
        prefill_tiles = prefill_dense = prefill_blk = prefill_off = None
        if gathers and self.use_triton:
            nqs = [qe - qs for qs, qe, _, _ in prefill]
            if max(nqs) <= FUSED_MAX_CHUNK:
                prefill_tiles = build_tiles(prefill, prefill_reqs, self.gqa, dev)
            else:
                ctxs = [ctx for _, _, ctx, _ in prefill]
                q_rows = torch.cat([torch.arange(qs, qe) for qs, qe, _, _ in prefill])
                cu_q = torch.tensor([0] + list(itertools.accumulate(nqs)), dtype=torch.int32)
                cu_k = torch.tensor([0] + list(itertools.accumulate(ctxs)), dtype=torch.int32)
                prefill_dense = (
                    build_dequant_work(prefill, prefill_reqs, bs, dev),
                    g_start,
                    to_device(q_rows, dev),
                    to_device(cu_q, dev),
                    to_device(cu_k, dev),
                    max(nqs),
                    max(ctxs),
                )
        elif gathers:
            blks, offs = [], []
            for r, ctx in gathers:
                pos = torch.arange(ctx, device=dev)
                blks.append(cm.block_table_tensor[r, pos // bs].long())
                offs.append(pos % bs)
            prefill_blk, prefill_off = torch.cat(blks), torch.cat(offs)
        return EchoKVMetadata(
            num_actual_tokens=t,
            num_reqs=n,
            query_start_loc_cpu=cm.query_start_loc_cpu[: n + 1].cpu(),
            seq_lens_cpu=seq_lens_cpu,
            block_table=cm.block_table_tensor,
            slot_mapping=cm.slot_mapping,
            write=write,
            canary_rows=canary_rows,
            fire=fire,
            decode_rows=decode_rows,
            decode_reqs=decode_reqs,
            decode_lens=decode_lens,
            num_splits=self.num_splits,
            prefill=prefill,
            prefill_reqs=prefill_reqs,
            prefill_tiles=prefill_tiles,
            prefill_dense=prefill_dense,
            prefill_blk=prefill_blk,
            prefill_off=prefill_off,
        )

    def _plan(self, slots_cpu: torch.Tensor, canary: np.ndarray, decode: tuple):
        """Host side of the step's writes for every layer, then one upload."""
        shared = self.shared
        plan = shared.planner.plan(slots_cpu)
        # Which blocks open is the same for every layer: offset-0 writes, plus
        # writes into blocks never opened (vLLM warmup batches), which are
        # never-opened in every layer alike. The tier each opens at is not.
        store0 = self._impls[0].store
        opens = plan.open_np
        if plan.touched_np.size:
            fresh = plan.touched_np[store0._tag_h[plan.touched_np] == 0]
            if fresh.size:
                opens = np.union1d(opens, fresh)
        opened = []
        if opens.size or plan.evictions:
            for impl in self._impls:
                st = impl.store
                for ev in plan.evictions:  # rare; before the slot is reused below
                    st._seal(ev)
                if opens.size:
                    opened.append(st.open_blocks(opens, impl.controller.bits, protect=plan.touched_np))
        shared.commit(plan)
        tail = [plan.res_map_block, plan.res_map_value, plan.res_group_slot, plan.res_group_value]
        for blocks, headers in opened:
            tail += [blocks, headers.view(np.int64).reshape(-1)]
        fin_count, canary_rows, dec, tok, fin, views = self._staging.stage(
            plan.n_fin, canary, decode,
            (plan.tok_slot, plan.tok_res, plan.tok_sealed, plan.tok_kref), plan.fin, tail,
        )
        shared.apply_res_map(*views[:4])
        for impl, (ids, hdr) in zip(self._impls, zip(views[4::2], views[5::2])):
            impl.store.write_headers(ids, hdr.view(-1, 2).view(torch.uint8))
        write = WriteMeta(tok=tok, fin=fin, fin_count=fin_count, n_rows=plan.n_rows, n_fin=plan.n_fin)
        return write, canary_rows, dec


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
        self.cfg = cfg = load_config()
        self.controller = LayerController(cfg.controller)
        # Allocated here, during model construction, so vLLM's memory profiler
        # counts them: full-precision K for each block's open group (+1 spare
        # row), the K_ref ring, and the canary's stashed query rows.
        self.residual = torch.zeros(
            _residual_slots(cfg) + 1, cfg.k_group, self.num_kv_heads, head_size, dtype=_model_dtype()
        )
        # Explicit dtype: vLLM builds layers under the model's default dtype.
        self.kref_buf = torch.zeros(cfg.kref_window, self.num_kv_heads, head_size, dtype=torch.float32)
        self.stash = torch.zeros(cfg.canary_max_q, num_heads, head_size, dtype=_model_dtype())
        self.store: TieredKVCache | None = None
        self.forward_calls = 0  # eager forwards; CUDA-graph replays do not enter Python

    def bind(self, kv_cache: torch.Tensor, shared: SharedState) -> None:
        """Wrap ``kv_cache`` (this layer's slice of vLLM's pool) in a tiered store."""
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
        dev = pages.device
        self.residual = self.residual.to(dev)
        self.kref_buf = self.kref_buf.to(dev)
        self.stash = self.stash.to(dev)
        self.store = TieredKVCache(layout, pages, residual=self.residual, shared=shared)

    # -- write path -------------------------------------------------------

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        md = get_forward_context().attn_metadata
        if isinstance(md, list):  # speculative decoding: [0] is the base model
            md = md[0]
        if isinstance(md, dict):
            md = md[layer.layer_name]
        if md is None or md.write is None:
            return
        with record_function("echokv.write"):
            self.store.apply(md.write, key, value, self.kref_buf, capturing=_capturing(key))

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
        md = attn_metadata
        if md is None or md.write is None:  # profiling / warmup run
            return output
        self.forward_calls += 1
        store = self.store
        if md.prefill:
            # Triton: one fused kernel reading the tiered pages directly.
            # PyTorch: one gather for all contexts, then SDPA per request.
            with record_function("echokv.prefill_attn"):
                gather = None if md.prefill_blk is None else (md.prefill_blk, md.prefill_off)
                prefill_attention(
                    store, query, output, md.prefill, md.prefill_reqs, md.block_table,
                    self.scale, tiles=md.prefill_tiles, gather=gather, dense=md.prefill_dense,
                )
        if md.decode_rows is not None:
            with record_function("echokv.decode_attn"):
                # Reads its query / block-table rows by index and writes its
                # output rows directly: no gathers, casts or scatters around it.
                decode_attention(
                    store,
                    query,
                    md.block_table,
                    md.decode_lens,
                    self.scale,
                    q_rows=md.decode_rows,
                    bt_rows=md.decode_reqs,
                    out=output.view(output.shape[0], -1, query.shape[-1]),
                    num_splits=md.num_splits,
                )
        # The canary's query rows; the builder measures δ̂ at the next step. A
        # graph captures the copy unconditionally (its rows are valid either way).
        if md.fire or _capturing(query):
            torch.index_select(query, 0, md.canary_rows, out=self.stash)
        return output


def _capturing(t: torch.Tensor) -> bool:
    return t.is_cuda and torch.cuda.is_current_stream_capturing()


def _block_multiple() -> int:
    return max(16, load_config().k_group)


def _residual_slots(cfg, vllm_config=None) -> int:
    if cfg.residual_slots is not None:
        return cfg.residual_slots
    if vllm_config is None:
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
    max_seqs = vllm_config.scheduler_config.max_num_seqs if vllm_config is not None else 256
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
