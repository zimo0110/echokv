"""Decode attention kernel time: EchoKV (INT8 / INT4 / INT2) vs flash-attention (bf16).

One decode step of one layer, Llama-3.1-8B shapes per GPU (8 KV heads, 32
query heads, head size 128, block size 64), captured in a CUDA graph and
replayed as the backend does, with the backend's split grid. Prints
microseconds per call and the KV bytes each read per microsecond.

    python evaluation/serving/echokv_decode_bench.py --out decode_bench.json

The ``decode`` step of scripts/echokv_bench_graphs.sh runs it (~3 min).
"""

from __future__ import annotations

import argparse
import json

import torch

from echokv_vllm.kernels.decode import BLOCKS_PER_SPLIT, decode_attention
from echokv_vllm.tiered_cache import TieredKVCache, TieredLayout

H, HQ, D, BS = 8, 32, 128, 64
MAX_SPLITS = 32  # backend.MAX_DECODE_SPLITS


def build_store(seqs: int, ctx: int, bits: int, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    nb = -(-ctx // BS)
    layout = TieredLayout(num_kv_heads=H, head_size=D, block_size=BS, mem_bits=8)
    pages = torch.zeros(seqs * nb + 2, BS * TieredLayout.per_token_bytes(H, D, 8), dtype=torch.uint8, device="cuda")
    store = TieredKVCache(layout, pages, residual_slots=seqs + 2)
    perm = torch.randperm(seqs * nb, device="cuda", generator=g)
    table = perm.view(seqs, nb).to(torch.int32)
    for s in range(seqs):
        slots = (table[s].long()[:, None] * BS + torch.arange(BS, device="cuda")).view(-1)[:ctx]
        k = torch.randn(ctx, H, D, device="cuda", generator=g, dtype=torch.bfloat16)
        v = torch.randn(ctx, H, D, device="cuda", generator=g, dtype=torch.bfloat16)
        store.write(k, v, slots, bits)
    return store, table


def time_graph(fn, iters: int) -> float:
    """Microseconds per call of ``fn`` replayed from a CUDA graph."""
    for _ in range(3):  # compile / autotune outside the graph
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters


def bench_echokv(seqs: int, ctx: int, bits: int, max_model_len: int, iters: int) -> dict:
    store, table = build_store(seqs, ctx, bits)
    q = torch.randn(seqs, HQ, D, device="cuda", dtype=torch.bfloat16)
    lens = torch.full((seqs,), ctx, device="cuda", dtype=torch.int32)
    rows = torch.arange(seqs, device="cuda")
    out = torch.empty_like(q)
    max_blocks = -(-max_model_len // BS)
    num_splits = max(1, min(-(-max_blocks // BLOCKS_PER_SPLIT), MAX_SPLITS))
    us = time_graph(
        lambda: decode_attention(store, q, table, lens, D**-0.5, BLOCKS_PER_SPLIT, q_rows=rows, bt_rows=rows,
                                 out=out, num_splits=num_splits),
        iters,
    )
    per_tok = H * D * bits // 4 + H * (D // 32) * 4 + H * D * 4 // 32  # K+V codes, V stats, K stats
    return {"us": us, "bytes_per_us": seqs * ctx * per_tok / us, "num_splits": num_splits}


def bench_flash(seqs: int, ctx: int, iters: int) -> dict:
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    nb = -(-ctx // BS)
    k = torch.randn(seqs * nb, BS, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    table = torch.arange(seqs * nb, device="cuda", dtype=torch.int32).view(seqs, nb)
    q = torch.randn(seqs, HQ, D, device="cuda", dtype=torch.bfloat16)
    cu_q = torch.arange(seqs + 1, device="cuda", dtype=torch.int32)
    lens = torch.full((seqs,), ctx, device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)
    us = time_graph(
        lambda: flash_attn_varlen_func(q, k, v, 1, cu_q, ctx, seqused_k=lens, softmax_scale=D**-0.5, causal=True,
                                       block_table=table, out=out),
        iters,
    )
    return {"us": us, "bytes_per_us": seqs * ctx * H * D * 4 / us}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shapes", nargs="*", default=["64x1024", "16x8192", "32x8192", "4x32768"],
                   help="SEQSxCTX decode batches")
    p.add_argument("--bits", type=int, nargs="*", default=[8, 4, 2])
    p.add_argument("--max-model-len", type=int, default=0,
                   help="sets the split grid as the backend would (default: each shape's context + 512)")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--label", default="")
    p.add_argument("--out", default="")
    args = p.parse_args()

    rows = []
    for shape in args.shapes:
        seqs, ctx = map(int, shape.split("x"))
        mml = args.max_model_len or ctx + 512
        row = {"shape": shape}
        try:
            row["flash_bf16"] = bench_flash(seqs, ctx, args.iters)
        except Exception as e:  # noqa: BLE001 - the baseline is optional
            row["flash_bf16"] = {"error": repr(e)[:200]}
        for b in args.bits:
            row[f"int{b}"] = bench_echokv(seqs, ctx, b, mml, args.iters)
            torch.cuda.empty_cache()
        rows.append(row)

    name = torch.cuda.get_device_name()
    print(f"\n{args.label or 'decode'} on {name}: us per call (KV bytes read per us)")
    cols = ["flash_bf16"] + [f"int{b}" for b in args.bits]
    print(f"{'shape':>10} " + " ".join(f"{c:>18}" for c in cols))
    for row in rows:
        cells = []
        for c in cols:
            r = row[c]
            cells.append("n/a" if "error" in r else f"{r['us']:8.1f} ({r['bytes_per_us'] / 1e3:5.0f}K)")
        print(f"{row['shape']:>10} " + " ".join(f"{x:>18}" for x in cells))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"label": args.label, "gpu": name, "blocks_per_split": BLOCKS_PER_SPLIT, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
