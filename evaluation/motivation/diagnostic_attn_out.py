"""20-request diagnostic: does attention-output L2 distance correlate with ΔNLL
better than KL on attention weights does?

For each of n GSM8K requests, capture per-layer (Q, K, V), then compute:

    δ_KL  = mean_ell  KL(softmax(QK_int4/√d) ‖ softmax(QK_fp16/√d))   (current)
    δ_OUT = mean_ell  ‖softmax(QK_int4/√d)·V_int4 − softmax(QK_fp16/√d)·V_fp16‖₂
                       / √d_head                                          (new)
    ΔNLL  = NLL_INT4 − NLL_FP16   (full K+V quantisation, teacher-forcing)

Then print Spearman ρ for both candidates against ΔNLL.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from .analyze import spearman_rho
from .delta import _attention_weights, compute_request_delta
from .kivi import simulate_quantization
from .run import (
    AttentionTraceHook,
    INT4KVHook,
    _compute_nll,
    _tokenize,
    load_model,
)
from .workloads import load_workload


def attention_output_distance(
    traces, *, bits: int = 4, group_size: int = 32, causal: bool = True
) -> list[float]:
    """Per-layer ‖attn_out_int4 − attn_out_fp16‖₂ / √d_head.

    K and V are both quantised (matching the production INT4 KV cache).
    """
    out = []
    for trace in traces:
        d_head = trace.q.shape[-1]
        q = trace.q.float()
        k_fp16 = trace.k.float()
        v_fp16 = trace.v.float()

        k_int4 = simulate_quantization(
            trace.k, bits=bits, group_size=group_size, group_axis=-1
        ).float()
        v_int4 = simulate_quantization(
            trace.v, bits=bits, group_size=group_size, group_axis=-1
        ).float()

        w_fp16 = _attention_weights(q, k_fp16, causal=causal)  # (B, H, Lq, Lk)
        w_int4 = _attention_weights(q, k_int4, causal=causal)
        out_fp16 = torch.matmul(w_fp16, v_fp16)  # (B, H, Lq, D)
        out_int4 = torch.matmul(w_int4, v_int4)

        # mean over batch / heads / query positions of per-position L2
        per_pos = (out_int4 - out_fp16).norm(dim=-1)  # (B, H, Lq)
        out.append(float(per_pos.mean() / (d_head ** 0.5)))
    return out


def main(
    n: int = 40,
    seed: int = 0,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
) -> None:
    device = "cuda:0"
    model, tok = load_model(model_name, device)
    hook = AttentionTraceHook()
    hooked = hook.install(model)
    print(f"hooked {hooked} attention layers (n={n}, seed={seed})", flush=True)

    rows = []
    for i, req in enumerate(load_workload("gsm8k_cot", n, seed=seed)):
        try:
            ids = _tokenize(tok, req)["input_ids"].to(device)

            # FP16 prefill — captures Q/K/V per layer
            hook.begin_request()
            with torch.no_grad():
                model(input_ids=ids, use_cache=False)
            traces = hook.end_request()

            # δ_KL (current): per-layer KL on attention weights, K-only
            delta_kls = compute_request_delta(traces, bits=4, group_size=32).tolist()
            delta_kl_mean = statistics.fmean(delta_kls)

            # δ_OUT (new): per-layer attention-output distance, K+V quantised
            with torch.no_grad():
                delta_outs = attention_output_distance(traces, bits=4, group_size=32)
            delta_out_mean = statistics.fmean(delta_outs)

            # Free traces before generate
            del traces
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # FP16 completion (teacher-forcing target)
            with torch.no_grad():
                gen = model.generate(
                    input_ids=ids, max_new_tokens=512, do_sample=False
                )
            comp_ids = gen[0, ids.shape[-1]:].unsqueeze(0)
            if comp_ids.shape[-1] == 0:
                print(f"[{i:02d}] {req.request_id}: empty completion, skipping")
                continue

            nll_fp16 = _compute_nll(model, ids, comp_ids, eos_token_id=tok.eos_token_id)

            # NLL_INT4 with full K+V quantisation
            int4_hook = INT4KVHook(bits=4, group_size=32, k_only=False)
            int4_hook.install(model)
            try:
                nll_int4 = _compute_nll(model, ids, comp_ids, eos_token_id=tok.eos_token_id)
            finally:
                int4_hook.remove()

            delta_nll = nll_int4 - nll_fp16
            rows.append((req.request_id, delta_kl_mean, delta_out_mean, delta_nll))
            print(
                f"[{i:02d}] {req.request_id:30s}  "
                f"δ_KL={delta_kl_mean:.5f}  δ_OUT={delta_out_mean:.5f}  "
                f"ΔNLL={delta_nll:+.4f}",
                flush=True,
            )
        except Exception as e:
            print(f"[{i:02d}] {req.request_id}: FAILED {e}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not rows:
        print("no rows collected")
        return

    delta_kls = [r[1] for r in rows]
    delta_outs = [r[2] for r in rows]
    delta_nlls = [r[3] for r in rows]
    rho_kl = spearman_rho(delta_kls, delta_nlls)
    rho_out = spearman_rho(delta_outs, delta_nlls)

    print()
    print("=" * 62)
    print(f"n = {len(rows)} (GSM8K, seed=99)")
    print(f"  ρ(δ_KL,  ΔNLL) = {rho_kl:+.3f}" if rho_kl is not None else "  ρ(δ_KL, ΔNLL) = na")
    print(f"  ρ(δ_OUT, ΔNLL) = {rho_out:+.3f}" if rho_out is not None else "  ρ(δ_OUT, ΔNLL) = na")
    print(f"  mean δ_KL  = {statistics.fmean(delta_kls):.5f}")
    print(f"  mean δ_OUT = {statistics.fmean(delta_outs):.5f}")
    print(f"  mean ΔNLL  = {statistics.fmean(delta_nlls):+.4f}")
    print(f"  σ    ΔNLL  = {statistics.pstdev(delta_nlls):.4f}")
    print(f"  σ    δ_KL  = {statistics.pstdev(delta_kls):.5f}")
    print(f"  σ    δ_OUT = {statistics.pstdev(delta_outs):.5f}")

    out_path = Path("results/motivation/diagnostic_attn_out.tsv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("request_id\tdelta_kl\tdelta_out\tdelta_nll\n")
        for rid, dkl, dout, dnll in rows:
            f.write(f"{rid}\t{dkl:.6f}\t{dout:.6f}\t{dnll:.6f}\n")
    print(f"\nrows written -> {out_path}")

    hook.remove()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
