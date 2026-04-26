"""Motivation study package.

Purpose: collect evidence that per-layer KV quantization error δ(ℓ, r)
is a runtime-meaningful signal — i.e. that its distribution varies
across workloads and that its magnitude correlates with task
accuracy. If the signal isn't there, the EchoKV architecture is
moot (see REPORT.md § Motivation Study gate: Spearman ρ ≥ 0.5).

The package is deliberately decoupled from vLLM / FlashInfer so it
can run with plain ``transformers`` on any PyTorch+CUDA box. The
production canary kernel (``kernels/canary_attention.py``) will
measure the same quantity at runtime with <2% TPOT overhead; this
offline harness measures it exactly, without a performance
constraint, on a controlled set of workloads.
"""
