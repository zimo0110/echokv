"""OpenAI-compatible HTTP server backed by vllm.LLM with our forward-
pre-hook quantization installed via apply_model.

This unblocks running benchmarks that talk to the model over HTTP
(Terminal-Bench, SWE-Bench-Verified, anything driven by litellm /
the OpenAI Python SDK) while still using our existing static-quant
hook installers and the EchoKV-3C closed-loop harness.

The standard `vllm.entrypoints.openai.api_server` constructs an
AsyncLLMEngine and doesn't expose a hook-insertion point. This
module instead uses the synchronous `vllm.LLM` class — same one
our in-process drivers (run_session14_70b_*.py,
run_session15_70b_taubench.py) use — so we can call
``llm.apply_model(install_<cfg>_hooks)`` between LLM construction
and serving. Sync LLM batches concurrent requests less efficiently
than the async engine, but for benchmarks at n_concurrent ≤ 4 the
difference is invisible.

Endpoints:
    GET  /v1/models                  — single entry: served-model-name
    POST /v1/chat/completions        — non-streaming and streaming

Usage:
    /opt/conda/envs/scalable/bin/python \
        -m evaluation.serving.vllm_server_with_hooks \
        --model /mnt/nvme_tmp6/llama-70b \
        --tensor-parallel-size 8 \
        --port 8000 \
        --served-model-name llama-3.1-70b \
        --config nvfp4

For EchoKV cells, supply a JSONL of warmup chat messages via
--echokv-warmup-jsonl; each line is a JSON object with a
``messages`` list (OpenAI chat schema). The server runs warmup
through the canary harness (controller live), consolidates the
tier vector, then either freezes (default) or stays continuously
dynamic (--echokv-dynamic).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse


log = logging.getLogger("vllm_server_hooks")
DEFAULT_70B = "/mnt/nvme_tmp6/llama-70b"


# ---------------------------------------------------------------------------
# Hook installation dispatch
# ---------------------------------------------------------------------------

def _install_static_int4(llm: Any) -> None:
    from evaluation.serving.static_int4_tp import install_static_int4_hooks
    counts = llm.apply_model(install_static_int4_hooks)
    log.info("static_int4: hooks per rank: %s", counts)


def _install_static_int8(llm: Any) -> None:
    from evaluation.serving.static_int8_tp import install_static_int8_hooks
    counts = llm.apply_model(install_static_int8_hooks)
    log.info("static_int8: hooks per rank: %s", counts)


def _install_nvfp4(llm: Any) -> None:
    from evaluation.serving.nvfp4_tp import install_nvfp4_hooks
    counts = llm.apply_model(install_nvfp4_hooks)
    log.info("nvfp4: hooks per rank: %s", counts)


# ---------------------------------------------------------------------------
# EchoKV warmup-then-serve
# ---------------------------------------------------------------------------

def _consolidate_tiers(per_rank_snapshots: list[dict]) -> list[int]:
    n = per_rank_snapshots[0]["num_layers"]
    return [max(s["tier_vector"][li] for s in per_rank_snapshots)
            for li in range(n)]


def _setup_echokv(
    llm: Any,
    *,
    warmup_jsonl: Optional[Path],
    tau_budget: float,
    observability_rate: float,
    tick_interval: int,
    kref_window_size: int,
    multi_precision: bool,
    absolute_canary: bool,
    measurement_canary_rate: float,
    measurement_freeze: bool,
    seed: int,
) -> dict:
    """Run the 3-phase warmup → consolidate → measure protocol on the
    server before opening the HTTP listener. Returns a summary blob
    with the consolidated tier vector + memory projection so callers
    can record what configuration is being served.

    The "measurement" phase here is the server's steady-state config:
    either frozen-tier (canary off, controller frozen) or continuous-
    dynamic (canary at p_c=1/64, controller unfrozen). Once setup
    completes the harness stays installed for the lifetime of the
    server.
    """
    from vllm import SamplingParams
    from evaluation.serving.canary_tp import TPCanaryHandle

    if warmup_jsonl is None or not warmup_jsonl.exists():
        raise FileNotFoundError(
            f"--echokv-warmup-jsonl is required for EchoKV configs and must "
            f"point to an existing JSONL file (got: {warmup_jsonl})"
        )

    warmup_messages: list[list[dict]] = []
    for line in warmup_jsonl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        warmup_messages.append(obj["messages"])
    log.info("EchoKV warmup: %d prompts loaded from %s",
             len(warmup_messages), warmup_jsonl)

    sampling = SamplingParams(max_tokens=512, temperature=0.0, seed=seed)

    # Phase 1 — warmup with canary observing + tier_application on
    log.info("=== EchoKV phase 1: warmup (n=%d, observability=%.2f) ===",
             len(warmup_messages), observability_rate)
    warmup_handle = TPCanaryHandle(
        llm,
        harness_kwargs={
            "canary_rate": observability_rate,
            "tau": 0.01,
            "tick_interval": tick_interval,
            "tier_application": True,
            "tier_application_quant_v": True,
            "freeze_controller": False,
            "multi_precision": multi_precision,
            "tau_budget": tau_budget,
            "absolute_canary": absolute_canary,
            "kref_window_size": kref_window_size,
            "seed": seed,
        },
    )
    warmup_handle.install()
    try:
        for msgs in warmup_messages:
            llm.chat([msgs], sampling_params=sampling)
        warmup_snapshots = warmup_handle.snapshot()
    finally:
        warmup_handle.remove()

    # Phase 2 — consolidate
    consolidated = _consolidate_tiers(warmup_snapshots)
    counts = dict(Counter(consolidated))
    avg_bits = sum(consolidated) / len(consolidated)
    log.info("    consolidated tier vector: counts=%s avg_bits=%.2f",
             counts, avg_bits)

    # Phase 3 — install the steady-state harness for serving
    proto = "continuous-dynamic" if not measurement_freeze else "frozen-tiers"
    log.info("=== EchoKV phase 3: serving with %s protocol ===", proto)
    serve_handle = TPCanaryHandle(
        llm,
        harness_kwargs={
            "canary_rate": measurement_canary_rate if not measurement_freeze else 1e-6,
            "tau": 0.01,
            "tick_interval": tick_interval,
            "tier_application": True,
            "tier_application_quant_v": True,
            "freeze_controller": measurement_freeze,
            "multi_precision": multi_precision,
            "tau_budget": tau_budget,
            "absolute_canary": absolute_canary,
            "kref_window_size": kref_window_size,
            "seed": seed + 1,
        },
    )
    serve_handle.install()
    serve_handle.seed_tiers(consolidated)

    return {
        "consolidated_tiers": consolidated,
        "tier_counts": counts,
        "avg_bits": avg_bits,
        "savings_vs_fp16_pct": 100 * (1.0 - avg_bits / 16.0),
        "protocol": proto,
        "warmup_n": len(warmup_messages),
        "warmup_tiers_per_rank": [s["tier_vector"] for s in warmup_snapshots],
    }


# ---------------------------------------------------------------------------
# OpenAI chat-completions API surface
# ---------------------------------------------------------------------------

def make_app(llm: Any, served_model_name: str, echokv_summary: dict) -> FastAPI:
    """Build a FastAPI app that exposes /v1/models and /v1/chat/completions
    backed by the in-process LLM. Streaming requests are supported via
    fake-streaming (run inference, then chunk the result into SSE events
    a few tokens at a time). True per-token streaming would require the
    async engine, which we deliberately avoided to keep apply_model
    accessible."""
    from vllm import SamplingParams

    app = FastAPI(title="EchoKV vLLM server with hooks")

    @app.get("/v1/models")
    def list_models() -> JSONResponse:
        return JSONResponse({
            "object": "list",
            "data": [{
                "id": served_model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "echokv",
                # Custom fields with our quantization config metadata
                "echokv_config": echokv_summary,
            }],
        })

    @app.get("/v1/echokv/status")
    def echokv_status() -> JSONResponse:
        """Custom endpoint exposing the current quantization summary. Lets
        a benchmark harness record which config produced its results."""
        return JSONResponse(echokv_summary)

    def _build_sampling_params(req: dict) -> Any:
        return SamplingParams(
            max_tokens=int(req.get("max_tokens", 512)),
            temperature=float(req.get("temperature", 0.7)),
            top_p=float(req.get("top_p", 1.0)),
            stop=req.get("stop"),
            seed=req.get("seed"),
        )

    def _generate_blocking(messages: list[dict], sampling: Any) -> tuple[str, dict]:
        """Run llm.chat in a worker thread. Returns (text, usage_dict)."""
        out = llm.chat([messages], sampling_params=sampling)[0]
        text = out.outputs[0].text
        n_in = len(out.prompt_token_ids) if out.prompt_token_ids else 0
        n_out = len(out.outputs[0].token_ids)
        usage = {
            "prompt_tokens": n_in,
            "completion_tokens": n_out,
            "total_tokens": n_in + n_out,
        }
        return text, usage

    def _wrap_chat_response(text: str, usage: dict, model_name: str) -> dict:
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
                "logprobs": None,
            }],
            "usage": usage,
        }

    async def _fake_stream(text: str, usage: dict, model_name: str,
                            chunk_chars: int = 16
                            ) -> AsyncGenerator[bytes, None]:
        """Fake per-token streaming by chunking the completed text into
        SSE deltas. Avoids the async-engine requirement of true
        streaming. Compatible with litellm and the OpenAI Python SDK."""
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        # Initial role event
        first = {
            "id": cid, "object": "chat.completion.chunk",
            "created": created, "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(first)}\n\n".encode()
        # Body chunks
        for i in range(0, len(text), chunk_chars):
            chunk = {
                "id": cid, "object": "chat.completion.chunk",
                "created": created, "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"content": text[i:i+chunk_chars]},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()
            await asyncio.sleep(0)  # cooperative yield
        # Final stop event
        last = {
            "id": cid, "object": "chat.completion.chunk",
            "created": created, "model": model_name,
            "choices": [{
                "index": 0, "delta": {}, "finish_reason": "stop",
            }],
            "usage": usage,
        }
        yield f"data: {json.dumps(last)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            req = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"bad JSON: {exc}")
        messages = req.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="missing 'messages'")

        # Strip OpenAI tool-call fields the chat template can't handle
        clean_messages = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                # OpenAI multimodal content list — flatten text parts only
                content = "".join(p.get("text", "") for p in content
                                  if p.get("type") == "text")
            clean_messages.append({"role": role, "content": content})

        sampling = _build_sampling_params(req)
        loop = asyncio.get_running_loop()

        try:
            text, usage = await loop.run_in_executor(
                None, _generate_blocking, clean_messages, sampling,
            )
        except Exception as exc:
            log.exception("chat_completions failed: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

        if req.get("stream"):
            return StreamingResponse(
                _fake_stream(text, usage, served_model_name),
                media_type="text/event-stream",
            )
        return JSONResponse(_wrap_chat_response(text, usage, served_model_name))

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_70B)
    p.add_argument("--tensor-parallel-size", type=int, default=8)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-num-seqs", type=int, default=8)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--served-model-name", default="llama-3.1-70b")
    p.add_argument(
        "--config", required=True,
        choices=["fp16", "static_int4", "static_int8", "nvfp4",
                 "echokv_3c_frozen", "echokv_3c_dynamic"],
    )
    # EchoKV-only flags
    p.add_argument("--echokv-warmup-jsonl", type=Path, default=None,
                   help="JSONL of {messages: [...]} entries for the EchoKV "
                        "warmup phase. Required for echokv_3c_* configs.")
    p.add_argument("--tau-budget", type=float, default=0.022)
    p.add_argument("--observability-rate", type=float, default=0.25)
    p.add_argument("--tick-interval", type=int, default=10)
    p.add_argument("--kref-window-size", type=int, default=64)
    p.add_argument("--measurement-canary-rate", type=float, default=1.0/64.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    from vllm import LLM
    log.info("loading %s tp=%d", a.model, a.tensor_parallel_size)
    t0 = time.perf_counter()
    llm = LLM(
        model=a.model,
        tensor_parallel_size=a.tensor_parallel_size,
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=a.gpu_memory_utilization,
        max_model_len=a.max_model_len,
        enable_prefix_caching=False,
        max_num_seqs=a.max_num_seqs,
    )
    log.info("loaded in %.1fs", time.perf_counter() - t0)

    # Install hooks per --config
    echokv_summary: dict = {"config": a.config}
    if a.config == "fp16":
        pass  # no hooks
    elif a.config == "static_int4":
        _install_static_int4(llm)
        echokv_summary.update({"avg_bits": 4.0})
    elif a.config == "static_int8":
        _install_static_int8(llm)
        echokv_summary.update({"avg_bits": 8.0})
    elif a.config == "nvfp4":
        _install_nvfp4(llm)
        echokv_summary.update({"avg_bits": 4.0})
    elif a.config in ("echokv_3c_frozen", "echokv_3c_dynamic"):
        ek = _setup_echokv(
            llm,
            warmup_jsonl=a.echokv_warmup_jsonl,
            tau_budget=a.tau_budget,
            observability_rate=a.observability_rate,
            tick_interval=a.tick_interval,
            kref_window_size=a.kref_window_size,
            multi_precision=True,
            absolute_canary=True,
            measurement_canary_rate=a.measurement_canary_rate,
            measurement_freeze=(a.config == "echokv_3c_frozen"),
            seed=a.seed,
        )
        echokv_summary.update(ek)
    else:
        raise ValueError(f"unhandled config: {a.config}")

    log.info("config summary: %s", echokv_summary)

    # Build and run the FastAPI app
    app = make_app(llm, a.served_model_name, echokv_summary)
    log.info("listening on http://%s:%d", a.host, a.port)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
