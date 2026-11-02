"""Boots real vLLM with the EchoKV backend. Slow (~1 min on CPU); run with ``pytest -m slow``."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

DRIVER = Path(__file__).resolve().parents[2] / "evaluation" / "serving" / "echokv_plugin_smoke.py"
MAX_TOKENS = 120


def run_smoke(tmp_path, *extra) -> dict:
    out = tmp_path / "smoke.json"
    subprocess.run(
        [sys.executable, str(DRIVER), "--tau0", "0.6", "--interval", "8", "--max-tokens", str(MAX_TOKENS),
         "--mem-bits", "8", "--init-bits", "8",
         "--out", str(out), *extra],
        check=True,
        cwd=tmp_path,
        # Every step: plan from the host slot mapping and check it against vLLM's.
        env=dict(os.environ, ECHOKV_CHECK_HOST_SLOTS="1"),
    )
    e = json.loads(out.read_text())["echokv"]
    st = e["state"]
    assert e["ranks_agree"]
    assert all(layer["fires"] > 0 for layer in st["layers"])
    # The loose budget moves some layers to INT4 mid-generation, so those
    # layers must hold blocks at two tiers at once.
    assert any(sum(1 for v in layer["blocks_opened"].values() if v) > 1 for layer in st["layers"])
    assert all(len(t.strip()) > 0 for t in e["texts"])
    return st


@pytest.mark.slow
def test_live_mixed_tiers_in_vllm(tmp_path):
    pytest.importorskip("vllm")
    run_smoke(tmp_path)


@pytest.mark.slow
def test_decode_steps_replay_cuda_graphs(tmp_path):
    """With CUDA graphs on, decode-only steps replay a full graph: the canary
    and controller still work, and forward() runs only for the mixed steps."""
    pytest.importorskip("vllm")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    st = run_smoke(tmp_path, "--cudagraph", "--dtype", "bfloat16")  # as real models run
    assert all(layer["forward_calls"] < MAX_TOKENS // 2 for layer in st["layers"])
