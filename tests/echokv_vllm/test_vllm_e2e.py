"""Boots real vLLM with the EchoKV backend. Slow (~1 min on CPU); run with ``pytest -m slow``."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

DRIVER = Path(__file__).resolve().parents[2] / "evaluation" / "serving" / "echokv_plugin_smoke.py"


@pytest.mark.slow
def test_live_mixed_tiers_in_vllm(tmp_path):
    pytest.importorskip("vllm")
    out = tmp_path / "smoke.json"
    subprocess.run(
        [sys.executable, str(DRIVER), "--tau0", "0.6", "--interval", "8", "--max-tokens", "120",
         "--out", str(out)],
        check=True,
        cwd=tmp_path,
    )
    e = json.loads(out.read_text())["echokv"]
    st = e["state"]
    assert e["ranks_agree"]
    assert all(layer["fires"] > 0 for layer in st["layers"])
    # The loose budget moves some layers to INT4 mid-generation, so those
    # layers must hold blocks at two tiers at once.
    assert any(sum(1 for v in layer["blocks_opened"].values() if v) > 1 for layer in st["layers"])
    assert all(len(t.strip()) > 0 for t in e["texts"])
