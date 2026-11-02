"""Run scripts/echokv_bench_h100.sh on Modal H100s instead of a local machine.

One-time setup:

    pip install modal
    modal token new                       # or set MODAL_TOKEN_ID / MODAL_TOKEN_SECRET

Run:

    modal run scripts/modal_bench_h100.py                         # smoke test, 1x H100, ~15 min
    modal run --detach scripts/modal_bench_h100.py --steps all    # full run, 4x H100, ~2.5 h

--steps takes the script's step names ("setup unit e2e profile8b" by
default, "all" for every step). Any 70B step gets 4 GPUs, otherwise 1.
--detach keeps the run going if this terminal closes. Results are written
to the Modal volume "echokv-results" as they are produced and copied to
results/echokv_gpu/ when the run ends; for a detached run, fetch them with
the `modal volume get` command printed at the start.

The models are ungated mirrors, so no Hugging Face token is needed. Weights
are cached in the volume "echokv-hf-cache" (~16 GB for 8B, ~140 GB for 70B).
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

LOCAL_REPO = Path(__file__).resolve().parents[1]
REMOTE_REPO = "/root/echokv"
SMOKE = "setup unit e2e profile8b"
MODEL70 = "unsloth/Llama-3.3-70B-Instruct"  # MODEL70 default in echokv_bench_h100.sh

# torch 2.13 (pinned by vLLM 0.32.0) is built for CUDA 13.0; FlashInfer
# JIT-compiles kernels at first use, so the image needs a matching nvcc.
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "build-essential")
    .pip_install("vllm==0.32.0", "datasets", "pytest", "huggingface_hub[cli]", "ninja")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .add_local_dir(
        LOCAL_REPO,
        REMOTE_REPO,
        copy=True,
        ignore=[".git", ".venv*", "**/__pycache__", "results", "LOGS", "paper", "figures"],
    )
    .run_commands(f"pip install --no-deps -e {REMOTE_REPO}")
)

app = modal.App("echokv-bench-h100", image=image)
hf_cache = modal.Volume.from_name("echokv-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("echokv-results", create_if_missing=True)


@app.function(gpu="H100", timeout=3600, volumes={"/cache": hf_cache, "/results": results})
def bench(steps: str, run_name: str) -> int:
    out = Path("/results") / run_name
    out.mkdir(parents=True, exist_ok=True)
    (Path(REMOTE_REPO) / "results").mkdir(exist_ok=True)
    (Path(REMOTE_REPO) / "results" / "echokv_gpu").symlink_to(out)

    done = threading.Event()

    def commit_periodically():  # keep partial results if the run dies or times out
        while not done.wait(300):
            results.commit()
            hf_cache.commit()

    threading.Thread(target=commit_periodically, daemon=True).start()
    env = {**os.environ, "PY": sys.executable, "HF_HOME": "/cache/hf"}
    if steps != "all":
        env["STEPS"] = steps
    rc = subprocess.run(["bash", "scripts/echokv_bench_h100.sh"], cwd=REMOTE_REPO, env=env).returncode
    done.set()
    results.commit()
    hf_cache.commit()
    return rc


@app.function(cpu=8, timeout=3600, volumes={"/cache": hf_cache})
def prefetch(model: str) -> None:
    """Download weights without paying for GPUs to wait on them."""
    from huggingface_hub import snapshot_download

    os.environ["HF_HOME"] = "/cache/hf"
    snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
    hf_cache.commit()


@app.local_entrypoint()
def main(steps: str = SMOKE, hours: float = 0):
    ngpu = 4 if steps == "all" or "70b" in steps else 1
    hours = hours or (5 if ngpu == 4 else 1.5)
    run_name = time.strftime("%Y%m%d_%H%M%S")
    print(f"== {ngpu}x H100, steps: {steps}, timeout {hours} h, run {run_name}")
    print(f"== if this terminal disconnects: modal volume get echokv-results /{run_name} results/echokv_gpu/")
    if ngpu == 4:
        print(f"== downloading {MODEL70} on CPU first")
        prefetch.remote(MODEL70)
    rc = bench.with_options(gpu=f"H100:{ngpu}", timeout=int(hours * 3600)).remote(steps, run_name)
    dest = LOCAL_REPO / "results" / "echokv_gpu"
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "modal", "volume", "get", "echokv-results", f"/{run_name}", f"{dest}/"], check=True)
    print(f"== results: {dest / run_name}/")
    if rc:
        sys.exit(rc)
