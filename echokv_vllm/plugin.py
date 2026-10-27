"""vLLM general plugin: installs the EchoKV backend when ``ECHOKV_ENABLE=1``.

Registered under the ``vllm.general_plugins`` entry point, so vLLM calls
``register`` in every process (engine core and each worker) before any
model is built. On CUDA select it with ``attention_config={"backend":
"CUSTOM"}``; the CPU platform always picks ``CPU_ATTN``, so there we
override that slot instead.
"""

from __future__ import annotations

import os

BACKEND_PATH = "echokv_vllm.backend.EchoKVAttentionBackend"


def register() -> None:
    if os.environ.get("ECHOKV_ENABLE", "0") != "1":
        return
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    from echokv_vllm.host_slots import install

    install()
    register_backend(AttentionBackendEnum.CUSTOM, BACKEND_PATH)
    if current_platform.is_cpu():
        register_backend(AttentionBackendEnum.CPU_ATTN, BACKEND_PATH)
