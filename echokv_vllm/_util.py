"""Small helpers shared by the backend and kernels."""

from __future__ import annotations

import torch


def to_device(t: torch.Tensor, device) -> torch.Tensor:
    """Host tensor -> ``device`` without blocking the CPU.

    A plain ``.to("cuda")`` from pageable memory synchronizes the stream, so
    the CPU stalls until all queued GPU work finishes. That serializes the
    host-side write path with the GPU, layer by layer. A pinned source plus a
    non-blocking copy keeps the CPU running ahead; PyTorch's pinned-memory
    allocator keeps the source alive until the copy has executed.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return t.to(device)
    return t.pin_memory().to(device, non_blocking=True)
