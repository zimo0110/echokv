"""TP-aware install/read helpers for ``CanaryHarness``.

The single-GPU ``CanaryHarness`` reaches into vLLM's engine to find
the rank-0 model. With TP > 1 and the ``mp`` distributed executor,
that path only sees rank 0's slice of the layers — the other workers'
attention modules never get hooked.

This module provides helpers shippable via ``llm.apply_model`` (or
``llm.collective_rpc``) so each TP worker constructs its own
``CanaryHarness`` against its local model. Replicated controllers and
estimators are intentional and cheap: each worker observes the same
prompts at the same decode steps with the same RNG seed, so the
controllers converge to the same tier vector independently — no
cross-worker sync is needed for the controller decision.

Why this is a separate module
-----------------------------
Functions shipped via ``llm.apply_model`` are pickled by name. Picklng
fails for functions defined in ``__main__`` (the driver script) because
the worker can't resolve ``__main__.foo`` back to the same object. So
the install/read helpers must live in a regular importable module.
``static_int4_tp.py`` exists for the same reason.

Usage
-----
::

    # Driver side
    from evaluation.serving.canary_tp import (
        TPCanaryHandle, install_canary_on_worker,
    )

    handle = TPCanaryHandle(
        llm,
        harness_kwargs={"canary_rate": 1e-6, "tier_application": True,
                        "force_bits": 4, "seed": 0},
    )
    handle.install()
    try:
        outputs = llm.generate(prompts, sampling_params)
    finally:
        handle.remove()

    state = handle.snapshot()       # per-rank state for diagnostics
    tiers = state[0]["tier_vector"]  # rank-0 tiers
"""

from __future__ import annotations

import os
import sys
from functools import partial
from pathlib import Path
from typing import Any

# Ensure the worker's PYTHONPATH includes the repo root so it can
# import ``evaluation.serving.canary_vllm``. The driver typically does
# this; do it again here defensively in case the worker was spawned
# without the env var inherited.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Attribute name used to stash the worker-side harness on the model
# object. Using a model attribute (not a module-level dict) avoids
# id() collisions across worker subprocesses and survives multiple
# install/remove cycles cleanly.
_HARNESS_ATTR = "_echokv_canary_harness"


def install_canary_on_worker(model: Any, *, harness_kwargs: dict) -> int:
    """Construct a ``CanaryHarness`` against the worker's local model
    and install its forward-pre-hooks. Returns the number of hooked
    Attention modules (one per layer).

    Idempotent: if a harness is already attached to ``model`` from a
    previous install, it's removed first.

    Per-rank seeding (Phase G(e) follow-up): when
    ``harness_kwargs["per_rank_seed"]`` is True, each TP rank's
    canary RNG is seeded with ``base_seed + rank_id`` instead of
    ``base_seed`` directly. Without this, every rank sees identical
    Bernoulli canary-fire patterns but operates on different head
    subsets, producing correlated-but-different EMAs. With it, each
    rank's canary trajectory is reproducible across runs at the same
    seed, collapsing one big source of warmup nondeterminism.
    """
    from evaluation.serving.canary_vllm import CanaryHarness

    existing = getattr(model, _HARNESS_ATTR, None)
    if existing is not None:
        existing.remove()
        try:
            delattr(model, _HARNESS_ATTR)
        except AttributeError:
            pass

    kwargs = dict(harness_kwargs)
    if kwargs.pop("per_rank_seed", False):
        try:
            from vllm.distributed import get_tp_group
            rank = get_tp_group().rank_in_group
        except Exception:
            rank = 0
        kwargs["seed"] = int(kwargs.get("seed", 0)) + int(rank)

    h = CanaryHarness(model=model, **kwargs)
    h.install()
    setattr(model, _HARNESS_ATTR, h)
    return h.num_layers


def remove_canary_on_worker(model: Any) -> bool:
    """Remove the harness's hooks from the worker's model and detach.
    Returns True if a harness was attached, False otherwise."""
    h = getattr(model, _HARNESS_ATTR, None)
    if h is None:
        return False
    h.remove()
    try:
        delattr(model, _HARNESS_ATTR)
    except AttributeError:
        pass
    return True


def seed_tiers_on_worker(model: Any, *, levels: list[int]) -> int:
    """Set the worker's controller tier vector to ``levels`` (one int
    per layer, value in {2, 4, 8}). Used between warmup and measurement
    phases so all ranks measure with the same converged tier vector.
    Returns the number of layers seeded."""
    from core.block_allocator import Precision

    h = getattr(model, _HARNESS_ATTR, None)
    if h is None:
        raise RuntimeError("no harness attached on this worker")
    if len(levels) != h.num_layers:
        raise ValueError(
            f"expected {h.num_layers} levels, got {len(levels)}"
        )
    for li, lv in enumerate(levels):
        h.controller._levels[li] = Precision(int(lv))
    return h.num_layers


def snapshot_canary_on_worker(model: Any) -> dict:
    """Read back per-rank state for diagnostics. Returns a dict with:

    - ``tier_vector``: list[int] of length ``num_layers`` (each in
      {2, 4, 8})
    - ``per_layer_delta``: list[float | None] of length ``num_layers``
      — current EMA of δ̂ per layer (None if no canary has fired on
      that layer yet)
    - ``stats``: small dict of attention_calls / canary_fires /
      tier_applications counts
    - ``num_layers``: int

    Returns an empty dict if no harness is attached.
    """
    h = getattr(model, _HARNESS_ATTR, None)
    if h is None:
        return {}

    tiers = [int(h.controller.level(i)) for i in range(h.num_layers)]
    # CanaryEstimator.snapshot() returns the per-layer EMA. 0.0 is the
    # "no canary has fired yet on this layer" sentinel — distinguishing
    # it from a real-but-tiny EMA is the analysis caller's job.
    deltas = [float(v) for v in h.estimator.snapshot()]
    mp_deltas = None
    if getattr(h, "_mp_estimator", None) is not None:
        mp_deltas = h._mp_estimator.snapshot_per_tier()

    stats = {
        "attention_calls": int(h.stats.attention_calls),
        "canary_fires": int(h.stats.canary_fires),
        "tier_applications": int(h.stats.tier_applications),
    }
    out = {
        "tier_vector": tiers,
        "per_layer_delta": deltas,
        "stats": stats,
        "num_layers": h.num_layers,
    }
    if mp_deltas is not None:
        out["per_tier_delta"] = mp_deltas
    return out


# ---------------------------------------------------------------------------
# Driver-side handle: a thin facade that hides the apply_model plumbing.
# ---------------------------------------------------------------------------


class TPCanaryHandle:
    """Driver-side wrapper around a per-worker ``CanaryHarness`` state.

    Acts as a context-manager-friendly handle: ``install()`` ships
    construction kwargs to every TP worker, ``remove()`` tears down,
    ``snapshot()`` reads back per-rank state.

    All workers run the same harness construction kwargs and the same
    seed, so their tier vectors should converge identically. After a
    run, call ``snapshot()`` to verify.
    """

    def __init__(self, llm: Any, *, harness_kwargs: dict) -> None:
        self.llm = llm
        self.harness_kwargs = dict(harness_kwargs)
        self._installed = False

    def install(self) -> list[int]:
        # ``LLM.apply_model(func)`` calls ``func(model)`` on each
        # worker; it does not forward extra args. Bind ``harness_kwargs``
        # via ``functools.partial`` — the result is a picklable
        # callable that the worker invokes as
        # ``install_canary_on_worker(model, harness_kwargs=...)``.
        counts = self.llm.apply_model(
            partial(install_canary_on_worker,
                    harness_kwargs=self.harness_kwargs)
        )
        self._installed = True
        return counts

    def remove(self) -> list[bool]:
        results = self.llm.apply_model(remove_canary_on_worker)
        self._installed = False
        return results

    def seed_tiers(self, levels: list[int]) -> list[int]:
        return self.llm.apply_model(
            partial(seed_tiers_on_worker, levels=list(levels))
        )

    def snapshot(self) -> list[dict]:
        return self.llm.apply_model(snapshot_canary_on_worker)

    def __enter__(self) -> "TPCanaryHandle":
        self.install()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._installed:
            self.remove()
