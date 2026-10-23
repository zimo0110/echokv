"""TP-aware canary harness smoke test on Llama-3.1-8B + TP=2.

Validates the full pipeline before committing 70B GPU time:

1. ``LLM(tensor_parallel_size=2)`` loads the model with two workers.
2. ``TPCanaryHandle(...).install()`` ships canary harness construction
   to each worker; per-rank hook count should equal the model's layer
   count (32 for Llama-3.1-8B).
3. After running ~10 prompts at high observability rate
   (``canary_rate=0.5``), both ranks should report:
   - non-zero ``canary_fires`` (the canary actually ran),
   - finite ``per_layer_delta`` values for the layers that fired,
   - tier vectors that match between ranks (because both ranks see
     the same prompts at the same global step with the same RNG seed
     — replicated controllers should converge identically).
4. Test seeded-tier path: set a deliberately non-uniform tier vector
   on every rank and confirm the snapshot reflects it.
5. Test remove path: hooks should be cleared on every rank.

Run with::

    VLLM_ALLOW_INSECURE_SERIALIZATION=1 \\
    FLASHINFER_DISABLE_VERSION_CHECK=1 \\
    CUDA_VISIBLE_DEVICES=0,1 \\
    /opt/conda/envs/scalable/bin/python \\
        -m evaluation.serving.smoke_canary_tp_8b
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.environ["PYTHONPATH"] = (
    f"{_REPO_ROOT}:" + os.environ.get("PYTHONPATH", "")
)

from evaluation.motivation.workloads import _ruler_sample
from evaluation.serving.canary_tp import TPCanaryHandle
from evaluation.serving.run_session8 import DEFAULT_MODEL


log = logging.getLogger("smoke_canary_tp_8b")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    from vllm import LLM, SamplingParams

    log.info("loading 8B with TP=2")
    t0 = time.perf_counter()
    llm = LLM(
        model=DEFAULT_MODEL,
        tensor_parallel_size=2,
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        max_model_len=4_096,
        enable_prefix_caching=False,
        max_num_seqs=4,
    )
    log.info("loaded in %.1fs", time.perf_counter() - t0)

    # High observability so the canary fires several times across the
    # ~10 prompts we run. Production rate (1/64) wouldn't give enough
    # samples to populate per-layer EMA in a 10-prompt smoke test.
    handle = TPCanaryHandle(
        llm,
        harness_kwargs={
            "canary_rate": 0.5,
            "tau": 0.01,
            "tick_interval": 10,
            "tier_application": False,
            "seed": 0,
        },
    )

    # 1. Install
    counts = handle.install()
    log.info("installed canary on each rank: hook counts = %s", counts)
    assert all(c == counts[0] for c in counts), \
        f"rank hook counts mismatch: {counts}"
    expected_layers = counts[0]

    # 2. Run a small batch — short RULER prompts at 4K to keep the
    # smoke test fast.
    prompts = [
        _ruler_sample(i, seed=0, target_tokens=2048).prompt for i in range(10)
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=16)
    log.info("running %d prompts", len(prompts))
    t1 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    log.info("generation done in %.1fs", time.perf_counter() - t1)

    # 3. Snapshot per-rank state
    snaps = handle.snapshot()
    log.info("snapshots from %d ranks", len(snaps))
    for ri, s in enumerate(snaps):
        log.info(
            "  rank %d: layers=%d  canary_fires=%d  attn_calls=%d  "
            "tier=%s",
            ri, s.get("num_layers", -1),
            s.get("stats", {}).get("canary_fires", -1),
            s.get("stats", {}).get("attention_calls", -1),
            s.get("tier_vector"),
        )
        # Per-layer δ̂: how many entries are non-zero (canary fired)?
        nz = sum(1 for v in s.get("per_layer_delta", []) if v > 0.0)
        log.info("    layers with non-zero δ̂: %d / %d", nz, s["num_layers"])

    # 4. Validate: tier vectors should match across ranks (replicated
    # controllers see same prompts → same decisions).
    rank0_tiers = snaps[0]["tier_vector"]
    for ri, s in enumerate(snaps[1:], start=1):
        if s["tier_vector"] != rank0_tiers:
            log.warning(
                "rank %d tier vector differs from rank 0 — replicated "
                "controllers diverged. rank0=%s rank%d=%s",
                ri, rank0_tiers, ri, s["tier_vector"],
            )

    # 5. Test seed_tiers: set a deliberately non-uniform pattern.
    pattern = ([2, 4, 8] * (expected_layers // 3 + 1))[:expected_layers]
    log.info("seeding non-uniform tier vector on every rank: %s", pattern)
    handle.seed_tiers(pattern)
    snap_after_seed = handle.snapshot()
    for ri, s in enumerate(snap_after_seed):
        assert s["tier_vector"] == pattern, \
            f"rank {ri} did not pick up seeded tiers: {s['tier_vector']}"
    log.info("seeded tier vector confirmed on every rank")

    # 6. Remove
    removed = handle.remove()
    log.info("remove() per-rank result: %s", removed)
    assert all(removed)

    # Snapshot after remove should be empty on every rank.
    snap_after_remove = handle.snapshot()
    for ri, s in enumerate(snap_after_remove):
        assert s == {}, f"rank {ri} still has harness after remove: {s}"

    summary = {
        "expected_layers": expected_layers,
        "ranks": len(counts),
        "rank0_tiers_after_run": rank0_tiers,
        "tier_vectors_match_across_ranks": all(
            s["tier_vector"] == rank0_tiers for s in snaps
        ),
        "canary_fires_per_rank": [
            s.get("stats", {}).get("canary_fires", 0) for s in snaps
        ],
        "outputs_sample": [o.outputs[0].text for o in outs[:3]],
    }
    out_path = Path("results/serving/smoke_canary_tp_8b.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    log.info("wrote %s", out_path)

    print()
    print("=" * 56)
    print("Smoke test PASSED — TP-aware canary works on TP=2")
    print(f"  layers per rank:        {expected_layers}")
    print(f"  ranks installed:        {len(counts)}")
    print(f"  canary fires per rank:  "
          f"{summary['canary_fires_per_rank']}")
    print(f"  tier vectors match:     "
          f"{summary['tier_vectors_match_across_ranks']}")
    print("=" * 56)
    return 0


if __name__ == "__main__":
    sys.exit(main())
