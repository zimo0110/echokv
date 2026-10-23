"""Pytest root configuration: put repo root on sys.path + register markers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: long-running tests (production canary rate, large n_steps). "
        "Skipped by default; opt in with `pytest -m slow`.",
    )


def pytest_collection_modifyitems(config, items):
    """Skip ``@pytest.mark.slow`` tests unless explicitly requested.

    ``pytest`` runs all collected tests by default; we don't want the
    slow production-rate convergence variant to run on every commit
    (it's a 10 000-step simulation). Opt in with ``pytest -m slow``
    or ``pytest --run-slow`` (alias). Honours ``-m`` overrides.
    """
    import pytest

    markexpr = config.getoption("-m") or ""
    if "slow" in markexpr:
        return  # user explicitly asked for slow tests
    skip_slow = pytest.mark.skip(reason="slow; run with `pytest -m slow`")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
