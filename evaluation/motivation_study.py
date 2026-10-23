"""Motivation study — top-level entry point.

Thin wrapper over :mod:`evaluation.motivation`. Usage::

    python -m evaluation.motivation_study run   [args]   # collect records
    python -m evaluation.motivation_study analyze --in out.jsonl --out analysis/

Run the collection + analysis end-to-end with ``all``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .motivation import analyze as _analyze
from .motivation import run as _run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="collect δ records")
    _run._parse_args  # re-use run's flag surface via argparse passthrough
    run_p.add_argument("run_args", nargs=argparse.REMAINDER)

    ana_p = sub.add_parser("analyze", help="compute stats + plots")
    ana_p.add_argument("--in", dest="input", type=Path, required=True)
    ana_p.add_argument("--out", dest="output", type=Path, required=True)

    all_p = sub.add_parser(
        "all", help="run and then analyze using matching paths"
    )
    all_p.add_argument("run_args", nargs=argparse.REMAINDER)

    args = parser.parse_args(argv)

    if args.cmd == "run":
        return _run.main(args.run_args)
    if args.cmd == "analyze":
        result = _analyze.analyze(args.input, args.output)
        print(_analyze.format_summary(result))
        return 0
    if args.cmd == "all":
        rc = _run.main(args.run_args)
        if rc != 0:
            return rc
        # Find the output path from the parsed run config.
        cfg = _run._parse_args(args.run_args)
        result = _analyze.analyze(
            cfg.output_path, cfg.output_path.parent / "analysis"
        )
        print(_analyze.format_summary(result))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
