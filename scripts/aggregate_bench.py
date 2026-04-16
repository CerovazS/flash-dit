#!/usr/bin/env python
"""Walk a directory of bench runs and emit a combined CSV + markdown summary.

Usage:
    uv run python scripts/aggregate_bench.py --root outputs/bench
    uv run python scripts/aggregate_bench.py --root outputs/bench/sweeps/20260417_00
"""
from __future__ import annotations

import argparse
from pathlib import Path

from flash_dit.bench.reporting import aggregate_markdown


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory containing one or more result.csv files (searched recursively).",
    )
    p.add_argument(
        "--print",
        action="store_true",
        help="Also print the markdown table to stdout.",
    )
    args = p.parse_args()

    md = aggregate_markdown(args.root)
    if args.print or not (args.root / "summary.md").exists():
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
