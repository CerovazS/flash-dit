#!/usr/bin/env python
"""Build a KAD reference manifest from the FMA tracks.csv.

FMA's audio tree (``fma_medium/000/000002.mp3`` …) is organised by ID, not by
split. The canonical split assignments live in ``fma_metadata/tracks.csv``
under the ``(set, subset)`` and ``(set, split)`` columns. This script reads
that CSV, filters by subset + split, resolves each ``track_id`` to the
corresponding audio file path, and writes a line-delimited manifest that
:func:`flash_dit.evaluation.kad.compute_kad` (and the training-time
validation KAD) can consume directly via ``reference_manifest``.

Usage:
    uv run python scripts/build_kad_reference.py \\
        --tracks-csv /leonardo_scratch/.../fma_metadata/tracks.csv \\
        --fma-dir    /leonardo_scratch/.../fma_medium \\
        --subset     medium \\
        --split      test \\
        --output     $FAST/kad_ref/fma_medium_test.txt

Defaults are wired to the CINECA paths already used by
``scripts/precompute_latents.py``. Environment variables
``FMA_AUDIO_DIR`` and ``FMA_METADATA_DIR`` override them.

The manifest is a plain text file; one absolute path per line, ``#``
comments supported. Re-run the script whenever the filter changes.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from flash_dit.utils.console import error, info, ok, warn


DEFAULT_FMA_DIR = Path(os.environ.get(
    "FMA_AUDIO_DIR",
    "/leonardo_scratch/large/userexternal/lcerovaz/fma/fma_medium",
))
DEFAULT_TRACKS_CSV = Path(os.environ.get(
    "FMA_METADATA_DIR",
    "/leonardo_scratch/large/userexternal/lcerovaz/fma/fma_metadata",
)) / "tracks.csv"


def _track_id_to_path(fma_dir: Path, tid: int) -> Path:
    """FMA convention: track 2 → ``<fma_dir>/000/000002.mp3``."""
    tid_str = f"{tid:06d}"
    return fma_dir / tid_str[:3] / f"{tid_str}.mp3"


def _load_tracks(tracks_csv: Path, subset: str, split: str) -> list[int]:
    """Return the list of track IDs matching ``(subset, split)``.

    tracks.csv has a two-row header; pandas' ``header=[0, 1]`` gives a
    MultiIndex so columns are referenced as ``("set", "subset")`` etc.
    The ``split`` column uses the long names ``training/validation/test``.
    """
    import pandas as pd

    df = pd.read_csv(str(tracks_csv), index_col=0, header=[0, 1])
    df = df[df[("set", "subset")] == subset]

    split_map = {"train": "training", "val": "validation", "test": "test"}
    split_long = split_map.get(split, split)
    df = df[df[("set", "split")] == split_long]

    return [int(tid) for tid in df.index]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--tracks-csv",
        type=Path,
        default=DEFAULT_TRACKS_CSV,
        help=f"FMA metadata tracks.csv (default: {DEFAULT_TRACKS_CSV})",
    )
    p.add_argument(
        "--fma-dir",
        type=Path,
        default=DEFAULT_FMA_DIR,
        help=f"FMA audio root (default: {DEFAULT_FMA_DIR})",
    )
    p.add_argument(
        "--subset",
        default="medium",
        choices=["small", "medium", "large"],
        help="FMA subset filter (default: medium)",
    )
    p.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="FMA split filter; long forms 'training/validation/test' are mapped (default: test)",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Manifest path to write (one absolute audio path per line).",
    )
    p.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip track IDs whose mp3 is missing on disk (default: skip + warn).",
    )
    args = p.parse_args()

    if not args.tracks_csv.exists():
        error(f"tracks.csv not found: {args.tracks_csv}")
        return 2
    if not args.fma_dir.is_dir():
        error(f"FMA audio dir not found: {args.fma_dir}")
        return 2

    info(f"Reading {args.tracks_csv}")
    info(f"Filter: subset={args.subset}, split={args.split}")
    tids = _load_tracks(args.tracks_csv, args.subset, args.split)
    info(f"Matched {len(tids)} track IDs")

    paths: list[Path] = []
    missing: list[int] = []
    for tid in tids:
        p_ = _track_id_to_path(args.fma_dir, tid)
        if p_.exists():
            paths.append(p_)
        else:
            missing.append(tid)

    if missing:
        warn(f"{len(missing)}/{len(tids)} track files missing on disk")
        if not args.skip_missing:
            # Default behaviour already drops them — flag kept for explicitness
            # and future "strict" mode.
            pass

    if not paths:
        error("No existing audio files matched — refusing to write empty manifest")
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = [
        f"# KAD reference manifest",
        f"# source: {args.tracks_csv}",
        f"# audio_root: {args.fma_dir}",
        f"# subset={args.subset} split={args.split}",
        f"# tracks_matched={len(tids)} files_present={len(paths)} missing={len(missing)}",
    ]
    with open(args.output, "w") as f:
        for line in header:
            f.write(line + "\n")
        for p_ in paths:
            f.write(str(p_.resolve()) + "\n")

    ok(f"Wrote manifest with {len(paths)} paths → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
