#!/usr/bin/env python
"""Standalone FAD evaluation between a generated directory and reference directory.

Usage:
    uv run python scripts/eval_fad.py \\
        --generated outputs/samples_20260415/ \\
        --reference /leonardo_scratch/large/userexternal/lcerovaz/fma/fma_medium_wav/
"""
from __future__ import annotations

import argparse
from pathlib import Path

from flash_dit.evaluation.fad import compute_fad
from flash_dit.utils.console import info, ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--generated",  required=True, help="Directory of generated WAV files")
    p.add_argument("--reference",  required=True, help="Directory of reference (real) WAV files")
    args = p.parse_args()

    gen_dir = Path(args.generated)
    ref_dir = Path(args.reference)

    n_gen = len(list(gen_dir.glob("*.wav")))
    n_ref = len(list(ref_dir.glob("*.wav")))
    info(f"Generated: {n_gen} files   Reference: {n_ref} files")

    fad = compute_fad(gen_dir, ref_dir)
    ok(f"FAD (CLAP-LAION-audio): {fad:.4f}")


if __name__ == "__main__":
    main()
