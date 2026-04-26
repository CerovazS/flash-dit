#!/usr/bin/env python
"""Standalone KAD evaluation between a generated WAV directory and a reference WAV directory.

Pipeline:

    generated/*.wav  ─┐
                      ├─► kadtk_vendored (PANNs) ──► embeddings
    reference/*.wav  ─┘

    embeddings ──► flash_dit.evaluation.mmd.mmd_unbiased
                  (chunked, bandwidth from reference, scale × 100)

Usage (directory of audio files):
    uv run python scripts/eval_kad.py \\
        --generated outputs/runs/<run_id>/generated/epoch_000123/ \\
        --reference /leonardo_scratch/large/userexternal/lcerovaz/fma/fma_medium_wav/ \\
        --model-name panns-wavegram-logmel \\
        --device cuda

Usage (manifest of absolute audio paths — e.g. FMA test split):
    uv run python scripts/eval_kad.py \\
        --generated outputs/runs/<run_id>/generated/epoch_000123/ \\
        --reference-manifest $FAST/kad_ref/fma_medium_test.txt \\
        --reference-cache-dir $FAST/kad_ref/fma_medium_test.kad_cache
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flash_dit.evaluation.kad import compute_kad
from flash_dit.evaluation.kadtk_vendored import list_models
from flash_dit.utils.console import error, info, ok


def main() -> int:
    p = argparse.ArgumentParser(description="Compute KAD between two audio sets.")
    p.add_argument("--generated", required=True, help="Directory of generated audio files")
    ref_group = p.add_mutually_exclusive_group(required=True)
    ref_group.add_argument(
        "--reference",
        default=None,
        help="Directory of reference audio files (embeddings cached in-place).",
    )
    ref_group.add_argument(
        "--reference-manifest",
        default=None,
        help="Text file with one absolute audio path per line (e.g. FMA test split).",
    )
    p.add_argument(
        "--reference-cache-dir",
        default=None,
        help="Cache dir for --reference-manifest mode (defaults to <manifest>.kad_cache).",
    )
    p.add_argument(
        "--model-name",
        default="panns-wavegram-logmel",
        choices=list_models(),
        help="embedding model (default: panns-wavegram-logmel)",
    )
    p.add_argument("--device", default="cuda", help="torch device (default: cuda)")
    p.add_argument("--workers", type=int, default=8, help="workers for embedding caching")
    p.add_argument(
        "--bandwidth-source",
        choices=["reference", "generated", "fixed"],
        default="reference",
        help="which set's median pairwise distance defines σ (default: reference, paper-correct)",
    )
    p.add_argument("--bandwidth", type=float, default=None, help="fixed σ (overrides source)")
    p.add_argument("--scale-factor", type=float, default=100.0)
    p.add_argument("--force-emb-encode", action="store_true", help="re-extract cached embeddings")
    p.add_argument(
        "--output-json",
        default=None,
        help="write metrics JSON to this path (defaults to <generated>/../metrics/kad.json)",
    )
    args = p.parse_args()

    gen = Path(args.generated)

    info(f"Generated: {gen}")
    if args.reference is not None:
        info(f"Reference: {args.reference}   (dir mode)")
    else:
        info(f"Reference: {args.reference_manifest}   (manifest mode)")
        if args.reference_cache_dir:
            info(f"Cache dir: {args.reference_cache_dir}")
    info(f"Model:     {args.model_name}   device={args.device}   workers={args.workers}")

    if args.output_json is None:
        metrics_dir = (
            gen.parent.parent / "metrics"
            if gen.parent.name.startswith("epoch_")
            else gen.parent / "metrics"
        )
        metrics_dir.mkdir(parents=True, exist_ok=True)
        output_json = metrics_dir / "kad.json"
    else:
        output_json = Path(args.output_json)

    try:
        result = compute_kad(
            generated_dir=gen,
            reference_dir=args.reference,
            reference_manifest=args.reference_manifest,
            reference_cache_dir=args.reference_cache_dir,
            model_name=args.model_name,
            device=args.device,
            workers=args.workers,
            bandwidth_source=args.bandwidth_source,
            bandwidth=args.bandwidth,
            scale_factor=args.scale_factor,
            force_emb_encode=args.force_emb_encode,
            output_json=output_json,
        )
    except Exception as e:
        error(str(e))
        return 2

    ok(
        f"KAD ({result.model_name}): raw={result.mmd2:.6f}  "
        f"scaled×{result.scale_factor:g}={result.score:.4f}  "
        f"σ={result.bandwidth:.4f} ({result.bandwidth_source})"
    )
    info(f"Wrote: {output_json}")
    print(json.dumps(result.__dict__, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
