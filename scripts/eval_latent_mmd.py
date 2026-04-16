#!/usr/bin/env python
"""Standalone latent-space MMD evaluation from a flash-dit checkpoint.

Generates N latents from the EMA model in a checkpoint, loads a reference
feature matrix from the HDF5 latent dataset, and writes the unbiased MMD^2
score (raw and scaled) to both a JSON record and a JSONL row. No VAE
decoding happens so this runs in seconds.

Usage:
    uv run python scripts/eval_latent_mmd.py \\
        --checkpoint outputs/runs/<run_id>/checkpoints/last.ckpt \\
        --h5-path /leonardo_scratch/large/userexternal/lcerovaz/flash-dit-latents/stable_audio_open/fma_medium.h5 \\
        --split val \\
        --n-generated 1024 --n-reference 1024 \\
        --feature-types frame clip_pooled \\
        --cfg-scale 4.0 --sampler heun --n-steps 50 \\
        --output-dir outputs/latent_mmd_eval
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

from flash_dit.evaluation.latent_mmd import (
    ReferenceFeatureSpec,
    compute_latent_mmd,
    load_reference_features,
)
from flash_dit.utils.console import info, ok, warn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute latent-space MMD from a checkpoint.")
    p.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    p.add_argument("--h5-path", required=True, help="Path to pre-computed latent HDF5 dataset")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--n-generated", type=int, default=1024)
    p.add_argument("--n-reference", type=int, default=1024)
    p.add_argument(
        "--feature-types",
        nargs="+",
        default=["frame", "clip_pooled"],
        choices=["frame", "clip_pooled"],
    )
    p.add_argument("--frame-subsample-per-clip", type=int, default=32)
    p.add_argument("--clip-pool-num-chunks", type=int, default=4)
    p.add_argument("--kernel", default="gaussian", choices=["gaussian"])
    p.add_argument(
        "--bandwidth-source",
        default="reference",
        choices=["reference", "generated", "fixed"],
    )
    p.add_argument("--bandwidth", type=float, default=None, help="fixed bandwidth (overrides source)")
    p.add_argument("--bandwidth-subsample", type=int, default=10_000)
    p.add_argument("--scale-factor", type=float, default=100.0)
    p.add_argument("--seq-len", type=int, default=172)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--normalized", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cfg-scale", type=float, default=4.0)
    p.add_argument("--sampler", choices=["euler", "heun"], default="heun")
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--genre", type=int, default=None, help="Fixed genre id; random if unset")
    p.add_argument("--device", default="cuda")
    p.add_argument("--cache-dir", default=None, help="Reference feature cache dir")
    p.add_argument("--output-dir", default=None, help="Directory for metric JSON/JSONL output")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if str(device) != args.device:
        warn(f"CUDA not available; falling back to {device}")

    # ------------------------------------------------------------------
    # Load checkpoint (reuse FlashDiTModule.load_from_checkpoint)
    # ------------------------------------------------------------------
    info(f"Loading checkpoint: {args.checkpoint}")
    from flash_dit.training.lit_module import FlashDiTModule

    lit = FlashDiTModule.load_from_checkpoint(args.checkpoint, map_location=str(device))
    ema_model = lit.ema.ema_model.eval().to(device)

    n_classes = int(getattr(ema_model.y_embedder, "null_class", 16))

    # ------------------------------------------------------------------
    # Output layout
    # ------------------------------------------------------------------
    run_root = Path(args.checkpoint).resolve().parent.parent
    out_dir = Path(args.output_dir) if args.output_dir else run_root / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = args.cache_dir or str(run_root / ".cache" / "latent_mmd")

    # ------------------------------------------------------------------
    # Generate latents (normalised, as produced by training)
    # ------------------------------------------------------------------
    from flash_dit.diffusion.samplers import euler_sample, heun_sample

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    if args.genre is not None:
        y = torch.full((args.n_generated,), int(args.genre), dtype=torch.long, device=device)
    else:
        y = torch.randint(0, n_classes, (args.n_generated,), generator=g).to(device)
    noise = torch.randn(args.n_generated, ema_model.in_channels, args.seq_len, device=device)

    info(
        f"Generating {args.n_generated} latents "
        f"(sampler={args.sampler}, cfg={args.cfg_scale}, steps={args.n_steps})"
    )
    t0 = time.time()
    fn = heun_sample if args.sampler == "heun" else euler_sample
    with torch.no_grad():
        latents = fn(ema_model, noise, y, n_steps=args.n_steps, cfg_scale=args.cfg_scale)
    ok(f"Generation done in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Compute latent MMD per feature type
    # ------------------------------------------------------------------
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = f"eval_{stamp}"
    json_path = out_dir / f"latent_mmd_{run_id}.json"
    jsonl_path = out_dir / "latent_mmd.jsonl"

    config_block = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "h5_path": args.h5_path,
        "split": args.split,
        "n_generated": args.n_generated,
        "n_reference": args.n_reference,
        "frame_subsample_per_clip": args.frame_subsample_per_clip,
        "clip_pool_num_chunks": args.clip_pool_num_chunks,
        "kernel": args.kernel,
        "bandwidth_source": args.bandwidth_source,
        "bandwidth": args.bandwidth,
        "bandwidth_subsample": args.bandwidth_subsample,
        "scale_factor": args.scale_factor,
        "seq_len": args.seq_len,
        "normalized": args.normalized,
        "seed": args.seed,
        "cfg_scale": args.cfg_scale,
        "sampler": args.sampler,
        "n_steps": args.n_steps,
        "genre": args.genre,
    }
    results: dict[str, dict] = {}

    for ft in args.feature_types:
        spec = ReferenceFeatureSpec(
            h5_path=args.h5_path,
            split=args.split,
            feature_type=ft,  # type: ignore[arg-type]
            n_reference=args.n_reference,
            seq_len=args.seq_len,
            frame_subsample_per_clip=(
                args.frame_subsample_per_clip if ft == "frame" else None
            ),
            clip_pool_num_chunks=args.clip_pool_num_chunks,
            normalized=args.normalized,
            seed=args.seed,
        )
        info(f"Loading reference features for feature_type={ft}")
        ref = load_reference_features(spec, cache_dir=cache_dir)

        res = compute_latent_mmd(
            latents,
            ref.to(device),
            feature_type=ft,  # type: ignore[arg-type]
            frame_subsample_per_clip=args.frame_subsample_per_clip,
            clip_pool_num_chunks=args.clip_pool_num_chunks,
            kernel=args.kernel,
            bandwidth=args.bandwidth,
            bandwidth_source=args.bandwidth_source,  # type: ignore[arg-type]
            bandwidth_subsample=args.bandwidth_subsample,
            scale_factor=args.scale_factor,
            seed=args.seed,
            normalized=args.normalized,
            device=device,
        )
        results[ft] = res.__dict__
        ok(
            f"latent_mmd[{ft}]: raw={res.mmd2:.6f}  "
            f"scaled×{res.scale_factor:g}={res.scaled:.4f}  "
            f"bw={res.bandwidth:.4f} (n_ref={res.n_reference}, n_gen={res.n_generated})"
        )

    payload = {
        "run_id": run_id,
        "timestamp": stamp,
        "config": config_block,
        "results": results,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    with open(jsonl_path, "a") as f:
        f.write(json.dumps({"run_id": run_id, "timestamp": stamp, **{f"{ft}/{k}": v for ft, r in results.items() for k, v in r.items() if isinstance(v, (int, float, str))}}) + "\n")

    info(f"Wrote {json_path}")
    info(f"Appended to {jsonl_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
