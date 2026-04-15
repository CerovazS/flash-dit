#!/usr/bin/env python
"""Generate audio samples from a trained flash-dit checkpoint.

Usage:
    uv run python scripts/sample.py \\
        --checkpoint outputs/run_name/checkpoints/last.ckpt \\
        --n-samples 8 --genre 3 --cfg-scale 4.0 \\
        --sampler heun --n-steps 50 --output-dir outputs/samples
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch

from flash_dit.utils.console import info, ok, warn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    p.add_argument("--config",     default=None,  help="Hydra config dir (default: inferred from checkpoint path)")
    p.add_argument("--n-samples",  type=int, default=8)
    p.add_argument("--genre",      type=int, default=None, help="Genre class (0-15); random if not set")
    p.add_argument("--cfg-scale",  type=float, default=4.0)
    p.add_argument("--sampler",    choices=["euler", "heun"], default="heun")
    p.add_argument("--n-steps",    type=int, default=50)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--device",     default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    out_dir = Path(args.output_dir or f"outputs/samples_{datetime.now():%Y%m%d_%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    info(f"Loading checkpoint: {args.checkpoint}")
    from flash_dit.training.lit_module import FlashDiTModule

    lit = FlashDiTModule.load_from_checkpoint(args.checkpoint, map_location=device)
    model = lit.ema.ema_model.eval().to(device)
    info(f"Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

    # ------------------------------------------------------------------
    # Load VAE decoder
    # ------------------------------------------------------------------
    vae = _load_vae(device)
    mean, std = _load_norm_stats_from_lit(lit, device)

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------
    from flash_dit.evaluation.generation import decode_latents_to_wav, generate_latents, save_wavs

    n = args.n_samples
    genre_labels = None
    if args.genre is not None:
        genre_labels = torch.full((n,), args.genre, dtype=torch.long, device=device)

    info(f"Generating {n} samples (sampler={args.sampler}, cfg={args.cfg_scale}, steps={args.n_steps})")
    latents = generate_latents(
        model, n, model.in_channels, 64,
        genre_labels=genre_labels,
        cfg_scale=args.cfg_scale,
        sampler=args.sampler,
        n_steps=args.n_steps,
        device=device,
    )

    if vae is not None:
        wavs = decode_latents_to_wav(latents, vae, mean, std)
        save_wavs(wavs, out_dir)
        ok(f"Saved {len(wavs)} WAVs to {out_dir}")
    else:
        # Save raw latents as fallback
        torch.save(latents.cpu(), out_dir / "latents.pt")
        warn(f"VAE not available — saved raw latents to {out_dir}/latents.pt")


def _load_vae(device: torch.device):
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        warn("HF_TOKEN not set — cannot load VAE decoder.")
        return None
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from stable_audio_tools.models.factory import create_model_from_config

        config_path = hf_hub_download("stabilityai/stable-audio-open-1.0", "model_config.json", token=hf_token)
        ckpt_path   = hf_hub_download("stabilityai/stable-audio-open-1.0", "model.safetensors", token=hf_token)
        with open(config_path) as f:
            model_cfg = json.load(f)
        model = create_model_from_config(model_cfg)
        state = load_file(ckpt_path)
        pt_state = {k.removeprefix("pretransform."): v for k, v in state.items() if k.startswith("pretransform.")}
        model.pretransform.load_state_dict(pt_state, strict=False)
        return model.pretransform.eval().to(device)
    except Exception as exc:
        warn(f"Failed to load VAE: {exc}")
        return None


def _load_norm_stats_from_lit(lit, device) -> tuple:
    if hasattr(lit, "latent_mean"):
        return lit.latent_mean.to(device), lit.latent_std.to(device)
    # Fallback: zero-mean, unit-std (no denormalisation)
    warn("No normalisation stats in checkpoint — assuming zero-mean / unit-std.")
    return torch.zeros(64, device=device), torch.ones(64, device=device)


if __name__ == "__main__":
    main()
