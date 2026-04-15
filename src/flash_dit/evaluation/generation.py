"""Batch generation utilities: latent sampling + VAE decoding."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def generate_latents(
    model: nn.Module,
    n_samples: int,
    in_channels: int,
    seq_len: int,
    genre_labels: torch.Tensor | None = None,
    n_classes: int = 16,
    cfg_scale: float = 4.0,
    sampler: str = "heun",
    n_steps: int = 50,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Sample latents from the DiT model.

    Args:
        model:        DiffusionTransformer in eval mode.
        n_samples:    number of samples to generate.
        in_channels:  latent channel dimension (e.g. 64).
        seq_len:      temporal length of latent (e.g. 64 for 3s @ 2048× compression).
        genre_labels: (n_samples,) int64 labels; if None, samples uniformly from all genres.
        n_classes:    number of genre classes (used when genre_labels is None).
        cfg_scale:    CFG guidance scale.
        sampler:      'euler' or 'heun'.
        n_steps:      ODE steps.
        device:       target device.

    Returns:
        (n_samples, in_channels, seq_len) float32 normalised latents.
    """
    from ..diffusion.samplers import euler_sample, heun_sample

    device = torch.device(device)
    if genre_labels is None:
        genre_labels = torch.randint(0, n_classes, (n_samples,), device=device)
    else:
        genre_labels = genre_labels.to(device)

    noise = torch.randn(n_samples, in_channels, seq_len, device=device)

    fn = heun_sample if sampler == "heun" else euler_sample
    with torch.no_grad():
        latents = fn(model, noise, genre_labels, n_steps=n_steps, cfg_scale=cfg_scale)

    return latents


def decode_latents_to_wav(
    latents: torch.Tensor,
    vae: nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    sample_rate: int = 44100,
) -> list:
    """Decode normalised latents to audio arrays using the VAE decoder.

    Args:
        latents: (B, C, T) normalised latents (as output by generate_latents).
        vae:     stable-audio-tools AudioAutoencoder in eval mode.
        mean:    (C,) per-channel normalisation mean.
        std:     (C,) per-channel normalisation std.
        sample_rate: audio sample rate (44100 Hz).

    Returns:
        List of (2, T_audio) numpy float32 stereo audio arrays.
    """
    import numpy as np

    device = latents.device
    mean = mean.to(device)
    std  = std.to(device)

    with torch.no_grad():
        latents_raw = latents * (std[:, None] + 1e-6) + mean[:, None]
        audio, _ = vae.decode_audio(latents_raw)  # (B, 2, T_audio)

    return [audio[i].cpu().float().clamp(-1.0, 1.0).numpy() for i in range(audio.shape[0])]


def save_wavs(wav_arrays: list, output_dir: str | Path, sample_rate: int = 44100) -> None:
    """Write a list of audio arrays to WAV files."""
    import numpy as np
    import soundfile as sf

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, wav in enumerate(wav_arrays):
        sf.write(str(out / f"sample_{i:05d}.wav"), np.asarray(wav).T, samplerate=sample_rate)
