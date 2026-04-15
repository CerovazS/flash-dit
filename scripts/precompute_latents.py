#!/usr/bin/env python
"""Pre-compute audio latents from the official stable-audio-open-1.0 VAE.

Usage:
    uv run python scripts/precompute_latents.py [--dry-run] [--split all|train|val|test]

Writes:
    /leonardo_scratch/large/userexternal/lcerovaz/flash-dit-latents/
        stable_audio_open/fma_medium.h5

HDF5 layout:
    latents   float16  (N, 64, 64)
    genres    int64    (N,)
    track_ids bytes    (N,)
    split     bytes    (N,)

HDF5 attrs:
    mean      float32  (64,)  per-channel mean over training split
    std       float32  (64,)  per-channel std  over training split
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from flash_dit.utils.console import error, info, ok, warn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE    = 44100
CHUNK_SAMPLES  = 131072          # ≈ 3 s at 44.1 kHz
VAE_STRIDE     = 2048            # temporal compression ratio of Oobleck
LATENT_FRAMES  = CHUNK_SAMPLES // VAE_STRIDE   # = 64
LATENT_CHANNELS = 64

HF_REPO   = "stabilityai/stable-audio-open-1.0"
CACHE_DIR = Path(os.environ.get(
    "FLASH_DIT_CACHE",
    "/leonardo_scratch/fast/IscrC_LENS/lcerovaz/flash-dit-cache"
))
FMA_DIR    = Path("/leonardo_scratch/large/userexternal/lcerovaz/fma/fma_medium")
# Standard FMA tracks.csv contains: track_id, genre_top, split (train/val/test)
TRACKS_CSV = Path("/leonardo_scratch/large/userexternal/lcerovaz/fma/fma_metadata/tracks.csv")
OUT_DIR    = Path("/leonardo_scratch/large/userexternal/lcerovaz/flash-dit-latents/stable_audio_open")
BATCH_SIZE = 16


# ---------------------------------------------------------------------------
# VAE loading
# ---------------------------------------------------------------------------

def load_vae(hf_token: str) -> torch.nn.Module:
    """Download (or load from cache) and return the Oobleck VAE encoder."""
    from huggingface_hub import hf_hub_download
    from stable_audio_tools.models.factory import create_model_from_config

    info("Downloading model config from HuggingFace…")
    config_path = hf_hub_download(
        HF_REPO, "model_config.json", token=hf_token,
        cache_dir=str(CACHE_DIR / "hf_cache"),
    )
    ckpt_path = hf_hub_download(
        HF_REPO, "model.safetensors", token=hf_token,
        cache_dir=str(CACHE_DIR / "hf_cache"),
    )

    with open(config_path) as f:
        config = json.load(f)

    model = create_model_from_config(config)

    from safetensors.torch import load_file
    state = load_file(ckpt_path)
    # Load only pretransform (VAE) weights to avoid OOM with the full DiT
    pretransform_state = {
        k.removeprefix("pretransform."): v
        for k, v in state.items()
        if k.startswith("pretransform.")
    }
    model.pretransform.load_state_dict(pretransform_state, strict=False)

    vae = model.pretransform.eval().cuda()
    ok("VAE loaded and ready on GPU.")
    return vae


# ---------------------------------------------------------------------------
# Audio processing
# ---------------------------------------------------------------------------

def load_audio_chunks(path: Path) -> list[torch.Tensor]:
    """Load a track, resample to 44100 Hz stereo, chunk into CHUNK_SAMPLES frames.

    Returns a list of (2, CHUNK_SAMPLES) float32 tensors.
    """
    try:
        wav, sr = torchaudio.load(str(path))
    except Exception as exc:
        warn(f"Cannot load {path}: {exc}")
        return []

    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

    # Ensure stereo
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]

    # Split into non-overlapping chunks
    n_chunks = wav.shape[1] // CHUNK_SAMPLES
    chunks = []
    for i in range(n_chunks):
        chunk = wav[:, i * CHUNK_SAMPLES : (i + 1) * CHUNK_SAMPLES]
        chunks.append(chunk)
    return chunks


@torch.no_grad()
def encode_chunks(vae, chunks: list[torch.Tensor]) -> np.ndarray:
    """Encode a list of audio chunks through the VAE encoder.

    stable-audio-tools AudioAutoencoder.encode returns (latents, info_dict).
    We take the mean from the VAE distribution (deterministic for pre-computation).

    Returns:
        (len(chunks), LATENT_CHANNELS, LATENT_FRAMES) float16 array.
    """
    out = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = torch.stack(chunks[i : i + BATCH_SIZE]).cuda().to(torch.bfloat16)
        result = vae.encode(batch)
        # stable-audio-tools returns (latent, info) or just latent depending on bottleneck
        if isinstance(result, tuple):
            latents, info = result
            # For VAE bottleneck, prefer mean over sampled latent
            if isinstance(info, dict) and "mean" in info:
                latents = info["mean"]
        else:
            latents = result
        latents = latents.float().cpu().numpy().astype(np.float16)
        out.append(latents)
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _track_id_to_path(tid: int) -> Path:
    """FMA convention: track 2 → fma_medium/000/000002.mp3"""
    tid_str = f"{tid:06d}"
    return FMA_DIR / tid_str[:3] / f"{tid_str}.mp3"


def _load_fma_tracks() -> list[tuple[str, int, Path]]:
    """Read tracks.csv and return (split, genre_id, path) triples.

    tracks.csv is a multi-level CSV where the first two rows are header rows.
    Relevant columns (after flattening):
        - (track, id)       → track ID
        - (track, split)    → 'training' / 'validation' / 'test'
        - (track, genre_top) → top-level genre string
    """
    import pandas as pd

    df = pd.read_csv(str(TRACKS_CSV), index_col=0, header=[0, 1])
    # Keep only rows that belong to fma_medium
    df = df[df[("set", "subset")] == "medium"]

    genre2id: dict[str, int] = {}
    tracks = []
    split_map = {"training": "train", "validation": "val", "test": "test"}

    for tid, row in df.iterrows():
        split_raw = row[("set", "split")]
        sp = split_map.get(split_raw, None)
        if sp is None:
            continue
        genre = str(row[("track", "genre_top")] or "Unknown")
        path  = _track_id_to_path(int(tid))
        if not path.exists():
            continue
        if genre not in genre2id:
            genre2id[genre] = len(genre2id)
        tracks.append((sp, genre2id[genre], path))

    return tracks, genre2id


def main(dry_run: bool = False) -> None:
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        error("HF_TOKEN environment variable not set.")
        raise SystemExit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "fma_medium.h5"

    info("Reading FMA tracks.csv…")
    tracks, genre2id = _load_fma_tracks()
    info(f"Found {len(tracks)} tracks. Genre map ({len(genre2id)} classes): {genre2id}")

    if dry_run:
        tracks = tracks[:10]
        info("[dry-run] processing only 10 tracks")

    vae = load_vae(hf_token)

    # Estimate total chunks for HDF5 pre-allocation
    avg_chunks_per_track = 10  # 30s / 3s
    n_est = len(tracks) * avg_chunks_per_track

    info(f"Writing to {out_path} (estimated {n_est} chunks)")

    with h5py.File(out_path, "w") as f:
        ds_lat = f.create_dataset(
            "latents", shape=(0, LATENT_CHANNELS, LATENT_FRAMES),
            maxshape=(None, LATENT_CHANNELS, LATENT_FRAMES),
            dtype="float16", chunks=(64, LATENT_CHANNELS, LATENT_FRAMES),
        )
        ds_gen = f.create_dataset("genres",    shape=(0,), maxshape=(None,), dtype="int64")
        ds_tid = f.create_dataset("track_ids", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype())
        ds_spl = f.create_dataset("split",     shape=(0,), maxshape=(None,), dtype=h5py.string_dtype())

        written = 0
        for sp, genre_id, path in tqdm(tracks, desc="encoding"):
            chunks = load_audio_chunks(path)
            if not chunks:
                continue

            latents = encode_chunks(vae, chunks)  # (K, C, T)
            K = latents.shape[0]

            for ds in (ds_lat, ds_gen, ds_tid, ds_spl):
                ds.resize(written + K, axis=0)

            ds_lat[written : written + K] = latents
            ds_gen[written : written + K] = np.full(K, genre_id, dtype=np.int64)
            ds_tid[written : written + K] = [path.stem] * K
            ds_spl[written : written + K] = [sp.encode()] * K
            written += K

        ok(f"Wrote {written} latent chunks.")

        # Compute normalisation stats over training split
        info("Computing normalisation stats over training split…")
        train_mask = ds_spl[:] == b"train"
        train_latents = ds_lat[train_mask].astype(np.float32)  # (N_train, C, T)
        mean = train_latents.mean(axis=(0, 2))   # (C,)
        std  = train_latents.std(axis=(0, 2))    # (C,)
        f.attrs["mean"] = mean
        f.attrs["std"]  = std
        ok(f"Normalisation stats: mean={mean.mean():.4f}, std={std.mean():.4f}")

    ok(f"Done → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Process only 10 tracks")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
