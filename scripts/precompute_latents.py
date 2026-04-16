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
# 215 × 2048 = 440320 samples = 9.985 s ≈ 10 s.
# Chosen so that a standard 30 s FMA track yields exactly 3 non-overlapping chunks
# (440320 × 3 = 1320960 < 1323000 = 30 s × 44100).
# Next multiple up (216 × 2048 = 442368 = 10.03 s) would give only 2 chunks per track.
CHUNK_SAMPLES  = 440320          # 9.985 s at 44.1 kHz
VAE_STRIDE     = 2048            # temporal compression ratio of Oobleck
LATENT_FRAMES  = CHUNK_SAMPLES // VAE_STRIDE   # = 215
LATENT_CHANNELS = 64

HF_REPO   = "stabilityai/stable-audio-open-1.0"
CACHE_DIR = Path(os.environ.get(
    "FLASH_DIT_CACHE",
    "/leonardo_scratch/fast/IscrC_LENS/lcerovaz/flash-dit-cache"
))
# Audio root: fma_medium on CINECA (only medium downloaded), fma_large on gauss
# (large is a superset; the subset filter in tracks.csv handles selection).
FMA_DIR    = Path(os.environ.get("FMA_AUDIO_DIR",
                  "/leonardo_scratch/large/userexternal/lcerovaz/fma/fma_medium"))
TRACKS_CSV = Path(os.environ.get("FMA_METADATA_DIR",
                  "/leonardo_scratch/large/userexternal/lcerovaz/fma/fma_metadata")) / "tracks.csv"
OUT_DIR    = Path(os.environ.get("FLASH_DIT_LATENTS_DIR",
                  "/leonardo_scratch/large/userexternal/lcerovaz/flash-dit-latents/stable_audio_open"))
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

def load_audio_chunk(path: Path) -> torch.Tensor | None:
    """Load a track and return a single CHUNK_SAMPLES-long stereo window.

    Takes the first CHUNK_SAMPLES frames after a short intro offset (2s) to
    skip typical silence/noise at the start of FMA tracks.
    Returns None if the track is too short or cannot be loaded.
    """
    try:
        wav, sr = torchaudio.load(str(path))
    except Exception as exc:
        warn(f"Cannot load {path}: {exc}")
        return None

    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

    # Ensure stereo
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]

    # Skip 2s intro, then take one CHUNK_SAMPLES window
    offset = 2 * SAMPLE_RATE
    if wav.shape[1] < offset + CHUNK_SAMPLES:
        # Track too short: fall back to the beginning
        offset = 0
    if wav.shape[1] < CHUNK_SAMPLES:
        return None

    return wav[:, offset : offset + CHUNK_SAMPLES]  # (2, CHUNK_SAMPLES)


@torch.no_grad()
def encode_chunk(vae, chunk: torch.Tensor) -> np.ndarray:
    """Encode a single (2, CHUNK_SAMPLES) audio tensor through the VAE encoder.

    stable-audio-tools AudioAutoencoder.encode returns (latents, info_dict).
    We use the distribution mean for deterministic, reproducible pre-computation.
    Encoding runs in float32 (VAE weights are float32); output is cast to float16
    only for compact HDF5 storage.

    Returns:
        (1, LATENT_CHANNELS, LATENT_FRAMES) float16 array.
    """
    batch = chunk.unsqueeze(0).cuda().float()  # (1, 2, T) — float32, matches VAE weights
    result = vae.encode(batch)
    if isinstance(result, tuple):
        latents, info = result
        if isinstance(info, dict) and "mean" in info:
            latents = info["mean"]
    else:
        latents = result
    return latents.float().cpu().numpy().astype(np.float16)  # (1, C, T_lat)


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
        - (set, subset)      → 'medium' / 'small' / 'large'
        - (set, split)       → 'training' / 'validation' / 'test'
        - (track, genre_top) → top-level genre string
    """
    import pandas as pd

    df = pd.read_csv(str(TRACKS_CSV), index_col=0, header=[0, 1])
    # Keep only rows that belong to fma_medium
    df = df[df[("set", "subset")] == "medium"]

    # Canonical alphabetically-sorted genre map — stable across runs and machines.
    all_genres = sorted(df[("track", "genre_top")].dropna().unique().tolist())
    genre2id: dict[str, int] = {g: i for i, g in enumerate(all_genres)}
    genre2id["Unknown"] = len(genre2id)   # fallback for missing labels

    tracks = []
    split_map = {"training": "train", "validation": "val", "test": "test"}

    for tid, row in df.iterrows():
        split_raw = row[("set", "split")]
        sp = split_map.get(split_raw, None)
        if sp is None:
            continue
        genre = str(row[("track", "genre_top")]) if pd.notna(row[("track", "genre_top")]) else "Unknown"
        path  = _track_id_to_path(int(tid))
        if not path.exists():
            warn(f"Audio file not found, skipping: {path}")
            continue
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
            chunk = load_audio_chunk(path)
            if chunk is None:
                continue

            latents = encode_chunk(vae, chunk)  # (1, C, T_lat)

            for ds in (ds_lat, ds_gen, ds_tid, ds_spl):
                ds.resize(written + 1, axis=0)

            ds_lat[written] = latents[0]
            ds_gen[written] = genre_id
            ds_tid[written] = path.stem
            ds_spl[written] = sp.encode()
            written += 1

        ok(f"Wrote {written} latent chunks.")

        # Compute normalisation stats over training split in batches —
        # avoids loading all training latents into RAM at once (~2 GB).
        info("Computing normalisation stats over training split…")
        ch_sum  = np.zeros(LATENT_CHANNELS, dtype=np.float64)
        ch_sum2 = np.zeros(LATENT_CHANNELS, dtype=np.float64)
        n_vals  = 0  # total number of (channel-wise) scalar values seen

        STAT_BATCH    = 512
        train_indices = np.where(ds_spl[:] == b"train")[0]
        for i in range(0, len(train_indices), STAT_BATCH):
            batch_idx = train_indices[i : i + STAT_BATCH]
            batch = ds_lat[batch_idx].astype(np.float64)       # (B, C, T)
            flat  = batch.transpose(1, 0, 2).reshape(LATENT_CHANNELS, -1)  # (C, B*T)
            ch_sum  += flat.sum(axis=1)
            ch_sum2 += (flat ** 2).sum(axis=1)
            n_vals  += flat.shape[1]

        mean = (ch_sum / n_vals).astype(np.float32)
        std  = np.sqrt(np.maximum(ch_sum2 / n_vals - (ch_sum / n_vals) ** 2, 0.0)).astype(np.float32)
        f.attrs["mean"] = mean
        f.attrs["std"]  = std
        ok(f"Normalisation stats: mean={mean.mean():.4f}, std={std.mean():.4f}")

    ok(f"Done → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Process only 10 tracks")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
