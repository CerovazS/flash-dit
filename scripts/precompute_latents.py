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
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from flash_dit.autoencoder import build_autoencoder
from flash_dit.utils.console import error, info, ok, warn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# 215 × 2048 = 440320 samples = 9.985 s ≈ 10 s.
# Chosen so that a standard 30 s FMA track yields exactly 3 non-overlapping chunks
# (440320 × 3 = 1320960 < 1323000 = 30 s × 44100).
# Next multiple up (216 × 2048 = 442368 = 10.03 s) would give only 2 chunks per track.
# These constants are SAO-specific; once we add a non-SAO encoder the per-encoder
# values come from the autoencoder metadata (sample_rate, compression_ratios[0]).
SAMPLE_RATE     = 44100
CHUNK_SAMPLES   = 440320          # 9.985 s at 44.1 kHz
VAE_STRIDE      = 2048
LATENT_FRAMES   = CHUNK_SAMPLES // VAE_STRIDE   # = 215
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
BATCH_SIZE = 8   # encode 8 tracks per GPU call (iterate_batch disabled in load_vae)


# ---------------------------------------------------------------------------
# VAE loading
# ---------------------------------------------------------------------------

def load_vae(hf_token: str):
    """Build the SAO autoencoder via the flash_dit.autoencoder registry.

    This used to be a 30-line snippet duplicated in three scripts; now lives
    in :mod:`flash_dit.autoencoder.stable_audio_open`. The wrapper handles
    weight download, prefix stripping, and the iterate_batch fix internally.
    """
    ae = build_autoencoder({
        "kind": "stable_audio_open",
        "hf_repo": HF_REPO,
        "cache_dir": str(CACHE_DIR / "hf_cache"),
        "hf_token": hf_token,
        "device": "cuda",
    })
    ok(f"Autoencoder {ae.metadata.encoder_id!r} loaded on {ae.device}.")
    return ae


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


def encode_batch(vae, chunks: list[torch.Tensor]) -> np.ndarray:
    """Encode a batch of ``(2, CHUNK_SAMPLES)`` tensors through the autoencoder.

    The wrapper's ``encode()`` already handles tuple/dict return unwrapping
    and stereo coercion. Output is cast to float16 only for compact HDF5
    storage.

    Args:
        chunks: list of B tensors each shaped (2, CHUNK_SAMPLES).

    Returns:
        ``(B, LATENT_CHANNELS, LATENT_FRAMES)`` float16 array.
    """
    batch = torch.stack(chunks, dim=0)  # (B, 2, T) — wrapper sends to its device
    latents = vae.encode(batch)
    return latents.float().cpu().numpy().astype(np.float16)


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

    # One latent chunk per track (single 10 s window).
    n_est = len(tracks)
    info(f"Writing to {out_path} (estimated {n_est} latents, batch_size={BATCH_SIZE})")

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
        pending_chunks: list[torch.Tensor] = []
        pending_meta:   list[tuple[int, str, bytes]] = []  # (genre_id, stem, split_bytes)

        def flush() -> None:
            nonlocal written
            if not pending_chunks:
                return
            latents = encode_batch(vae, pending_chunks)  # (B, C, T_lat)
            b = len(pending_chunks)
            for ds in (ds_lat, ds_gen, ds_tid, ds_spl):
                ds.resize(written + b, axis=0)
            ds_lat[written : written + b] = latents
            ds_gen[written : written + b] = [m[0] for m in pending_meta]
            ds_tid[written : written + b] = [m[1] for m in pending_meta]
            ds_spl[written : written + b] = [m[2] for m in pending_meta]
            written += b
            pending_chunks.clear()
            pending_meta.clear()

        for sp, genre_id, path in tqdm(tracks, desc="encoding"):
            chunk = load_audio_chunk(path)
            if chunk is None:
                continue
            pending_chunks.append(chunk)
            pending_meta.append((genre_id, path.stem, sp.encode()))
            if len(pending_chunks) >= BATCH_SIZE:
                flush()

        flush()  # remainder
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

        # ----------------------------------------------------------------
        # Schema v2 attrs — derived from the autoencoder's metadata plus
        # the encoding-job context (chunk length, git SHA, timestamp).
        # See plan.md §2 for the full schema.
        # ----------------------------------------------------------------
        f.attrs["schema_version"] = 2
        for k, v in vae.metadata.to_h5_attrs().items():
            f.attrs[k] = v
        f.attrs["chunk_samples"] = CHUNK_SAMPLES
        f.attrs["storage_dtype"] = "float16"
        f.attrs["created_utc"] = _now_utc()
        f.attrs["git_commit"] = _git_commit()
        ok(f"Wrote v2 schema attrs (encoder_id={vae.metadata.encoder_id!r}).")

    ok(f"Done → {out_path}")


def _now_utc() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Process only 10 tracks")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
