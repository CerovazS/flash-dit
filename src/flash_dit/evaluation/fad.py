"""Fréchet Audio Distance (FAD) computation using CLAP-LAION-audio embeddings.

Uses the `frechet_audio_distance` library with the CLAP backend,
consistent with ml-training.md rules for audio quality metrics.
"""
from __future__ import annotations

from pathlib import Path


def compute_fad(generated_dir: str | Path, reference_dir: str | Path) -> float:
    """Compute FAD between a directory of generated WAVs and reference WAVs.

    Args:
        generated_dir: directory containing generated .wav files.
        reference_dir: directory containing reference (real) .wav files.

    Returns:
        FAD scalar (lower is better).
    """
    try:
        from frechet_audio_distance import FrechetAudioDistance
    except ImportError as e:
        raise ImportError(
            "frechet-audio-distance is required for FAD computation. "
            "Install with: uv add frechet-audio-distance"
        ) from e

    fad = FrechetAudioDistance(
        model_name="clap",
        use_pca=False,
        use_activation=False,
        verbose=False,
    )
    return fad.score(str(reference_dir), str(generated_dir))


def compute_fad_from_arrays(
    generated: list,  # list of (T,) or (2, T) numpy float32 arrays
    reference: list,
    sample_rate: int = 44100,
    tmp_dir: str | Path = "/tmp/flash_dit_fad",
) -> float:
    """Convenience wrapper that writes arrays to disk then calls compute_fad."""
    import numpy as np
    import soundfile as sf

    tmp = Path(tmp_dir)
    gen_dir = tmp / "generated"
    ref_dir = tmp / "reference"
    gen_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    for i, wav in enumerate(generated):
        sf.write(str(gen_dir / f"{i:05d}.wav"), np.asarray(wav).T, samplerate=sample_rate)
    for i, wav in enumerate(reference):
        sf.write(str(ref_dir / f"{i:05d}.wav"), np.asarray(wav).T, samplerate=sample_rate)

    return compute_fad(gen_dir, ref_dir)
