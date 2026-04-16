"""Audiobox Aesthetics scoring for generated audio files."""
from __future__ import annotations

from pathlib import Path


def score_wavs(wav_paths: list[str | Path]) -> dict[str, float]:
    """Score a list of WAV files with Audiobox Aesthetics.

    Runs the predictor on all provided paths and returns mean scores.

    Args:
        wav_paths: list of paths to .wav files to score.

    Returns:
        dict with mean scores for keys 'CE', 'CU', 'PC', 'PQ':
            CE  Content Enjoyment
            CU  Content Usefulness
            PC  Production Complexity
            PQ  Production Quality
    """
    try:
        from audiobox_aesthetics.infer import initialize_predictor
    except ImportError as e:
        raise ImportError(
            "audiobox-aesthetics is required. "
            "Install with: uv add audiobox-aesthetics"
        ) from e

    predictor = initialize_predictor()
    inputs = [{"path": str(p)} for p in wav_paths]
    results = predictor.forward(inputs)

    keys = ["CE", "CU", "PC", "PQ"]
    n = len(results)
    return {k: sum(r[k] for r in results) / n for k in keys}
