"""Audiobox Aesthetics scoring for generated audio files."""
from __future__ import annotations

from pathlib import Path

_predictor = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        from audiobox_aesthetics.infer import initialize_predictor
        _predictor = initialize_predictor()
    return _predictor


def score_wavs(wav_paths: list[str | Path]) -> dict[str, float]:
    """Score a list of WAV files with Audiobox Aesthetics.

    The predictor is initialised once and reused across calls.

    Returns:
        dict with mean scores for keys 'CE', 'CU', 'PC', 'PQ'.
    """
    try:
        predictor = _get_predictor()
    except ImportError as e:
        raise ImportError(
            "audiobox-aesthetics is required. "
            "Install with: uv add audiobox-aesthetics"
        ) from e

    inputs = [{"path": str(p)} for p in wav_paths]
    results = predictor.forward(inputs)

    keys = ["CE", "CU", "PC", "PQ"]
    n = len(results)
    return {k: sum(r[k] for r in results) / n for k in keys}
