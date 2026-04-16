"""Kernel Audio Distance (KAD) — file-based evaluation over decoded audio.

Splits responsibility clearly:

1. **Audio → embeddings**: handled by the vendored ``kadtk_vendored``
   subpackage (PANNs backbone, torchaudio resampling, .npy caching).
2. **Embeddings → MMD²**: handled by ``flash_dit.evaluation.mmd``
   (unbiased, chunked, Gaussian RBF, paper-correct default bandwidth).

This is a deliberate improvement over upstream kadtk's ``calc_kernel_audio_distance``:
the upstream code materialises the full N×N distance matrix (OOM risk on
large reference sets) and uses ``median_pairwise_distance(y)`` — i.e. the
*generated* set — as the bandwidth source, contradicting the KAD paper/README
which specify the *reference* set. We keep the audio pipeline intact and
plug the validated PANNs embeddings into our chunked reference-bandwidth MMD.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from .kadtk_vendored import EmbeddingLoader, cache_embedding_files, get_model
from .kadtk_vendored.emb_loader import _collect_audio_files
from .kadtk_vendored.model_loader import ModelLoader
from .mmd import mmd_unbiased


@dataclass
class KADResult:
    """Machine-readable KAD evaluation record.

    ``score`` is the scaled value (compatible with the kadtk / KAD-paper
    convention of reporting 100 × MMD²). ``mmd2`` is the raw unbiased
    estimator, which may be slightly negative for near-identical distributions
    — this is expected and useful to log alongside the scaled number.
    """

    score: float
    mmd2: float
    scale_factor: float
    bandwidth: float
    bandwidth_source: str
    model_name: str
    generated_dir: str
    reference_dir: str
    n_generated: int
    n_reference: int
    n_generated_embeddings: int
    n_reference_embeddings: int
    device: str
    workers: int
    timestamp: str
    extra: dict = field(default_factory=dict)

    def to_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)


def _load_embedding_matrix(loader: EmbeddingLoader, audio_dir: Path) -> np.ndarray:
    """Read every cached .npy under ``audio_dir/embeddings/<model>/`` and concat.

    Returns an (N_total, D) float array, where N_total is the sum of
    per-clip frame counts (PANNs emits multiple frames per audio file).
    """
    files = _collect_audio_files(audio_dir)
    if not files:
        raise FileNotFoundError(f"no audio files under {audio_dir}")
    parts = [loader.read_embedding_file(f) for f in files]
    return np.concatenate(parts, axis=0)


def compute_kad(
    generated_dir: str | Path,
    reference_dir: str | Path,
    model_name: str = "panns-wavegram-logmel",
    device: str = "cuda",
    workers: int = 8,
    bandwidth_source: Literal["reference", "generated", "fixed"] = "reference",
    bandwidth: float | None = None,
    bandwidth_subsample: int | None = 10_000,
    scale_factor: float = 100.0,
    block_size: int = 4096,
    force_emb_encode: bool = False,
    output_json: str | Path | None = None,
    model: ModelLoader | None = None,
) -> KADResult:
    """Compute KAD between two directories of audio files.

    Args:
        generated_dir: directory of generated audio files (.wav/.flac/.mp3/...).
        reference_dir: directory of ground-truth audio files.
        model_name: identifier in ``kadtk_vendored.list_models()``. Paper
            default ``panns-wavegram-logmel`` handles general audio.
        device: torch device used to compute the MMD². Audio embedding
            extraction always runs on whatever device the model loaded on.
        workers: worker count for embedding caching (``1`` = inline, used
            in tests).
        bandwidth_source: whose pairwise-distance median becomes σ. Default
            ``'reference'`` matches the KAD paper/README.
        bandwidth: fixed σ value; when set, ``bandwidth_source`` is ignored
            and recorded as ``'fixed'`` in the result.
        bandwidth_subsample: rows subsampled for the median heuristic.
        scale_factor: multiplicative scale on MMD²; 100 matches upstream kadtk.
        block_size: chunk size for the pairwise kernel sums — keep under
            available GPU memory for the (B, N) blocks used.
        force_emb_encode: re-extract embeddings even if .npy caches exist.
        output_json: if set, writes a JSON record to this path.
        model: pre-built ``ModelLoader`` (for testing with mocks). When
            given, ``model_name`` is overridden with ``model.name``.

    Returns:
        :class:`KADResult` with both raw and scaled MMD² and full metadata.
    """
    gen = Path(generated_dir)
    ref = Path(reference_dir)
    if not gen.is_dir():
        raise FileNotFoundError(f"generated_dir does not exist: {gen}")
    if not ref.is_dir():
        raise FileNotFoundError(f"reference_dir does not exist: {ref}")

    ml = model if model is not None else get_model(model_name)
    model_name = ml.name

    # 1. Populate the on-disk embedding cache for both sides.
    cache_embedding_files(ref, ml, workers=workers, force_emb_encode=force_emb_encode)
    cache_embedding_files(gen, ml, workers=workers, force_emb_encode=force_emb_encode)

    # 2. Load concatenated embedding matrices.
    #    The loader that extracted them keeps the already-loaded model;
    #    for reading cached .npy we don't need it again, but reusing the
    #    instance keeps behaviour consistent with upstream.
    reader = EmbeddingLoader(ml, load_model=False)
    ref_emb = _load_embedding_matrix(reader, ref)
    gen_emb = _load_embedding_matrix(reader, gen)

    # 3. Compute unbiased MMD² with our chunked estimator.
    device_t = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    ref_t = torch.from_numpy(ref_emb).to(device=device_t, dtype=torch.float32)
    gen_t = torch.from_numpy(gen_emb).to(device=device_t, dtype=torch.float32)

    mmd_res = mmd_unbiased(
        ref_t, gen_t,
        kernel="gaussian",
        bandwidth=bandwidth,
        bandwidth_source=bandwidth_source,
        bandwidth_subsample=bandwidth_subsample,
        scale_factor=scale_factor,
        block_size=block_size,
    )

    result = KADResult(
        score=mmd_res.scaled,
        mmd2=mmd_res.mmd2,
        scale_factor=mmd_res.scale_factor,
        bandwidth=mmd_res.bandwidth,
        bandwidth_source=mmd_res.bandwidth_source,
        model_name=model_name,
        generated_dir=str(gen.resolve()),
        reference_dir=str(ref.resolve()),
        n_generated=len(_collect_audio_files(gen)),
        n_reference=len(_collect_audio_files(ref)),
        n_generated_embeddings=int(gen_emb.shape[0]),
        n_reference_embeddings=int(ref_emb.shape[0]),
        device=str(device_t),
        workers=workers,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    if output_json is not None:
        result.to_json(output_json)

    return result
