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

Reference audio can be supplied two ways:

- ``reference_dir``: a directory containing audio files (original upstream flow).
- ``reference_manifest``: a text file with one absolute audio path per line
  (``#`` comments and blank lines allowed). Enables reusing an existing audio
  tree (e.g. FMA) without copying/symlinking files. In manifest mode the
  embedding cache is redirected to ``reference_cache_dir`` so the source tree
  stays pristine.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from ..utils.console import warn
from .kadtk_vendored import EmbeddingLoader, cache_embedding_files, get_model
from .kadtk_vendored.emb_loader import _collect_audio_files
from .kadtk_vendored.model_loader import ModelLoader
from .kadtk_vendored.utils import get_cache_embedding_path
from .mmd import mmd_unbiased


@dataclass
class KADResult:
    """Machine-readable KAD evaluation record.

    ``score`` is the scaled value (compatible with the kadtk / KAD-paper
    convention of reporting 100 × MMD²). ``mmd2`` is the raw unbiased
    estimator, which may be slightly negative for near-identical distributions
    — this is expected and useful to log alongside the scaled number.

    ``reference_mode`` is ``'dir'`` or ``'manifest'``; exactly one of
    ``reference_dir`` / ``reference_manifest`` is populated accordingly.
    """

    score: float
    mmd2: float
    scale_factor: float
    bandwidth: float
    bandwidth_source: str
    model_name: str
    generated_dir: str
    reference_dir: str | None
    reference_manifest: str | None
    reference_mode: str
    reference_cache_dir: str | None
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


def _read_reference_manifest(path: Path) -> list[Path]:
    """Parse a line-delimited manifest of absolute audio paths.

    - Lines starting with ``#`` and blank lines are skipped.
    - Paths must be absolute so the manifest is machine-portable in the
      narrow sense of "works from any cwd on the same filesystem".
    """
    lines = Path(path).read_text().splitlines()
    out: list[Path] = []
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        p = Path(s)
        if not p.is_absolute():
            raise ValueError(
                f"manifest {path}: non-absolute path on line {lines.index(raw) + 1}: {s}"
            )
        out.append(p)
    return out


def _load_embedding_matrix(
    loader: EmbeddingLoader,
    files: list[Path],
) -> np.ndarray:
    """Concatenate cached embeddings for ``files`` into an ``(N_total, D)`` array.

    ``N_total`` is the sum of per-clip frame counts (PANNs emits multiple
    frames per audio file).
    """
    if not files:
        raise FileNotFoundError("no audio files to read embeddings from")
    parts = [loader.read_embedding_file(f) for f in files]
    return np.concatenate(parts, axis=0)


def compute_kad(
    generated_dir: str | Path,
    reference_dir: str | Path | None = None,
    reference_manifest: str | Path | None = None,
    reference_cache_dir: str | Path | None = None,
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
    """Compute KAD between a generated audio directory and a reference set.

    Exactly one of ``reference_dir`` or ``reference_manifest`` must be given.

    Args:
        generated_dir: directory of generated audio files (.wav/.flac/.mp3/...).
        reference_dir: directory of ground-truth audio files. Embeddings are
            cached in-place under ``<reference_dir>/embeddings/<model>/``.
        reference_manifest: text file listing one absolute audio path per line.
            Enables reusing an existing dataset tree (e.g. FMA) without
            copying files into a dedicated directory.
        reference_cache_dir: cache directory for the manifest path (ignored
            in ``reference_dir`` mode). If omitted in manifest mode, defaults
            to ``<manifest>.kad_cache/`` next to the manifest file.
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
    if not gen.is_dir():
        raise FileNotFoundError(f"generated_dir does not exist: {gen}")

    if (reference_dir is None) == (reference_manifest is None):
        raise ValueError(
            "compute_kad: pass exactly one of reference_dir or reference_manifest"
        )

    ml = model if model is not None else get_model(model_name)
    model_name = ml.name

    # Resolve reference side: dir or manifest.
    if reference_dir is not None:
        ref_mode = "dir"
        ref_path = Path(reference_dir)
        if not ref_path.is_dir():
            raise FileNotFoundError(f"reference_dir does not exist: {ref_path}")
        ref_files = _collect_audio_files(ref_path)
        if not ref_files:
            raise FileNotFoundError(f"no audio files under {ref_path}")
        ref_cache_root: Path | None = None
    else:
        ref_mode = "manifest"
        ref_path = Path(reference_manifest)
        if not ref_path.is_file():
            raise FileNotFoundError(f"reference_manifest does not exist: {ref_path}")
        all_paths = _read_reference_manifest(ref_path)
        if not all_paths:
            raise FileNotFoundError(f"manifest {ref_path} lists no audio paths")
        missing = [p for p in all_paths if not p.exists()]
        if missing:
            warn(
                f"[kad] manifest {ref_path.name}: "
                f"{len(missing)}/{len(all_paths)} listed files missing — skipping them"
            )
        ref_files = [p for p in all_paths if p.exists()]
        if not ref_files:
            raise FileNotFoundError(
                f"manifest {ref_path} has no existing audio files on disk"
            )
        ref_cache_root = (
            Path(reference_cache_dir)
            if reference_cache_dir is not None
            else ref_path.with_suffix(ref_path.suffix + ".kad_cache")
        )
        ref_cache_root.mkdir(parents=True, exist_ok=True)

    # 1. Populate the on-disk embedding cache for both sides.
    cache_embedding_files(
        ref_files, ml,
        workers=workers,
        force_emb_encode=force_emb_encode,
        cache_root=ref_cache_root,
    )
    cache_embedding_files(gen, ml, workers=workers, force_emb_encode=force_emb_encode)

    # 1b. Drop audio files that failed to cache (corrupt mp3, unreadable codec,
    #     etc.). ``_cache_batch_worker`` catches per-file errors and continues,
    #     but the audio path is still in ``ref_files`` / gen listing with no
    #     corresponding .npy — reading it would raise FileNotFoundError here.
    def _keep_cached(files: list[Path], cache_root: Path | None) -> list[Path]:
        return [f for f in files if get_cache_embedding_path(ml.name, f, cache_root).exists()]

    ref_files_ok = _keep_cached(ref_files, ref_cache_root)
    gen_files = _collect_audio_files(gen)
    gen_files_ok = _keep_cached(gen_files, None)
    if len(ref_files_ok) < len(ref_files):
        warn(
            f"[kad] {len(ref_files) - len(ref_files_ok)}/{len(ref_files)} "
            f"reference files failed to cache — continuing with {len(ref_files_ok)}"
        )
    if len(gen_files_ok) < len(gen_files):
        warn(
            f"[kad] {len(gen_files) - len(gen_files_ok)}/{len(gen_files)} "
            f"generated files failed to cache — continuing with {len(gen_files_ok)}"
        )
    if not ref_files_ok or not gen_files_ok:
        raise FileNotFoundError("KAD: no usable embeddings after caching")

    # 2. Load concatenated embedding matrices.
    #    The loader that extracted them keeps the already-loaded model;
    #    for reading cached .npy we don't need it again, but reusing the
    #    instance keeps behaviour consistent with upstream.
    ref_reader = EmbeddingLoader(ml, load_model=False, cache_root=ref_cache_root)
    gen_reader = EmbeddingLoader(ml, load_model=False)
    ref_emb = _load_embedding_matrix(ref_reader, ref_files_ok)
    gen_emb = _load_embedding_matrix(gen_reader, gen_files_ok)

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
        reference_dir=str(ref_path.resolve()) if ref_mode == "dir" else None,
        reference_manifest=str(ref_path.resolve()) if ref_mode == "manifest" else None,
        reference_mode=ref_mode,
        reference_cache_dir=str(ref_cache_root.resolve()) if ref_cache_root is not None else None,
        n_generated=len(gen_files_ok),
        n_reference=len(ref_files_ok),
        n_generated_embeddings=int(gen_emb.shape[0]),
        n_reference_embeddings=int(ref_emb.shape[0]),
        device=str(device_t),
        workers=workers,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    if output_json is not None:
        result.to_json(output_json)

    return result
