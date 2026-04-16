"""Audio embedding extraction and caching (trimmed vendor of kadtk.emb_loader).

Changes vs upstream:

- **torchaudio-only** resampling path — drop the sox/ffmpeg alternative
  branch. Modern torchaudio handles every format we care about via libsox
  under the hood, with higher-quality sinc-interp resampling.
- **No hypy_utils dependency** — worker pool uses
  ``concurrent.futures.ProcessPoolExecutor`` and progress via ``tqdm``.
- **Serial fallback** for tests: ``cache_embedding_files(..., workers=1)``
  runs inline without spawning subprocesses (avoids the pickled-ModelLoader
  round-trip that requires heavy deps to be importable in workers).

Cache layout is kept identical to upstream so caches are interchangeable:

    <audio_dir>/convert/<sr>/<stem>.wav       # resampled mono @ model sr
    <audio_dir>/embeddings/<model_name>/<stem>.npy
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torchaudio

from .model_loader import ModelLoader
from .utils import get_cache_embedding_path

PathLike = Union[str, Path]
_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}


class EmbeddingLoader:
    """Extract embeddings for audio files, with on-disk caching."""

    def __init__(self, model: ModelLoader, load_model: bool = True) -> None:
        self.ml = model
        if load_model:
            self.ml.load_model()
            self.loaded = True

    def load_audio(self, f: PathLike) -> np.ndarray:
        """Resample ``f`` to mono @ model sample rate, cache the resampled WAV."""
        f = Path(f)
        cache_dir = f.parent / "convert" / str(self.ml.sr)
        new = (cache_dir / f.name).with_suffix(".wav")
        if not new.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            wav, sr0 = torchaudio.load(str(f))
            wav = torch.mean(wav, dim=0, keepdim=True)  # → mono (1, T)
            if sr0 != self.ml.sr:
                resampler = torchaudio.transforms.Resample(
                    sr0,
                    self.ml.sr,
                    lowpass_filter_width=64,
                    rolloff=0.9475937167399596,
                    resampling_method="sinc_interp_kaiser",
                    beta=14.769656459379492,
                )
                wav = resampler(wav)
            torchaudio.save(
                str(new), wav, self.ml.sr,
                encoding="PCM_S", bits_per_sample=16,
            )
        return self.ml.load_wav(new)

    def read_embedding_file(self, audio_path: PathLike) -> np.ndarray:
        cache = get_cache_embedding_path(self.ml.name, audio_path)
        if not cache.exists():
            raise FileNotFoundError(f"Embedding cache missing: {cache}")
        return np.load(cache)

    def cache_embedding_file(self, audio_path: PathLike) -> Path:
        """Compute + cache the embedding for one file, return the cache path."""
        cache = get_cache_embedding_path(self.ml.name, audio_path)
        if cache.exists():
            return cache
        wav = self.load_audio(audio_path)
        emb = self.ml.get_embedding(wav)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, emb)
        return cache

    def load_embeddings(
        self, audio_dir: PathLike, max_count: int = -1, concat: bool = True,
    ) -> np.ndarray:
        """Read every cached embedding under ``audio_dir`` (flat glob)."""
        files = _collect_audio_files(audio_dir)
        if not files:
            raise ValueError(f"No audio files found under {audio_dir}")
        return self._load_embeddings(files, max_count=max_count, concat=concat)

    def _load_embeddings(
        self, files: list[Path], max_count: int = -1, concat: bool = True,
    ):
        if max_count == -1:
            embds = [self.read_embedding_file(f) for f in files]
        else:
            embds = []
            total = 0
            for f in files:
                e = self.read_embedding_file(f)
                embds.append(e)
                total += e.shape[0]
                if total > max_count:
                    break
        if concat:
            return np.concatenate(embds, axis=0)
        return embds, files


def _collect_audio_files(audio_dir: PathLike) -> list[Path]:
    d = Path(audio_dir)
    return sorted(p for p in d.glob("*") if p.suffix.lower() in _AUDIO_EXTS)


def _cache_batch_worker(args: tuple[list[Path], ModelLoader]) -> int:
    files, ml = args
    loader = EmbeddingLoader(ml, load_model=True)
    n = 0
    for f in files:
        loader.cache_embedding_file(f)
        n += 1
    return n


def cache_embedding_files(
    files_or_dir: Union[list[Path], str, Path],
    ml: ModelLoader,
    workers: int = 8,
    force_emb_encode: bool = False,
) -> int:
    """Populate the embedding cache for every audio file under ``files_or_dir``.

    With ``workers=1`` runs inline (no subprocess, no pickling) — this is
    the path used in unit tests with mock model loaders. With ``workers>1``
    uses a ``ProcessPoolExecutor`` so each worker can hold the pretrained
    model in its own GPU memory.

    Returns the number of files that were (re)cached this call.
    """
    import shutil

    files = (
        list(_collect_audio_files(files_or_dir))
        if isinstance(files_or_dir, (str, Path))
        else list(files_or_dir)
    )
    if not files:
        return 0

    if force_emb_encode:
        emb_root = files[0].parent / "embeddings" / ml.name
        if emb_root.exists():
            shutil.rmtree(emb_root)

    to_encode = [f for f in files if not get_cache_embedding_path(ml.name, f).exists()]
    if not to_encode:
        return 0

    if workers <= 1:
        return _cache_batch_worker((to_encode, ml))

    batches = [list(b) for b in np.array_split(to_encode, workers) if len(b) > 0]
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n in pool.map(_cache_batch_worker, [(b, ml) for b in batches]):
            done += n
    return done
