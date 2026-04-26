"""Minimal helpers vendored from kadtk.utils.

Only keeps the cache-path helper we need. Sox/ffmpeg probing is dropped
because the embedding pipeline here uses torchaudio resampling exclusively.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def get_cache_embedding_path(
    model: str,
    audio_dir: PathLike,
    cache_root: PathLike | None = None,
) -> Path:
    """Return the canonical .npy cache path for an audio file's embedding.

    When ``cache_root`` is ``None`` (default), mirrors the upstream kadtk
    layout: ``<audio_parent>/embeddings/<model>/<stem>.npy`` — so caches stay
    next to the audio tree and are interchangeable with upstream kadtk.

    When ``cache_root`` is given, caches are redirected to
    ``<cache_root>/embeddings/<model>/<stem>.npy``. This is used by the
    manifest-based reference path in :func:`flash_dit.evaluation.kad.compute_kad`
    so that scattered audio files (e.g. FMA's ``000/``, ``001/``, … tree)
    share a single cache directory instead of polluting each subfolder.

    Note: stems must be unique across all audio files that share the same
    ``cache_root``, otherwise caches would collide. FMA track IDs are globally
    unique (six-digit), so this holds for the FMA reference flow.
    """
    p = Path(audio_dir)
    root = Path(cache_root) if cache_root is not None else p.parent
    return root / "embeddings" / model / p.with_suffix(".npy").name


def download_file(url: str, dest: PathLike, chunk: int = 1 << 20) -> Path:
    """Inline replacement for ``hypy_utils.downloader.download_file``.

    Streams ``url`` to ``dest`` via ``urllib.request``. Shows a tqdm progress
    bar when a content-length is available. Intentionally minimal: no
    resume, no retries. Raises on HTTP errors.
    """
    import urllib.request

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    with urllib.request.urlopen(url) as resp:  # noqa: S310 (known URL)
        total = int(resp.headers.get("Content-Length", 0)) or None
        try:
            from tqdm.auto import tqdm
            bar = tqdm(total=total, unit="B", unit_scale=True, desc=dest.name)
        except ImportError:
            bar = None

        with open(tmp, "wb") as f:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                if bar is not None:
                    bar.update(len(buf))
        if bar is not None:
            bar.close()

    tmp.replace(dest)
    return dest
