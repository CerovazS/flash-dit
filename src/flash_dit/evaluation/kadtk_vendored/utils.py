"""Minimal helpers vendored from kadtk.utils.

Only keeps the cache-path helper we need. Sox/ffmpeg probing is dropped
because the embedding pipeline here uses torchaudio resampling exclusively.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def get_cache_embedding_path(model: str, audio_dir: PathLike) -> Path:
    """Return the canonical .npy cache path for an audio file's embedding.

    Mirrors the upstream kadtk layout: ``<audio_parent>/embeddings/<model>/<stem>.npy``.
    Keeping the same layout means we can mix files produced by upstream kadtk
    and by this vendored copy in the same tree without re-extracting.
    """
    p = Path(audio_dir)
    return p.parent / "embeddings" / model / p.with_suffix(".npy").name


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
