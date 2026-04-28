"""HDF5-backed dataset of pre-computed audio (or vision) latents.

Schema v2 (see ``plan.md`` §2). Pre-v2 files have only ``mean`` / ``std``
attrs and are silently treated as ``stable_audio_open`` audio data via
the backward-compat shim — a one-shot warning fires the first time we
open such a file.

Datasets:
    latents    float16  (N, C, T)   for audio
                       (N, C, H, W) for vision
    genres     int64    (N,)
    track_ids  bytes    (N,)
    split      bytes    (N,)   — b'train' / b'val' / b'test'

Attrs (v2):
    schema_version   int
    encoder_id       str    e.g. "stable_audio_open"
    domain           str    "audio" | "vision"
    input_format     str    "BCT" | "BCHW"
    latent_format    str    "BCT" | "BCHW"
    latent_channels  int
    compression_ratios list[int]
    sample_rate      int (audio) | -1 (vision)
    mean             float32 (C,)   — over training split
    std              float32 (C,)   — over training split
    …                see plan.md
"""
from __future__ import annotations

import warnings

import numpy as np
import torch
from torch.utils.data import Dataset


_BACKCOMPAT_WARNED = False


def _decode_attr(v):
    """h5py returns bytes for str attrs; decode for comparison."""
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.ndarray) and v.dtype.kind == "S":
        return v.item().decode("utf-8", errors="replace")
    return v


class EncoderMismatchError(ValueError):
    """Raised when an HDF5 file's encoder identity disagrees with the config."""


class LatentDataset(Dataset):
    """Loads normalised latents from a pre-computed HDF5 file.

    Args:
        h5_path: path to the HDF5 file produced by ``precompute_latents.py``.
        split:   one of ``'train'``, ``'val'``, ``'test'``.
        seq_len: if given, randomly crop the temporal dim to this many frames.
                 Audio only (1-D temporal latent).
        expected_encoder_id: if set, the file's ``encoder_id`` attr must match.
                             Pre-v2 files are accepted with a one-shot warning
                             and assumed to be ``"stable_audio_open"``.
        expected_domain:     if set, must match the file's ``domain`` attr
                             (``"audio"`` or ``"vision"``). Pre-v2 files
                             default to ``"audio"``.
    """

    def __init__(
        self,
        h5_path: str,
        split: str,
        seq_len: int | None = None,
        expected_encoder_id: str | None = None,
        expected_domain: str | None = None,
    ) -> None:
        import h5py

        self.h5_path = h5_path
        self.split = split.encode()
        self.seq_len = seq_len
        # Set early so __del__ doesn't AttributeError if validation raises.
        self._file = None

        # Read indices that belong to this split (done once at init).
        with h5py.File(h5_path, "r") as f:
            all_splits = f["split"][:]
            self.indices = np.where(all_splits == self.split)[0]

            self.mean = torch.tensor(f.attrs["mean"], dtype=torch.float32)
            self.std  = torch.tensor(f.attrs["std"],  dtype=torch.float32)

            self._validate_schema(
                f, h5_path,
                expected_encoder_id=expected_encoder_id,
                expected_domain=expected_domain,
            )

        # _file is opened per-worker in __getitem__ to be fork-safe; the
        # initial assignment above kept it None for __del__ safety.

    @staticmethod
    def _validate_schema(
        f,
        h5_path: str,
        expected_encoder_id: str | None,
        expected_domain: str | None,
    ) -> None:
        """Fail fast on encoder / domain mismatch; warn once on pre-v2 files.

        Pre-v2 files (no ``encoder_id`` attr) are accepted as
        ``stable_audio_open`` audio data — this is the backward-compat shim
        that lets old HDF5s keep working without re-encoding.
        """
        global _BACKCOMPAT_WARNED

        encoder_id = _decode_attr(f.attrs.get("encoder_id"))
        domain = _decode_attr(f.attrs.get("domain"))

        if encoder_id is None or domain is None:
            # Pre-v2: assume SAO audio. Warn once globally to avoid spamming
            # in DDP / multi-worker setups.
            if not _BACKCOMPAT_WARNED:
                warnings.warn(
                    f"{h5_path}: pre-v2 schema (no encoder_id/domain attrs). "
                    f"Assuming encoder_id='stable_audio_open', domain='audio'. "
                    f"Run scripts/migrate_latents_h5.py to upgrade.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                _BACKCOMPAT_WARNED = True
            encoder_id = encoder_id or "stable_audio_open"
            domain = domain or "audio"

        if expected_encoder_id is not None and encoder_id != expected_encoder_id:
            raise EncoderMismatchError(
                f"{h5_path}: encoder_id={encoder_id!r} on disk but config "
                f"expects {expected_encoder_id!r}. Either re-precompute the "
                f"latents with the configured encoder, or switch the "
                f"autoencoder.kind to match the file."
            )
        if expected_domain is not None and domain != expected_domain:
            raise EncoderMismatchError(
                f"{h5_path}: domain={domain!r} on disk but config expects "
                f"{expected_domain!r}. The autoencoder backbone produces "
                f"the wrong modality for this dataset."
            )

    def _open(self) -> None:
        import h5py
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r", swmr=True)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._open()
        real_idx = int(self.indices[idx])

        latent = torch.from_numpy(
            self._file["latents"][real_idx].astype(np.float32)
        )  # (C, T) for audio, (C, H, W) for vision
        genre = torch.tensor(self._file["genres"][real_idx], dtype=torch.long)

        # Channel-wise normalisation. Audio uses (C, 1) broadcast over T;
        # vision would need (C, 1, 1) — only the audio path is exercised today.
        if latent.dim() == 2:
            latent = (latent - self.mean[:, None]) / (self.std[:, None] + 1e-6)
        elif latent.dim() == 3:
            latent = (latent - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-6)
        else:
            raise ValueError(f"unexpected latent rank {latent.dim()} (shape {tuple(latent.shape)})")

        # Random temporal crop — audio path only; assumes T >= seq_len.
        if self.seq_len is not None and latent.dim() == 2 and latent.shape[1] > self.seq_len:
            max_start = latent.shape[1] - self.seq_len
            start = torch.randint(0, max_start + 1, ()).item()
            latent = latent[:, start : start + self.seq_len]

        return latent, genre

    def __del__(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
