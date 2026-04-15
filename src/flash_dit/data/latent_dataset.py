"""HDF5-backed dataset of pre-computed audio latents.

Expected HDF5 layout (written by scripts/precompute_latents.py):
    latents    float16  (N, 64, 64)
    genres     int64    (N,)
    track_ids  bytes    (N,)
    split      bytes    (N,)   — b'train' / b'val' / b'test'

HDF5 file attrs:
    mean       float32  (64,)  — per-channel mean over training split
    std        float32  (64,)  — per-channel std  over training split
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class LatentDataset(Dataset):
    """Loads normalised latents from a pre-computed HDF5 file.

    Args:
        h5_path: path to the HDF5 file produced by precompute_latents.py.
        split:   one of 'train', 'val', 'test'.
    """

    def __init__(self, h5_path: str, split: str) -> None:
        import h5py  # lazy import so the module loads without h5py installed

        self.h5_path = h5_path
        self.split = split.encode()

        # Read indices that belong to this split (done once at init).
        with h5py.File(h5_path, "r") as f:
            all_splits = f["split"][:]
            self.indices = np.where(all_splits == self.split)[0]

            # Normalization statistics computed over the training split
            self.mean = torch.tensor(f.attrs["mean"], dtype=torch.float32)  # (C,)
            self.std  = torch.tensor(f.attrs["std"],  dtype=torch.float32)  # (C,)

        # HDF5 file is opened per-worker in __getitem__ to be fork-safe.
        self._file = None

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
        )  # (C, T)
        genre = torch.tensor(self._file["genres"][real_idx], dtype=torch.long)

        # Channel-wise normalisation: (C, 1) broadcast
        latent = (latent - self.mean[:, None]) / (self.std[:, None] + 1e-6)

        return latent, genre

    def __del__(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
