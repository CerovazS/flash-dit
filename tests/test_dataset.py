"""Unit tests for the latent dataset (requires an HDF5 file or uses mock)."""
import numpy as np
import pytest
import torch


def _make_mock_h5(tmp_path, n=50) -> str:
    """Create a minimal mock HDF5 file for testing."""
    import h5py

    path = str(tmp_path / "test_latents.h5")
    rng = np.random.default_rng(42)

    latents = rng.standard_normal((n, 64, 64)).astype(np.float16)
    genres  = rng.integers(0, 16, size=n).astype(np.int64)
    splits  = np.array(
        [b"train"] * 40 + [b"val"] * 5 + [b"test"] * 5, dtype=object
    )

    mean = latents[:40].astype(np.float32).mean(axis=(0, 2))
    std  = latents[:40].astype(np.float32).std(axis=(0, 2)) + 1e-6

    with h5py.File(path, "w") as f:
        f.create_dataset("latents",    data=latents)
        f.create_dataset("genres",     data=genres)
        f.create_dataset("track_ids",  data=np.array([f"track_{i}".encode() for i in range(n)]))
        f.create_dataset("split",      data=splits)
        f.attrs["mean"] = mean
        f.attrs["std"]  = std

    return path


@pytest.mark.parametrize("split,expected", [("train", 40), ("val", 5), ("test", 5)])
def test_dataset_split_size(tmp_path, split, expected):
    from flash_dit.data.latent_dataset import LatentDataset

    h5 = _make_mock_h5(tmp_path)
    ds = LatentDataset(h5, split=split)
    assert len(ds) == expected


def test_dataset_item_shape(tmp_path):
    from flash_dit.data.latent_dataset import LatentDataset

    h5 = _make_mock_h5(tmp_path)
    ds = LatentDataset(h5, split="train")
    latent, genre = ds[0]

    assert latent.shape == (64, 64), f"Expected (64, 64), got {latent.shape}"
    assert latent.dtype == torch.float32
    assert genre.dtype == torch.long


def test_dataset_normalisation(tmp_path):
    """After normalisation, per-channel mean should be close to zero."""
    from flash_dit.data.latent_dataset import LatentDataset

    h5 = _make_mock_h5(tmp_path, n=200)
    ds = LatentDataset(h5, split="train")
    items = torch.stack([ds[i][0] for i in range(len(ds))])  # (N, C, T)
    channel_mean = items.mean(dim=(0, 2))  # (C,)
    assert channel_mean.abs().max() < 0.5, "Normalisation not applied correctly"
