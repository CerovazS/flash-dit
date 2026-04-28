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


# ---------------------------------------------------------------------------
# Schema v2 validation
# ---------------------------------------------------------------------------


def _make_v2_h5(tmp_path, encoder_id="stable_audio_open", domain="audio", n=20) -> str:
    """Mock HDF5 with the v2 schema attrs populated."""
    import h5py

    path = str(tmp_path / f"v2_{encoder_id}_{domain}.h5")
    rng = np.random.default_rng(0)
    latents = rng.standard_normal((n, 64, 64)).astype(np.float16)
    splits = np.array([b"train"] * (n - 5) + [b"val"] * 3 + [b"test"] * 2, dtype=object)

    with h5py.File(path, "w") as f:
        f.create_dataset("latents", data=latents)
        f.create_dataset("genres", data=rng.integers(0, 16, n).astype(np.int64))
        f.create_dataset("track_ids", data=np.array([f"t{i}".encode() for i in range(n)]))
        f.create_dataset("split", data=splits)
        f.attrs["mean"] = latents.mean(axis=(0, 2)).astype(np.float32)
        f.attrs["std"] = latents.std(axis=(0, 2)).astype(np.float32) + 1e-6
        f.attrs["schema_version"] = 2
        f.attrs["encoder_id"] = encoder_id
        f.attrs["domain"] = domain
        f.attrs["latent_format"] = "BCT" if domain == "audio" else "BCHW"
    return path


def test_dataset_validation_passes_on_match(tmp_path):
    from flash_dit.data.latent_dataset import LatentDataset

    h5 = _make_v2_h5(tmp_path, encoder_id="stable_audio_open", domain="audio")
    # No exception raised
    ds = LatentDataset(
        h5, split="train",
        expected_encoder_id="stable_audio_open",
        expected_domain="audio",
    )
    assert len(ds) > 0


def test_dataset_validation_fails_on_encoder_mismatch(tmp_path):
    from flash_dit.data.latent_dataset import EncoderMismatchError, LatentDataset

    h5 = _make_v2_h5(tmp_path, encoder_id="stable_audio_open")
    with pytest.raises(EncoderMismatchError, match="encoder_id="):
        LatentDataset(
            h5, split="train",
            expected_encoder_id="music2latent",
        )


def test_dataset_validation_fails_on_domain_mismatch(tmp_path):
    from flash_dit.data.latent_dataset import EncoderMismatchError, LatentDataset

    h5 = _make_v2_h5(tmp_path, domain="audio")
    with pytest.raises(EncoderMismatchError, match="domain="):
        LatentDataset(
            h5, split="train",
            expected_domain="vision",
        )


def test_dataset_backcompat_shim_warns_and_proceeds(tmp_path, recwarn):
    """Pre-v2 file (no encoder_id / domain) is accepted as SAO audio with a warning."""
    import flash_dit.data.latent_dataset as ds_mod
    from flash_dit.data.latent_dataset import LatentDataset

    # Reset the module-global once-flag so this test sees the warning.
    ds_mod._BACKCOMPAT_WARNED = False

    h5 = _make_mock_h5(tmp_path)  # legacy: only mean/std attrs
    ds = LatentDataset(
        h5, split="train",
        expected_encoder_id="stable_audio_open",  # matches the shim's default
    )
    assert len(ds) > 0
    deprecation_warnings = [w for w in recwarn if issubclass(w.category, DeprecationWarning)]
    assert any("pre-v2 schema" in str(w.message) for w in deprecation_warnings)


def test_dataset_backcompat_shim_fails_on_mismatch(tmp_path):
    """Even with the shim, a non-default expected_encoder_id raises."""
    import flash_dit.data.latent_dataset as ds_mod
    from flash_dit.data.latent_dataset import EncoderMismatchError, LatentDataset

    ds_mod._BACKCOMPAT_WARNED = False
    h5 = _make_mock_h5(tmp_path)
    with pytest.raises(EncoderMismatchError):
        LatentDataset(h5, split="train", expected_encoder_id="music2latent")
