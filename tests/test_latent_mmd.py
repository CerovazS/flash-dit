"""Tests for latent-space MMD feature extraction and end-to-end metric."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from flash_dit.evaluation.latent_mmd import (
    ReferenceFeatureSpec,
    compute_latent_mmd,
    extract_clip_pooled_features,
    extract_features,
    extract_frame_features,
    load_reference_features,
)


# ---------------------------------------------------------------------------
# Frame features
# ---------------------------------------------------------------------------


def test_frame_features_full_flatten_shape_and_values():
    torch.manual_seed(0)
    x = torch.randn(3, 4, 5)  # (B, C, T)
    feats = extract_frame_features(x, subsample_per_clip=None)
    assert feats.shape == (3 * 5, 4)
    # First B*T rows should be the permute-reshape layout
    expected = x.permute(0, 2, 1).reshape(-1, 4)
    assert torch.allclose(feats, expected)


def test_frame_features_subsample_shape_and_determinism():
    x = torch.randn(2, 3, 20)
    a = extract_frame_features(x, subsample_per_clip=5, seed=42)
    b = extract_frame_features(x, subsample_per_clip=5, seed=42)
    c = extract_frame_features(x, subsample_per_clip=5, seed=7)
    assert a.shape == (2 * 5, 3)
    assert torch.allclose(a, b)
    assert not torch.allclose(a, c)


def test_frame_features_subsample_larger_than_t_returns_full():
    x = torch.randn(2, 3, 4)
    feats = extract_frame_features(x, subsample_per_clip=99)
    assert feats.shape == (2 * 4, 3)
    assert torch.allclose(feats, x.permute(0, 2, 1).reshape(-1, 3))


# ---------------------------------------------------------------------------
# Clip pooled features
# ---------------------------------------------------------------------------


def test_clip_pooled_shape_and_order():
    torch.manual_seed(0)
    b, c, t = 2, 4, 16
    x = torch.randn(b, c, t)
    feats = extract_clip_pooled_features(x, num_chunks=4)
    # (2 + 4) * C columns
    assert feats.shape == (b, 6 * c)

    # Components should appear in order: mean, std, chunk_1..chunk_4
    expected_mean = x.mean(dim=2)
    expected_std = x.std(dim=2, unbiased=False)
    chunks = torch.tensor_split(x, 4, dim=2)
    expected_chunk_means = torch.cat([ck.mean(dim=2) for ck in chunks], dim=1)

    assert torch.allclose(feats[:, :c], expected_mean)
    assert torch.allclose(feats[:, c : 2 * c], expected_std)
    assert torch.allclose(feats[:, 2 * c :], expected_chunk_means)


def test_clip_pooled_handles_uneven_chunks():
    x = torch.randn(1, 2, 10)
    # 10 not divisible by 4 — tensor_split should still produce 4 chunks
    feats = extract_clip_pooled_features(x, num_chunks=4)
    assert feats.shape == (1, 6 * 2)


def test_clip_pooled_rejects_more_chunks_than_T():
    x = torch.randn(1, 2, 3)
    with pytest.raises(ValueError):
        extract_clip_pooled_features(x, num_chunks=4)


def test_extract_features_dispatch():
    x = torch.randn(2, 3, 8)
    a = extract_features(x, feature_type="frame", frame_subsample_per_clip=None)
    b = extract_features(x, feature_type="clip_pooled", clip_pool_num_chunks=4)
    c = extract_features(x, feature_type="frame", frame_subsample_per_clip=4, seed=0)
    assert a.shape == (16, 3)
    assert b.shape == (2, 6 * 3)
    assert c.shape == (8, 3)
    with pytest.raises(ValueError):
        extract_features(x, feature_type="bogus")  # type: ignore[arg-type]


def test_bad_input_shape_raises():
    with pytest.raises(ValueError):
        extract_frame_features(torch.randn(2, 3))  # missing T
    with pytest.raises(ValueError):
        extract_clip_pooled_features(torch.randn(2, 3))  # missing T


# ---------------------------------------------------------------------------
# End-to-end metric on synthetic latents (no HDF5)
# ---------------------------------------------------------------------------


def test_compute_latent_mmd_identical_distributions_is_small():
    torch.manual_seed(0)
    # Two independent batches from the same distribution
    ref_latents = torch.randn(48, 8, 32)
    gen_latents = torch.randn(48, 8, 32)
    ref_feats = extract_clip_pooled_features(ref_latents, num_chunks=4)
    res = compute_latent_mmd(
        gen_latents,
        ref_feats,
        feature_type="clip_pooled",
        clip_pool_num_chunks=4,
        bandwidth_source="reference",
        scale_factor=1.0,
    )
    assert res.feature_dim == 6 * 8
    assert res.n_reference == 48
    assert res.n_generated == 48
    assert abs(res.mmd2) < 0.2


def test_compute_latent_mmd_shifted_is_larger_than_identical():
    torch.manual_seed(1)
    ref_latents = torch.randn(64, 8, 32)
    close_latents = torch.randn(64, 8, 32)
    far_latents = torch.randn(64, 8, 32) + 2.5

    ref_feats = extract_clip_pooled_features(ref_latents, num_chunks=4)

    close = compute_latent_mmd(
        close_latents,
        ref_feats,
        feature_type="clip_pooled",
        clip_pool_num_chunks=4,
        scale_factor=1.0,
    )
    far = compute_latent_mmd(
        far_latents,
        ref_feats,
        feature_type="clip_pooled",
        clip_pool_num_chunks=4,
        scale_factor=1.0,
    )
    assert far.mmd2 > close.mmd2
    assert far.mmd2 > 0.05


def test_compute_latent_mmd_frame_feature_runs_on_cpu():
    torch.manual_seed(2)
    ref_latents = torch.randn(16, 6, 20)
    gen_latents = torch.randn(16, 6, 20)
    ref_feats = extract_frame_features(ref_latents, subsample_per_clip=8, seed=0)
    res = compute_latent_mmd(
        gen_latents,
        ref_feats,
        feature_type="frame",
        frame_subsample_per_clip=8,
        seed=0,
        scale_factor=1.0,
    )
    assert res.feature_type == "frame"
    assert res.n_reference == 16 * 8
    assert res.n_generated == 16 * 8
    assert res.bandwidth > 0.0


# ---------------------------------------------------------------------------
# Cache keying
# ---------------------------------------------------------------------------


def test_reference_feature_spec_key_depends_on_fields(tmp_path):
    fake_h5 = tmp_path / "fake.h5"
    fake_h5.write_bytes(b"not really h5")

    base = ReferenceFeatureSpec(
        h5_path=str(fake_h5),
        split="val",
        feature_type="clip_pooled",
        n_reference=128,
        seq_len=32,
        frame_subsample_per_clip=32,
        clip_pool_num_chunks=4,
        normalized=True,
        seed=0,
    )
    other = ReferenceFeatureSpec(**{**base.__dict__, "n_reference": 256})
    diff_seed = ReferenceFeatureSpec(**{**base.__dict__, "seed": 1})

    assert base.cache_key() == base.cache_key()
    assert base.cache_key() != other.cache_key()
    assert base.cache_key() != diff_seed.cache_key()


# ---------------------------------------------------------------------------
# HDF5 round-trip
# ---------------------------------------------------------------------------


def _make_fake_h5(path, n=12, c=4, t=16):
    import h5py

    rng = np.random.default_rng(0)
    latents = rng.normal(size=(n, c, t)).astype(np.float16)
    genres = rng.integers(0, 3, size=n).astype(np.int64)
    splits = np.array([b"train"] * (n // 2) + [b"val"] * (n - n // 2))
    mean = rng.normal(size=c).astype(np.float32)
    std = np.abs(rng.normal(size=c).astype(np.float32)) + 0.1

    with h5py.File(path, "w") as f:
        f.create_dataset("latents", data=latents)
        f.create_dataset("genres", data=genres)
        f.create_dataset("split", data=splits)
        f.attrs["mean"] = mean
        f.attrs["std"] = std


def test_load_reference_features_round_trip(tmp_path):
    h5_path = tmp_path / "fake.h5"
    _make_fake_h5(str(h5_path), n=20, c=4, t=24)

    spec = ReferenceFeatureSpec(
        h5_path=str(h5_path),
        split="val",
        feature_type="clip_pooled",
        n_reference=8,
        seq_len=16,
        frame_subsample_per_clip=4,
        clip_pool_num_chunks=4,
        normalized=True,
        seed=0,
    )

    cache_dir = tmp_path / "cache"
    feats1 = load_reference_features(spec, cache_dir=cache_dir)
    assert feats1.shape == (8, 6 * 4)
    # Second call must hit the cache and return identical tensor
    feats2 = load_reference_features(spec, cache_dir=cache_dir)
    assert torch.allclose(feats1, feats2)

    # Cached file exists on disk
    cached_files = list(cache_dir.glob("ref_clip_pooled_*.pt"))
    assert len(cached_files) == 1
