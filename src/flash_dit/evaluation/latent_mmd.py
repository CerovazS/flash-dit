"""Latent-space MMD for cheap training-time distribution monitoring.

Compares generated Stable Audio VAE latents against ground-truth latents
from the pre-computed HDF5 dataset without any VAE decoding.

Two feature transforms are supported, both taking (B, C, T) latents:

- ``frame``        : flatten to (B*T, C) and optionally subsample a fixed
                     number of frames per clip. Useful for local per-frame
                     distribution matching but treats temporally correlated
                     frames as independent evidence unless subsampled.

- ``clip_pooled``  : concatenate channel mean, channel std, and a small
                     fixed number of temporal-chunk means into (B, 6*C) for
                     num_chunks=4. Captures clip-level statistics without
                     high-dimensional C*T flattening.

Normalisation convention: training operates on normalised latents, so the
default compares normalised generated latents to normalised reference
latents. The caller is responsible for the normalisation state.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from .mmd import MMDResult, mmd_unbiased

FeatureType = Literal["frame", "clip_pooled"]


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------


def extract_frame_features(
    latents: torch.Tensor,
    subsample_per_clip: int | None = None,
    seed: int = 0,
) -> torch.Tensor:
    """Flatten (B, C, T) latents to frame-level (N, C).

    If subsample_per_clip is set, sample that many frames per clip without
    replacement using a deterministic generator seeded on (seed, clip_idx).
    """
    if latents.dim() != 3:
        raise ValueError(f"expected (B, C, T), got shape {tuple(latents.shape)}")
    b, c, t = latents.shape

    if subsample_per_clip is None or subsample_per_clip >= t:
        # Plain flatten
        return latents.permute(0, 2, 1).reshape(b * t, c).contiguous()

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    idx = torch.stack(
        [torch.randperm(t, generator=g)[:subsample_per_clip] for _ in range(b)],
        dim=0,
    )  # (B, K)
    gathered = torch.gather(
        latents.permute(0, 2, 1),  # (B, T, C)
        1,
        idx.unsqueeze(-1).expand(b, subsample_per_clip, c).to(latents.device),
    )  # (B, K, C)
    return gathered.reshape(b * subsample_per_clip, c).contiguous()


def extract_clip_pooled_features(
    latents: torch.Tensor,
    num_chunks: int = 4,
) -> torch.Tensor:
    """Pool (B, C, T) latents to per-clip descriptors of shape (B, (2+num_chunks)*C).

    Concatenates in fixed order:
      [mean_T, std_T, chunk_mean_1, ..., chunk_mean_{num_chunks}]

    Each component has shape (B, C) so the full descriptor is (B, (2+num_chunks)*C).
    When T is not divisible by num_chunks the chunk boundaries are computed
    via torch.tensor_split, so chunk sizes differ by at most 1.
    """
    if latents.dim() != 3:
        raise ValueError(f"expected (B, C, T), got shape {tuple(latents.shape)}")
    if num_chunks < 1:
        raise ValueError(f"num_chunks must be >= 1, got {num_chunks}")
    t = latents.shape[2]
    if num_chunks > t:
        raise ValueError(f"num_chunks ({num_chunks}) must be <= T ({t})")

    mean = latents.mean(dim=2)               # (B, C)
    # unbiased=False -> population std, matches numpy default and is defined for T=1
    std = latents.std(dim=2, unbiased=False)  # (B, C)

    chunks = torch.tensor_split(latents, num_chunks, dim=2)
    chunk_means = [chunk.mean(dim=2) for chunk in chunks]  # each (B, C)

    return torch.cat([mean, std, *chunk_means], dim=1).contiguous()


def extract_features(
    latents: torch.Tensor,
    feature_type: FeatureType,
    frame_subsample_per_clip: int | None = 32,
    clip_pool_num_chunks: int = 4,
    seed: int = 0,
) -> torch.Tensor:
    """Dispatch to the requested feature extractor."""
    if feature_type == "frame":
        return extract_frame_features(
            latents, subsample_per_clip=frame_subsample_per_clip, seed=seed
        )
    if feature_type == "clip_pooled":
        return extract_clip_pooled_features(latents, num_chunks=clip_pool_num_chunks)
    raise ValueError(f"unknown feature_type {feature_type!r}")


# ---------------------------------------------------------------------------
# Reference loading / caching
# ---------------------------------------------------------------------------


@dataclass
class ReferenceFeatureSpec:
    """Everything that uniquely identifies a reference feature matrix."""

    h5_path: str
    split: str
    feature_type: FeatureType
    n_reference: int
    seq_len: int
    frame_subsample_per_clip: int | None
    clip_pool_num_chunks: int
    normalized: bool
    seed: int

    def cache_key(self) -> str:
        h5_mtime = "0"
        try:
            h5_mtime = str(int(os.path.getmtime(self.h5_path)))
        except OSError:
            pass
        payload = {**asdict(self), "h5_mtime": h5_mtime}
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


def _load_reference_latents_from_h5(
    h5_path: str,
    split: str,
    n_reference: int,
    seq_len: int,
    normalized: bool,
    seed: int,
) -> torch.Tensor:
    """Load a deterministic subset of reference latents as (N, C, seq_len).

    Matches the LatentDataset normalisation convention: if ``normalized``
    is True, applies per-channel (x - mean) / (std + 1e-6) using stats
    stored in the HDF5 file attrs; otherwise returns the raw latents.
    """
    import h5py  # local import to avoid hard dep at module load

    with h5py.File(h5_path, "r") as f:
        all_splits = f["split"][:]
        split_idx = np.where(all_splits == split.encode())[0]
        if split_idx.size == 0:
            raise ValueError(f"no samples in split {split!r} of {h5_path}")

        rng = np.random.default_rng(seed)
        if split_idx.size < n_reference:
            chosen = split_idx
        else:
            chosen = rng.choice(split_idx, size=n_reference, replace=False)
        chosen = np.sort(chosen)  # contiguous reads

        # Read whole latents then crop centrally to seq_len.
        lats = f["latents"][chosen].astype(np.float32)  # (N, C, T)
        if lats.shape[2] < seq_len:
            raise ValueError(
                f"stored latents have T={lats.shape[2]} < seq_len={seq_len}"
            )
        if lats.shape[2] > seq_len:
            # Deterministic random crop seeded with the same RNG for stable caches
            starts = rng.integers(0, lats.shape[2] - seq_len + 1, size=lats.shape[0])
            cropped = np.empty((lats.shape[0], lats.shape[1], seq_len), dtype=np.float32)
            for i, s in enumerate(starts):
                cropped[i] = lats[i, :, s : s + seq_len]
            lats = cropped

        latents = torch.from_numpy(lats)

        if normalized:
            mean = torch.tensor(f.attrs["mean"], dtype=torch.float32)[None, :, None]
            std = torch.tensor(f.attrs["std"], dtype=torch.float32)[None, :, None]
            latents = (latents - mean) / (std + 1e-6)

    return latents


def load_reference_features(
    spec: ReferenceFeatureSpec,
    cache_dir: str | Path | None = None,
) -> torch.Tensor:
    """Return the reference feature matrix for ``spec``, caching to disk if asked.

    The cache is a single .pt file named after the spec cache_key() plus
    feature_type for readability.
    """
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"ref_{spec.feature_type}_{spec.cache_key()}.pt"
        if cache_path.exists():
            return torch.load(cache_path, map_location="cpu", weights_only=True)

    latents = _load_reference_latents_from_h5(
        spec.h5_path,
        spec.split,
        spec.n_reference,
        spec.seq_len,
        spec.normalized,
        spec.seed,
    )
    feats = extract_features(
        latents,
        feature_type=spec.feature_type,
        frame_subsample_per_clip=spec.frame_subsample_per_clip,
        clip_pool_num_chunks=spec.clip_pool_num_chunks,
        seed=spec.seed,
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(feats, cache_path)

    return feats


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


@dataclass
class LatentMMDResult:
    """Full machine-readable record for a single latent-MMD evaluation."""

    feature_type: str
    mmd2: float
    scaled: float
    kernel: str
    bandwidth: float
    scale_factor: float
    bandwidth_source: str
    n_reference: int
    n_generated: int
    feature_dim: int
    normalized: bool
    seed: int
    extra: dict = field(default_factory=dict)


def compute_latent_mmd(
    generated_latents: torch.Tensor,
    reference_features: torch.Tensor,
    feature_type: FeatureType,
    *,
    frame_subsample_per_clip: int | None = 32,
    clip_pool_num_chunks: int = 4,
    kernel: str = "gaussian",
    bandwidth: float | None = None,
    bandwidth_source: Literal["reference", "generated", "fixed"] = "reference",
    bandwidth_subsample: int | None = 10_000,
    scale_factor: float = 100.0,
    block_size: int = 4096,
    seed: int = 0,
    normalized: bool = True,
    device: torch.device | str | None = None,
) -> LatentMMDResult:
    """Compute unbiased MMD^2 between generated latents and a reference feature matrix.

    ``generated_latents`` is a (B, C, T) tensor; ``reference_features`` is
    a pre-extracted (N, D) feature matrix produced by ``load_reference_features``
    (or ``extract_features``) using the same ``feature_type`` and pooling config.
    """
    if device is not None:
        device = torch.device(device)
        generated_latents = generated_latents.to(device)
        reference_features = reference_features.to(device)

    gen_feats = extract_features(
        generated_latents,
        feature_type=feature_type,
        frame_subsample_per_clip=frame_subsample_per_clip,
        clip_pool_num_chunks=clip_pool_num_chunks,
        seed=seed,
    )

    if gen_feats.shape[1] != reference_features.shape[1]:
        raise ValueError(
            "feature dimension mismatch between generated and reference: "
            f"{gen_feats.shape[1]} vs {reference_features.shape[1]}"
        )

    # Promote to float32 for numerical stability in pairwise distances.
    ref = reference_features.to(dtype=torch.float32)
    gen = gen_feats.to(dtype=torch.float32)

    res: MMDResult = mmd_unbiased(
        ref,
        gen,
        kernel=kernel,
        bandwidth=bandwidth,
        bandwidth_source=bandwidth_source,
        bandwidth_subsample=bandwidth_subsample,
        scale_factor=scale_factor,
        block_size=block_size,
        seed=seed,
    )

    return LatentMMDResult(
        feature_type=feature_type,
        mmd2=res.mmd2,
        scaled=res.scaled,
        kernel=res.kernel,
        bandwidth=res.bandwidth,
        scale_factor=res.scale_factor,
        bandwidth_source=res.bandwidth_source,
        n_reference=res.n_x,
        n_generated=res.n_y,
        feature_dim=int(ref.shape[1]),
        normalized=bool(normalized),
        seed=int(seed),
    )
