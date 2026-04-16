"""Unbiased MMD^2 estimator shared by latent_mmd and KAD.

Implements the finite-sample unbiased Maximum Mean Discrepancy squared
estimator used in `kadtk` and the KAD paper (arXiv 2502.15602):

    MMD^2(X, Y) = 1/(n(n-1)) sum_{i!=j} k(x_i, x_j)
                + 1/(m(m-1)) sum_{i!=j} k(y_i, y_j)
                - 2/(nm)     sum_i sum_j k(x_i, y_j)

For the Gaussian RBF kernel k(a,b) = exp(-||a-b||^2 / (2 sigma^2)) the
diagonal terms k(x_i, x_i) = 1 are subtracted out exactly when
accumulating the within-set sums, so the estimator has no diagonal leakage
and is negative-unbiased.

Pairwise squared-distance computations are chunked so large feature
matrices never materialise the full (n, m) kernel at once.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

KernelName = Literal["gaussian"]


def _sq_norms(x: torch.Tensor) -> torch.Tensor:
    return (x * x).sum(dim=-1)


def pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return (n, m) matrix of squared Euclidean distances between rows."""
    x_sq = _sq_norms(x).unsqueeze(1)   # (n, 1)
    y_sq = _sq_norms(y).unsqueeze(0)   # (1, m)
    d2 = x_sq + y_sq - 2.0 * (x @ y.T)
    return d2.clamp_min_(0.0)


def pairwise_sq_dists_chunked(
    x: torch.Tensor,
    y: torch.Tensor,
    block_size: int = 4096,
) -> torch.Tensor:
    """Memory-safe equivalent of pairwise_sq_dists via row-chunking of x.

    The output matrix has shape (n, m) and is identical (up to floating
    point non-associativity) to pairwise_sq_dists(x, y).
    """
    n = x.shape[0]
    m = y.shape[0]
    out = x.new_empty((n, m))
    y_sq = _sq_norms(y).unsqueeze(0)
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        chunk = x[start:end]
        chunk_sq = _sq_norms(chunk).unsqueeze(1)
        d2 = chunk_sq + y_sq - 2.0 * (chunk @ y.T)
        out[start:end] = d2.clamp_min_(0.0)
    return out


def median_pairwise_distance(
    x: torch.Tensor,
    subsample: int | None = None,
    seed: int | None = 0,
) -> torch.Tensor:
    """Median heuristic for the Gaussian RBF bandwidth sigma.

    Computes the median of ||x_i - x_j||_2 over off-diagonal pairs
    (i != j) of the given feature matrix. Optionally subsamples rows to
    keep the computation tractable on large sets.
    """
    n = x.shape[0]
    if subsample is not None and n > subsample:
        g = torch.Generator(device="cpu")
        if seed is not None:
            g.manual_seed(int(seed))
        idx = torch.randperm(n, generator=g)[:subsample]
        x = x[idx]
        n = x.shape[0]

    d2 = pairwise_sq_dists(x, x)
    mask = ~torch.eye(n, dtype=torch.bool, device=d2.device)
    d = d2[mask].clamp_min(0.0).sqrt()
    return d.median()


def _gaussian_kernel_sum_offdiag(
    x: torch.Tensor,
    inv_two_sigma_sq: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, int]:
    """Return (sum_{i!=j} k(x_i, x_j), n*(n-1)) for Gaussian RBF.

    Uses that k(x_i, x_i) = 1 exactly, so we can subtract n from the
    full-matrix sum to exclude the diagonal without ever materialising
    or reading it.
    """
    n = x.shape[0]
    total = x.new_zeros(())
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        d2 = pairwise_sq_dists_chunked(x[start:end], x, block_size=block_size)
        total = total + torch.exp(-d2 * inv_two_sigma_sq).sum()
    total = total - float(n)  # remove diagonal contributions (exact for Gaussian)
    return total, n * (n - 1)


def _gaussian_kernel_sum_cross(
    x: torch.Tensor,
    y: torch.Tensor,
    inv_two_sigma_sq: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, int]:
    """Return (sum_i sum_j k(x_i, y_j), n*m) for Gaussian RBF."""
    n = x.shape[0]
    m = y.shape[0]
    total = x.new_zeros(())
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        d2 = pairwise_sq_dists_chunked(x[start:end], y, block_size=block_size)
        total = total + torch.exp(-d2 * inv_two_sigma_sq).sum()
    return total, n * m


@dataclass
class MMDResult:
    """Container for unbiased MMD^2 output and diagnostic metadata."""

    mmd2: float
    scaled: float
    kernel: str
    bandwidth: float
    scale_factor: float
    n_x: int
    n_y: int
    bandwidth_source: str


def mmd_unbiased(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: KernelName = "gaussian",
    bandwidth: float | None = None,
    bandwidth_source: Literal["reference", "generated", "fixed"] = "reference",
    bandwidth_subsample: int | None = 10_000,
    scale_factor: float = 100.0,
    block_size: int = 4096,
    seed: int | None = 0,
) -> MMDResult:
    """Unbiased finite-sample MMD^2 with a Gaussian RBF kernel.

    Args:
        x: reference feature matrix (n, d).
        y: generated/eval feature matrix (m, d).
        kernel: only 'gaussian' is supported.
        bandwidth: RBF sigma. If None, computed by the median heuristic on
            the set selected by bandwidth_source.
        bandwidth_source: which set to use for the median heuristic when
            bandwidth is None. 'reference' matches the KAD paper/README;
            'generated' matches the current kadtk code path.
        bandwidth_subsample: rows to subsample when computing the median
            pairwise distance. None uses all rows.
        scale_factor: multiplicative scale applied to MMD^2 for reporting.
            Both raw and scaled values are returned.
        block_size: chunk size used for pairwise kernel sums.
        seed: deterministic seed for the bandwidth subsample.

    Returns:
        MMDResult with raw MMD^2, scaled value, and metadata. The raw value
        may be slightly negative for close distributions — this is expected
        for the unbiased estimator.
    """
    if kernel != "gaussian":
        raise NotImplementedError(f"kernel '{kernel}' is not supported")

    if x.dim() != 2 or y.dim() != 2:
        raise ValueError(f"x, y must be 2-D matrices, got {x.shape=}, {y.shape=}")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"feature dims differ: {x.shape[1]} vs {y.shape[1]}")
    if x.shape[0] < 2 or y.shape[0] < 2:
        raise ValueError("need at least 2 samples per set for the unbiased estimator")

    if bandwidth is None:
        source = x if bandwidth_source == "reference" else y
        sigma = median_pairwise_distance(source, subsample=bandwidth_subsample, seed=seed)
        sigma_val = float(sigma.item())
        if sigma_val <= 0.0:
            raise ValueError("median pairwise distance is zero; bandwidth cannot be computed")
        bandwidth_used = sigma_val
    else:
        bandwidth_used = float(bandwidth)

    inv_two_sigma_sq = x.new_tensor(1.0 / (2.0 * bandwidth_used * bandwidth_used))

    k_xx_sum, xx_denom = _gaussian_kernel_sum_offdiag(x, inv_two_sigma_sq, block_size)
    k_yy_sum, yy_denom = _gaussian_kernel_sum_offdiag(y, inv_two_sigma_sq, block_size)
    k_xy_sum, xy_denom = _gaussian_kernel_sum_cross(x, y, inv_two_sigma_sq, block_size)

    mmd2 = k_xx_sum / xx_denom + k_yy_sum / yy_denom - 2.0 * k_xy_sum / xy_denom
    mmd2_val = float(mmd2.item())

    return MMDResult(
        mmd2=mmd2_val,
        scaled=mmd2_val * float(scale_factor),
        kernel=kernel,
        bandwidth=bandwidth_used,
        scale_factor=float(scale_factor),
        n_x=int(x.shape[0]),
        n_y=int(y.shape[0]),
        bandwidth_source=str(bandwidth_source) if bandwidth is None else "fixed",
    )
