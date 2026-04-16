"""Tests for the unbiased MMD^2 estimator."""
from __future__ import annotations

import math

import pytest
import torch

from flash_dit.evaluation.mmd import (
    mmd_unbiased,
    pairwise_sq_dists,
    pairwise_sq_dists_chunked,
    median_pairwise_distance,
)


def _naive_mmd2_gaussian(x: torch.Tensor, y: torch.Tensor, sigma: float) -> float:
    """Reference implementation using explicit nested loops."""
    n = x.shape[0]
    m = y.shape[0]
    s = 1.0 / (2.0 * sigma * sigma)

    def k(a, b):
        return math.exp(-s * float(((a - b) ** 2).sum().item()))

    xx = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            xx += k(x[i], x[j])

    yy = 0.0
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            yy += k(y[i], y[j])

    xy = 0.0
    for i in range(n):
        for j in range(m):
            xy += k(x[i], y[j])

    return xx / (n * (n - 1)) + yy / (m * (m - 1)) - 2.0 * xy / (n * m)


# ---------------------------------------------------------------------------
# Pairwise distance helpers
# ---------------------------------------------------------------------------


def test_pairwise_sq_dists_shape_and_values():
    torch.manual_seed(0)
    x = torch.randn(7, 5)
    y = torch.randn(11, 5)
    d2 = pairwise_sq_dists(x, y)
    assert d2.shape == (7, 11)
    # Spot-check against a loop
    for i in [0, 3, 6]:
        for j in [0, 5, 10]:
            expected = float(((x[i] - y[j]) ** 2).sum())
            assert d2[i, j].item() == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_pairwise_sq_dists_diagonal_is_zero():
    x = torch.randn(8, 4)
    d2 = pairwise_sq_dists(x, x)
    assert torch.allclose(d2.diagonal(), torch.zeros(8), atol=1e-5)


def test_chunked_matches_unchunked():
    torch.manual_seed(1)
    x = torch.randn(50, 6)
    y = torch.randn(37, 6)
    ref = pairwise_sq_dists(x, y)
    for block in [1, 7, 16, 64]:
        out = pairwise_sq_dists_chunked(x, y, block_size=block)
        assert torch.allclose(ref, out, atol=1e-5)


# ---------------------------------------------------------------------------
# Median heuristic
# ---------------------------------------------------------------------------


def test_median_pairwise_distance_is_deterministic():
    torch.manual_seed(2)
    x = torch.randn(200, 8)
    a = median_pairwise_distance(x, subsample=50, seed=123)
    b = median_pairwise_distance(x, subsample=50, seed=123)
    assert a.item() == pytest.approx(b.item())

    c = median_pairwise_distance(x, subsample=50, seed=999)
    # Different seed should generally give a different subsample (and value).
    assert a.item() != pytest.approx(c.item())


# ---------------------------------------------------------------------------
# MMD correctness
# ---------------------------------------------------------------------------


def test_mmd_matches_naive_reference():
    torch.manual_seed(0)
    x = torch.randn(16, 4, dtype=torch.float64)
    y = torch.randn(18, 4, dtype=torch.float64) + 0.7

    sigma = 1.5
    res = mmd_unbiased(x, y, bandwidth=sigma, scale_factor=1.0, block_size=8)
    expected = _naive_mmd2_gaussian(x, y, sigma)
    assert res.mmd2 == pytest.approx(expected, rel=1e-8, abs=1e-10)


def test_mmd_chunked_block_invariance():
    torch.manual_seed(0)
    x = torch.randn(40, 5, dtype=torch.float64)
    y = torch.randn(55, 5, dtype=torch.float64) + 0.2
    sigma = 1.2

    ref = mmd_unbiased(x, y, bandwidth=sigma, scale_factor=1.0, block_size=1024).mmd2
    for block in [3, 8, 32]:
        out = mmd_unbiased(x, y, bandwidth=sigma, scale_factor=1.0, block_size=block).mmd2
        assert out == pytest.approx(ref, rel=1e-10, abs=1e-12)


def test_mmd_identical_distributions_is_near_zero():
    torch.manual_seed(7)
    # Two large independent samples from the same distribution — unbiased MMD^2
    # should be small relative to the shifted case.
    x = torch.randn(300, 3, dtype=torch.float64)
    y = torch.randn(300, 3, dtype=torch.float64)
    same = mmd_unbiased(x, y, bandwidth=1.0, scale_factor=1.0).mmd2
    assert abs(same) < 1e-2


def test_mmd_shifted_distribution_is_larger():
    torch.manual_seed(3)
    x = torch.randn(300, 3, dtype=torch.float64)
    y_close = torch.randn(300, 3, dtype=torch.float64)
    y_far = torch.randn(300, 3, dtype=torch.float64) + 3.0

    close = mmd_unbiased(x, y_close, bandwidth=1.0, scale_factor=1.0).mmd2
    far = mmd_unbiased(x, y_far, bandwidth=1.0, scale_factor=1.0).mmd2
    assert far > close
    assert far > 0.05  # shifted distributions clearly distinguishable


def test_mmd_no_diagonal_leakage_vs_manual():
    # Very small case where we can compute the exact off-diagonal sums.
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    y = torch.tensor([[0.5, 0.5], [1.0, 1.0]], dtype=torch.float64)
    sigma = 0.8
    res = mmd_unbiased(x, y, bandwidth=sigma, scale_factor=1.0, block_size=2)
    expected = _naive_mmd2_gaussian(x, y, sigma)
    assert res.mmd2 == pytest.approx(expected, rel=1e-10, abs=1e-12)


def test_mmd_scale_factor_applied():
    torch.manual_seed(0)
    x = torch.randn(50, 3, dtype=torch.float64)
    y = torch.randn(50, 3, dtype=torch.float64) + 0.5
    raw = mmd_unbiased(x, y, bandwidth=1.0, scale_factor=1.0).mmd2
    res100 = mmd_unbiased(x, y, bandwidth=1.0, scale_factor=100.0)
    assert res100.mmd2 == pytest.approx(raw, rel=1e-12)
    assert res100.scaled == pytest.approx(raw * 100.0, rel=1e-12)


def test_mmd_reports_metadata():
    torch.manual_seed(0)
    x = torch.randn(50, 3)
    y = torch.randn(60, 3)
    res = mmd_unbiased(x, y, bandwidth=None, bandwidth_source="reference")
    assert res.kernel == "gaussian"
    assert res.n_x == 50 and res.n_y == 60
    assert res.bandwidth > 0.0
    assert res.bandwidth_source == "reference"


def test_mmd_requires_matching_feature_dims():
    x = torch.randn(10, 4)
    y = torch.randn(10, 5)
    with pytest.raises(ValueError):
        mmd_unbiased(x, y, bandwidth=1.0)
