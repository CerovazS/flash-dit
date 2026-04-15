"""Unit tests for flow matching components."""
import pytest
import torch

from flash_dit.diffusion.flow_matching import flow_matching_loss, interpolate
from flash_dit.diffusion.schedules import sample_logit_normal, sample_uniform
from flash_dit.models.dit import DiffusionTransformer


def test_interpolate_endpoints():
    """At t=0, x_t should be x0; at t=1, x_t should be noise."""
    B, C, T = 2, 8, 16
    x0 = torch.randn(B, C, T)
    noise = torch.randn(B, C, T)

    t0 = torch.zeros(B)
    t1 = torch.ones(B)

    assert torch.allclose(interpolate(x0, noise, t0), x0)
    assert torch.allclose(interpolate(x0, noise, t1), noise)


def test_logit_normal_range():
    t = sample_logit_normal(1000, device=torch.device("cpu"))
    assert t.min() > 0.0
    assert t.max() < 1.0
    # Should be approximately centred around 0.5
    assert 0.4 < t.mean().item() < 0.6


def test_flow_matching_loss_tiny_model():
    """Loss should be finite and positive on a tiny model."""
    model = DiffusionTransformer(
        in_channels=8,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        n_classes=4,
        use_compile=False,
        max_seq_len=32,
    )
    model.eval()

    B, C, T = 2, 8, 8
    x0 = torch.randn(B, C, T)
    y  = torch.randint(0, 4, (B,))

    with torch.no_grad():
        loss = flow_matching_loss(model, x0, y)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
