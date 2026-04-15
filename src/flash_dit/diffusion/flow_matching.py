"""Rectified flow matching training objective.

Interpolant:  x_t = (1 - t) * x_0 + t * ε,  ε ~ N(0, I)
Target:       v   = ε - x_0                  (velocity field)
Loss:         MSE(model(x_t, t, cond), v)

Reference: Liu et al., "Flow Straight and Fast" (2022).
"""
import torch
import torch.nn.functional as F

from .schedules import sample_logit_normal


def interpolate(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Linearly interpolate between data and noise at timestep t.

    Args:
        x0:    (B, C, T) — clean latent.
        noise: (B, C, T) — standard Gaussian noise.
        t:     (B,)      — timestep in [0, 1].

    Returns:
        (B, C, T) noisy latent x_t.
    """
    t_ = t[:, None, None]  # broadcast over (C, T)
    return (1.0 - t_) * x0 + t_ * noise


def flow_matching_loss(
    model: torch.nn.Module,
    x0: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the rectified flow MSE loss.

    Args:
        model: DiffusionTransformer.
        x0:    (B, C, T) normalised clean latents.
        y:     (B,) int64 genre labels.
        t:     (B,) optional pre-sampled timesteps; sampled internally if None.

    Returns:
        Scalar loss tensor.
    """
    B = x0.shape[0]
    device = x0.device

    if t is None:
        t = sample_logit_normal(B, device)

    noise = torch.randn_like(x0)
    x_t = interpolate(x0, noise, t)
    target = noise - x0  # target velocity

    pred = model(x_t, t, y)
    return F.mse_loss(pred, target)
