"""ODE samplers for inference: Euler and Heun.

All samplers integrate the learned velocity field from t=1 (pure noise)
back to t=0 (clean data) using the reverse-time ODE:

    dx/dt = -v_θ(x_t, t, cond)

With the rectified-flow convention (x_t = (1-t)*x0 + t*ε), moving from
t=1 → t=0 means denoising from noise to data.
"""
from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def euler_sample(
    model: nn.Module,
    noise: torch.Tensor,
    y: torch.Tensor,
    n_steps: int = 50,
    cfg_scale: float = 1.0,
) -> torch.Tensor:
    """Euler ODE sampler.

    Args:
        model:     DiffusionTransformer.
        noise:     (B, C, T) starting noise (t=1).
        y:         (B,) int64 genre labels.
        n_steps:   number of function evaluations.
        cfg_scale: classifier-free guidance scale (1.0 = no guidance).

    Returns:
        (B, C, T) generated latents.
    """
    dt = -1.0 / n_steps
    x = noise.clone()
    ts = torch.linspace(1.0, 0.0, n_steps + 1, device=noise.device)[:-1]

    for t_val in ts:
        t = t_val.expand(x.shape[0])
        v = _model_call(model, x, t, y, cfg_scale)
        x = x + dt * v

    return x


@torch.no_grad()
def heun_sample(
    model: nn.Module,
    noise: torch.Tensor,
    y: torch.Tensor,
    n_steps: int = 50,
    cfg_scale: float = 1.0,
) -> torch.Tensor:
    """Heun (2nd-order Runge–Kutta) ODE sampler.

    Uses one predictor + one corrector step per NFE pair.
    Quality is noticeably better than Euler at the same n_steps.

    Args:
        Same as euler_sample.

    Returns:
        (B, C, T) generated latents.
    """
    dt = -1.0 / n_steps
    x = noise.clone()
    ts = torch.linspace(1.0, 0.0, n_steps + 1, device=noise.device)

    for i in range(n_steps):
        t0 = ts[i].expand(x.shape[0])
        t1 = ts[i + 1].expand(x.shape[0])

        v0 = _model_call(model, x, t0, y, cfg_scale)       # predictor
        x_pred = x + dt * v0
        v1 = _model_call(model, x_pred, t1, y, cfg_scale)  # corrector
        x = x + dt * 0.5 * (v0 + v1)

    return x


def _model_call(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    cfg_scale: float,
) -> torch.Tensor:
    """Apply the model with optional classifier-free guidance.

    When cfg_scale > 1: batched CFG — concatenate conditioned and unconditioned
    inputs along the batch dimension for a single forward pass, then split.
    """
    if cfg_scale == 1.0:
        return model(x, t, y)

    # Batched CFG: single forward pass with doubled batch
    x2 = torch.cat([x, x], dim=0)
    t2 = torch.cat([t, t], dim=0)
    # First half: conditioned (genre label as-is)
    # Second half: unconditioned (null class via force_drop)
    y_cond   = y
    y_null   = torch.full_like(y, model.y_embedder.null_class)
    y2 = torch.cat([y_cond, y_null], dim=0)

    out = model(x2, t2, y2, force_drop_cond=False)
    v_cond, v_uncond = out.chunk(2, dim=0)

    return v_uncond + cfg_scale * (v_cond - v_uncond)
