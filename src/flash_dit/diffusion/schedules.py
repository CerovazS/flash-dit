"""Timestep sampling schedules for flow matching training."""
import torch


def sample_logit_normal(n: int, device: torch.device, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    """Sample timesteps from a logit-normal distribution.

    Maps a Gaussian through the sigmoid, concentrating samples in (0, 1)
    but de-emphasising t≈0 and t≈1 which are trivially easy for the model.
    Reference: Esser et al., "Scaling Rectified Flow Transformers" (SD3).

    Args:
        n:      number of samples.
        device: target device.
        mean:   mean of the underlying Gaussian (default 0 → median at 0.5).
        std:    std of the underlying Gaussian (default 1).

    Returns:
        (n,) float tensor in (0, 1).
    """
    z = torch.randn(n, device=device) * std + mean
    return torch.sigmoid(z)


def sample_uniform(n: int, device: torch.device) -> torch.Tensor:
    """Uniform timestep sampling in (0, 1) — kept as a simple baseline."""
    return torch.rand(n, device=device)
