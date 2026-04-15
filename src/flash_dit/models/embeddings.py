"""Timestep and positional embeddings for the DiT."""
import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------

def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Create sinusoidal timestep embeddings.

    Args:
        t: (B,) float tensor of timesteps in [0, 1].
        dim: embedding dimension (must be even).
        max_period: controls the minimum frequency.

    Returns:
        (B, dim) float tensor.
    """
    assert dim % 2 == 0
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into a d_model-dimensional vector via MLP."""

    def __init__(self, d_model: int, freq_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.freq_dim = freq_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Args:
            t: (B,) float timesteps in [0, 1].
        Returns:
            (B, d_model) embedding.
        """
        freq_emb = timestep_embedding(t, self.freq_dim)
        return self.mlp(freq_emb)


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE) — 1D temporal
# ---------------------------------------------------------------------------

def precompute_freqs_cis(head_dim: int, seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute the complex exponentials for RoPE.

    Returns:
        (seq_len, head_dim // 2) complex tensor.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (seq_len, head_dim // 2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """Apply RoPE to query and key tensors.

    Args:
        xq: (B, T, H, head_dim) float tensor.
        xk: (B, T, Hkv, head_dim) float tensor.
        freqs_cis: (T, head_dim // 2) complex tensor.

    Returns:
        xq_rotated, xk_rotated — same shapes as inputs.
    """
    # View as complex: (..., head_dim // 2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # freqs_cis: (T, head_dim//2) → broadcast over (B, T, H, head_dim//2)
    freqs = freqs_cis[None, :, None, :]  # (1, T, 1, head_dim//2)

    xq_out = torch.view_as_real(xq_ * freqs).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs).flatten(-2)

    return xq_out.to(xq.dtype), xk_out.to(xk.dtype)


class RotaryEmbedding(nn.Module):
    """Caches and applies 1D RoPE for fixed or dynamic sequence lengths."""

    def __init__(self, head_dim: int, max_seq_len: int = 2048, theta: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        freqs_cis = precompute_freqs_cis(head_dim, max_seq_len, theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, xq: torch.Tensor, xk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        T = xq.shape[1]
        return apply_rotary_emb(xq, xk, self.freqs_cis[:T])
