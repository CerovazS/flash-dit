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


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    rot_dim: int | None = None,
):
    """Apply RoPE to query and key tensors.

    Args:
        xq: (B, T, H, head_dim) float tensor.
        xk: (B, T, Hkv, head_dim) float tensor.
        freqs_cis: (T, rot_dim // 2) complex tensor. When ``rot_dim`` is set
            and is < head_dim, only the first ``rot_dim`` head dimensions are
            rotated and the unrotated tail is concatenated back (SAO-style
            partial RoPE).

    Returns:
        xq_rotated, xk_rotated — same shapes as inputs.
    """
    if rot_dim is None or rot_dim == xq.shape[-1]:
        xq_to_rot, xq_pass = xq, None
        xk_to_rot, xk_pass = xk, None
    else:
        xq_to_rot, xq_pass = xq[..., :rot_dim], xq[..., rot_dim:]
        xk_to_rot, xk_pass = xk[..., :rot_dim], xk[..., rot_dim:]

    xq_ = torch.view_as_complex(xq_to_rot.float().reshape(*xq_to_rot.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk_to_rot.float().reshape(*xk_to_rot.shape[:-1], -1, 2))

    freqs = freqs_cis[None, :, None, :]  # (1, T, 1, rot_dim//2)

    xq_out = torch.view_as_real(xq_ * freqs).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs).flatten(-2)

    if xq_pass is not None:
        xq_out = torch.cat([xq_out.to(xq.dtype), xq_pass], dim=-1)
        xk_out = torch.cat([xk_out.to(xk.dtype), xk_pass], dim=-1)
        return xq_out, xk_out

    return xq_out.to(xq.dtype), xk_out.to(xk.dtype)


class RotaryEmbedding(nn.Module):
    """Caches and applies 1D RoPE for fixed or dynamic sequence lengths.

    Args:
        head_dim: full per-head dimension.
        max_seq_len: max sequence length to precompute frequencies for.
        theta: base period (default 10000).
        rot_dim: number of head dimensions to rotate. ``None`` (default) =
            full RoPE on the whole head; otherwise rotate only the first
            ``rot_dim`` dims and pass the unrotated tail through. SAO uses
            ``rot_dim = max(head_dim // 2, 32)``.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        rot_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self.rot_dim = rot_dim if rot_dim is not None else head_dim
        assert self.rot_dim <= head_dim and self.rot_dim % 2 == 0
        freqs_cis = precompute_freqs_cis(self.rot_dim, max_seq_len, theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, xq: torch.Tensor, xk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        T = xq.shape[1]
        return apply_rotary_emb(xq, xk, self.freqs_cis[:T], rot_dim=self.rot_dim)


class SinCosPosEmbedding1D(nn.Module):
    """Fixed 1D sinusoidal positional embedding (frozen, never trained).

    1D port of the 2D sin-cos pos embed used in DiT (``models.py:174,191-193``):
    upstream registers it as a Parameter with ``requires_grad=False``. Here we
    use a non-persistent buffer, frozen by construction.
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 2048,
        max_period: int = 10000,
    ) -> None:
        super().__init__()
        assert d_model % 2 == 0
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(max_period) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D) with positional embedding added."""
        return x + self.pe[:, : x.size(1)].to(dtype=x.dtype)
