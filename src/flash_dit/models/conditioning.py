"""Conditioning modules: genre embedding, AdaLN-Zero modulation."""
import torch
import torch.nn as nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply AdaLN shift and scale: x * (1 + scale) + shift.

    Args:
        x:     (B, T, D) — layernorm output.
        shift: (B, D)   — additive shift.
        scale: (B, D)   — multiplicative scale (applied as 1 + scale).

    Returns:
        (B, T, D)
    """
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class GenreEmbedder(nn.Module):
    """Learned genre class embeddings with a dedicated null token for CFG.

    Token IDs:
        0 … n_classes-1  — actual genre classes
        n_classes         — null token (used when conditioning is dropped for CFG)
    """

    def __init__(self, n_classes: int, d_model: int, dropout_prob: float = 0.1) -> None:
        super().__init__()
        self.null_class = n_classes
        self.dropout_prob = dropout_prob
        self.embedding = nn.Embedding(n_classes + 1, d_model)

    def forward(self, labels: torch.Tensor, force_drop: bool = False) -> torch.Tensor:
        """Args:
            labels:     (B,) int64 genre labels.
            force_drop: if True, replace all labels with the null token.

        Returns:
            (B, d_model) genre embedding.
        """
        if force_drop:
            labels = torch.full_like(labels, self.null_class)
        elif self.training and self.dropout_prob > 0.0:
            mask = torch.rand(labels.shape, device=labels.device) < self.dropout_prob
            labels = labels.clone()
            labels[mask] = self.null_class
        return self.embedding(labels)


class AdaLNModulation(nn.Module):
    """Projects the combined conditioning vector into AdaLN-Zero scale/shift/gate.

    Outputs 6 × d_model values per block:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp

    The final linear layer is zero-initialized so that all blocks start as
    identity functions at the beginning of training (AdaLN-Zero trick from DiT).
    """

    def __init__(self, d_cond: int, d_model: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(d_cond, 6 * d_model, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, c: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """c: (B, d_cond) → 6 tensors of shape (B, d_model)."""
        return self.linear(self.silu(c)).chunk(6, dim=-1)
