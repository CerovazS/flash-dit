"""Diffusion Transformer (DiT) core model.

Supports three baseline variants via config:
  - vanilla_sit:       MHA + AdamW, no compile   (reference)
  - flash_dit:         GQA + FA2 + Muon + compile (optimized)
  - flash_dit_recurse: GQA + FA2 + depth recurrence (~40M params)
"""
import torch
import torch.nn as nn

from .attention import build_attention
from .conditioning import AdaLNModulation, GenreEmbedder, modulate
from .embeddings import RotaryEmbedding, TimestepEmbedder
from .mlp import build_mlp


class DiTBlock(nn.Module):
    """Single transformer block with AdaLN-Zero conditioning.

    Forward pass:
        x = x + gate_attn * Attn(AdaLN(x, shift_a, scale_a))
        x = x + gate_mlp  * MLP(AdaLN(x, shift_m, scale_m))
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        mlp_mult: float,
        attention_type: str,
        mlp_type: str,
        rope: RotaryEmbedding,
        dropout: float = 0.0,
        use_fa3: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.attn = build_attention(
            d_model, n_heads, n_kv_heads, rope, attention_type, dropout, use_fa3
        )
        self.mlp = build_mlp(d_model, mlp_mult, mlp_type)
        self.adaLN = AdaLNModulation(d_cond=d_model, d_model=d_model)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model) — input sequence.
            c: (B, d_model)    — combined conditioning (timestep + genre).
        Returns:
            (B, T, d_model)
        """
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN(c)

        x = x + gate_a[:, None, :] * self.attn(modulate(self.norm1(x), shift_a, scale_a))
        x = x + gate_m[:, None, :] * self.mlp(modulate(self.norm2(x), shift_m, scale_m))
        return x


class FinalLayer(nn.Module):
    """Final AdaLN + linear projection back to latent channels."""

    def __init__(self, d_model: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.adaLN = AdaLNModulation(d_cond=d_model, d_model=d_model)
        self.linear = nn.Linear(d_model, out_channels, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale, *_ = self.adaLN(c)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


class DiffusionTransformer(nn.Module):
    """Modular Diffusion Transformer for audio latents.

    Input:  x  (B, in_channels, T_lat) — normalised latent
            t  (B,)                    — diffusion timestep in [0, 1]
            y  (B,)                    — genre label (int64, or null class for CFG)

    Output: (B, in_channels, T_lat)    — predicted velocity field

    Depth recurrence:
        When n_unique_layers < n_layers the same n_unique_layers blocks are
        applied cyclically until n_layers total passes have been made.
        Example: n_layers=12, n_unique_layers=4 → 4 blocks × 3 loops.
    """

    def __init__(
        self,
        in_channels: int = 64,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        n_kv_heads: int = 2,
        mlp_mult: float = 8 / 3,
        n_classes: int = 16,
        cfg_dropout: float = 0.1,
        attention_type: str = "gqa",
        mlp_type: str = "swiglu",
        dropout: float = 0.0,
        use_fa3: bool = False,
        use_compile: bool = False,
        n_unique_layers: int | None = None,
        max_seq_len: int = 256,
        name: str = "",          # used by Hydra configs for run_name interpolation
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.n_layers = n_layers

        # Depth recurrence: n_unique_layers defines the unique blocks to cycle
        n_unique = n_unique_layers if n_unique_layers is not None else n_layers
        assert n_layers % n_unique == 0, (
            f"n_layers ({n_layers}) must be divisible by n_unique_layers ({n_unique})"
        )
        self.n_unique = n_unique
        self.n_loops = n_layers // n_unique

        # Shared RoPE instance (weights are not learned, just cached buffers)
        self.rope = RotaryEmbedding(head_dim=d_model // n_heads, max_seq_len=max_seq_len)

        # Input/output projections
        self.x_embedder = nn.Linear(in_channels, d_model, bias=False)
        self.final_layer = FinalLayer(d_model, in_channels)

        # Conditioning
        self.t_embedder = TimestepEmbedder(d_model)
        self.y_embedder = GenreEmbedder(n_classes, d_model, dropout_prob=cfg_dropout)

        # Transformer blocks
        block_kwargs = dict(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            mlp_mult=mlp_mult,
            attention_type=attention_type,
            mlp_type=mlp_type,
            rope=self.rope,
            dropout=dropout,
            use_fa3=use_fa3,
        )
        self.blocks = nn.ModuleList([DiTBlock(**block_kwargs) for _ in range(n_unique)])

        if use_compile:
            self.blocks = nn.ModuleList(
                [torch.compile(b, mode="reduce-overhead") for b in self.blocks]
            )

        self._init_weights()

    def _init_weights(self) -> None:
        # Input embedder: small init
        nn.init.xavier_uniform_(self.x_embedder.weight)
        # Timestep MLP
        for m in self.t_embedder.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        force_drop_cond: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:               (B, C, T) normalised latent.
            t:               (B,) float timestep in [0, 1].
            y:               (B,) int64 genre label.
            force_drop_cond: replace all genre labels with null (for CFG evaluation).

        Returns:
            (B, C, T) predicted velocity field.
        """
        # (B, C, T) → (B, T, C) → (B, T, d_model)
        x = x.permute(0, 2, 1)
        x = self.x_embedder(x)

        # Conditioning vector: (B, d_model) + (B, d_model) → (B, d_model)
        c = self.t_embedder(t) + self.y_embedder(y, force_drop=force_drop_cond)

        # N transformer passes (with cyclic weight sharing when n_loops > 1)
        for _ in range(self.n_loops):
            for block in self.blocks:
                x = block(x, c)

        # Final projection: (B, T, d_model) → (B, T, C) → (B, C, T)
        x = self.final_layer(x, c)
        return x.permute(0, 2, 1)
