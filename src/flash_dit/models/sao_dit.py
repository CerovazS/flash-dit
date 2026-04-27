"""Stable Audio Open-style Diffusion Transformer (1D, class-conditional).

Architectural identity preserved from SAO 1.0 (`stable_audio_tools` repo):

  - bias-less LayerNorm with learnable γ (β=0 frozen)
  - SwiGLU FFN with mlp_mult=4
  - partial RoPE on self-attention only (rot_dim = max(head_dim/2, 32))
  - residual zero-init on out_proj of self-attention and on down of SwiGLU
    (instead of AdaLN gate zero-init)
  - residual zero-init Conv1d wrappers (preprocess + postprocess)
  - prepend-token conditioning: a single (t_emb + y_emb) summary token is
    prepended to the sequence at input and sliced off at output

Deliberately diverges from SAO 1.0 on the conditioning *content*:

  - **No cross-attention.** Upstream cross-attn carries T5 prompt tokens
    (128 tok). With class-only conditioning the cross-attn context would
    collapse to a single y_emb token, making cross-attention degenerate to
    a position-independent additive bias `MLP(y_emb)`. We drop the cross-attn
    entirely; the prepend token (which carries `t_emb + y_emb` and
    self-attends through the standard softmax with all audio tokens) is
    sufficient to inject conditioning.
  - **No per-block AdaLN.** Matches SAO 1.0's `global_cond_type="prepend"`,
    which sets `global_cond_dim=None` and skips `to_scale_shift_gate`
    allocation (`stable_audio_tools/models/transformer.py:653-654`).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .attention import build_attention
from .conditioning import GenreEmbedder
from .embeddings import RotaryEmbedding, TimestepEmbedder
from .mlp import build_mlp


# ---------------------------------------------------------------------------
# Norm
# ---------------------------------------------------------------------------

def _layer_norm_biasless(d_model: int, eps: float = 1e-6) -> nn.Module:
    """LayerNorm with learnable γ but β=0 (frozen).

    Matches SAO's custom ``LayerNorm`` at ``transformer.py:215-241``. PyTorch
    added ``bias=`` to ``nn.LayerNorm`` in 2.1; we use it directly when
    available and fall back to a manual implementation otherwise.
    """
    try:
        return nn.LayerNorm(d_model, elementwise_affine=True, bias=False, eps=eps)
    except TypeError:  # pragma: no cover — torch < 2.1
        return _LayerNormBiaslessManual(d_model, eps)


class _LayerNormBiaslessManual(nn.Module):
    """Fallback for torch < 2.1 — bias-less LayerNorm with learnable γ."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------

class SAOBlock(nn.Module):
    """SAO non-AdaLN transformer block: SelfAttn + FFN, both pre-norm,
    both with zero-init residual contributions.

    Mirrors the else-branch of upstream
    ``TransformerBlock.forward`` (``transformer.py:704-712``) when
    ``global_cond_dim`` is ``None`` (= prepend conditioning mode).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_mult: float,
        rope: RotaryEmbedding,
        mlp_type: str = "swiglu",
        dropout: float = 0.0,
        use_fa3: bool = False,
    ) -> None:
        super().__init__()
        self.pre_norm = _layer_norm_biasless(d_model)
        self.ff_norm = _layer_norm_biasless(d_model)
        # MHA with full-RoPE-on-rot_dim (handled by the shared RotaryEmbedding
        # which carries its own rot_dim). zero_init_out=True implements the
        # SAO residual zero-init at the projection rather than via a gate.
        self.self_attn = build_attention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_heads,           # SAO uses MHA, not GQA
            rope=rope,
            attention_type="mha",
            dropout=dropout,
            use_fa3=use_fa3,
            qkv_bias=False,
            zero_init_out=True,
            rope_active=True,
        )
        self.ff = build_mlp(d_model, mlp_mult, mlp_type)
        # Zero-init on the FFN's "down" projection so the residual contribution
        # starts at exactly 0 (mirrors transformer.py:311-314 zero_init).
        self._zero_init_ff_down()

    def _zero_init_ff_down(self) -> None:
        """Zero-init the FFN's last linear so the post-residual contribution
        starts at 0 — irrespective of whether ``ff`` is SwiGLU (`down`) or
        GELUMlp (`fc2`)."""
        for attr in ("down", "fc2"):
            layer = getattr(self.ff, attr, None)
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                return
        raise RuntimeError(
            f"SAOBlock could not find a final Linear to zero-init in {type(self.ff).__name__}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1+T, D) → (B, 1+T, D). The prepended summary token is the
        position 0; self-attention mixes it with all audio tokens."""
        x = x + self.self_attn(self.pre_norm(x))
        x = x + self.ff(self.ff_norm(x))
        return x


# ---------------------------------------------------------------------------
# Final layer
# ---------------------------------------------------------------------------

class SAOFinalLayer(nn.Module):
    """Bias-less LayerNorm + zero-init Linear projection.

    No AdaLN modulation — conditioning is fully consumed by the prepended
    token via self-attention.
    """

    def __init__(self, d_model: int, out_channels: int) -> None:
        super().__init__()
        self.norm = _layer_norm_biasless(d_model)
        self.linear = nn.Linear(d_model, out_channels, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

class SAODiTTransformer(nn.Module):
    """Stable Audio Open-style 1D Diffusion Transformer (class-conditional).

    Input/output contract is *identical* to ``DiffusionTransformer``:
        forward(x, t, y, force_drop_cond=False) → (B, C, T)

    Conditioning is built from the same primitives as vanilla_sit
    (``TimestepEmbedder`` + ``GenreEmbedder``), but routed differently:
    instead of AdaLN-Zero per block, ``(t_emb + y_emb)`` is prepended as a
    single summary token at position 0 of the sequence and removed before
    the final projection. Zero-init residual Conv1d wraps the entire
    transformer at input and output (SAO 1.0 ``preprocess_conv`` /
    ``postprocess_conv``).
    """

    def __init__(
        self,
        in_channels: int = 64,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        mlp_mult: float = 4.0,
        n_classes: int = 16,
        cfg_dropout: float = 0.1,
        mlp_type: str = "swiglu",
        dropout: float = 0.0,
        use_fa3: bool = False,
        max_seq_len: int = 256,
        partial_rope: bool = True,
        rope_theta: float = 10000.0,
        name: str = "",          # used by Hydra configs for run_name interpolation
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.in_channels = in_channels
        self.d_model = d_model
        self.n_layers = n_layers

        head_dim = d_model // n_heads
        # SAO formula: rot_dim = max(head_dim // 2, 32) when partial_rope
        rot_dim = max(head_dim // 2, 32) if partial_rope else head_dim
        # +1 sequence slot for the prepended global token
        self.rope = RotaryEmbedding(
            head_dim=head_dim,
            max_seq_len=max_seq_len + 1,
            theta=rope_theta,
            rot_dim=rot_dim,
        )

        # Zero-init residual Conv1d wrappers (SAO dit.py:120-123, :193, :224).
        # bias=False to match upstream and to avoid a trainable bias drifting
        # away from the zero-residual identity at init.
        self.preprocess_conv = nn.Conv1d(in_channels, in_channels, 1, bias=False)
        self.postprocess_conv = nn.Conv1d(in_channels, in_channels, 1, bias=False)
        nn.init.zeros_(self.preprocess_conv.weight)
        nn.init.zeros_(self.postprocess_conv.weight)

        # Input projection (no patch — patch_size=1)
        self.x_embedder = nn.Linear(in_channels, d_model, bias=False)
        self.final_layer = SAOFinalLayer(d_model, in_channels)

        # Conditioning — IDENTICAL primitives to vanilla_sit / dit_meta
        self.t_embedder = TimestepEmbedder(d_model)
        self.y_embedder = GenreEmbedder(n_classes, d_model, dropout_prob=cfg_dropout)

        self.blocks = nn.ModuleList([
            SAOBlock(
                d_model=d_model,
                n_heads=n_heads,
                mlp_mult=mlp_mult,
                rope=self.rope,
                mlp_type=mlp_type,
                dropout=dropout,
                use_fa3=use_fa3,
            )
            for _ in range(n_layers)
        ])

        self._init_weights()

    def _init_weights(self) -> None:
        """Init policy mirroring vanilla_sit; AdaLN-style zero-inits are not
        applicable here (no AdaLN). Zero-inits are applied at construction
        time on each module that needs them."""
        nn.init.xavier_uniform_(self.x_embedder.weight)
        for m in self.t_embedder.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.y_embedder.embedding.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        force_drop_cond: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) normalised latent.
            t: (B,) float timestep in [0, 1].
            y: (B,) int64 genre label.
            force_drop_cond: replace all genre labels with the null token
                (used by samplers for batched CFG).

        Returns:
            (B, C, T) predicted velocity field.
        """
        # Zero-init residual preprocess conv (operates on (B, C, T) directly)
        x = x + self.preprocess_conv(x)

        # (B, C, T) → (B, T, C) → (B, T, D)
        x = x.permute(0, 2, 1)
        x = self.x_embedder(x)

        t_emb = self.t_embedder(t)                                # (B, D)
        y_emb = self.y_embedder(y, force_drop=force_drop_cond)    # (B, D)

        # Prepend a single (t_emb + y_emb) summary token at position 0
        g = (t_emb + y_emb).unsqueeze(1)                          # (B, 1, D)
        x = torch.cat([g, x], dim=1)                              # (B, 1+T, D)

        for block in self.blocks:
            x = block(x)

        # Drop the prepended token before final projection
        x = x[:, 1:, :]                                           # (B, T, D)
        x = self.final_layer(x)                                   # (B, T, C)

        # (B, T, C) → (B, C, T) and zero-init residual postprocess conv
        x = x.permute(0, 2, 1)
        x = x + self.postprocess_conv(x)
        return x
