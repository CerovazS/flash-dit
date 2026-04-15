"""Attention variants: MHA and GQA, with FA2/FA3/SDPA backends."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .embeddings import RotaryEmbedding

try:
    from flash_attn import flash_attn_func
    _HAS_FA2 = True
except ImportError:
    _HAS_FA2 = False

try:
    from flash_attn_interface import flash_attn_func as fa3_func  # H100 only
    _HAS_FA3 = True
except ImportError:
    _HAS_FA3 = False


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dropout_p: float) -> torch.Tensor:
    """SDPA fallback: q/k/v in (B, H, T, D) layout → returns (B, T, H*D)."""
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
    return rearrange(out, "b h t d -> b t (h d)")


class MultiHeadAttention(nn.Module):
    """Standard MHA — baseline for vanilla_sit.

    Uses Flash Attention 2 if available, otherwise SDPA.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rope: RotaryEmbedding,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.rope = rope

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → (B, T, C)."""
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = rearrange(q, "b t (h d) -> b t h d", h=self.n_heads)
        k = rearrange(k, "b t (h d) -> b t h d", h=self.n_heads)
        v = rearrange(v, "b t (h d) -> b t h d", h=self.n_heads)

        q, k = self.rope(q, k)

        dp = self.dropout if self.training else 0.0

        if _HAS_FA2:
            out = flash_attn_func(q, k, v, dropout_p=dp)
            out = rearrange(out, "b t h d -> b t (h d)")
        else:
            out = _sdpa(
                rearrange(q, "b t h d -> b h t d"),
                rearrange(k, "b t h d -> b h t d"),
                rearrange(v, "b t h d -> b h t d"),
                dp,
            )

        return self.out_proj(out)


class GroupedQueryAttention(nn.Module):
    """GQA — n_kv_heads key/value heads shared across query groups.

    On A100 (CINECA): uses Flash Attention 2.
    On H100 (RunPod pod): uses Flash Attention 3 when use_fa3=True.
    Falls back to SDPA if neither is available.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        rope: RotaryEmbedding,
        dropout: float = 0.0,
        use_fa3: bool = False,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.use_fa3 = use_fa3 and _HAS_FA3
        self.rope = rope

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → (B, T, C)."""
        q = rearrange(self.q_proj(x), "b t (h d) -> b t h d", h=self.n_heads)
        k = rearrange(self.k_proj(x), "b t (h d) -> b t h d", h=self.n_kv_heads)
        v = rearrange(self.v_proj(x), "b t (h d) -> b t h d", h=self.n_kv_heads)

        q, k = self.rope(q, k)

        dp = self.dropout if self.training else 0.0

        if self.use_fa3 and _HAS_FA3:
            # FA3 natively supports GQA: pass q (B,T,H,D) and k/v (B,T,Hkv,D) directly
            out = fa3_func(q.contiguous(), k.contiguous(), v.contiguous())
            out = rearrange(out, "b t h d -> b t (h d)")
        elif _HAS_FA2:
            # FA2: expand kv to match query head count
            k_exp = k.repeat_interleave(self.n_groups, dim=2)
            v_exp = v.repeat_interleave(self.n_groups, dim=2)
            out = flash_attn_func(q, k_exp, v_exp, dropout_p=dp)
            out = rearrange(out, "b t h d -> b t (h d)")
        else:
            k_exp = rearrange(k, "b t h d -> b h t d").repeat_interleave(self.n_groups, dim=1)
            v_exp = rearrange(v, "b t h d -> b h t d").repeat_interleave(self.n_groups, dim=1)
            out = _sdpa(rearrange(q, "b t h d -> b h t d"), k_exp, v_exp, dp)

        return self.out_proj(out)


def build_attention(
    d_model: int,
    n_heads: int,
    n_kv_heads: int,
    rope: RotaryEmbedding,
    attention_type: str,
    dropout: float = 0.0,
    use_fa3: bool = False,
) -> nn.Module:
    """Factory: returns the correct attention module given attention_type."""
    if attention_type == "mha":
        return MultiHeadAttention(d_model, n_heads, rope, dropout)
    if attention_type == "gqa":
        return GroupedQueryAttention(d_model, n_heads, n_kv_heads, rope, dropout, use_fa3)
    raise ValueError(f"Unknown attention_type: {attention_type!r}. Choose 'mha' or 'gqa'.")
