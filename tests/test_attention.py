"""Unit tests for attention modules."""
import pytest
import torch

from flash_dit.models.attention import GroupedQueryAttention, MultiHeadAttention
from flash_dit.models.embeddings import RotaryEmbedding


@pytest.fixture
def rope():
    return RotaryEmbedding(head_dim=64, max_seq_len=128)


def test_mha_output_shape(rope):
    B, T, D = 2, 32, 768
    mha = MultiHeadAttention(D, n_heads=12, rope=rope)
    x = torch.randn(B, T, D)
    out = mha(x)
    assert out.shape == (B, T, D)


def test_gqa_output_shape(rope):
    B, T, D = 2, 32, 768
    gqa = GroupedQueryAttention(D, n_heads=12, n_kv_heads=2, rope=rope)
    x = torch.randn(B, T, D)
    out = gqa(x)
    assert out.shape == (B, T, D)


def test_mha_gqa_similar_output(rope):
    """MHA and GQA with same random init should produce outputs in similar range."""
    B, T, D = 1, 16, 192
    rope_small = RotaryEmbedding(head_dim=192 // 6, max_seq_len=128)

    mha = MultiHeadAttention(D, n_heads=6, rope=rope_small)
    gqa = GroupedQueryAttention(D, n_heads=6, n_kv_heads=2, rope=rope_small)

    x = torch.randn(B, T, D)
    with torch.no_grad():
        out_mha = mha(x)
        out_gqa = gqa(x)

    # Just check they're finite and in a reasonable range (not numerically exploding)
    assert torch.isfinite(out_mha).all()
    assert torch.isfinite(out_gqa).all()
    assert out_mha.abs().mean() < 10.0
    assert out_gqa.abs().mean() < 10.0
