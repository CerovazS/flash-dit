"""Unit tests for attention modules."""
import pytest
import torch
import torch.nn.functional as F

import flash_dit.models.attention as attention
from flash_dit.models.attention import GroupedQueryAttention, MultiHeadAttention, _sdpa
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


def test_sdpa_native_gqa_matches_manual_kv_expansion():
    """Native SDPA GQA should match the old explicit KV expansion path."""
    q = torch.randn(2, 6, 11, 32)
    k = torch.randn(2, 2, 11, 32)
    v = torch.randn(2, 2, 11, 32)

    out = _sdpa(q, k, v, dropout_p=0.0, enable_gqa=True)
    ref = _sdpa(
        q,
        k.repeat_interleave(3, dim=1),
        v.repeat_interleave(3, dim=1),
        dropout_p=0.0,
    )

    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)


def test_gqa_sdpa_path_keeps_compressed_kv_heads(monkeypatch, rope):
    """Fallback SDPA should receive K/V with n_kv_heads and enable_gqa=True."""
    calls = []

    def fake_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
        calls.append((q.shape, k.shape, v.shape, enable_gqa))
        return torch.zeros_like(q)

    monkeypatch.setattr(attention, "_HAS_FA2", False)
    monkeypatch.setattr(attention, "_HAS_FA3", False)
    monkeypatch.setattr(F, "scaled_dot_product_attention", fake_sdpa)

    B, T, D = 2, 16, 768
    gqa = GroupedQueryAttention(D, n_heads=12, n_kv_heads=2, rope=rope)
    x = torch.randn(B, T, D)
    out = gqa(x)

    assert out.shape == (B, T, D)
    assert calls == [
        (
            torch.Size([B, 12, T, 64]),
            torch.Size([B, 2, T, 64]),
            torch.Size([B, 2, T, 64]),
            True,
        )
    ]


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
