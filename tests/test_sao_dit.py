"""Tests for the sao_dit variant.

Verifies:
  - end-to-end forward shape (with the prepend-then-slice mechanism)
  - attributes used by lit_module / samplers
  - the architectural primitives are wired correctly (bias-less LN, partial
    RoPE, SwiGLU mult=4, zero-init out_proj / FFN-down / Conv1d wrappers)
  - the model output is exactly 0 at init (zero-init residuals + zero-init
    final.linear)
  - gradients reach the trunk after the residuals are perturbed
"""
from __future__ import annotations

import torch

from flash_dit.models.mlp import SwiGLU
from flash_dit.models.sao_dit import SAODiTTransformer


def _build_sao_dit(**overrides) -> SAODiTTransformer:
    cfg = dict(
        in_channels=64,
        d_model=192,
        n_layers=2,
        n_heads=6,
        mlp_mult=4.0,
        n_classes=17,
        cfg_dropout=0.1,
        mlp_type="swiglu",
        use_fa3=False,
        max_seq_len=256,
        partial_rope=True,
    )
    cfg.update(overrides)
    return SAODiTTransformer(**cfg)


def test_sao_dit_forward_shape():
    model = _build_sao_dit().eval()
    x = torch.randn(2, 64, 172)
    t = torch.rand(2)
    y = torch.randint(0, 16, (2,))
    out = model(x, t, y)
    assert out.shape == x.shape


def test_sao_dit_attributes_for_sampler():
    model = _build_sao_dit()
    assert model.in_channels == 64
    assert model.y_embedder.null_class == 17
    # force_drop kwarg path runs end-to-end
    x = torch.randn(1, 64, 32)
    t = torch.rand(1)
    y = torch.randint(0, 16, (1,))
    out_a = model(x, t, y, force_drop_cond=False)
    out_b = model(x, t, y, force_drop_cond=True)
    assert out_a.shape == out_b.shape == (1, 64, 32)
    # The y_emb path differs; check directly since the model itself outputs 0
    # at init.
    e_cond = model.y_embedder(y, force_drop=False)
    e_null = model.y_embedder(y, force_drop=True)
    assert not torch.allclose(e_cond, e_null)


def test_sao_dit_uses_correct_primitives():
    model = _build_sao_dit()
    block = model.blocks[0]
    # FFN must be SwiGLU
    assert isinstance(block.ff, SwiGLU)
    # Bias-less LayerNorm: γ learnable (weight is a Parameter), β=0 frozen
    # (bias is None on torch>=2.1; for the manual fallback there is no bias
    # attribute at all)
    has_bias = getattr(block.pre_norm, "bias", None)
    assert has_bias is None
    assert block.pre_norm.weight is not None
    # Partial RoPE: rot_dim should be max(head_dim/2, 32) = max(16, 32) = 32
    head_dim = 192 // 6
    expected_rot_dim = max(head_dim // 2, 32)
    assert model.rope.rot_dim == expected_rot_dim
    # Self-attn out_proj must be zero-init
    assert torch.allclose(
        block.self_attn.out_proj.weight, torch.zeros_like(block.self_attn.out_proj.weight)
    )
    # FFN.down must be zero-init
    assert torch.allclose(block.ff.down.weight, torch.zeros_like(block.ff.down.weight))
    # qkv must NOT have bias
    assert block.self_attn.qkv.bias is None
    # Conv1d wrappers must be zero-init (residual identity at start) and
    # bias-less (matching upstream — no trainable bias to drift).
    assert torch.allclose(
        model.preprocess_conv.weight, torch.zeros_like(model.preprocess_conv.weight)
    )
    assert torch.allclose(
        model.postprocess_conv.weight, torch.zeros_like(model.postprocess_conv.weight)
    )
    assert model.preprocess_conv.bias is None
    assert model.postprocess_conv.bias is None


def test_sao_dit_output_is_zero_at_init():
    """Zero-init final.linear + zero-init residual contributions in every
    block + zero-init postprocess conv → output is exactly 0 at init.
    Because of the cascade of zero-inits, even non-zero hidden activations
    cannot leak to the output."""
    model = _build_sao_dit().eval()
    x = torch.randn(2, 64, 172)
    t = torch.rand(2)
    y = torch.randint(0, 16, (2,))
    with torch.no_grad():
        out = model(x, t, y)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_sao_dit_grads_flow():
    """Perturb the zero-inits to verify gradients reach the trunk."""
    model = _build_sao_dit()
    with torch.no_grad():
        model.final_layer.linear.weight.normal_(std=0.02)
        for b in model.blocks:
            b.self_attn.out_proj.weight.normal_(std=0.02)
            b.ff.down.weight.normal_(std=0.02)
    x = torch.randn(2, 64, 32)
    t = torch.rand(2)
    y = torch.randint(0, 16, (2,))
    loss = model(x, t, y).pow(2).mean()
    loss.backward()
    n_with_grad = sum(
        1 for p in model.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    assert n_with_grad > 10, f"expected ≥10 params to receive grads, got {n_with_grad}"


def test_sao_dit_prepend_then_slice_consistency():
    """If we change the audio sequence length, the output length must follow
    (prepend/slice must not leak the global token into the output)."""
    model = _build_sao_dit().eval()
    for T in (16, 64, 172):
        x = torch.randn(1, 64, T)
        t = torch.rand(1)
        y = torch.randint(0, 16, (1,))
        out = model(x, t, y)
        assert out.shape == (1, 64, T), f"failed for T={T}"
