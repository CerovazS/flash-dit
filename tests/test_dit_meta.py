"""Tests for the dit_meta variant of DiffusionTransformer.

Verifies the architectural switches (LayerNorm-no-affine, GELU MLP, sin-cos
abs positional embedding, qkv_bias=True) produce a model that:
  - has the correct shapes end-to-end,
  - exposes the attributes used by lit_module / samplers,
  - behaves differently when classifier-free guidance is forced on,
  - has zero-init AdaLN and final layer (output ≈ 0 at init).
"""
from __future__ import annotations

import torch

from flash_dit.models.dit import DiffusionTransformer
from flash_dit.models.embeddings import SinCosPosEmbedding1D
from flash_dit.models.mlp import GELUMlp


def _build_dit_meta(**overrides) -> DiffusionTransformer:
    cfg = dict(
        in_channels=64,
        d_model=192,
        n_layers=2,
        n_heads=6,
        n_kv_heads=6,
        mlp_mult=4.0,
        n_classes=17,
        cfg_dropout=0.1,
        attention_type="mha",
        mlp_type="gelu",
        norm_type="layernorm_no_affine",
        pos_embed_type="sincos_abs",
        qkv_bias=True,
        use_fa3=False,
        use_compile=False,
        dropout=0.0,
        max_seq_len=256,
    )
    cfg.update(overrides)
    return DiffusionTransformer(**cfg)


def test_dit_meta_forward_shape():
    model = _build_dit_meta().eval()
    x = torch.randn(2, 64, 172)
    t = torch.rand(2)
    y = torch.randint(0, 16, (2,))
    out = model(x, t, y)
    assert out.shape == x.shape


def test_dit_meta_attributes_for_sampler():
    """lit_module and samplers rely on these attributes."""
    model = _build_dit_meta()
    assert model.in_channels == 64
    # GenreEmbedder.null_class equals the n_classes argument; the embedding
    # table has n_classes + 1 rows with the last being the null token.
    assert model.y_embedder.null_class == 17
    # Both branches must run end-to-end. At init the model outputs 0 by
    # design (zero-init AdaLN + zero-init final.linear), so we instead
    # verify that the genre embedding *itself* differs when force_drop is on
    # — which is what the sampler relies on for batched CFG.
    y = torch.randint(0, 16, (1,))
    e_cond = model.y_embedder(y, force_drop=False)
    e_null = model.y_embedder(y, force_drop=True)
    assert not torch.allclose(e_cond, e_null)
    # And the model forward must accept the kwarg.
    x = torch.randn(1, 64, 32)
    t = torch.rand(1)
    out_a = model(x, t, y, force_drop_cond=False)
    out_b = model(x, t, y, force_drop_cond=True)
    assert out_a.shape == out_b.shape == (1, 64, 32)


def test_dit_meta_uses_correct_primitives():
    """Sanity check the wiring picked the DiT-Meta primitives, not the SiT ones."""
    model = _build_dit_meta()
    assert isinstance(model.abs_pos, SinCosPosEmbedding1D)
    block = model.blocks[0]
    # MLP must be GELUMlp, not SwiGLU
    assert isinstance(block.mlp, GELUMlp)
    # LayerNorm with elementwise_affine=False has no learnable γ/β
    assert block.norm1.weight is None and block.norm1.bias is None
    # qkv_bias=True
    assert block.attn.qkv.bias is not None
    # rope is instantiated but not active in this path
    assert model.rope is not None
    assert block.attn.rope_active is False


def test_dit_meta_output_is_zero_at_init():
    """Zero-init AdaLN gates and zero-init final.linear should make the
    initial output exactly 0 — this is what makes the AdaLN-Zero recipe
    trainable from random weights."""
    model = _build_dit_meta().eval()
    x = torch.randn(2, 64, 172)
    t = torch.rand(2)
    y = torch.randint(0, 16, (2,))
    with torch.no_grad():
        out = model(x, t, y)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_dit_meta_grads_flow():
    """Zero-init AdaLN makes the initial output exactly 0, so we perturb the
    AdaLN finals before backward to verify gradients reach the trunk."""
    model = _build_dit_meta()
    # Break zero-init on a single block's AdaLN final and on the output linear
    # so the loss is non-zero and backprop can reach the trunk.
    with torch.no_grad():
        model.blocks[0].adaLN.linear.weight.normal_(std=0.02)
        model.final_layer.linear.weight.normal_(std=0.02)
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
