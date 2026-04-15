# flash-dit

Optimized audio Diffusion Transformer with modern training speedups:
**Muon optimizer** · **GQA** · **FlashAttention 2/3** · **torch.compile** · **Depth Recurrence**

Built on [stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) latents
and trained on [FMA Medium](https://github.com/mdeff/fma) (19,922 tracks, 16 genres).

## Baselines

| Model | Attention | Optimizer | Compile | Params |
|-------|-----------|-----------|---------|--------|
| `vanilla_sit` | MHA | AdamW | — | 115M |
| `flash_dit` | GQA (12Q/2KV) + FA2 | Muon+AdamW | ✓ | 115M |
| `flash_dit_recurse` | GQA + FA2 | Muon+AdamW | ✓ | ~40M |

## Quick Start

```bash
# 0. Setup
cd flash-dit && uv sync

# 1. Pre-compute latents (GPU node, ~3-4h)
sbatch slurm/precompute_latents.sbatch

# 2. Train
RUN_NAME=flash_dit_001 MODEL=flash_dit sbatch slurm/train_single.sbatch

# 3. Sample
uv run python scripts/sample.py \
    --checkpoint outputs/.../checkpoints/last.ckpt \
    --n-samples 8 --cfg-scale 4.0

# 4. Evaluate FAD
uv run python scripts/eval_fad.py \
    --generated outputs/samples_*/ \
    --reference /path/to/real/wavs/

# 5. Tests
uv run pytest tests/ -v
```

## Architecture

- **Diffusion**: Rectified flow matching, logit-normal timestep sampling
- **Conditioning**: Genre CFG (16 classes + null token, p_drop=0.1)
- **VAE**: stable-audio-open-1.0 Oobleck, 44.1kHz stereo, 2048× compression, 64D latents
- **Chunk size**: 131072 samples (≈3s) → 64 latent frames
- **Precision**: bf16 mixed, EMA β=0.9999
