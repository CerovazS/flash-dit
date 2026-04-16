# Flash-DiT: Optimized Audio Diffusion Transformer

## Context

The audio generation ecosystem (stable-audio-tools, DiTTo, EzAudio) uses DiT/SiT architectures
but has not fully integrated the optimizer, attention, and compile-level speedups that are now
standard in the NLP/vision DiT world. The opportunity is to build a clean, open-source audio DiT
that closes this gap — shipping a repo where every component is justified by a measured gain.

**Key insight**: stable-audio-tools already has GQA, FA2, torch.compile, SwiGLU, and RoPE.
Genuine gaps are: **Muon optimizer**, **FSDP2**, **FA3 flag for H100**, **Differential Attention**,
**depth recurrence / weight tying**, and a **clean latent-precompute + modular DiT framework**
that lets anyone swap components and benchmark independently.

**Existing assets on Leonardo:**
- **VAE**: `stabilityai/stable-audio-open-1.0` — download via HuggingFace Hub with `$HF_TOKEN`
  - Oobleck encoder, 44.1kHz stereo, 2048× compression, latent_dim=64
  - 440320 samples (≈10s) → 215 latent frames per chunk
- FMA Medium: `/leonardo_scratch/large/userexternal/lcerovaz/fma/fma_medium/` — 19,922 train tracks, 30s MP3s
- FMA metadata JSON: train/val/test split pre-labeled, 16 genre classes
- CINECA: A100 64GB, account `IscrC_LENS`, partition `boost_usr_prod`
- RunPod H100 pod: `root@103.207.149.120:14754` — FA3 available here

---

## Resolved Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| VAE | Official `stabilityai/stable-audio-open-1.0` (HuggingFace) | Fully converged, trained on large diverse audio corpus, open-source |
| Latent dim | 64 (Oobleck default), 2048× compression @ 44.1kHz | ~21.5 latent frames/s; 10s chunk → 215 tokens |
| Conditioning | Genre + unconditional via CFG dropout (p_drop=0.1) | FMA Medium 16 genres annotated; CFG scale at inference |
| Model scale | Small — 115M (d_model=768, 12L, 12H GQA) | ~8-12h/run on 1 A100; good for ablations |
| Baselines | 3 progressive: vanilla_sit → flash_dit → flash_dit_recurse | Clean ablation table for paper |
| Audio chunk | 440320 samples @ 44100Hz ≈ 10s → 215 latent frames | 3 chunks per 30s FMA track; enough musical context |

---

## Repo Structure

```
flash-dit/
├── conf/                              # Hydra configs
│   ├── config.yaml                    # Main sweep config
│   ├── model/
│   │   ├── vanilla_sit.yaml           # Baseline: MHA + AdamW (no speedups)
│   │   ├── flash_dit.yaml             # Optimized: GQA + FA2 + Muon + compile
│   │   ├── flash_dit_diff.yaml        # + Differential Attention
│   │   └── flash_dit_recurse.yaml     # + Depth recurrence (weight tying)
│   ├── diffusion/
│   │   └── flow_matching.yaml         # Rectified flow (SiT objective)
│   ├── optimizer/
│   │   ├── muon.yaml                  # Muon (2D params) + AdamW (scalars)
│   │   └── adamw.yaml                 # AdamW baseline
│   ├── data/
│   │   ├── fma_medium.yaml
│   │   └── fma_small.yaml
│   └── trainer/
│       ├── single_gpu.yaml
│       └── multi_gpu_ddp.yaml
├── src/flash_dit/
│   ├── models/
│   │   ├── dit.py                     # DiffusionTransformer (main module)
│   │   ├── attention.py               # GQA, FA2/FA3, Differential Attention
│   │   ├── mlp.py                     # SwiGLU, ReLU², activations
│   │   ├── embeddings.py              # RoPE 1D, patch embed, timestep embed
│   │   └── conditioning.py            # AdaLN-Zero, cross-attn, CFG
│   ├── diffusion/
│   │   ├── flow_matching.py           # Rectified flow training objective
│   │   ├── schedules.py               # Timestep sampling (logit-normal)
│   │   └── samplers.py                # Euler, Heun, DPM-Solver++
│   ├── data/
│   │   ├── latent_dataset.py          # HDF5-backed LatentDataset
│   │   └── datamodule.py              # LightningDataModule
│   ├── training/
│   │   ├── lit_module.py              # LightningModule: train/val/configure_optimizers
│   │   └── ema.py                     # EMA wrapper (beta=0.9999)
│   ├── evaluation/
│   │   ├── fad.py                     # FAD with CLAP-LAION-audio backend
│   │   └── generation.py              # Decode latents → WAV during validation
│   └── utils/
│       └── console.py                 # Rich console (ok/info/warn/error)
├── scripts/
│   ├── precompute_latents.py          # Encode FMA → HDF5 latent file
│   ├── train.py                       # Main entry: uv run python scripts/train.py
│   ├── sample.py                      # Inference / generation
│   └── eval_fad.py                    # Standalone FAD evaluation
├── slurm/
│   ├── precompute_latents.sbatch
│   ├── train_single.sbatch
│   └── train_multi.sbatch
├── tests/
│   ├── test_attention.py
│   ├── test_flow_matching.py
│   └── test_dataset.py
├── pyproject.toml
└── README.md
```

---

## Phase 1 — Dependencies & Project Bootstrap

**Add via `uv add`:**
```
hydra-core>=1.3
lightning>=2.4
einops
rich
h5py
muon-optim           # or inline Newton-Schulz (5 iterations)
frechet-audio-distance
laion-clap
tqdm
```

**Stable-audio-tools**: import directly from the sibling repo (add as editable install):
```
uv add --editable ../stable-audio-tools
```
This gives access to `AudioAutoencoder` and model loading utilities without copying code.

**HuggingFace setup** (needed for VAE download):
```bash
export HF_TOKEN="$(grep HF_TOKEN ~/.bashrc | grep -oP '(?<=")hf_[^"]+(?=")')"
huggingface-cli login --token $HF_TOKEN
```
VAE checkpoint cached to `$FAST/flash-dit-cache/stable-audio-open/` to avoid re-downloading.

---

## Phase 2 — Latent Pre-computation

**Script**: `scripts/precompute_latents.py`  
**SLURM**: 1× A100 64GB, ~3-4h for FMA Medium (~400K chunks)

### Algorithm
1. Load official VAE from HuggingFace:
   ```python
   from stable_audio_tools.models.factory import create_model_from_config
   from huggingface_hub import hf_hub_download
   config_path = hf_hub_download("stabilityai/stable-audio-open-1.0", "model_config.json", token=HF_TOKEN)
   ckpt_path   = hf_hub_download("stabilityai/stable-audio-open-1.0", "model.safetensors", token=HF_TOKEN)
   model = create_model_from_config(json.load(open(config_path)))
   # load only the pretransform (VAE) weights; skip the DiT weights
   model.pretransform.eval().to("cuda", dtype=torch.bfloat16)
   ```
2. For each track in FMA Medium (train/val/test split from metadata JSON):
   - Load audio at 44100 Hz stereo with `torchaudio`
   - Chunk into non-overlapping **440320-sample** (≈10s) windows, drop remainder
   - Batch encode: `latent = model.pretransform.encode(audio)` → `(B, 64, 64)`
   - Store to HDF5:
     - `latents`: float16, shape `(N, 64, 215)` — 64 channels × 215 temporal frames
     - `genres`: int64 genre label (0–15) per chunk
     - `track_ids`: string track IDs
     - `split`: byte-string `train`/`val`/`test` per chunk
3. Compute channel-wise mean/std over training split; store as HDF5 attrs

**Output**: `/leonardo_scratch/large/userexternal/lcerovaz/flash-dit-latents/stable_audio_open/fma_medium.h5`

**Estimated size**: ~17,000 tracks × 3 chunks/track × 64×215×2 bytes ≈ **~14 GB**

**Note on stable-audio-open VAE**: the official checkpoint is a `DiffusionTransformer` with a
`pretransform` (the Oobleck VAE). We use only `model.pretransform` for encoding/decoding latents.

### Latent normalization
Before DiT training, normalize latents to zero mean / unit variance per channel.
Stats stored inside HDF5 as metadata attrs. Applied in `LatentDataset.__getitem__`.

---

## Phase 3 — Data Pipeline

### `LatentDataset` (`src/flash_dit/data/latent_dataset.py`)
- Opens HDF5 in read-only mode; supports `split` filter
- `__getitem__` returns `(latent, label)` as float32 tensors
- Applies stored channel-wise normalization

### `LatentDataModule` (`src/flash_dit/data/datamodule.py`)
- Standard Lightning DataModule
- Config: `batch_size`, `num_workers`, `pin_memory`
- `setup(stage)` opens HDF5 (one per process for DDP safety)

---

## Phase 4 — Model Architecture

### Core: `DiffusionTransformer` (`src/flash_dit/models/dit.py`)

```
Input: x (B, C, T_lat), σ (B,), cond (B, d_cond) [optional]
  ↓ flatten C × T_lat → sequence (B, T_lat, C)
  ↓ linear patch embed: C → d_model
  ↓ + RoPE 1D positional encoding
  ↓ N × TransformerBlock(d_model, n_heads, n_kv_heads, mlp_mult, cond_dim)
  ↓ final layernorm
  ↓ linear project: d_model → C
  ↓ reshape → (B, C, T_lat)
Output: predicted velocity v
```

### `TransformerBlock`
- Pre-norm (RMSNorm for efficiency)
- Self-attention: GQA (8 Q heads → 1-2 KV heads by default)
- AdaLN-Zero: scale/shift/gate modulated by `MLP(σ_embed + cond_embed)`
- MLP: SwiGLU with 8/3 × d_model expansion (matches Llama standard)
- Flash Attention 2 by default; FA3 if `use_fa3=True` (H100 only)
- `torch.compile` applied at TransformerBlock level (regional compile)

### Attention variants (config-selectable):
| Name | Config key | Notes |
|------|-----------|-------|
| MHA | `attention: mha` | Baseline, no KV compression |
| GQA | `attention: gqa` (default) | `n_kv_heads: 2` (16:1 default ratio) |
| Differential | `attention: diff` | Two softmax maps, subtract |

### MLP variants (config-selectable):
| Name | Config key |
|------|-----------|
| SwiGLU | `mlp: swiglu` (default) |
| ReLU² | `mlp: relu2` |

### Conditioning: Genre CFG (`src/flash_dit/models/conditioning.py`)

```python
# 17 classes total: 16 genres + class 16 = null token for CFG
genre_embed = nn.Embedding(17, d_model)   # learned class embeddings
```

- During **training**: drop genre label → null class with `p_drop=0.1`
- **AdaLN modulation input**: `concat(timestep_embed, genre_embed)` → 2-layer MLP → `(scale, shift, gate)` per block
- During **inference**: CFG = `model(x, t, null) + cfg_scale * (model(x, t, genre) - model(x, t, null))`
- `cfg_scale` default: 4.0 (sweep 2.0–7.0 for quality vs diversity)

### Depth Recurrence variant (`flash_dit_recurse.yaml`):
- Define `n_unique_blocks` < `n_blocks`; reuse same block weights cyclically
- Equivalent to `n_blocks // n_unique_blocks` passes through `n_unique_blocks` blocks
- Parameter reduction: ~3-5× vs unique blocks

### Model size targets (tentative):
| Name | d_model | n_heads | n_layers | Params |
|------|---------|---------|----------|--------|
| Tiny | 384 | 6 | 12 | ~28M |
| Small | 768 | 12 | 12 | ~115M |
| Base | 1024 | 16 | 24 | ~380M |

---

## Phase 5 — Diffusion: Rectified Flow

**Objective** (`src/flash_dit/diffusion/flow_matching.py`):
- Interpolant: `x_t = (1-t) x_0 + t ε` where `ε ~ N(0,I)`
- Target velocity: `v = ε - x_0`
- Loss: `MSE(model(x_t, t), v)`
- Timestep sampling: logit-normal `t ~ LogitNormal(0, 1)` (better than uniform)

**Samplers** (`src/flash_dit/diffusion/samplers.py`):
- `EulerSampler(n_steps)`: simple baseline
- `HeunSampler(n_steps)`: 2nd-order, better quality at same NFE
- `DPMSolverPP(n_steps, order)`: adaptive, fewest NFE for quality target

---

## Phase 6 — Training

### `FlashDiTModule` (`src/flash_dit/training/lit_module.py`)
- `training_step`: sample t → interpolate → predict v → MSE loss
- `validation_step`: val loss + every `cfg.val.generate_every_n_epochs` → generate N samples → FAD
- `configure_optimizers`:
  - **Muon** for all 2D weight matrices (`W_q, W_k, W_v, W_o, W_up, W_down, W_gate`)
  - **AdamW** for embeddings, final head, norms, 1D params
  - Single `lr_scheduler` with warmup + cosine decay

### EMA (`src/flash_dit/training/ema.py`):
- `EMA(model, beta=0.9999, update_every=1)`
- EMA model used for validation and inference

### Precision: BF16 mixed precision (per global rules)

### Distributed:
- Default: DDP (Lightning `ddp` strategy)
- Optional: FSDP2 for Base+ models (config flag `strategy: fsdp2`)

### Logging:
- Lightning CSV logger (always)
- WandB optional (`use_wandb: false` default)
- Rich console for step-level progress

---

## Phase 7 — Validation & FAD

Every `val.generate_every_n_epochs` epochs (suggested: 5):
1. Use EMA model to generate `val.n_samples=64` latents via Heun sampler (50 NFE)
2. Decode with frozen VAE decoder → WAV files
3. Compute FAD against real FMA validation clips
4. Log: `val/fad`, `val/loss`, `val/generated_wav` (sample to WandB if enabled)

**FAD backend**: CLAP-LAION-audio (per `ml-training.md` rules)

---

## Phase 8 — Inference / Sampling Script

`scripts/sample.py`:
- Load checkpoint + EMA weights
- Load VAE decoder from stable-audio-tools
- Sample N latents from model
- Decode to WAV, save to `outputs/sample_YYYYMMDD_HHMMSS/`
- Accepts: `--n_samples`, `--sampler euler|heun|dpm`, `--n_steps`, `--cfg_scale`

---

## Phase 9 — SLURM Scripts

`slurm/precompute_latents.sbatch`:
- Account: `IscrC_LENS`, partition: `boost_usr_prod`
- 1 GPU, 40GB RAM, 4 hours
- Outputs to `$SCRATCH/flash-dit-latents/`

`slurm/train_single.sbatch`:
- 1 A100, 40GB RAM, 24 hours
- Env var: `RUN_NAME`, `MODEL_CONFIG`, `DATA_CONFIG`

`slurm/train_multi.sbatch`:
- 4 A100s, DDP
- For Base+ scale models

---

## Baseline Comparison Plan (3 progressive)

| Baseline | Optimizer | Attention | Compile | Depth | Param count | Story |
|----------|-----------|-----------|---------|-------|-------------|-------|
| `vanilla_sit` | AdamW | MHA (full heads) | No | 12 unique blocks | 115M | Reference / SiT-style |
| `flash_dit` | Muon+AdamW | GQA (12Q/2KV) + FA2 | Yes | 12 unique blocks | 115M | Speed & convergence |
| `flash_dit_recurse` | Muon+AdamW | GQA + FA2 | Yes | 4 unique, 3× looped | ~40M | Parameter efficiency |

**Ablation story**: vanilla_sit establishes quality ceiling → flash_dit shows same quality faster
(Muon convergence + throughput gains) → flash_dit_recurse shows quality vs parameter efficiency trade-off.

All runs: same latents, same data splits, same total training tokens (not same wall-clock time).

**Conditioning in all baselines**: genre CFG with p_drop=0.1; evaluate both unconditional
(null class token) and genre-conditional generation.

---

## Verification Checklist

1. **Latent precompute**: `uv run python scripts/precompute_latents.py --dry-run` — 10 files, check output shapes `(N, 64, 215)`
2. **Dataset**: `uv run pytest tests/test_dataset.py` — check item shapes, normalization
3. **Attention**: `uv run pytest tests/test_attention.py` — MHA==GQA output within tolerance, FA2 matches naive
4. **Flow matching**: `uv run pytest tests/test_flow_matching.py` — loss decreases on toy data
5. **Smoke train**: `uv run python scripts/train.py data.batch_size=2 trainer.max_steps=100` — no crash, loss goes down
6. **Generation**: `uv run python scripts/sample.py n_samples=2 sampler=euler n_steps=10` — WAV files produced
7. **FAD**: `uv run python scripts/eval_fad.py --generated outputs/sample_* --real $FMA_VAL_DIR` — produces scalar
