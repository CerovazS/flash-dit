# flash-dit

Optimized audio Diffusion Transformer for class-conditional music generation.
Built on [stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0)
latents, trained on [FMA Medium](https://github.com/mdeff/fma) (19,922 tracks,
16 genres), and instrumented for both training-time monitoring and paper-grade
throughput benchmarking.

> [!NOTE]
> Three baselines share the same parameter budget but differ in attention,
> optimizer, and compile policy — making it easy to attribute speedups to
> individual stack changes.

| Model               | Attention            | Optimizer    | `torch.compile` | Params |
| ------------------- | -------------------- | ------------ | --------------- | ------ |
| `vanilla_sit`       | MHA                  | AdamW        | —               | 115 M  |
| `flash_dit`         | GQA (12 Q / 2 KV)    | Muon + AdamW | ✓               | 115 M  |
| `flash_dit_recurse` | GQA + depth recurrence (4 unique × 3 loops) | Muon + AdamW | ✓ | ~40 M |

## Highlights

- **Rectified flow matching** with logit-normal timestep sampling.
- **GQA** with a 6:1 query-to-KV ratio — fewer KV-cache bytes, free quality on FMA.
- **Flash Attention 2/3** with runtime guards: FA2 on Ampere+, FA3 on Hopper, SDPA fallback elsewhere.
- **Muon + AdamW hybrid**: Muon orthogonalizes hidden 2-D projections via 5 Newton-Schulz iterations, AdamW handles embeddings/norms/biases.
- **Depth recurrence** variant: 4 unique blocks × 3 loops = 12 passes, ~3× parameter compression.
- **Validation-time orchestrator**: one sampling pass feeds latent MMD, KAD, Audiobox Aesthetics, and WandB audio uploads.
- **Standalone benchmark suite** (no Lightning, synthetic inputs) with throughput / micro-bench / DDP modes.

## Quick Start

```bash
# 0. Setup (uv-managed; FA2 wheel pinned for torch 2.5 / cu12 / py311)
cd flash-dit && uv sync

# 1. Pre-compute latents (GPU, ~3-4h on 1× A100)
sbatch slurm/precompute_latents.sbatch

# 2. Train (single GPU)
RUN_NAME=flash_dit_001 MODEL=flash_dit sbatch slurm/train_single.sbatch

# 3. Generate samples from a checkpoint
uv run python scripts/sample.py \
    --checkpoint outputs/.../checkpoints/last.ckpt \
    --n-samples 8 --cfg-scale 4.0 --sampler heun --n-steps 50

# 4. Run the standard eval triplet
uv run python scripts/eval_fad.py --generated outputs/samples_*/ --reference $FAST/fma_medium_wav/
uv run python scripts/eval_kad.py --generated outputs/samples_*/ --reference-manifest $FAST/kad_ref/fma_medium_test.txt
uv run python scripts/eval_latent_mmd.py --checkpoint outputs/.../checkpoints/last.ckpt --h5-path $FAST/fma_medium.h5

# 5. Tests
uv run pytest tests/ -q
```

> [!IMPORTANT]
> The VAE decoder is downloaded from `stabilityai/stable-audio-open-1.0`; export
> `HF_TOKEN` before training or sampling. Without it the run continues but
> validation generation and FAD are skipped.

## Architecture

- **Diffusion**: rectified flow matching, target velocity `v = ε - x_0`, MSE loss, logit-normal `t ~ N(0, 1) → σ(·)`.
- **Conditioning**: 16 genre classes + null token (CFG dropout `p = 0.1`), AdaLN-Zero modulation.
- **VAE**: stable-audio-open-1.0 Oobleck, 44.1 kHz stereo, 2048× temporal compression, 64-D latents.
- **Chunk size**: 131 072 samples (~3 s) → 64 latent frames during pre-compute; 215 frames (~10 s) per stored clip; **172 frames (~8 s)** randomly cropped at training.
- **Precision**: bf16 mixed; **EMA** with `β = 0.9999`.
- **Sampler**: Heun (2nd-order RK) by default, Euler optional; reverse-time ODE `dx/dt = -v_θ(x_t, t)` integrated over `n_steps`.

> [!TIP]
> `flash_dit_recurse` recovers most of the full-depth quality at ~40 M params
> by sharing weights across loop iterations — a useful baseline for
> compute-vs-quality comparisons under a fixed step budget.

## Configuration

Hydra composes the training config from these groups:

| Group       | Files                                       | Selected by `defaults:` |
| ----------- | ------------------------------------------- | ----------------------- |
| `model`     | `flash_dit`, `flash_dit_recurse`, `vanilla_sit` | `flash_dit`         |
| `diffusion` | `flow_matching`                             | `flow_matching`         |
| `optimizer` | `muon`, `adamw`                             | `muon`                  |
| `data`      | `fma_medium`, `fma_small`                   | `fma_medium`            |
| `trainer`   | `single_gpu`, `multi_gpu_ddp`               | `single_gpu`            |

Override anything from the command line:

```bash
uv run python scripts/train.py model=vanilla_sit data.batch_size=64 optimizer=adamw
uv run python scripts/train.py trainer=multi_gpu_ddp trainer.devices=4
uv run python scripts/train.py model.use_fa3=true   # H100 only
```

## Validation-time evaluation

Every `val.generate_every_n_epochs` ticks, `FlashDiTModule.on_validation_epoch_end`
samples `val.n_samples` latents **once** and dispatches them to every enabled
metric whose own cadence is also due. Decoding to WAV is skipped entirely when
no WAV-consuming metric fires.

| Metric          | Decode WAV? | Source                         | Default |
| --------------- | ----------- | ------------------------------ | ------- |
| `val/loss` (MSE)  | no        | per-batch flow-matching loss   | every val tick |
| `latent_mmd`    | no          | normalised latents vs HDF5 reference | enabled |
| `kad`           | yes         | PANNs `wavegram-logmel` embeddings | **enabled** |
| `audiobox`      | yes         | Audiobox Aesthetics (CE/CU/PC/PQ) | disabled |
| WandB audio     | yes         | first 2 WAVs uploaded as `wandb.Audio` + Artifact | when WandB run is active |

> [!NOTE]
> MSE (`val/loss`) fires at every val-loop tick governed by
> `trainer.check_val_every_n_epoch`. Pick that value as the MSE cadence and
> set `val.generate_every_n_epochs` to a multiple of it.

> [!WARNING]
> Per-epoch generated dirs grow linearly with the number of validations.
> `val.keep_n_wavs` (default `2`) bounds disk usage — extra WAVs and the
> per-epoch `embeddings/` + `convert/` PANNs caches are deleted after metrics
> and WandB upload complete. The WandB artifact upload is awaited so cleanup
> never races the upload.

### KAD reference: directory or manifest

KAD accepts the reference set in two mutually exclusive ways:

```yaml
evaluation:
  kad:
    enabled: true
    reference_dir: null              # (a) directory of audio files
    reference_manifest: null         # (b) text file, one absolute path per line
    reference_cache_dir: null        # cache dir for manifest mode (default: <manifest>.kad_cache/)
```

- **`reference_dir`** caches embeddings in-place under `<dir>/embeddings/<model>/`.
  Best when the reference audio lives in a dedicated directory.
- **`reference_manifest`** points at scattered audio (e.g. the FMA `000/`,
  `001/`, … tree) without copying or symlinking. Embedding caches are
  redirected to `reference_cache_dir` so the source tree stays pristine.

Build a manifest from FMA's `tracks.csv`:

```bash
uv run python scripts/build_kad_reference.py \
    --tracks-csv $FAST/fma/fma_metadata/tracks.csv \
    --fma-dir    $FAST/fma/fma_medium \
    --subset medium --split test \
    --output     $FAST/kad_ref/fma_medium_test.txt
```

> [!CAUTION]
> Stems must be unique across all audio sharing the same `cache_root` —
> caches are keyed by stem. FMA track IDs are six-digit globally unique, so
> the manifest flow is safe; if you build a custom manifest with colliding
> filenames the cache will silently overwrite entries.

## Hardware notes

> [!IMPORTANT]
> Flash Attention 2 requires Ampere (`sm_80`) or newer. Flash Attention 3
> requires Hopper (`sm_90`). The runtime guards in `models/attention.py`
> (`_fa2_runnable` / `_fa3_runnable`) check the device's compute capability
> at every forward and silently fall back to PyTorch SDPA when the kernel
> isn't supported — including on heterogeneous DDP jobs.

> [!TIP]
> Set `FLASH_DIT_DISABLE_FA2=1` to force the SDPA fallback even when FA2 is
> available — useful for ablation benchmarks that isolate the FA2 contribution.

Installing FA2/FA3:

```bash
# CINECA / A100 (requires nvcc)
module load cuda/12.2 && uv add flash-attn --no-build-isolation

# H100 (prebuilt cu130 abi3 wheel — installs in seconds, no compilation)
pip install --no-cache-dir \
    "https://download.pytorch.org/whl/cu130/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"
```

## Standalone benchmark suite

A separate Hydra root (`conf/bench_config.yaml`) drives kernel-level and
end-to-end throughput measurements **without** Lightning, callbacks, or
disk I/O. See [`docs/bench.md`](docs/bench.md) for the full guide.

```bash
# Default profile — flash_dit, full step, 50 measure × 3 repeats
uv run python scripts/bench.py

# Compare flash_dit vs vanilla_sit across batch sizes
uv run python scripts/bench.py -m model=flash_dit,vanilla_sit bench.batch_size=32,64,128

# Kernel-only attention benchmark
uv run python scripts/bench.py bench=microbench_attention

# DDP on four local GPUs
uv run python -m torch.distributed.run --standalone --nproc-per-node=4 scripts/bench.py

# Aggregate every result.csv under outputs/bench/ into a ranked summary
uv run python scripts/aggregate_bench.py --root outputs/bench --print
```

## Newton-Schulz Triton kernel ablation

`scripts/bench_ns_kernels.py` benchmarks Muon's NS iteration with the three
Triton kernels (`XXT`, `XTX`, `ba_plus_cAA`) from
[KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt/blob/master/triton_kernels.py).

> [!WARNING]
> On RTX 3090 (`sm_86`), **every Triton-kernel configuration is slower than the
> `torch.matmul` baseline** — the kernels were tuned for Hopper's wider
> register file and 128×128×64 block tiles, and `num_stages=2` on Ampere
> halves the software pipelining versus `num_stages=4` on H100.
> See [`docs/ns_kernels_ablation.md`](docs/ns_kernels_ablation.md) for the full
> per-shape breakdown and reasoning.

## Repository layout

```
flash-dit/
├── conf/                       # Hydra root + groups
│   ├── config.yaml             # training root
│   ├── bench_config.yaml       # bench root
│   ├── bench/                  # bench profiles (throughput, microbench_*, sweep)
│   ├── data/                   # fma_medium, fma_small
│   ├── diffusion/              # flow_matching
│   ├── model/                  # flash_dit, flash_dit_recurse, vanilla_sit
│   ├── optimizer/              # muon, adamw
│   └── trainer/                # single_gpu, multi_gpu_ddp
│
├── docs/
│   ├── bench.md                # benchmark suite guide
│   └── ns_kernels_ablation.md  # NS Triton kernel ablation report
│
├── scripts/
│   ├── train.py                # Lightning training entry point
│   ├── sample.py               # CLI inference from a checkpoint
│   ├── precompute_latents.py   # FMA → HDF5 latent dataset
│   ├── eval_fad.py             # FAD vs reference dir
│   ├── eval_kad.py             # KAD (PANNs) vs dir or manifest
│   ├── eval_latent_mmd.py      # latent-space MMD from a checkpoint
│   ├── build_kad_reference.py  # FMA tracks.csv → KAD manifest
│   ├── bench.py                # bench Hydra entry point
│   ├── bench_ns_kernels.py     # NS Triton kernel sweep
│   └── aggregate_bench.py      # walks outputs/, emits summary.{md,csv}
│
├── slurm/                      # CINECA Leonardo sbatch templates
│   ├── precompute_latents.sbatch
│   ├── train_single.sbatch
│   └── train_multi.sbatch
│
├── src/flash_dit/
│   ├── bench/                  # CudaEventTimer, runner, reporting, DDP helpers
│   ├── data/                   # LatentDataset + LatentDataModule (HDF5)
│   ├── diffusion/              # flow matching, Euler/Heun samplers, schedules
│   ├── evaluation/             # FAD, KAD (vendored kadtk), latent_mmd, audiobox, generation
│   ├── models/                 # DiT, attention (MHA/GQA), MLP (SwiGLU), conditioning, RoPE
│   ├── training/               # Lightning module, EMA, Muon optimizer + Triton kernels, JSONL callback
│   └── utils/                  # rich-based console helpers
│
└── tests/                      # pytest suite (attention, MMD, KAD, optimizer, dataset, bench, ...)
```

## Outputs

Every training run writes to `outputs/<run_name>/`:

```
outputs/<run_name>/
├── checkpoints/
│   ├── last.ckpt                              # full optimizer state (resumable)
│   └── epoch_NNNN-val_loss=X.XXXX.ckpt        # top-3 weights-only
├── csv/                                       # Lightning CSVLogger
├── metrics/                                   # JSONL per-step + per-metric JSON dumps
└── generated/epoch_NNNNNN/                    # validation WAVs (bounded by val.keep_n_wavs)
```

Bench runs land under `outputs/bench/<mode>_<model>_<timestamp>/` with
`config.yaml`, `result.json`, and `result.csv` per run.

## License

Research code, no license declared. The vendored
`evaluation/kadtk_vendored/` directory is a stripped-down copy of
[KAD](https://github.com/microsoft/kad) (PANNs backbone) reformatted for this
project; upstream license applies to those files.
