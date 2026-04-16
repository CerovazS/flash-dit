# Program: Latent MMD and KAD Evaluation for flash-dit

## Goal

Integrate two complementary distribution metrics into `flash-dit`:

1. `latent_mmd`: an internal, fast metric computed directly on Stable Audio VAE latent features.
2. `kad`: an external audio metric computed on decoded WAV files through `kadtk`.

Both metrics must support:

- periodic evaluation during training;
- standalone evaluation at the end of training from a checkpoint or generated directory;
- local machine-readable artifacts under `outputs/`;
- enough metadata to compare runs reproducibly.

This plan assumes the implementing agent has no previous context about KAD, `kadtk`, or the planned latent metric.

## Source Material to Reopen First

Before coding, reread these sources and record any findings that affect implementation:

- Paper: `KAD: No More FAD! An Effective and Efficient Evaluation Metric for Audio Generation`, arXiv `2502.15602`.
- Implementation repo: `https://github.com/YoonjinXD/kadtk`.
- Local repo files:
  - `src/flash_dit/training/lit_module.py`
  - `src/flash_dit/evaluation/fad.py`
  - `src/flash_dit/evaluation/audiobox.py`
  - `src/flash_dit/evaluation/generation.py`
  - `scripts/sample.py`
  - `scripts/eval_fad.py`
  - `conf/config.yaml`
  - `pyproject.toml`

## KAD Summary

KAD is based on the unbiased finite-sample MMD estimator:

```text
MMD^2_unbiased(X, Y) =
  1/(n(n-1)) sum_{i!=j} k(x_i, x_j)
+ 1/(m(m-1)) sum_{i!=j} k(y_i, y_j)
- 2/(nm)    sum_i sum_j k(x_i, y_j)
```

The default kernel in `kadtk` is Gaussian RBF:

```text
k(a, b) = exp(-||a - b||^2 / (2 sigma^2))
```

The paper defines KAD as a scaled MMD value. The `kadtk` code currently uses:

```python
SCALE_FACTOR = 100
```

The paper summary reports a scale factor of 1000. Treat this as a source/code discrepancy to verify before finalizing the implementation defaults.

The metric is unbiased but noisy at small sample sizes. It can become negative when the sample count is small or the two distributions are very close. Therefore:

- training-time metrics can be noisy trend signals;
- final evaluation should use a larger fixed sample count;
- every logged value must include `n_reference`, `n_generated`, `kernel`, `bandwidth`, `scale_factor`, and feature type.

## `kadtk` Implementation Structure

Important modules in `kadtk`:

- `kadtk.__main__`
  - CLI entrypoint: `kadtk <model_name> <baseline_dir> <eval_dir>`.
  - Calls `cache_embedding_files` on both directories.
  - Creates `KernelAudioDistance` or `FrechetAudioDistance`.
  - Writes optional CSV output.

- `kadtk.emb_loader`
  - `EmbeddingLoader.load_audio(path)` converts audio to mono, resamples to the embedding model sample rate, and caches converted WAVs under `convert/<sr>/`.
  - `cache_embedding_files(files_or_dir, model, workers, force_emb_encode)` extracts and caches embeddings.
  - Embeddings are saved under an `embeddings/<model_name>/...` cache path.
  - The public API is file-based, but the metric itself is matrix-based.

- `kadtk.model_loader`
  - Defines supported audio embedding models.
  - `panns-wavegram-logmel` is a good first default for general audio KAD.
  - Some models pull in heavy dependencies, including TensorFlow or model-specific packages.

- `kadtk.kad`
  - `calc_kernel_audio_distance(x, y, cache_dirs, device, bandwidth=None, kernel="gaussian")` is the core metric.
  - `KernelAudioDistance.score(baseline, eval)` loads cached embedding arrays and calls the core metric.
  - `KernelAudioDistance.score_inf(...)` samples different evaluation sizes and extrapolates against `1/n`.

Important implementation caveat:

- The README states that the bandwidth is computed from the reference set.
- The current `calc_kernel_audio_distance(x, y, ...)` code uses `median_pairwise_distance(y)` when `bandwidth is None`.
- Since `KernelAudioDistance.score(baseline, eval)` passes `x=baseline`, `y=eval`, current code appears to use the generated/eval set for bandwidth.
- Decide whether `flash-dit` should exactly mirror `kadtk` code behavior or follow the paper/README convention using the reference set.

## Metric Definitions for flash-dit

### 1. `kad`

Purpose:

- External perceptual/distribution metric over decoded audio.
- Used for final model comparison and periodic heavy validation.

Default behavior:

- Decode generated latents to WAV.
- Keep generated WAVs as curated artifacts for audit.
- Use `kadtk` to extract embeddings and compute KAD.
- Cache reference embeddings so final evaluation does not repeatedly re-embed ground-truth audio.

Proposed first default:

```yaml
evaluation:
  kad:
    enabled: false
    model_name: panns-wavegram-logmel
    reference_dir: null
    workers: 8
    device: cuda
    kernel: gaussian
    bandwidth_source: reference
    scale_factor: kadtk
    every_n_epochs: 20
    n_samples_train: 256
    n_samples_final: 1024
```

The first implementation can be file-based for reliability. A later optimization can compute and store generated embeddings directly.

### 2. `latent_mmd`

Purpose:

- Internal metric that compares generated Stable Audio VAE latents against ground-truth latentized audio.
- Fast enough to run during training.
- Does not require VAE decoding, WAV files, or external embedding models.

This is not KAD. Use names like:

```text
val/latent_mmd_frame
val/latent_mmd_clip
test/latent_mmd_frame
test/latent_mmd_clip
```

Feature sets:

- Reference set `X`: latents from the validation/test split in the HDF5 latent dataset.
- Generated set `Y`: sampled latents from the current EMA model using the same latent normalization convention as training.

Feature transforms:

1. `frame`
   - Input latent shape: `(B, C, T)`.
   - Transform to `(B*T, C)` via `latents.permute(0, 2, 1).reshape(-1, C)`.
   - Optional subsampling is required to avoid treating huge numbers of correlated frames as independent evidence.

2. `clip_pooled`
   - Input latent shape: `(B, C, T)`.
   - Compute per-clip features:
     - channel mean over time: `(B, C)`;
     - channel std over time: `(B, C)`;
     - optional temporal chunk means with a small fixed number of chunks.
   - Concatenate into `(B, D)`.
   - This captures clip-level distribution better than frame-only features and avoids very high-dimensional `C*T` flattening.

Avoid initially:

- Full flattening `(B, C*T)` for long sequences such as 2048 latent frames. It is high-dimensional and likely suffers from distance concentration unless carefully normalized and tested.

Proposed first default:

```yaml
evaluation:
  latent_mmd:
    enabled: true
    every_n_epochs: 4
    n_reference: 1024
    n_generated_train: 256
    n_generated_final: 1024
    feature_types: [frame, clip_pooled]
    frame_subsample_per_clip: 32
    clip_pool_num_chunks: 4
    kernel: gaussian
    bandwidth_source: reference
    bandwidth_subsample: 10000
    scale_factor: 100
    precision: float32
    max_pairwise_block: 4096
```

## Implementation Plan

### Phase 0: Dependency and Compatibility Check

Success criteria:

- Determine whether `kadtk` can be added with `uv add kadtk` without breaking the current Torch/CUDA stack.
- If dependency constraints conflict with the existing `torch` version, do not force the install.
- In that case, implement `latent_mmd` independently and make KAD an optional wrapper that raises a clear install error.

Actions:

1. Inspect current Torch version in the `.venv`.
2. Inspect `kadtk` dependency constraints.
3. Decide whether to:
   - add `kadtk` to `pyproject.toml`; or
   - add an optional extra such as `evaluation = ["kadtk"]`; or
   - leave `kadtk` optional and document manual install.

Use `uv add` for any dependency changes. Do not use `uv pip install`.

If `kadtk` is not usable with uv add try cloning the repo and importing the relevant file. Do not reinvent the wheel. Try always use files that have scientific grounding and are already evaluated and validated so pay attention to the files inside https://github.com/YoonjinXD/kadtk for metrics implementation.

### Phase 1: Shared MMD Core

Create:

```text
src/flash_dit/evaluation/mmd.py
```

Responsibilities:

- `pairwise_sq_dists_chunked(x, y, block_size, device)`.
- `median_pairwise_distance(x, subsample=None, seed=...)`.
- `mmd_unbiased(x, y, kernel="gaussian", bandwidth=None, scale_factor=100, ...)`.
- Support Gaussian RBF first.
- Optionally expose `iq` and `imq` only if low effort and tested.
- Use chunked pairwise computation to avoid OOM.
- Return both scalar and metadata.

Tests:

```text
tests/test_mmd.py
```

Required checks:

- identical sets produce near-zero MMD within finite-sample tolerance;
- shifted distributions produce larger MMD;
- result matches a small naive implementation;
- chunked and unchunked results match;
- no diagonal leakage in the unbiased within-set terms.

### Phase 2: Latent Feature Extraction

Create:

```text
src/flash_dit/evaluation/latent_mmd.py
```

Responsibilities:

- Convert latent tensors into feature matrices.
- Implement `frame` and `clip_pooled`.
- Load reference latents from `LatentDataset` or directly from HDF5.
- Sample a fixed reference subset per run/evaluation seed.
- Compute `latent_mmd` for generated latents.
- Save reference feature cache under the run output directory or a deterministic cache path keyed by:
  - HDF5 path;
  - split;
  - feature type;
  - sequence length;
  - pooling config;
  - sample count;
  - seed.

Important:

- Use the same normalized latent convention consistently. Since training operates on normalized latents, the default should compare normalized generated latents to normalized reference latents.
- Record whether latents are normalized or raw in metadata.

Tests:

```text
tests/test_latent_mmd.py
```

Required checks:

- feature shapes are correct for `(B, C, T)`;
- frame subsampling is deterministic with a seed;
- clip pooled features contain mean/std and chunk means in the documented order;
- metric runs on CPU with small tensors.

### Phase 3: KAD Wrapper

Create:

```text
src/flash_dit/evaluation/kad.py
scripts/eval_kad.py
```

Wrapper behavior:

- `compute_kad(generated_dir, reference_dir, model_name, device, workers, ...)`.
- If `kadtk` is missing, raise an actionable error explaining how to install it via `uv add kadtk` or the chosen optional dependency path.
- Use `kadtk.emb_loader.cache_embedding_files`.
- Use `kadtk.kad.KernelAudioDistance`.
- Save a metrics JSON next to generated audio with:
  - score;
  - model name;
  - generated dir;
  - reference dir;
  - number of files;
  - device;
  - bandwidth policy;
  - timestamp.

Tests:

- Unit-test import error handling without requiring `kadtk`.
- Unit-test CLI argument parsing where possible.
- Do not require heavy embedding model downloads in normal tests.

### Phase 4: Training Integration

Modify:

```text
src/flash_dit/training/lit_module.py
conf/config.yaml
scripts/train.py
```

Training-time behavior:

- Keep lightweight sample generation unchanged.
- Add optional latent MMD computation during validation using generated latents before VAE decode.
- Log to Lightning:
  - `val/latent_mmd_frame`
  - `val/latent_mmd_clip`
  - metadata as JSON artifact/file, not as scalar spam.
- Add optional KAD computation only at lower frequency because it requires decoding and embedding extraction.
- Ensure rank-zero-only execution for file writing and heavy metrics in distributed training.
- Avoid overwriting metric artifacts from older evaluations.

Suggested output layout:

```text
outputs/runs/<run_id>/
  metrics/
    latent_mmd_epoch_000123.json
    kad_epoch_000123.json
  generated/
    epoch_000123/
      sample_000.wav
      ...
```

### Phase 5: Final Evaluation CLI

Create:

```text
scripts/eval_metrics.py
```

or extend separate scripts:

```text
scripts/eval_latent_mmd.py
scripts/eval_kad.py
```

Preferred first version:

- `scripts/eval_latent_mmd.py` for latent-only evaluation from generated latent tensors or a checkpoint.
- `scripts/eval_kad.py` for file-based audio KAD.

Success criteria:

- Can compute latent MMD from a checkpoint without decoding audio.
- Can compute KAD from an existing generated WAV directory.
- Writes JSON and CSV/JSONL metrics under `outputs/.../metrics/`.

### Phase 6: Verification

Commands to run:

```bash
uv run pytest tests/test_mmd.py tests/test_latent_mmd.py -v
uv run pytest tests/test_attention.py tests/test_flow_matching.py -v
uv run python scripts/eval_latent_mmd.py --help
uv run python scripts/eval_kad.py --help
```

If `kadtk` is installed and reference/generated WAV directories are available:

```bash
uv run python scripts/eval_kad.py \
  --generated <generated_wav_dir> \
  --reference <reference_wav_dir> \
  --model-name panns-wavegram-logmel \
  --device cuda
```

For training smoke test:

```bash
uv run python scripts/train.py trainer.fast_dev_run=true val.generate_every_n_epochs=1 evaluation.latent_mmd.enabled=true
```

Adjust the exact command to match the repo's existing Hydra config structure.

## Key Decisions Needed From User

1. Bandwidth policy:
   - `reference`: match paper/README intent and keep scores comparable across generated sets.
   - `generated`: mirror current `kadtk` code behavior.
   - `fixed`: precompute and store one bandwidth value for a benchmark suite.

   Recommended: `reference`.

2. KAD scale factor:
   - `kadtk`: use code value, currently 100.
   - `paper`: use paper-reported value, 1000.
   - configurable: store raw MMD and scaled values.

   Recommended: configurable, always log raw MMD and scaled score.

3. Default KAD embedding model:
   - `panns-wavegram-logmel`: good first default for general audio.
   - `clap-laion-music`: potentially music-aligned but may behave differently.
   - multiple models: slower but more robust.

   Recommended first pass: `panns-wavegram-logmel`.

4. Latent MMD reference split:
   - validation split for training-time model selection;
   - test split only for final reports.

   Recommended: validation during training, test only for final evaluation.

5. Training-time sample budget:
   - small budget, noisy: 64-128 generated clips;
   - balanced: 256 generated clips;
   - stronger but expensive: 512-1024 generated clips.

   Recommended: 256 during training, 1024 final.

6. Whether generated validation WAVs should remain permanent artifacts:
   - keep all KAD-evaluated WAVs for audit;
   - keep only a curated subset plus embeddings/metrics.

   Recommended: keep all WAVs used for KAD in final evaluation; keep a smaller subset during training if storage becomes a concern.

## Non-Goals

- Do not replace FAD/Audiobox immediately.
- Do not call latent MMD "KAD".
- Do not implement a full online exact accumulator until the batch metric is tested.
- Do not add heavy embedding model downloads to normal unit tests.
- Do not delete existing run artifacts or checkpoints.

## Initial Implementation Order

1. Implement and test `evaluation/mmd.py`.
2. Implement and test `evaluation/latent_mmd.py`.
3. Add standalone `scripts/eval_latent_mmd.py`.
4. Add optional training-time latent MMD logging.
5. Add `evaluation/kad.py` and `scripts/eval_kad.py`.
6. Add optional training-time KAD at low frequency.
7. Run smoke tests and document final commands in README or a short evaluation note.
