# flash-dit benchmark suite

A standalone benchmarking suite for the Diffusion Transformer, designed
for paper-grade throughput and kernel-level ablations. It runs **without
Lightning** so the numbers reflect the raw model cost — no callback, hook,
or logger overhead — and uses **synthetic inputs** so results are portable
across any machine (local, CINECA, cloud pods) without dataset setup.

## Table of contents

- [What it measures](#what-it-measures)
- [Quick start](#quick-start)
- [Module layout](#module-layout)
- [Configuration](#configuration)
- [Bench modes](#bench-modes)
- [Running on a single GPU](#running-on-a-single-gpu)
- [GPU pinning](#gpu-pinning)
- [Hydra multirun sweeps](#hydra-multirun-sweeps)
- [DDP (multi-GPU)](#ddp-multi-gpu)
- [Outputs](#outputs)
- [Aggregating many runs](#aggregating-many-runs)
- [Extending the suite](#extending-the-suite)
- [Metrics glossary](#metrics-glossary)
- [Caveats and gotchas](#caveats-and-gotchas)

## What it measures

Each bench run produces a `BenchResult` record with:

- **Throughput**: `step/s` (± std across repeats), `samples/s`, `tokens/s`.
- **Per-stage latency**: `fwd_ms`, `bwd_ms`, `opt_ms` (mean over all measured steps).
- **Memory**: peak VRAM in MB.
- **Wall time**: total time of the measurement loop.
- **Model metadata**: parameter counts, attention type, compile flag, precision.
- **Repro metadata**: git SHA, torch version, timestamp.

The suite intentionally does **not** track training loss. Convergence
benchmarks (loss-vs-step) are out of scope here — this suite measures
efficiency only.

## Quick start

```bash
# Run the default config: flash_dit model, full fwd+bwd+optimizer, 50 steps × 3 repeats
uv run python scripts/bench.py

# Kernel-only throughput of the GQA attention module
uv run python scripts/bench.py bench=microbench_attention

# Compare flash_dit vs vanilla_sit across three batch sizes (6 runs)
uv run python scripts/bench.py -m model=flash_dit,vanilla_sit bench.batch_size=32,64,128

# DDP on four local GPUs
uv run python -m torch.distributed.run --standalone --nproc-per-node=4 scripts/bench.py

# Combine all runs under outputs/bench/ into a ranked markdown table
uv run python scripts/aggregate_bench.py --root outputs/bench --print
```

## Module layout

```
src/flash_dit/bench/
  core.py         # CudaEventTimer, CpuTimer, synthetic_batch, memory_snapshot, set_determinism
  ddp.py          # setup_dist, cleanup_dist, is_global_zero, maybe_wrap_ddp
  runner.py       # ThroughputRunner, MicroRunner, BenchResult dataclass
  reporting.py    # write_result (JSON+CSV), aggregate_markdown

scripts/
  bench.py             # Hydra entry point
  aggregate_bench.py   # Walks outputs/ and emits summary.md + summary.csv

conf/
  bench_config.yaml    # Root config composed for scripts/bench.py
  bench/
    throughput.yaml           # default profile
    attention_sweep.yaml      # sweep model × batch_size
    batch_scaling.yaml        # scan batch_size to find peak / OOM
    microbench_attention.yaml # attention module only
    microbench_mlp.yaml       # MLP module only
    microbench_block.yaml     # full DiTBlock
```

Nothing in the suite modifies existing training code. The only module
touched in the main codebase is
[`models/attention.py`](../src/flash_dit/models/attention.py), which gained
runtime guards (`_fa2_runnable`, `_fa3_runnable`) checking the GPU compute
capability before dispatching to Flash Attention kernels. This fixes
heterogeneous-GPU jobs (e.g. DDP across Ampere + Turing) where FA2 is only
supported on Ampere+.

## Configuration

The bench suite uses its own Hydra root
([`conf/bench_config.yaml`](../conf/bench_config.yaml)), distinct from the
training root (`conf/config.yaml`) so it does not pull in unused groups
(data, diffusion, val, evaluation). The root composes three groups:

| Group       | Source                    | Notes                                                        |
| ----------- | ------------------------- | ------------------------------------------------------------ |
| `bench`     | `conf/bench/*.yaml`       | Profile — timing budget, batch/seq, bench mode, precision    |
| `model`     | `conf/model/*.yaml`       | Reused verbatim from training (flash_dit, vanilla_sit, ...)  |
| `optimizer` | `conf/optimizer/*.yaml`   | Consulted only when `bench.include_optimizer=true`           |

Override anything from the command line:

```bash
# Pick a different model and raise the batch size
uv run python scripts/bench.py model=vanilla_sit bench.batch_size=128

# Disable optimizer step (measure fwd+bwd only)
uv run python scripts/bench.py bench.include_optimizer=false

# Disable backward too (inference-only throughput)
uv run python scripts/bench.py bench.include_backward=false bench.include_optimizer=false

# FP32 for comparison against mixed precision
uv run python scripts/bench.py bench.precision=fp32
```

### Key bench options (from `conf/bench/throughput.yaml`)

| Field                 | Default              | Meaning                                                         |
| --------------------- | -------------------- | --------------------------------------------------------------- |
| `mode`                | `throughput_step`    | `throughput_step` or `microbench_{attention,mlp,block}`         |
| `warmup_steps`        | `10`                 | Steps executed but not timed; absorbs `torch.compile` + cuDNN   |
| `measure_steps`       | `50`                 | Timed steps per repeat                                          |
| `repeats`             | `3`                  | Independent repeats; std reflects cross-repeat variability      |
| `batch_size`          | `32`                 | Batch dimension `B`                                             |
| `seq_len`             | `172`                | Temporal dimension `T` (matches 8 s @ Stable Audio VAE rate)    |
| `in_channels`         | `64`                 | Latent channels `C`                                             |
| `n_classes`           | `17`                 | Genre classes + CFG null token                                  |
| `precision`           | `bf16`               | `bf16`, `fp16`, or `fp32`; applied via `torch.autocast` on CUDA |
| `include_backward`    | `true`               | If false, only measures forward pass                            |
| `include_optimizer`   | `true`               | If false, skips `optimizer.step()`                              |

## Bench modes

### `throughput_step` — full training step

The standard benchmark. Each step runs:

1. Build a synthetic batch on device (`torch.randn` for latents, `torch.randint` for labels).
2. Forward through the full `DiffusionTransformer` inside `flow_matching_loss`.
3. Backward on the MSE loss.
4. `optimizer.step()` + `zero_grad`.

### `microbench_attention` — attention module only

Builds just the attention module (`MultiHeadAttention` or
`GroupedQueryAttention`) with RoPE + projections + output projection, and
runs forward/backward with random `(B, T, d_model)` inputs. Isolates the
attention cost from the rest of the DiT block.

### `microbench_mlp` — MLP module only

Same pattern, but for the MLP (`SwiGLU` or `ReLU²` depending on
`model.mlp_type`).

### `microbench_block` — full DiTBlock

Norms + attention + MLP + AdaLN modulation, but no input/output projections,
no timestep/genre embedding, no stacking of blocks. A useful mid-point
between kernel-only and full-model benchmarks for sanity-checking layer
counts.

## Running on a single GPU

```bash
uv run python scripts/bench.py
```

The CLI prints a single summary line like:

```
OK throughput_step  samples/s=440.6  step/s=13.77±0.01  fwd=20.42ms bwd=40.28ms opt=11.61ms  VRAM=3419MB
INFO wrote: outputs/bench/throughput_step_flash_dit_20260417_012707/result.json outputs/bench/.../result.csv
```

## GPU pinning

Pin the run to a specific device with `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python scripts/bench.py model=vanilla_sit
```

The bench initialises on the first visible device (index 0 inside the
process), but the physical device is whichever one `CUDA_VISIBLE_DEVICES`
selects. Use `nvidia-smi` to confirm the correct GPU is being driven.

## Hydra multirun sweeps

Hydra's multirun mode (`-m` or `--multirun`) expands every configured
dimension into separate jobs run back-to-back. For example:

```bash
uv run python scripts/bench.py -m \
  model=flash_dit,vanilla_sit \
  bench.batch_size=32,64,128
```

runs six jobs (2 models × 3 batch sizes), each in its own output
subdirectory under
`outputs/bench/sweeps/<timestamp>_<mode>/<job>_<name>_B<batch>_T<seq>/`.

Three pre-made sweep profiles live under `conf/bench/`:

- **`attention_sweep`**: `model ∈ {flash_dit, vanilla_sit}` × `batch_size ∈ {32, 64, 128}`.
- **`batch_scaling`**: `batch_size ∈ {8, 16, 32, 64, 128, 192, 256}` at the default model.
- **`microbench_attention`**: single-job profile, but composable with `-m`.

Pick a sweep with:

```bash
uv run python scripts/bench.py -m bench=attention_sweep
uv run python scripts/bench.py -m bench=batch_scaling
```

## DDP (multi-GPU)

The bench initialises `torch.distributed` from the env vars that
`torchrun` sets (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`). If those vars are
absent it stays single-process — no code changes needed between modes.

```bash
# Four GPUs on one node
uv run python -m torch.distributed.run --standalone --nproc-per-node=4 scripts/bench.py

# Two nodes × four GPUs = 8 processes total (coordinator on node 0)
# Node 0:
uv run python -m torch.distributed.run --nnodes=2 --nproc-per-node=4 \
  --master_addr=10.0.0.1 --master_port=29500 --node_rank=0 scripts/bench.py
# Node 1:
uv run python -m torch.distributed.run --nnodes=2 --nproc-per-node=4 \
  --master_addr=10.0.0.1 --master_port=29500 --node_rank=1 scripts/bench.py
```

Every rank runs the full measurement but **only rank 0 writes
`result.json` / `result.csv`** to disk. `samples_per_s` in the record is
already multiplied by `world_size`, so it reflects global throughput.

## Outputs

For a single run, the output directory contains:

```
outputs/bench/throughput_step_flash_dit_<timestamp>/
  config.yaml   # fully resolved Hydra config (for repro)
  result.json   # full BenchResult + per-repeat step/s list
  result.csv    # single-row flat CSV with the scalar fields
```

For a multirun sweep, each job gets its own subdirectory under the sweep
root and the structure above.

## Aggregating many runs

Once several runs have been produced (single or sweep), collapse them
into a ranked table:

```bash
uv run python scripts/aggregate_bench.py --root outputs/bench --print
```

This walks `--root` recursively, loads every `result.csv` it finds,
writes:

- `<root>/summary.csv` — concatenated CSV with all columns plus `_source`
  (the path of the original `result.csv`).
- `<root>/summary.md` — human-readable markdown table sorted by
  `samples_per_s` descending.

Example output:

```
| model       | attn | compile | prec | B  | T   | ws | mode                 | step/s | ±     | samples/s  | fwd(ms) | bwd(ms) | opt(ms) | VRAM(MB) |
|-------------|------|---------|------|----|-----|----|----------------------|--------|-------|------------|---------|---------|---------|----------|
| flash_dit   | gqa  | True    | bf16 | 32 | 172 | 1  | microbench_attention | 520.69 | 21.13 | 1.666e+04  | 0.64    | 1.23    | 0.00    | 173      |
| flash_dit   | gqa  | True    | bf16 | 32 | 172 | 1  | throughput_step      | 13.77  | 0.01  | 440.58     | 20.42   | 40.28   | 11.61   | 3419     |
| vanilla_sit | mha  | False   | bf16 | 32 | 172 | 1  | throughput_step      | 9.16   | 0.00  | 293.27     | 31.57   | 64.44   | 12.79   | 5647     |
```

## Extending the suite

### Adding a new bench profile

Drop a YAML into `conf/bench/` that inherits the defaults:

```yaml
# conf/bench/long_sequence.yaml
defaults:
  - throughput
  - _self_

seq_len: 512
warmup_steps: 20
measure_steps: 100
```

Then:

```bash
uv run python scripts/bench.py bench=long_sequence
```

### Adding a new microbench target

1. Add a `_build_<thing>(model_cfg, device)` helper in `runner.py` that
   returns the module and the expected input signature.
2. Register it under a new `mode` string in `MicroRunner._build`.
3. Create `conf/bench/microbench_<thing>.yaml` setting `mode:
   microbench_<thing>`.

Both runners share the same `BenchResult` schema, so the CSV/aggregator
pipeline keeps working unchanged.

### Adding a new model for the sweep

Any file in `conf/model/` is automatically sweepable:

```bash
uv run python scripts/bench.py -m model=flash_dit,vanilla_sit,my_new_model
```

No bench-side changes are needed.

## Metrics glossary

| Metric              | Unit      | Definition                                                      |
| ------------------- | --------- | --------------------------------------------------------------- |
| `step_per_s_mean`   | 1/s       | Mean over `repeats` of `measure_steps / elapsed_repeat_seconds` |
| `step_per_s_std`    | 1/s       | Std of the per-repeat `step/s` values (0 when `repeats=1`)      |
| `samples_per_s`     | samples/s | `step_per_s_mean × batch_size × world_size`                     |
| `tokens_per_s`      | tokens/s  | `samples_per_s × seq_len × in_channels`                         |
| `fwd_ms_mean`       | ms        | Mean elapsed time of the forward path (CUDA-event timed)        |
| `bwd_ms_mean`       | ms        | Mean elapsed time of `loss.backward()`                          |
| `opt_ms_mean`       | ms        | Mean elapsed time of `optimizer.step() + zero_grad()`           |
| `peak_vram_mb`      | MB        | `torch.cuda.max_memory_allocated()` during measurement          |
| `total_wall_s`      | s         | Wall time covering all repeats (includes per-repeat sync)       |
| `trainable_params`  | int       | `sum(p.numel() for p in params if p.requires_grad)`             |
| `total_params`      | int       | All parameters including frozen                                 |

## Caveats and gotchas

- **`torch.compile` warmup**. `flash_dit` sets `use_compile=true`, so the
  first `warmup_steps` include one or two very slow iterations while
  Inductor builds the kernel. The default `warmup_steps=10` is enough for
  `batch_size=32`. Raise it if you see the first repeat's `step/s` dip
  well below the rest (visible in `extra.per_repeat_step_per_s` inside
  `result.json`).

- **cuDNN autotune**. `set_determinism` keeps `cudnn.benchmark=True`
  because we *want* the fastest kernel cuDNN can find. This means the
  first epoch on a new shape may be slightly slower — the warmup
  absorbs it.

- **Flash Attention 2 on older GPUs**. FA2 requires Ampere (sm_80) or
  newer. On Turing (sm_75, e.g. 2080 Ti) or Volta the bench falls back
  to PyTorch SDPA automatically thanks to the runtime guard in
  `attention.py`. You will see lower throughput on those GPUs, but the
  run succeeds. For heterogeneous DDP (e.g. 3090 + 2080 Ti) the backward
  is dominated by the allreduce that waits on the slowest rank.

- **`time.perf_counter` fallback on CPU**. When run without CUDA,
  `CudaEventTimer` is replaced by `CpuTimer` — no sync issues, but the
  GPU-specific semantics (allreduce overhead, autotune) disappear.
  CPU runs are useful for unit tests, not for paper numbers.

- **Rank-zero writes only**. In DDP, every rank runs the measurement
  (gradients still need to allreduce), but only rank 0 writes the JSON /
  CSV. Don't be surprised to see `world_size=4` in the record even
  though there's a single output file.

- **No validation, no checkpoints**. The suite deliberately skips those.
  Use `scripts/train.py` for any run that needs loss curves, EMA, or
  saved weights.
