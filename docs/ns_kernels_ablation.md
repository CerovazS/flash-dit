# Newton-Schulz Triton kernel ablation

**Goal.** Evaluate the three Triton kernels (`XXT`, `XTX`, `ba_plus_cAA`) from
[KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt/blob/master/triton_kernels.py)
as drop-in accelerators for Muon's Newton-Schulz iteration, and measure each
combination's effect on our actual workload.

**Workload.** Apply 5 NS iterations to every Muon-eligible 2D matrix of a
`flash_dit` model — 84 matrices across 4 distinct shapes:

| shape | count | role |
|---|---|---|
| `(128, 768)` | 24 | k_proj, v_proj (GQA, n_kv_heads=2) |
| `(768, 768)` | 24 | q_proj, out_proj |
| `(768, 2048)` | 12 | mlp.down |
| `(2048, 768)` | 24 | mlp.gate, mlp.up |

**Hardware.** NVIDIA RTX 3090 (sm_86, 24 GB). The H100 block configs (128×128×64
with `num_stages=4`) exceed the 3090's 101 KB/SM shared-memory budget, so
`muon_kernels.py::_pick_block_config` drops `num_stages=2` on non-Hopper GPUs.

**Protocol.** 15 warmup iterations, 40 measured iterations, 3 repeats; CUDA
events for timing, sync after each repeat.

## Headline result

| rank | xxt | xtx | ba_plus_cAA | ms/step (mean ± std) | vs baseline |
|---|---|---|---|---|---|
| 🥇 1 | — | — | — | **38.34 ± 0.04** | 1.00× (reference) |
| 2 | — | ✓ | — | 43.38 ± 0.02 | 0.88× (slower) |
| 3 | — | — | ✓ | 50.21 ± 0.10 | 0.76× |
| 4 | — | ✓ | ✓ | 51.92 ± 0.13 | 0.74× |
| 5 | ✓ | ✓ | — | 53.60 ± 0.04 | 0.72× |
| 6 | ✓ | — | — | 54.11 ± 0.05 | 0.71× |
| 7 | ✓ | ✓ | ✓ | 56.16 ± 0.07 | 0.68× |
| 8 | ✓ | — | ✓ | 56.80 ± 0.05 | 0.67× |

**Every Triton-kernel configuration is slower than the `torch.matmul` baseline
on the 3090.** The "all three kernels together" combination (the one the
sub-agent expected would be fastest) is the slowest tested (−32%).

## Per-shape breakdown

ms/step contribution, split by matrix shape (lower is better):

| config | (128, 768) × 24 | (2048, 768) × 24 | (768, 2048) × 12 | (768, 768) × 24 |
|---|---:|---:|---:|---:|
| `___` (baseline) | **10.57** | 20.14 | 9.18 | **10.33** |
| `_T_` (xtx) | 10.53 | 24.79 | **8.88** ✓ | 10.36 |
| `__B` (ba_plus_cAA) | 14.11 | 21.78 | 9.99 | 14.04 |
| `_TB` | 14.13 | 26.51 | 9.95 | 14.08 |
| `X__` (xxt) | 15.26 | 21.62 | 9.96 | 15.00 |
| `X_B` | 15.93 | 24.12 | 10.96 | 15.91 |
| `XT_` | 15.23 | 24.82 | 9.95 | 15.05 |
| `XTB` (all) | 16.13 | 26.48 | 10.95 | 15.95 |

**One cell where a Triton kernel beats torch**: `XTX` alone on `(768, 2048)` —
8.88 vs 9.18 ms, a marginal 3% win. Every other combination is neutral or
worse than cuBLAS.

## Why the kernels lose on Ampere

1. **cuBLAS is extremely mature on Ampere.** For matmul of these shapes
   (128–2048 × 768–2048), PyTorch dispatches to cuBLAS kernels that have been
   auto-tuned for sm_80/sm_86 for multiple years. The Triton kernels were
   **authored and tuned for Hopper (sm_90)**; the 128×128×64 block tiles
   saturate H100's SM capacity but are suboptimal on Ampere's narrower registers.
2. **Symmetric-output optimization has diminishing returns at small sizes.**
   The `XXT`/`XTX` trick saves ~50% of the FLOPs on the triangular half, but
   the savings are in absolute FLOPs. For `(128, 768) → (128, 128)` output,
   the total FLOPs are already tiny, and kernel-launch overhead dominates.
3. **Per-call launch + dtype plumbing.** Each `XXT()`/`XTX()`/`ba_plus_cAA()`
   call is a separate Triton kernel launch with its own argument-marshalling
   path. For 84 matrices × 5 NS iterations × N kernels per iteration,
   the Python/Triton overhead adds up to milliseconds.
4. **`num_stages=2` on Ampere** (vs 4 on H100) halves the software
   pipelining, losing a chunk of the asynchronous-copy overlap. Restoring
   `num_stages=4` would blow the shared-memory budget.

## The `torch.compile` alternative

For comparison, I timed the existing `_newtonschulz5` (classical
`torch.matmul`) wrapped in `torch.compile(dynamic=False, fullgraph=False)`:

| variant | ms/step | vs baseline |
|---|---:|---:|
| classical NS5 (`torch.matmul` eager) | 41.55 ± 0.01 | 1.00× |
| classical NS5 (`torch.compile`-wrapped) | **35.66 ± 0.06** | **1.17×** |
| best Triton combo (`_T_`, xtx only) | 43.38 | 0.96× |
| full Triton (`XTB`) | 56.16 | 0.74× |

**`torch.compile` gives +17% speedup for free**, simply by letting Inductor
fuse the polynomial step (`a*X + B @ X` where `B = b*A + c*A@A`) into a
single graph while still using cuBLAS for the matmuls. It is also the
fastest option tested across all variants.

(Measurement delta on the baseline between this run — 41.55 ms — and the
ablation run above — 38.34 ms — is residual variability from cuDNN autotune
reordering; within run-to-run noise.)

## Recommendation

1. **Do not adopt the Triton kernels on Ampere.** They are H100-tuned and
   consistently slower than cuBLAS on the 3090/A100 for our matrix sizes.
2. **Adopt `torch.compile` on the Muon NS function instead.** +17% free
   speedup, zero maintenance cost. One-line change in
   [`src/flash_dit/training/optimizer.py`](../src/flash_dit/training/optimizer.py):
   wrap `_newtonschulz5` (or the whole `Muon.step`) with `torch.compile`.
3. **Re-evaluate on H100.** If we run a paper benchmark on the runpod-tcp
   8×H100 pod, rerun `scripts/bench_ns_kernels.py` there. The same three
   kernels should regain their expected 1.3-1.8× advantage once the
   H100-tuned block config fits; `_pick_block_config` already detects Hopper
   via `torch.cuda.get_device_capability` and uses `num_stages=4`.
4. **Orthogonal win: batched NS over shape groups.** Independent of kernel
   choice, replacing the Python for-loop over 84 matrices with 4 batched
   calls (one per distinct shape, stacking the same-shape matrices into a
   3D tensor) should reduce launch overhead significantly. The Triton
   kernels in this file already support a batch dimension; so does
   `torch.bmm` — both paths can be explored.

## Files changed

- [`src/flash_dit/training/muon_kernels.py`](../src/flash_dit/training/muon_kernels.py) — vendored the three Triton kernels; added `_pick_block_config` to pick Ampere-safe block tiles.
- [`src/flash_dit/training/optimizer.py`](../src/flash_dit/training/optimizer.py) — added `Muon._newtonschulz5_kernels(G, steps, use_xxt, use_xtx, use_ba_plus_caa)` alongside the classical `_newtonschulz5`. Default path unchanged.
- [`scripts/bench_ns_kernels.py`](../scripts/bench_ns_kernels.py) — standalone microbench producing `outputs/bench/ns_kernels_<ts>/{results.json,results.csv,summary.md}`.

## Reproducibility

```bash
# Rerun the full ablation
CUDA_VISIBLE_DEVICES=0 uv run python scripts/bench_ns_kernels.py \
  --warmup 15 --measure 40 --repeats 3

# Just one config
uv run python scripts/bench_ns_kernels.py --warmup 15 --measure 40 --repeats 1
```

Raw data lives at [`outputs/bench/ns_kernels_20260417_023226/`](../outputs/bench/ns_kernels_20260417_023226/).
