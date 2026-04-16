"""Primitive building blocks used by every bench runner.

All helpers are side-effect-light and GPU/CPU dual: on CUDA we use
``torch.cuda.Event`` for accurate timing; on CPU we fall back to
``time.perf_counter``. Keeping the APIs identical lets tests exercise
the full pipeline without a GPU.
"""
from __future__ import annotations

import os
import random
import time
from typing import Union

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


class CudaEventTimer:
    """Records a (start, end) CUDA event pair without synchronising.

    Call :pyattr:`elapsed_ms` *after* an explicit ``torch.cuda.synchronize()``
    has been issued downstream; reading earlier raises because the events
    have not completed yet. This deferred pattern is key for accurate
    throughput: we never force a sync inside the hot loop.
    """

    __slots__ = ("device", "start", "end")

    def __init__(self, device: torch.device | str = "cuda") -> None:
        self.device = torch.device(device)
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self) -> "CudaEventTimer":
        self.start.record()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.end.record()

    @property
    def elapsed_ms(self) -> float:
        return float(self.start.elapsed_time(self.end))


class CpuTimer:
    """``time.perf_counter``-based fallback with the same context API."""

    __slots__ = ("_t0", "_elapsed_ms")

    def __init__(self) -> None:
        self._t0 = 0.0
        self._elapsed_ms = 0.0

    def __enter__(self) -> "CpuTimer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._elapsed_ms = (time.perf_counter() - self._t0) * 1000.0

    @property
    def elapsed_ms(self) -> float:
        return self._elapsed_ms


def make_timer(device: torch.device | str):
    """Factory: CUDA event timer when ``device`` is CUDA, else CPU timer."""
    d = torch.device(device)
    if d.type == "cuda":
        return CudaEventTimer(d)
    return CpuTimer()


def sync_device(device: torch.device | str) -> None:
    """Synchronise the given device if CUDA; no-op otherwise."""
    d = torch.device(device)
    if d.type == "cuda":
        torch.cuda.synchronize(d)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def memory_snapshot(device: torch.device | str = "cuda") -> dict:
    """Return a dict of CUDA memory usage in MB. Zeros on CPU."""
    d = torch.device(device)
    if d.type != "cuda":
        return {"peak_mb": 0.0, "current_mb": 0.0, "reserved_mb": 0.0}
    return {
        "peak_mb": torch.cuda.max_memory_allocated(d) / 1e6,
        "current_mb": torch.cuda.memory_allocated(d) / 1e6,
        "reserved_mb": torch.cuda.memory_reserved(d) / 1e6,
    }


def reset_peak_memory(device: torch.device | str = "cuda") -> None:
    d = torch.device(device)
    if d.type == "cuda":
        torch.cuda.reset_peak_memory_stats(d)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


DType = Union[torch.dtype, str]
_DTYPE_MAP: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
}


def resolve_dtype(dtype: DType) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype in _DTYPE_MAP:
        return _DTYPE_MAP[dtype]
    raise ValueError(f"unknown dtype {dtype!r}; expected one of {list(_DTYPE_MAP)} or a torch.dtype")


def synthetic_batch(
    batch_size: int,
    in_channels: int,
    seq_len: int,
    n_classes: int,
    device: torch.device | str,
    dtype: DType = torch.float32,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a random ``(x, y)`` batch matching the DiT training signature.

    - ``x``: ``(B, C, T)`` normalised latent-shaped tensor. Always drawn on
      CPU then moved to ``device`` (deterministic across CPU/GPU for a fixed
      generator seed).
    - ``y``: ``(B,)`` int64 class labels in ``[0, n_classes)``.
    """
    d = torch.device(device)
    torch_dtype = resolve_dtype(dtype)

    # CPU generation keeps the bench portable (no cuda RNG dependency) and
    # lets us share a single seed across ranks when wanted.
    x = torch.randn((batch_size, in_channels, seq_len), generator=generator)
    y = torch.randint(0, n_classes, (batch_size,), generator=generator)

    return x.to(device=d, dtype=torch_dtype, non_blocking=True), y.to(device=d, non_blocking=True)


# ---------------------------------------------------------------------------
# Determinism / repro
# ---------------------------------------------------------------------------


def set_determinism(seed: int, cudnn_benchmark: bool = True) -> None:
    """Seed every RNG we touch. ``cudnn_benchmark=True`` keeps kernel auto-tune
    enabled — benchmarks SHOULD see the fastest kernel cuDNN picks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
