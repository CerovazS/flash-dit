"""Unit tests for flash_dit.bench.core primitives."""
from __future__ import annotations

import torch
import pytest

from flash_dit.bench.core import (
    CpuTimer,
    make_timer,
    memory_snapshot,
    reset_peak_memory,
    resolve_dtype,
    set_determinism,
    sync_device,
    synthetic_batch,
)


def test_synthetic_batch_shape_and_dtype_cpu():
    x, y = synthetic_batch(batch_size=4, in_channels=8, seq_len=16, n_classes=7, device="cpu", dtype=torch.float32)
    assert x.shape == (4, 8, 16)
    assert x.dtype == torch.float32
    assert y.shape == (4,)
    assert y.dtype == torch.int64
    assert y.min() >= 0 and y.max() < 7


def test_synthetic_batch_accepts_string_dtype():
    x, _ = synthetic_batch(2, 4, 8, 3, "cpu", dtype="bf16")
    assert x.dtype == torch.bfloat16


def test_synthetic_batch_deterministic_with_seed():
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    x1, y1 = synthetic_batch(3, 4, 5, 3, "cpu", generator=g1)
    x2, y2 = synthetic_batch(3, 4, 5, 3, "cpu", generator=g2)
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


def test_resolve_dtype_strings():
    assert resolve_dtype("fp32") == torch.float32
    assert resolve_dtype("bf16") == torch.bfloat16
    assert resolve_dtype("fp16") == torch.float16
    assert resolve_dtype(torch.float64) == torch.float64
    with pytest.raises(ValueError):
        resolve_dtype("int4")


def test_cpu_timer_context_ok():
    t = CpuTimer()
    with t:
        _ = sum(range(1000))
    assert t.elapsed_ms >= 0.0


def test_make_timer_returns_cpu_on_cpu():
    t = make_timer(torch.device("cpu"))
    assert isinstance(t, CpuTimer)


def test_memory_snapshot_cpu_is_zero():
    snap = memory_snapshot(torch.device("cpu"))
    assert snap == {"peak_mb": 0.0, "current_mb": 0.0, "reserved_mb": 0.0}


def test_reset_peak_memory_cpu_is_noop():
    # Should not raise.
    reset_peak_memory(torch.device("cpu"))


def test_sync_device_cpu_is_noop():
    # Should not raise.
    sync_device(torch.device("cpu"))


def test_set_determinism_reproducible_randn():
    set_determinism(1234)
    a = torch.randn(10)
    set_determinism(1234)
    b = torch.randn(10)
    assert torch.equal(a, b)
