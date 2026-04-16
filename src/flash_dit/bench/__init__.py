"""Benchmarking suite for paper-grade throughput and kernel ablations.

Designed to run without Lightning so the numbers reflect the raw model cost
— no callback/hook overhead. Sweepable through Hydra, DDP-enabled through
``torchrun``. See ``conf/bench/*`` for ready-made profiles.
"""
from .core import (
    CudaEventTimer,
    CpuTimer,
    make_timer,
    memory_snapshot,
    reset_peak_memory,
    set_determinism,
    sync_device,
    synthetic_batch,
)
from .ddp import (
    barrier,
    cleanup_dist,
    is_dist,
    is_global_zero,
    maybe_wrap_ddp,
    setup_dist,
    world_size,
)
from .reporting import BenchResult, aggregate_markdown, write_result
from .runner import MicroRunner, ThroughputRunner

__all__ = [
    # core
    "CudaEventTimer",
    "CpuTimer",
    "make_timer",
    "memory_snapshot",
    "reset_peak_memory",
    "set_determinism",
    "sync_device",
    "synthetic_batch",
    # ddp
    "barrier",
    "cleanup_dist",
    "is_dist",
    "is_global_zero",
    "maybe_wrap_ddp",
    "setup_dist",
    "world_size",
    # results
    "BenchResult",
    "aggregate_markdown",
    "write_result",
    # runners
    "MicroRunner",
    "ThroughputRunner",
]
