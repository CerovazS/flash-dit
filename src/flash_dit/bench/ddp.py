"""Minimal torchrun-compatible distributed helpers.

The bench runs outside Lightning, so we manage ``torch.distributed`` manually.
The contract:

- If ``RANK``/``WORLD_SIZE``/``LOCAL_RANK`` are set in the environment (as
  torchrun does) we initialise a process group and pin the local rank.
- Otherwise we stay single-process: every helper returns sensible defaults
  (world_size=1, is_global_zero=True, ``maybe_wrap_ddp`` is a no-op).

This means the *same* ``scripts/bench.py`` works for:
    uv run python scripts/bench.py ...
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/bench.py ...
    torchrun --nproc-per-node=4 scripts/bench.py ...
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except ValueError:
        return default


def setup_dist(backend: str = "nccl") -> dict:
    """Initialise ``torch.distributed`` from torchrun env vars.

    Returns a dict with ``world_size``, ``rank``, ``local_rank``, ``device``.
    Always returns a usable dict even when running single-process.
    """
    rank = _env_int("RANK", 0)
    world = _env_int("WORLD_SIZE", 1)
    local_rank = _env_int("LOCAL_RANK", 0)

    if world > 1 and not dist.is_initialized():
        # nccl requires CUDA; gloo works on CPU for tests.
        chosen_backend = backend if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=chosen_backend, init_method="env://")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return {
        "world_size": world,
        "rank": rank,
        "local_rank": local_rank,
        "device": device,
    }


def cleanup_dist() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_global_zero() -> bool:
    return (not is_dist()) or dist.get_rank() == 0


def barrier() -> None:
    if is_dist():
        dist.barrier()


def maybe_wrap_ddp(
    model: torch.nn.Module,
    local_rank: int,
    find_unused_parameters: bool = False,
) -> torch.nn.Module:
    """Wrap the model with DDP when world_size>1, else return as-is."""
    if not is_dist() or world_size() == 1:
        return model
    from torch.nn.parallel import DistributedDataParallel as DDP

    device_ids = [local_rank] if torch.cuda.is_available() else None
    return DDP(model, device_ids=device_ids, find_unused_parameters=find_unused_parameters)
