#!/usr/bin/env python
"""Throughput / microbench entry point for flash-dit.

Examples:

    # Single GPU, default config (flash_dit + throughput_step)
    uv run python scripts/bench.py

    # GPU pinning + different model
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/bench.py model=vanilla_sit

    # Hydra multirun sweep
    uv run python scripts/bench.py -m model=flash_dit,vanilla_sit bench.batch_size=32,64,128

    # DDP on 4 GPUs (local, torchrun)
    torchrun --standalone --nproc-per-node=4 scripts/bench.py

    # Microbenchmark (attention kernel only)
    uv run python scripts/bench.py bench=microbench_attention
"""
from __future__ import annotations

import os

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from flash_dit.bench import (
    MicroRunner,
    ThroughputRunner,
    cleanup_dist,
    is_global_zero,
    maybe_wrap_ddp,
    set_determinism,
    setup_dist,
    write_result,
)
from flash_dit.training.optimizer import build_optimizer
from flash_dit.utils.console import info, ok


@hydra.main(version_base=None, config_path="../conf", config_name="bench_config")
def main(cfg: DictConfig) -> None:
    dist = setup_dist()
    set_determinism(int(cfg.seed))

    device: torch.device = dist["device"]
    if is_global_zero():
        info(f"device={device}  world_size={dist['world_size']}  mode={cfg.bench.mode}")

    if str(cfg.bench.mode).startswith("microbench"):
        # Microbench builds the sub-module itself from the model config; it
        # doesn't need a full DiT instantiation or an optimizer.
        runner = MicroRunner(cfg.model, cfg.bench, device=device)
    else:
        model = instantiate(cfg.model).to(device)
        model = maybe_wrap_ddp(model, dist["local_rank"])

        optimizer = None
        if bool(cfg.bench.include_optimizer):
            # build_optimizer returns a list (Muon+AdamW or AdamW only); for
            # the bench we step all of them together.
            opts = build_optimizer(
                model.module if hasattr(model, "module") else model,
                optimizer_type=str(cfg.optimizer.type),
                lr_muon=float(cfg.optimizer.get("lr_muon", 0.02)),
                lr_adamw=float(cfg.optimizer.get("lr_adamw", 3e-4)),
                weight_decay=float(cfg.optimizer.get("weight_decay", 0.01)),
                momentum=float(cfg.optimizer.get("momentum", 0.95)),
                ns_steps=int(cfg.optimizer.get("ns_steps", 5)),
            )
            optimizer = _ComposedOptimizer(opts)

        model_meta = {
            "name": str(cfg.model.get("name", "unknown")),
            "attention_type": str(cfg.model.get("attention_type", "unknown")),
            "use_compile": bool(cfg.model.get("use_compile", False)),
            "use_fa3": bool(cfg.model.get("use_fa3", False)),
        }
        runner = ThroughputRunner(
            model=model,
            optimizer=optimizer,
            cfg=cfg.bench,
            device=device,
            model_meta=model_meta,
        )

    result = runner.run()

    if is_global_zero():
        out_dir = cfg.output_dir
        os.makedirs(out_dir, exist_ok=True)
        # Save the full resolved config next to the result for repro.
        with open(os.path.join(out_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))
        json_path, csv_path = write_result(result, out_dir)
        ok(
            f"{result.mode}  samples/s={result.samples_per_s:.1f}  "
            f"step/s={result.step_per_s_mean:.2f}±{result.step_per_s_std:.2f}  "
            f"fwd={result.fwd_ms_mean:.2f}ms bwd={result.bwd_ms_mean:.2f}ms opt={result.opt_ms_mean:.2f}ms  "
            f"VRAM={result.peak_vram_mb:.0f}MB"
        )
        info(f"wrote: {json_path}  {csv_path}")

    cleanup_dist()


class _ComposedOptimizer:
    """Treat a list of torch.optim.Optimizer as a single optimizer.

    build_optimizer can return Muon+AdamW together. The bench runner only
    wants a single ``.step()`` / ``.zero_grad()`` interface, so we chain them.
    """

    def __init__(self, opts: list[torch.optim.Optimizer]) -> None:
        self._opts = opts

    def step(self) -> None:
        for o in self._opts:
            o.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for o in self._opts:
            o.zero_grad(set_to_none=set_to_none)


if __name__ == "__main__":
    main()
