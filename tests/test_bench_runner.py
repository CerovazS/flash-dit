"""Runners tests on CPU with a tiny DiT (fast, no GPU required)."""
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from flash_dit.bench.runner import MicroRunner, ThroughputRunner
from flash_dit.models.dit import DiffusionTransformer


@pytest.fixture()
def tiny_bench_cfg():
    return OmegaConf.create({
        "mode": "throughput_step",
        "warmup_steps": 1,
        "measure_steps": 2,
        "repeats": 2,
        "batch_size": 2,
        "seq_len": 8,
        "in_channels": 8,
        "n_classes": 4,
        "include_optimizer": True,
        "include_backward": True,
        "precision": "fp32",
    })


@pytest.fixture()
def tiny_model_cfg():
    # Keeps FA2 off by choosing MHA + CPU; tiny dims keep tests < 1 s.
    return OmegaConf.create({
        "name": "tiny",
        "d_model": 32,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 4,
        "mlp_mult": 2.0,
        "attention_type": "mha",
        "mlp_type": "relu2",
        "use_compile": False,
        "use_fa3": False,
        "max_seq_len": 64,
    })


def _build_tiny_dit(in_channels: int, n_classes: int, model_cfg) -> DiffusionTransformer:
    return DiffusionTransformer(
        in_channels=in_channels,
        d_model=model_cfg.d_model,
        n_layers=model_cfg.n_layers,
        n_heads=model_cfg.n_heads,
        n_kv_heads=model_cfg.n_kv_heads,
        mlp_mult=model_cfg.mlp_mult,
        n_classes=n_classes,
        attention_type=model_cfg.attention_type,
        mlp_type=model_cfg.mlp_type,
        use_fa3=False,
        use_compile=False,
        max_seq_len=model_cfg.max_seq_len,
    )


def test_throughput_runner_cpu_produces_result(tiny_bench_cfg, tiny_model_cfg):
    model = _build_tiny_dit(tiny_bench_cfg.in_channels, tiny_bench_cfg.n_classes, tiny_model_cfg)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    runner = ThroughputRunner(
        model=model, optimizer=optim, cfg=tiny_bench_cfg, device=torch.device("cpu"),
        model_meta={"name": "tiny", "attention_type": "mha", "use_compile": False, "use_fa3": False},
    )
    res = runner.run()
    assert res.mode == "throughput_step"
    assert res.step_per_s_mean > 0
    assert res.samples_per_s > 0
    assert res.fwd_ms_mean > 0
    assert res.bwd_ms_mean > 0
    assert res.trainable_params > 0
    assert res.total_wall_s > 0
    # CPU fallback: peak_vram_mb must be 0.
    assert res.peak_vram_mb == 0.0


def test_throughput_runner_no_optimizer_no_backward(tiny_bench_cfg, tiny_model_cfg):
    cfg = OmegaConf.merge(tiny_bench_cfg, {"include_optimizer": False, "include_backward": False})
    model = _build_tiny_dit(cfg.in_channels, cfg.n_classes, tiny_model_cfg)
    runner = ThroughputRunner(
        model=model, optimizer=None, cfg=cfg, device=torch.device("cpu"),
        model_meta={"name": "tiny", "attention_type": "mha", "use_compile": False, "use_fa3": False},
    )
    res = runner.run()
    # bwd and opt timers still present but ≈ 0 (no work).
    assert res.fwd_ms_mean > 0
    # tolerate tiny CPU noise on the "empty" timers
    assert res.bwd_ms_mean >= 0.0
    assert res.opt_ms_mean >= 0.0


@pytest.mark.parametrize("mode", ["microbench_attention", "microbench_mlp", "microbench_block"])
def test_microrunner_modes_cpu(mode, tiny_bench_cfg, tiny_model_cfg):
    cfg = OmegaConf.merge(tiny_bench_cfg, {"mode": mode, "include_optimizer": False})
    runner = MicroRunner(model_cfg=tiny_model_cfg, bench_cfg=cfg, device=torch.device("cpu"))
    res = runner.run()
    assert res.mode == mode
    assert res.step_per_s_mean > 0
    assert res.fwd_ms_mean > 0
    if cfg.include_backward:
        assert res.bwd_ms_mean > 0


def test_microrunner_rejects_bad_mode(tiny_bench_cfg, tiny_model_cfg):
    cfg = OmegaConf.merge(tiny_bench_cfg, {"mode": "microbench_unknown"})
    with pytest.raises(ValueError):
        MicroRunner(model_cfg=tiny_model_cfg, bench_cfg=cfg, device=torch.device("cpu"))
