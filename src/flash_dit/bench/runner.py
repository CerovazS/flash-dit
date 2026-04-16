"""Benchmark runners — where the actual measurement happens.

Two runner classes:

- :class:`ThroughputRunner` exercises the full ``forward + backward + optimizer``
  pipeline on a :class:`DiffusionTransformer` via :func:`flow_matching_loss`.
  This is what matches a training iteration.

- :class:`MicroRunner` isolates a sub-module (attention kernel, MLP, or a
  single :class:`DiTBlock`) and runs forward+backward on it with random
  inputs, so architecture changes can be profiled without the noise of
  the full model.

Both return a :class:`BenchResult`. Output contract lives in
``reporting.py``; we intentionally keep result construction here so the
runner fully owns timing semantics.
"""
from __future__ import annotations

import statistics
import subprocess
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from ..diffusion.flow_matching import flow_matching_loss
from .core import (
    make_timer,
    memory_snapshot,
    reset_peak_memory,
    resolve_dtype,
    sync_device,
    synthetic_batch,
)
from .ddp import is_global_zero, world_size


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    """Serialisable record of a single bench run.

    All ``_ms`` fields are averages over measure_steps × repeats.
    ``step_per_s_{mean,std}`` reflect cross-repeat variability (the most
    honest std for a benchmark: within a repeat, steps are IID on the same
    kernel cache; across repeats you also capture cuDNN heuristics drift).
    """

    mode: str
    model_name: str
    attention_type: str
    use_compile: bool
    use_fa3: bool
    batch_size: int
    seq_len: int
    in_channels: int
    n_classes: int
    precision: str
    device: str
    world_size: int
    warmup_steps: int
    measure_steps: int
    repeats: int
    include_optimizer: bool
    include_backward: bool
    step_per_s_mean: float
    step_per_s_std: float
    samples_per_s: float
    tokens_per_s: float
    fwd_ms_mean: float
    bwd_ms_mean: float
    opt_ms_mean: float
    peak_vram_mb: float
    total_wall_s: float
    trainable_params: int
    total_params: int
    git_sha: str
    torch_version: str
    timestamp: str
    extra: dict[str, Any] = field(default_factory=dict)


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False, timeout=2,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Throughput runner
# ---------------------------------------------------------------------------


class ThroughputRunner:
    """Full training-step throughput: forward → backward → optimizer.step."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        cfg: DictConfig,
        device: torch.device,
        model_meta: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg
        self.device = device
        self.model_meta = model_meta or {}

        self._model_for_params = model.module if hasattr(model, "module") else model
        amp_dtype = resolve_dtype(cfg.precision) if cfg.precision in {"bf16", "fp16"} else None
        # Only autocast on CUDA; CPU tests fall through to fp32.
        self._amp_ctx = (
            (lambda: torch.autocast(device_type="cuda", dtype=amp_dtype))
            if (amp_dtype is not None and self.device.type == "cuda")
            else (lambda: nullcontext())
        )

    # One full training step; returns the (fwd, bwd, opt) timers.
    def _step(self) -> tuple[Any, Any, Any]:
        x, y = synthetic_batch(
            self.cfg.batch_size,
            self.cfg.in_channels,
            self.cfg.seq_len,
            self.cfg.n_classes,
            self.device,
            dtype=torch.float32,
        )

        t_fwd = make_timer(self.device)
        with t_fwd, self._amp_ctx():
            loss = flow_matching_loss(self.model, x, y)

        t_bwd = make_timer(self.device)
        if self.cfg.include_backward:
            with t_bwd:
                loss.backward()
        else:
            # Still enter/exit timer so elapsed_ms is defined.
            with t_bwd:
                pass

        t_opt = make_timer(self.device)
        if self.optimizer is not None and self.cfg.include_optimizer:
            with t_opt:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
        else:
            with t_opt:
                pass

        return t_fwd, t_bwd, t_opt

    def run(self) -> BenchResult:
        # Warmup (absorbs torch.compile build + cuDNN autotune).
        for _ in range(int(self.cfg.warmup_steps)):
            self._step()
        sync_device(self.device)
        reset_peak_memory(self.device)

        fwd_ms, bwd_ms, opt_ms = [], [], []
        per_repeat_sps: list[float] = []

        wall_start = time.perf_counter()
        for _ in range(int(self.cfg.repeats)):
            rep_timers: list[tuple[Any, Any, Any]] = []
            rep_start = time.perf_counter()
            for _ in range(int(self.cfg.measure_steps)):
                rep_timers.append(self._step())
            sync_device(self.device)  # finalise all pending events before reading
            rep_end = time.perf_counter()

            per_repeat_sps.append(int(self.cfg.measure_steps) / (rep_end - rep_start))
            for tf, tb, to in rep_timers:
                fwd_ms.append(tf.elapsed_ms)
                bwd_ms.append(tb.elapsed_ms)
                opt_ms.append(to.elapsed_ms)
        wall_end = time.perf_counter()

        peak_vram_mb = memory_snapshot(self.device)["peak_mb"]
        step_per_s_mean = statistics.mean(per_repeat_sps)
        step_per_s_std = statistics.stdev(per_repeat_sps) if len(per_repeat_sps) > 1 else 0.0
        ws = world_size()

        trainable = sum(p.numel() for p in self._model_for_params.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._model_for_params.parameters())

        return BenchResult(
            mode=self.cfg.mode,
            model_name=self.model_meta.get("name", "unknown"),
            attention_type=self.model_meta.get("attention_type", "unknown"),
            use_compile=bool(self.model_meta.get("use_compile", False)),
            use_fa3=bool(self.model_meta.get("use_fa3", False)),
            batch_size=int(self.cfg.batch_size),
            seq_len=int(self.cfg.seq_len),
            in_channels=int(self.cfg.in_channels),
            n_classes=int(self.cfg.n_classes),
            precision=str(self.cfg.precision),
            device=str(self.device),
            world_size=ws,
            warmup_steps=int(self.cfg.warmup_steps),
            measure_steps=int(self.cfg.measure_steps),
            repeats=int(self.cfg.repeats),
            include_optimizer=bool(self.cfg.include_optimizer),
            include_backward=bool(self.cfg.include_backward),
            step_per_s_mean=step_per_s_mean,
            step_per_s_std=step_per_s_std,
            samples_per_s=step_per_s_mean * int(self.cfg.batch_size) * ws,
            tokens_per_s=(
                step_per_s_mean * int(self.cfg.batch_size) * ws
                * int(self.cfg.seq_len) * int(self.cfg.in_channels)
            ),
            fwd_ms_mean=statistics.mean(fwd_ms),
            bwd_ms_mean=statistics.mean(bwd_ms),
            opt_ms_mean=statistics.mean(opt_ms),
            peak_vram_mb=peak_vram_mb,
            total_wall_s=wall_end - wall_start,
            trainable_params=trainable,
            total_params=total,
            git_sha=_git_sha(),
            torch_version=torch.__version__,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            extra={
                "per_repeat_step_per_s": per_repeat_sps,
            },
        )


# ---------------------------------------------------------------------------
# Microbench runners
# ---------------------------------------------------------------------------


def _build_attention_kernel(
    model_cfg: DictConfig, device: torch.device,
) -> tuple[torch.nn.Module, int, int, int]:
    """Return (attention_module, n_heads, n_kv_heads, head_dim) from a model config."""
    from ..models.attention import build_attention
    from ..models.embeddings import RotaryEmbedding

    d_model = int(model_cfg.d_model)
    n_heads = int(model_cfg.n_heads)
    n_kv_heads = int(model_cfg.get("n_kv_heads", n_heads))
    head_dim = d_model // n_heads
    rope = RotaryEmbedding(head_dim=head_dim, max_seq_len=int(model_cfg.get("max_seq_len", 256)))
    attn = build_attention(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        rope=rope,
        attention_type=str(model_cfg.attention_type),
        dropout=0.0,
        use_fa3=bool(model_cfg.get("use_fa3", False)),
    ).to(device)
    return attn, n_heads, n_kv_heads, d_model


def _build_mlp(model_cfg: DictConfig, device: torch.device) -> torch.nn.Module:
    from ..models.mlp import build_mlp

    return build_mlp(
        int(model_cfg.d_model), float(model_cfg.mlp_mult), str(model_cfg.mlp_type),
    ).to(device)


def _build_dit_block(model_cfg: DictConfig, device: torch.device) -> torch.nn.Module:
    from ..models.dit import DiTBlock
    from ..models.embeddings import RotaryEmbedding

    d_model = int(model_cfg.d_model)
    n_heads = int(model_cfg.n_heads)
    head_dim = d_model // n_heads
    rope = RotaryEmbedding(head_dim=head_dim, max_seq_len=int(model_cfg.get("max_seq_len", 256)))
    block = DiTBlock(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=int(model_cfg.get("n_kv_heads", n_heads)),
        mlp_mult=float(model_cfg.mlp_mult),
        attention_type=str(model_cfg.attention_type),
        mlp_type=str(model_cfg.mlp_type),
        rope=rope,
        dropout=0.0,
        use_fa3=bool(model_cfg.get("use_fa3", False)),
    ).to(device)
    return block


class MicroRunner:
    """Isolated sub-module benchmarks.

    ``mode`` selects which sub-module to exercise:
    - ``microbench_attention``: full attention module (proj + RoPE + attn + out_proj)
    - ``microbench_mlp``: MLP only
    - ``microbench_block``: full DiTBlock (norms + attn + MLP + AdaLN)

    Inputs are random tensors of the right shape on ``device``. Backward uses
    a random upstream gradient so we stress both directions without loss
    computation overhead.
    """

    def __init__(
        self,
        model_cfg: DictConfig,
        bench_cfg: DictConfig,
        device: torch.device,
    ) -> None:
        self.model_cfg = model_cfg
        self.cfg = bench_cfg
        self.device = device
        self.mode = str(bench_cfg.mode)
        amp_dtype = resolve_dtype(bench_cfg.precision) if bench_cfg.precision in {"bf16", "fp16"} else None
        self._amp_ctx = (
            (lambda: torch.autocast(device_type="cuda", dtype=amp_dtype))
            if (amp_dtype is not None and self.device.type == "cuda")
            else (lambda: nullcontext())
        )
        self.module, self._input_maker = self._build()

    # Build (module, input_maker) pair.
    # input_maker() returns (args_tuple, upstream_grad_like) used by _step.
    def _build(self):
        if self.mode == "microbench_attention":
            attn, *_ = _build_attention_kernel(self.model_cfg, self.device)

            def input_maker():
                d = int(self.model_cfg.d_model)
                x = torch.randn(
                    (int(self.cfg.batch_size), int(self.cfg.seq_len), d),
                    device=self.device, requires_grad=True,
                )
                return (x,), torch.randn_like(x)

            return attn, input_maker

        if self.mode == "microbench_mlp":
            mlp = _build_mlp(self.model_cfg, self.device)

            def input_maker():
                d = int(self.model_cfg.d_model)
                x = torch.randn(
                    (int(self.cfg.batch_size), int(self.cfg.seq_len), d),
                    device=self.device, requires_grad=True,
                )
                return (x,), torch.randn_like(x)

            return mlp, input_maker

        if self.mode == "microbench_block":
            block = _build_dit_block(self.model_cfg, self.device)

            def input_maker():
                d = int(self.model_cfg.d_model)
                x = torch.randn(
                    (int(self.cfg.batch_size), int(self.cfg.seq_len), d),
                    device=self.device, requires_grad=True,
                )
                c = torch.randn((int(self.cfg.batch_size), d), device=self.device)
                return (x, c), torch.randn_like(x)

            return block, input_maker

        raise ValueError(f"unknown microbench mode {self.mode!r}")

    def _step(self) -> tuple[Any, Any, Any]:
        args, grad_out = self._input_maker()

        t_fwd = make_timer(self.device)
        with t_fwd, self._amp_ctx():
            out = self.module(*args)

        t_bwd = make_timer(self.device)
        if self.cfg.include_backward:
            with t_bwd:
                out.backward(grad_out)
                for a in args:
                    if isinstance(a, torch.Tensor) and a.grad is not None:
                        a.grad = None
        else:
            with t_bwd:
                pass

        t_opt = make_timer(self.device)  # no optimizer step in microbench
        with t_opt:
            pass

        return t_fwd, t_bwd, t_opt

    def run(self) -> BenchResult:
        for _ in range(int(self.cfg.warmup_steps)):
            self._step()
        sync_device(self.device)
        reset_peak_memory(self.device)

        fwd_ms, bwd_ms, opt_ms = [], [], []
        per_repeat_sps: list[float] = []
        wall_start = time.perf_counter()
        for _ in range(int(self.cfg.repeats)):
            rep_timers = []
            rep_start = time.perf_counter()
            for _ in range(int(self.cfg.measure_steps)):
                rep_timers.append(self._step())
            sync_device(self.device)
            rep_end = time.perf_counter()
            per_repeat_sps.append(int(self.cfg.measure_steps) / (rep_end - rep_start))
            for tf, tb, to in rep_timers:
                fwd_ms.append(tf.elapsed_ms)
                bwd_ms.append(tb.elapsed_ms)
                opt_ms.append(to.elapsed_ms)
        wall_end = time.perf_counter()

        trainable = sum(p.numel() for p in self.module.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.module.parameters())
        step_per_s_mean = statistics.mean(per_repeat_sps)
        step_per_s_std = statistics.stdev(per_repeat_sps) if len(per_repeat_sps) > 1 else 0.0
        ws = world_size()

        return BenchResult(
            mode=self.mode,
            model_name=str(self.model_cfg.get("name", "unknown")),
            attention_type=str(self.model_cfg.attention_type),
            use_compile=bool(self.model_cfg.get("use_compile", False)),
            use_fa3=bool(self.model_cfg.get("use_fa3", False)),
            batch_size=int(self.cfg.batch_size),
            seq_len=int(self.cfg.seq_len),
            in_channels=int(self.model_cfg.d_model),  # feature dim for micro
            n_classes=0,
            precision=str(self.cfg.precision),
            device=str(self.device),
            world_size=ws,
            warmup_steps=int(self.cfg.warmup_steps),
            measure_steps=int(self.cfg.measure_steps),
            repeats=int(self.cfg.repeats),
            include_optimizer=False,
            include_backward=bool(self.cfg.include_backward),
            step_per_s_mean=step_per_s_mean,
            step_per_s_std=step_per_s_std,
            samples_per_s=step_per_s_mean * int(self.cfg.batch_size) * ws,
            tokens_per_s=(
                step_per_s_mean * int(self.cfg.batch_size) * ws * int(self.cfg.seq_len)
            ),
            fwd_ms_mean=statistics.mean(fwd_ms),
            bwd_ms_mean=statistics.mean(bwd_ms),
            opt_ms_mean=statistics.mean(opt_ms),
            peak_vram_mb=memory_snapshot(self.device)["peak_mb"],
            total_wall_s=wall_end - wall_start,
            trainable_params=trainable,
            total_params=total,
            git_sha=_git_sha(),
            torch_version=torch.__version__,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            extra={
                "per_repeat_step_per_s": per_repeat_sps,
                "sub_module": type(self.module).__name__,
            },
        )
