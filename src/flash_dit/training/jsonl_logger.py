"""Local JSONL metrics callback."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import lightning as L
import torch


class JSONLMetricsCallback(L.Callback):
    """Append scalar metrics to a local JSONL file during training.

    This complements Lightning's CSVLogger with append-only, machine-readable
    records that include wall-clock time for throughput analysis.
    """

    def __init__(self, output_dir: str | Path, every_n_train_steps: int = 50) -> None:
        self.output_dir = Path(output_dir)
        self.every_n_train_steps = every_n_train_steps
        self.path = self.output_dir / "jsonl" / "metrics.jsonl"
        self._start_time: float | None = None

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        self._start_time = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_record(trainer, "fit_start", {})

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if not trainer.is_global_zero:
            return
        step = int(trainer.global_step)
        if step == 0 or step % self.every_n_train_steps != 0:
            return

        metrics = self._collect_metrics(trainer)
        metrics.update(self._collect_lrs(trainer))
        self._write_record(trainer, "train_batch_end", metrics)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        metrics = self._collect_metrics(trainer)
        metrics.update(self._collect_lrs(trainer))
        self._write_record(trainer, "validation_epoch_end", metrics)

    def _write_record(self, trainer: L.Trainer, event: str, metrics: dict[str, Any]) -> None:
        now = time.time()
        start = self._start_time if self._start_time is not None else now
        record = {
            "event": event,
            "time": now,
            "elapsed_sec": now - start,
            "epoch": int(trainer.current_epoch),
            "step": int(trainer.global_step),
            **metrics,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    @staticmethod
    def _collect_metrics(trainer: L.Trainer) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for key, value in trainer.callback_metrics.items():
            scalar = _to_json_scalar(value)
            if scalar is not None:
                metrics[str(key)] = scalar
        return metrics

    @staticmethod
    def _collect_lrs(trainer: L.Trainer) -> dict[str, float]:
        lrs: dict[str, float] = {}
        for opt in trainer.optimizers:
            name = type(opt).__name__
            if len(opt.param_groups) == 1:
                lrs[f"lr-{name}"] = float(opt.param_groups[0]["lr"])
            else:
                for idx, group in enumerate(opt.param_groups, start=1):
                    lrs[f"lr-{name}/pg{idx}"] = float(group["lr"])
        return lrs


def _to_json_scalar(value: Any) -> int | float | bool | str | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value
    return None
