"""Write + aggregate round-trip for bench results."""
from __future__ import annotations

import csv
import json

import pytest

from flash_dit.bench.reporting import aggregate_markdown, write_result
from flash_dit.bench.runner import BenchResult


def _fake_result(**overrides) -> BenchResult:
    defaults = dict(
        mode="throughput_step",
        model_name="flash_dit",
        attention_type="gqa",
        use_compile=True,
        use_fa3=False,
        batch_size=32,
        seq_len=172,
        in_channels=64,
        n_classes=17,
        precision="bf16",
        device="cuda:0",
        world_size=1,
        warmup_steps=10,
        measure_steps=50,
        repeats=3,
        include_optimizer=True,
        include_backward=True,
        step_per_s_mean=5.8,
        step_per_s_std=0.12,
        samples_per_s=185.6,
        tokens_per_s=2_044_979.0,
        fwd_ms_mean=60.0,
        bwd_ms_mean=100.0,
        opt_ms_mean=10.0,
        peak_vram_mb=1500.0,
        total_wall_s=25.0,
        trainable_params=12_345_678,
        total_params=12_345_678,
        git_sha="deadbee",
        torch_version="2.5.1+cu121",
        timestamp="2026-04-17T00:00:00Z",
    )
    defaults.update(overrides)
    return BenchResult(**defaults)


def test_write_result_creates_json_and_csv(tmp_path):
    r = _fake_result()
    json_path, csv_path = write_result(r, tmp_path / "run01")
    assert json_path.exists() and csv_path.exists()

    data = json.loads(json_path.read_text())
    assert data["samples_per_s"] == pytest.approx(185.6)
    assert data["mode"] == "throughput_step"

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert float(rows[0]["samples_per_s"]) == pytest.approx(185.6)


def test_aggregate_markdown_sorts_and_writes_summary(tmp_path):
    write_result(_fake_result(model_name="flash_dit", samples_per_s=200.0), tmp_path / "a")
    write_result(_fake_result(model_name="vanilla_sit", samples_per_s=120.0), tmp_path / "b")
    write_result(_fake_result(model_name="flash_dit_recurse", samples_per_s=90.0), tmp_path / "c")

    md = aggregate_markdown(tmp_path)
    assert "flash_dit" in md
    assert "vanilla_sit" in md

    # Highest samples/s should appear before the lowest.
    assert md.index("flash_dit") < md.index("vanilla_sit")
    assert md.index("vanilla_sit") < md.index("flash_dit_recurse")

    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "summary.md").exists()


def test_aggregate_markdown_empty_dir(tmp_path):
    md = aggregate_markdown(tmp_path)
    assert "No result.csv files found" in md
