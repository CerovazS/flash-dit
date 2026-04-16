"""Serialise :class:`BenchResult` to disk and aggregate many of them.

File layout per run (one run == one BenchResult):

    <out_dir>/
      result.json   # full dataclass incl. extra.per_repeat_step_per_s
      result.csv    # single-row CSV with the same scalar fields

``aggregate_markdown(root)`` walks ``root`` recursively, loads every
``result.csv``, and produces a single combined CSV + a human-readable
markdown summary ranked by ``samples_per_s``.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .runner import BenchResult

# Columns that appear in the flat CSV row, in order. Kept stable so old
# result CSVs remain readable by newer aggregators.
CSV_FIELDS: tuple[str, ...] = (
    "timestamp", "git_sha", "mode",
    "model_name", "attention_type", "use_compile", "use_fa3",
    "precision", "device", "world_size",
    "batch_size", "seq_len", "in_channels", "n_classes",
    "warmup_steps", "measure_steps", "repeats",
    "include_backward", "include_optimizer",
    "step_per_s_mean", "step_per_s_std",
    "samples_per_s", "tokens_per_s",
    "fwd_ms_mean", "bwd_ms_mean", "opt_ms_mean",
    "peak_vram_mb", "total_wall_s",
    "trainable_params", "total_params",
    "torch_version",
)


def _result_to_row(r: BenchResult) -> dict[str, object]:
    d = asdict(r)
    d.pop("extra", None)
    return {k: d[k] for k in CSV_FIELDS}


def write_result(result: BenchResult, out_dir: str | Path) -> tuple[Path, Path]:
    """Atomically write ``result.json`` + ``result.csv`` under ``out_dir``.

    Returns the two file paths. Should be called from rank-zero only.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "result.json"
    csv_path = out / "result.csv"

    payload = asdict(result)

    tmp_json = json_path.with_suffix(".json.tmp")
    with open(tmp_json, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    tmp_json.replace(json_path)

    tmp_csv = csv_path.with_suffix(".csv.tmp")
    with open(tmp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(_result_to_row(result))
    tmp_csv.replace(csv_path)

    return json_path, csv_path


def _iter_result_csvs(root: Path) -> Iterable[Path]:
    for p in sorted(Path(root).rglob("result.csv")):
        if p.is_file():
            yield p


def _load_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for p in _iter_result_csvs(root):
        with open(p) as f:
            for row in csv.DictReader(f):
                row["_source"] = str(p)
                rows.append(row)
    return rows


def _fmt(v: object, digits: int = 2) -> str:
    try:
        fv = float(v)
    except (ValueError, TypeError):
        return str(v)
    if fv != fv:  # NaN
        return "-"
    if abs(fv) >= 1e4 or abs(fv) < 1e-2 and fv != 0:
        return f"{fv:.3e}"
    return f"{fv:.{digits}f}"


_MARKDOWN_COLUMNS: tuple[tuple[str, str], ...] = (
    ("model_name", "model"),
    ("attention_type", "attn"),
    ("use_compile", "compile"),
    ("precision", "prec"),
    ("batch_size", "B"),
    ("seq_len", "T"),
    ("world_size", "ws"),
    ("mode", "mode"),
    ("step_per_s_mean", "step/s"),
    ("step_per_s_std", "±"),
    ("samples_per_s", "samples/s"),
    ("fwd_ms_mean", "fwd(ms)"),
    ("bwd_ms_mean", "bwd(ms)"),
    ("opt_ms_mean", "opt(ms)"),
    ("peak_vram_mb", "VRAM(MB)"),
)


def aggregate_markdown(root: str | Path) -> str:
    """Scan ``root`` for ``result.csv`` files and return a markdown summary.

    Also writes ``<root>/summary.csv`` and ``<root>/summary.md`` atomically.
    """
    root_p = Path(root)
    rows = _load_rows(root_p)
    if not rows:
        return f"No result.csv files found under {root_p}.\n"

    # Sort by samples_per_s desc (paper-friendly ranking).
    def _sps(r: dict) -> float:
        try:
            return float(r.get("samples_per_s", 0.0))
        except ValueError:
            return 0.0

    rows.sort(key=_sps, reverse=True)

    # Combined CSV
    summary_csv = root_p / "summary.csv"
    tmp_csv = summary_csv.with_suffix(".csv.tmp")
    fieldnames = list(CSV_FIELDS) + ["_source"]
    with open(tmp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp_csv.replace(summary_csv)

    # Markdown table
    headers = [label for _, label in _MARKDOWN_COLUMNS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for r in rows:
        cells = []
        for key, _ in _MARKDOWN_COLUMNS:
            v = r.get(key, "-")
            cells.append(_fmt(v) if key.endswith("_mean") or key.endswith("_s") or key == "step_per_s_std" or key == "peak_vram_mb" else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"_{len(rows)} run(s) under `{root_p}`; sorted by samples/s desc._")
    md = "\n".join(lines) + "\n"

    summary_md = root_p / "summary.md"
    tmp_md = summary_md.with_suffix(".md.tmp")
    with open(tmp_md, "w") as f:
        f.write(md)
    tmp_md.replace(summary_md)

    return md
