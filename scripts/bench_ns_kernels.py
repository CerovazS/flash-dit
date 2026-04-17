#!/usr/bin/env python
"""Microbenchmark Muon's Newton-Schulz inner loop across the 2³=8 combinations
of the three vendored Triton kernels (XXT, XTX, ba_plus_cAA).

We time **only the NS5 iteration** applied to every Muon-eligible 2D matrix of
a flash_dit model (one step = 5 iterations × 84 matrices). No forward / backward
/ data loading is in the loop, so per-kernel contributions are clearly
separated.

Usage:
    uv run python scripts/bench_ns_kernels.py
    uv run python scripts/bench_ns_kernels.py --warmup 30 --measure 100 --repeats 5
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from flash_dit.models.dit import DiffusionTransformer
from flash_dit.training.optimizer import Muon, _is_muon_matrix_param
from flash_dit.utils.console import info, ok


FLAG_NAMES = ("use_xxt", "use_xtx", "use_ba_plus_caa")


@dataclass
class RunResult:
    tag: str
    use_xxt: bool
    use_xtx: bool
    use_ba_plus_caa: bool
    ms_per_step_mean: float   # per-step = 84 matrices × 5 NS iters
    ms_per_step_std: float
    ms_per_matrix_mean: float
    per_shape_ms: dict[str, float] = field(default_factory=dict)


def build_muon_grads(model: torch.nn.Module, device: torch.device):
    """Return a list of random gradient tensors shaped like the model's
    Muon-eligible 2D matrices."""
    grads: list[torch.Tensor] = []
    shape_keys: list[str] = []
    for name, p in model.named_parameters():
        if not _is_muon_matrix_param(name, p):
            continue
        g = torch.randn_like(p.data, device=device, dtype=torch.float32)
        grads.append(g)
        shape_keys.append(f"{tuple(p.shape)}")
    return grads, shape_keys


def run_ns_once(
    grads: list[torch.Tensor],
    steps: int,
    use_xxt: bool,
    use_xtx: bool,
    use_ba_plus_caa: bool,
) -> None:
    """Apply the chosen NS variant to every grad. Discards output."""
    for g in grads:
        _ = Muon._newtonschulz5_kernels(
            g, steps=steps,
            use_xxt=use_xxt, use_xtx=use_xtx, use_ba_plus_caa=use_ba_plus_caa,
        )


def time_config(
    grads: list[torch.Tensor],
    shape_keys: list[str],
    steps: int,
    warmup: int,
    measure: int,
    repeats: int,
    use_xxt: bool,
    use_xtx: bool,
    use_ba_plus_caa: bool,
) -> RunResult:
    device = grads[0].device

    # Warmup
    for _ in range(warmup):
        run_ns_once(grads, steps, use_xxt, use_xtx, use_ba_plus_caa)
    torch.cuda.synchronize(device)

    per_step_ms_all: list[float] = []
    per_shape_ms: dict[str, list[float]] = {}

    for _ in range(repeats):
        # One repeat: record a CUDA event pair around the whole measure window.
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(measure):
            run_ns_once(grads, steps, use_xxt, use_xtx, use_ba_plus_caa)
        end.record()
        torch.cuda.synchronize(device)
        per_step_ms_all.append(start.elapsed_time(end) / measure)

    # Per-shape timing (coarse — one event per distinct shape)
    unique_shapes = sorted(set(shape_keys))
    shape_to_grads: dict[str, list[torch.Tensor]] = {s: [] for s in unique_shapes}
    for g, s in zip(grads, shape_keys):
        shape_to_grads[s].append(g)
    for s, gs in shape_to_grads.items():
        starts, ends = [], []
        # warmup this shape
        for _ in range(5):
            for g in gs:
                _ = Muon._newtonschulz5_kernels(
                    g, steps=steps,
                    use_xxt=use_xxt, use_xtx=use_xtx, use_ba_plus_caa=use_ba_plus_caa,
                )
        torch.cuda.synchronize(device)
        for _ in range(20):
            st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            st.record()
            for g in gs:
                _ = Muon._newtonschulz5_kernels(
                    g, steps=steps,
                    use_xxt=use_xxt, use_xtx=use_xtx, use_ba_plus_caa=use_ba_plus_caa,
                )
            en.record()
            starts.append(st); ends.append(en)
        torch.cuda.synchronize(device)
        times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
        per_shape_ms[s] = sum(times) / len(times)

    import statistics
    tag = (
        ("X" if use_xxt else "_")
        + ("T" if use_xtx else "_")
        + ("B" if use_ba_plus_caa else "_")
    )
    mean = statistics.mean(per_step_ms_all)
    std = statistics.stdev(per_step_ms_all) if len(per_step_ms_all) > 1 else 0.0
    return RunResult(
        tag=tag,
        use_xxt=use_xxt, use_xtx=use_xtx, use_ba_plus_caa=use_ba_plus_caa,
        ms_per_step_mean=mean,
        ms_per_step_std=std,
        ms_per_matrix_mean=mean / len(grads),
        per_shape_ms=per_shape_ms,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--measure", type=int, default=30)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--ns-steps", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    info("Building flash_dit model + muon-eligible grads …")
    # Match conf/model/flash_dit.yaml (flash_dit defaults) but no compile:
    model = DiffusionTransformer(
        in_channels=64, d_model=768, n_layers=12, n_heads=12, n_kv_heads=2,
        mlp_mult=8 / 3, n_classes=17, cfg_dropout=0.1,
        attention_type="gqa", mlp_type="swiglu",
        use_fa3=False, use_compile=False,
        max_seq_len=256,
    ).to(device)
    model.eval()

    grads, shape_keys = build_muon_grads(model, device)
    info(f"  {len(grads)} muon-eligible matrices")
    from collections import Counter
    shape_counts = Counter(shape_keys)
    for s, n in sorted(shape_counts.items()):
        info(f"    {s} × {n}")

    results: list[RunResult] = []
    t_total0 = time.time()
    for flags in itertools.product([False, True], repeat=3):
        use_xxt, use_xtx, use_ba_plus_caa = flags
        info(
            f"→ config xxt={use_xxt} xtx={use_xtx} ba_plus_caa={use_ba_plus_caa}"
        )
        t0 = time.time()
        res = time_config(
            grads, shape_keys,
            steps=args.ns_steps,
            warmup=args.warmup, measure=args.measure, repeats=args.repeats,
            use_xxt=use_xxt, use_xtx=use_xtx, use_ba_plus_caa=use_ba_plus_caa,
        )
        ok(
            f"  tag={res.tag}  ms/step={res.ms_per_step_mean:.3f} ± {res.ms_per_step_std:.3f}  "
            f"ms/matrix={res.ms_per_matrix_mean:.3f}  wall={time.time()-t0:.1f}s"
        )
        results.append(res)

    # Sort by mean ms/step (fastest first for reporting)
    results.sort(key=lambda r: r.ms_per_step_mean)

    ok(f"All 8 configs done in {time.time()-t_total0:.1f}s total.")

    # Output
    out_dir = Path(args.output_dir or f"outputs/bench/ns_kernels_{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "results.json", "w") as f:
        json.dump(
            {"results": [asdict(r) for r in results],
             "config": {"warmup": args.warmup, "measure": args.measure, "repeats": args.repeats,
                        "ns_steps": args.ns_steps, "n_matrices": len(grads),
                        "gpu": torch.cuda.get_device_name(device)}},
            f, indent=2, sort_keys=True,
        )

    # Flat CSV
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "tag", "use_xxt", "use_xtx", "use_ba_plus_caa",
            "ms_per_step_mean", "ms_per_step_std", "ms_per_matrix_mean",
        ])
        for r in results:
            w.writerow([
                r.tag, r.use_xxt, r.use_xtx, r.use_ba_plus_caa,
                f"{r.ms_per_step_mean:.4f}", f"{r.ms_per_step_std:.4f}",
                f"{r.ms_per_matrix_mean:.4f}",
            ])

    # Markdown table
    baseline = next((r for r in results if not any([r.use_xxt, r.use_xtx, r.use_ba_plus_caa])), None)
    lines = [
        "# NS kernel ablation",
        "",
        f"**GPU**: {torch.cuda.get_device_name(device)}  ",
        f"**Matrices/step**: {len(grads)} (flash_dit default, 12 blocks × 7 shapes)  ",
        f"**NS iterations/call**: {args.ns_steps}  ",
        f"**Warmup**: {args.warmup} | **Measure**: {args.measure} | **Repeats**: {args.repeats}  ",
        "",
        "| rank | xxt | xtx | ba_cAA | ms/step (mean ± std) | ms/matrix | speedup vs baseline |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(results):
        speedup = (baseline.ms_per_step_mean / r.ms_per_step_mean) if baseline else 1.0
        lines.append(
            f"| {i+1} | {'✓' if r.use_xxt else '✗'} | {'✓' if r.use_xtx else '✗'} | "
            f"{'✓' if r.use_ba_plus_caa else '✗'} | "
            f"{r.ms_per_step_mean:.3f} ± {r.ms_per_step_std:.3f} | "
            f"{r.ms_per_matrix_mean:.3f} | {speedup:.3f}× |"
        )
    lines.append("")
    # Per-shape matrix (relative speedup per shape)
    all_shapes = sorted({s for r in results for s in r.per_shape_ms.keys()})
    lines.append("## Per-shape ms (lower is better)")
    lines.append("")
    header_cells = ["config"] + all_shapes
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cells)) + "|")
    for r in results:
        row = [r.tag] + [f"{r.per_shape_ms.get(s, 0.0):.3f}" for s in all_shapes]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines))
    info(f"wrote {out_dir}/summary.md")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
