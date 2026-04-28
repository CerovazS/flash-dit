#!/usr/bin/env python
"""Migrate a v1 latent HDF5 (SAO-only, no encoder metadata) to v2.

The v1 schema, written by the original ``precompute_latents.py``, has only
``mean`` / ``std`` attrs. v2 adds the encoder identity and shape metadata
required by the multi-VAE refactor (see ``plan.md`` §2).

Usage:
    # Default: assume Stable Audio Open (the only encoder used in v1)
    uv run python scripts/migrate_latents_h5.py \\
        --h5-path outputs/latents/stable_audio_open/fma_medium.h5

    # Override defaults if the file came from a different encoder
    uv run python scripts/migrate_latents_h5.py \\
        --h5-path /path/to/file.h5 \\
        --encoder-id stable_audio_open \\
        --sample-rate 44100 \\
        --audio-channels 2 \\
        --compression-ratio 2048

    # Inspect-only — print what would be added without writing
    uv run python scripts/migrate_latents_h5.py \\
        --h5-path /path/to/file.h5 --dry-run

The script is **idempotent**: running it twice on the same file is a no-op
once ``schema_version`` is set to 2. Existing v2 files raise a clear
"already migrated" message and exit 0.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

import h5py

# Stable Audio Open defaults — the only encoder produced by v1.
SAO_DEFAULTS = {
    "encoder_id": "stable_audio_open",
    "encoder_version": "stabilityai/stable-audio-open-1.0",
    "encoder_config": {
        "hf_repo": "stabilityai/stable-audio-open-1.0",
        "kind": "stable_audio_open",
    },
    "domain": "audio",
    "input_format": "BCT",
    "latent_format": "BCT",
    "input_channels": 2,
    "latent_channels": 64,
    "sample_rate": 44100,
    "compression_ratios": [2048],
    "chunk_samples": 440320,         # 215 latent frames × 2048 stride
    "storage_dtype": "float16",
}

SCHEMA_VERSION = 2


def _ok(msg: str) -> None:
    print(f"\033[1;32mOK\033[0m {msg}", flush=True)


def _info(msg: str) -> None:
    print(f"\033[36mINFO\033[0m {msg}", flush=True)


def _warn(msg: str) -> None:
    print(f"\033[1;33mWARN\033[0m {msg}", flush=True)


def _error(msg: str) -> None:
    print(f"\033[1;31mERROR\033[0m {msg}", flush=True)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _detect_existing(f: h5py.File) -> dict:
    """Return current attrs as a plain dict (decoding bytes to str)."""
    out = {}
    for k in f.attrs:
        v = f.attrs[k]
        if isinstance(v, bytes):
            v = v.decode("utf-8", errors="replace")
        out[k] = v
    return out


def _validate_existing_schema(attrs: dict, h5_path: Path) -> None:
    """Bail out cleanly if the file is already at v2 (idempotency)."""
    sv = attrs.get("schema_version")
    if sv is not None and int(sv) >= SCHEMA_VERSION:
        _ok(f"{h5_path.name} already at schema_version={int(sv)} — nothing to do")
        sys.exit(0)
    if "encoder_id" in attrs and "schema_version" not in attrs:
        _warn(
            f"{h5_path.name} has encoder_id={attrs['encoder_id']!r} but no "
            f"schema_version — proceeding to set schema_version=2"
        )


def _infer_shape_metadata(f: h5py.File) -> tuple[int, int]:
    """Read latent_channels and latent_frames from the latents dataset shape."""
    if "latents" not in f:
        raise KeyError("HDF5 has no 'latents' dataset — refusing to migrate")
    shape = f["latents"].shape
    if len(shape) != 3:
        raise ValueError(
            f"expected (N, C, T) 3-D latents, got shape {shape}"
        )
    _, c, _ = shape
    return c, shape[2]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--h5-path", type=Path, required=True,
                   help="HDF5 file to migrate in place")
    p.add_argument("--encoder-id", default=SAO_DEFAULTS["encoder_id"],
                   help=f"encoder identifier (default: {SAO_DEFAULTS['encoder_id']!r})")
    p.add_argument("--encoder-version", default=SAO_DEFAULTS["encoder_version"])
    p.add_argument("--domain", default=SAO_DEFAULTS["domain"],
                   choices=["audio", "vision"])
    p.add_argument("--input-format", default=SAO_DEFAULTS["input_format"],
                   help='"BCT" (audio) or "BCHW" (vision)')
    p.add_argument("--latent-format", default=SAO_DEFAULTS["latent_format"])
    p.add_argument("--input-channels", type=int, default=SAO_DEFAULTS["input_channels"])
    p.add_argument("--sample-rate", type=int, default=SAO_DEFAULTS["sample_rate"])
    p.add_argument("--compression-ratios", type=int, nargs="+",
                   default=SAO_DEFAULTS["compression_ratios"],
                   help="Per-axis stride: [2048] for audio, [8, 8] for SD VAE")
    p.add_argument("--chunk-samples", type=int, default=SAO_DEFAULTS["chunk_samples"])
    p.add_argument("--storage-dtype", default=SAO_DEFAULTS["storage_dtype"])
    p.add_argument("--dry-run", action="store_true",
                   help="Print attrs that would be added; do not write")
    args = p.parse_args()

    if not args.h5_path.is_file():
        _error(f"file not found: {args.h5_path}")
        return 2

    # Read first to validate idempotency.
    with h5py.File(args.h5_path, "r") as f:
        existing = _detect_existing(f)
        _info(f"existing attrs: {sorted(existing.keys())}")
        _validate_existing_schema(existing, args.h5_path)
        latent_channels, latent_frames = _infer_shape_metadata(f)

    # If the user explicitly passed --encoder-id different from SAO, build
    # an encoder_config that just records the kind (no full upstream config
    # available without instantiating the actual library).
    if args.encoder_id == SAO_DEFAULTS["encoder_id"]:
        encoder_config = SAO_DEFAULTS["encoder_config"]
    else:
        encoder_config = {"kind": args.encoder_id}

    new_attrs = {
        "schema_version": SCHEMA_VERSION,
        "encoder_id": args.encoder_id,
        "encoder_version": args.encoder_version,
        "encoder_config_json": json.dumps(encoder_config, sort_keys=True),
        "domain": args.domain,
        "input_format": args.input_format,
        "latent_format": args.latent_format,
        "input_channels": args.input_channels,
        "latent_channels": latent_channels,
        "compression_ratios": list(args.compression_ratios),
        "sample_rate": args.sample_rate if args.domain == "audio" else -1,
        "chunk_samples": args.chunk_samples,
        "storage_dtype": args.storage_dtype,
        "created_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
    }

    # Sanity: warn if shape doesn't match declared compression
    if args.encoder_id == "stable_audio_open" and latent_channels != 64:
        _warn(
            f"latent_channels in file = {latent_channels} but SAO default is 64. "
            f"Proceeding, but verify --encoder-id is correct."
        )

    _info(f"shape inferred from latents dataset: C={latent_channels}, T={latent_frames}")
    _info("attrs to add/update:")
    for k, v in new_attrs.items():
        marker = "+" if k not in existing else "~"
        existing_repr = f" (was {existing.get(k)!r})" if k in existing else ""
        print(f"  {marker} {k:22s} = {v!r}{existing_repr}")

    if args.dry_run:
        _ok(f"[dry-run] would write {len(new_attrs)} attrs to {args.h5_path}")
        return 0

    # Open in r+ to add attrs without rewriting datasets.
    with h5py.File(args.h5_path, "r+") as f:
        for k, v in new_attrs.items():
            f.attrs[k] = v

    # Verify
    with h5py.File(args.h5_path, "r") as f:
        sv = int(f.attrs["schema_version"])
        eid = f.attrs["encoder_id"]
        if isinstance(eid, bytes):
            eid = eid.decode("utf-8")
    _ok(f"migrated {args.h5_path} → schema_version={sv}, encoder_id={eid!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
