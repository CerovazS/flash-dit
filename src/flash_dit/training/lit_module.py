"""Lightning training module for the audio DiT."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from ..diffusion.flow_matching import flow_matching_loss
from ..utils.console import info, ok, warn
from .ema import EMA
from .optimizer import build_optimizer


class FlashDiTModule(L.LightningModule):
    """LightningModule that wraps DiffusionTransformer for training.

    Args:
        model:               DiffusionTransformer instance.
        optimizer_type:      'muon_adamw' or 'adamw'.
        lr_muon:             Muon learning rate.
        lr_adamw:            AdamW learning rate.
        weight_decay:        AdamW weight decay.
        warmup_steps:        linear warmup steps.
        total_steps:         total training steps (for cosine schedule).
        ema_beta:            EMA decay.
        val_generate_every:  generate audio samples every N epochs.
        val_n_samples:       number of latent samples to generate.
        val_seq_len:         latent temporal length to generate (frames).
                             172 frames × 2048 / 44100 ≈ 8 s.
        val_cfg_scale:       CFG guidance scale at validation.
        val_sampler:         'euler' or 'heun'.
        val_n_steps:         ODE sampler steps at validation.
        output_dir:          directory to save generated WAVs.
        metrics_dir:         where to write per-epoch metric JSON files.
                             Defaults to ``Path(output_dir).parent / "metrics"``.
        evaluation_cfg:      nested dict mirroring ``conf/config.yaml`` ``evaluation``.
        h5_path:             HDF5 path used to load reference latents for latent MMD.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_type: str = "muon_adamw",
        lr_muon: float = 0.02,
        lr_adamw: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        total_steps: int = 500_000,
        muon_momentum: float = 0.95,
        muon_ns_steps: int = 5,
        ema_beta: float = 0.9999,
        val_generate_every: int = 5,
        val_n_samples: int = 32,
        val_seq_len: int = 172,
        val_cfg_scale: float = 4.0,
        val_sampler: str = "heun",
        val_n_steps: int = 50,
        val_keep_n_wavs: int = 2,
        output_dir: str = "outputs",
        metrics_dir: str | None = None,
        evaluation_cfg: dict | None = None,
        h5_path: str | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.ema = EMA(model, beta=ema_beta)
        # Multiple optimizers (Muon + AdamW) require manual optimization
        self.automatic_optimization = False

        # Resolved at first use (rank-zero only) and cached per feature type.
        self._latent_mmd_ref_cache: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch  # (B, C, T), (B,)

        opts = self.optimizers()
        if not isinstance(opts, list):
            opts = [opts]
        for opt in opts:
            opt.zero_grad()

        loss = flow_matching_loss(self.model, x, y)
        self.manual_backward(loss)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        for opt in opts:
            opt.step()

        # Step LR scheduler(s) every batch
        schs = self.lr_schedulers()
        if schs is not None:
            if not isinstance(schs, list):
                schs = [schs]
            for sch in schs:
                sch.step()

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def on_fit_start(self) -> None:
        self.ema.ema_model.to(self.device)

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        self.ema.update(self.model)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        x, y = batch
        with torch.no_grad():
            loss = flow_matching_loss(self.ema.ema_model, x, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        """Single orchestrator for the whole validation pipeline.

        Runs once per ``val.generate_every_n_epochs`` ticks; samples
        ``val.n_samples`` latents ONCE and dispatches them to every enabled
        downstream metric whose own cadence is also due. Decoding to WAV is
        skipped when no WAV-consuming metric fires — cheap epochs stay cheap.
        """
        import time as _time

        hp = self.hparams
        epoch_1 = self.current_epoch + 1

        # --- sampling cadence ------------------------------------------------
        if hp.val_generate_every <= 0 or epoch_1 % hp.val_generate_every != 0:
            return

        ev_cfg = self._get_eval_cfg()
        lm_cfg = ev_cfg.get("latent_mmd", {}) if ev_cfg else {}
        kad_cfg = ev_cfg.get("kad", {}) if ev_cfg else {}
        ab_cfg = ev_cfg.get("audiobox", {}) if ev_cfg else {}

        def _due(cfg: dict) -> bool:
            if not cfg.get("enabled", False):
                return False
            every = int(cfg.get("every_n_epochs", hp.val_generate_every))
            return every > 0 and epoch_1 % every == 0

        due_latent_mmd = _due(lm_cfg)
        due_kad = _due(kad_cfg)
        due_audiobox = _due(ab_cfg)
        due_wandb = hasattr(self, "vae")  # wandb audio always follows generation when VAE present

        # --- 1. Sample ONCE ---------------------------------------------------
        info(f"[val] sampling {hp.val_n_samples} latents (epoch {epoch_1})")
        t0 = _time.time()
        latents = self._sample_validation_latents(hp.val_n_samples)
        t_sample = _time.time() - t0
        info(f"[val] sampling done in {t_sample:.2f}s")

        # --- 2. Decode ONCE if any WAV-consuming metric fires ---------------
        wav_paths: list[Path] | None = None
        needs_wav = due_kad or due_audiobox or due_wandb
        if needs_wav:
            if hasattr(self, "vae"):
                t1 = _time.time()
                wav_paths = self._decode_and_save(latents, self.current_epoch)
                info(f"[val] decoded {len(wav_paths)} latents to WAV in {_time.time()-t1:.2f}s")
            else:
                warn("[val] VAE not attached — skipping WAV-dependent metrics")
                due_kad = due_audiobox = due_wandb = False

        # --- 3. Metrics -------------------------------------------------------
        if due_latent_mmd:
            self._compute_latent_mmd(latents, lm_cfg, t_sample=t_sample)
        if wav_paths and due_audiobox:
            self._score_audiobox(wav_paths)
        if wav_paths and due_kad:
            # Free cached sampler/VAE activations before spawning PANNs workers.
            # Without this, the parent process's allocator reservations prevent
            # the spawn subprocesses from claiming GPU memory — OOM when the
            # main model has a large cache footprint (e.g. MHA / no torch.compile).
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._compute_kad(wav_paths[0].parent, kad_cfg)
        if wav_paths and due_wandb:
            self._log_audio_to_wandb(wav_paths[:2])
        if wav_paths:
            self._cleanup_generated_dir(wav_paths, keep_n=int(hp.val_keep_n_wavs))

    # ------------------------------------------------------------------
    # Sampling — single source of truth for validation latents
    # ------------------------------------------------------------------

    def _sample_validation_latents(self, n: int) -> torch.Tensor:
        """Draw ``n`` EMA-model latents used by every downstream metric."""
        from ..diffusion.samplers import euler_sample, heun_sample

        hp = self.hparams
        device = self.device
        n_classes = int(getattr(self.model.y_embedder, "null_class", 16))

        g = torch.Generator(device="cpu").manual_seed(42 + self.current_epoch)
        y = torch.randint(0, n_classes, (n,), generator=g).to(device)
        noise = torch.randn(n, self.model.in_channels, hp.val_seq_len, device=device)

        fn = heun_sample if hp.val_sampler == "heun" else euler_sample
        # bf16 autocast matches training precision. The compiled attention path
        # (flash-attn via torch.compile) rejects fp32 inputs, and without this
        # wrap sampling can hit "FlashAttention only support fp16 and bf16".
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return fn(
                self.ema.ema_model, noise, y,
                n_steps=hp.val_n_steps, cfg_scale=hp.val_cfg_scale,
            )

    def _decode_and_save(self, latents: torch.Tensor, epoch: int) -> list[Path]:
        """Decode latents to WAV, save to disk, and return list of file paths."""
        import soundfile as sf

        out_dir = Path(self.hparams.output_dir) / f"epoch_{epoch:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            # Denormalise latents using stored stats (attached by train.py)
            mean = self.latent_mean.to(latents)
            std  = self.latent_std.to(latents)
            latents_raw = latents * (std[:, None] + 1e-6) + mean[:, None]

            audio = self.vae.decode(latents_raw)  # (B, 2, T_audio)

        audio = audio.cpu().float().clamp(-1.0, 1.0).numpy()
        paths = []
        for i, wav in enumerate(audio):
            path = out_dir / f"sample_{i:03d}.wav"
            sf.write(str(path), wav.T, samplerate=44100)
            paths.append(path)

        ok(f"[val] saved {len(audio)} WAVs to {out_dir}")
        return paths

    @rank_zero_only
    def _log_audio_to_wandb(self, wav_paths: list[Path]) -> None:
        """Upload WAVs to WandB as playable media and downloadable artifacts."""
        import wandb

        if wandb.run is None:
            return
        if not wav_paths:
            return

        epoch = int(self.current_epoch + 1)
        step = int(self.global_step)
        clips = [
            wandb.Audio(str(p), sample_rate=44100, caption=f"epoch {epoch} - {p.name}")
            for p in wav_paths
        ]
        table = wandb.Table(columns=["epoch", "sample", "audio"])
        artifact = wandb.Artifact(
            name=f"validation-audio-epoch-{epoch:06d}",
            type="validation-audio",
            metadata={"epoch": epoch, "global_step": step, "sample_rate": 44100},
        )

        for path in wav_paths:
            table.add_data(
                epoch,
                path.name,
                wandb.Audio(str(path), sample_rate=44100, caption=f"epoch {epoch} - {path.name}"),
            )
            artifact.add_file(str(path), name=f"epoch_{epoch:06d}/{path.name}")

        wandb.log(
            {
                "val/audio": clips,
                "val/audio_table": table,
                "val/audio_epoch": epoch,
            },
            step=step,
            commit=True,
        )
        # wait() blocks until the artifact has finished uploading — otherwise
        # _cleanup_generated_dir may delete the source WAVs before wandb has
        # read them for upload, failing the artifact with missing-file errors.
        logged = wandb.log_artifact(artifact)
        try:
            logged.wait()
        except Exception:
            pass

    def _cleanup_generated_dir(self, wav_paths: list[Path], keep_n: int = 2) -> None:
        """Delete per-epoch generated artifacts after metrics + WandB upload.

        Keeps the first ``keep_n`` WAVs (the ones sent to WandB, preserved
        locally for manual inspection) and removes:
          - every other WAV
          - the ``embeddings/`` and ``convert/`` cache subdirs populated by
            kadtk_vendored for KAD

        Without this the per-epoch generated dirs accumulate linearly with
        the number of validations — for 1000 epochs and val every 10 epochs
        that is ~9 GB of disposable WAVs + ~300 MB of per-epoch PANNs .npy
        that are never read again after the KAD call in the same val.
        """
        import shutil

        if not wav_paths:
            return
        gen_dir = wav_paths[0].parent
        for sub in ("embeddings", "convert"):
            d = gen_dir / sub
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
        for p in wav_paths[keep_n:]:
            try:
                p.unlink()
            except (FileNotFoundError, OSError):
                pass

    def _score_audiobox(self, wav_paths: list[Path]) -> None:
        """Score generated WAVs with Audiobox Aesthetics and log mean metrics."""
        from ..evaluation.audiobox import score_wavs

        try:
            scores = score_wavs(wav_paths)
        except Exception as exc:
            warn(f"[val] Audiobox scoring failed: {exc}")
            return

        for key, val in scores.items():
            self.log(f"val/audiobox_{key.lower()}", val, prog_bar=False, sync_dist=True)

        ok(
            f"[val] Audiobox — CE={scores['CE']:.3f}  CU={scores['CU']:.3f}  "
            f"PC={scores['PC']:.3f}  PQ={scores['PQ']:.3f}"
        )

    # ------------------------------------------------------------------
    # Evaluation helpers (latent MMD + KAD)
    # ------------------------------------------------------------------

    def _get_eval_cfg(self) -> dict:
        cfg = self.hparams.get("evaluation_cfg") if hasattr(self.hparams, "get") else None
        if cfg is None:
            cfg = getattr(self.hparams, "evaluation_cfg", None)
        return dict(cfg) if cfg else {}

    def _metrics_dir(self) -> Path:
        md = self.hparams.get("metrics_dir", None) if hasattr(self.hparams, "get") else None
        if md:
            return Path(md)
        return Path(self.hparams.output_dir).parent / "metrics"

    def _compute_latent_mmd(
        self, latents: torch.Tensor, lm_cfg: dict, t_sample: float = 0.0,
    ) -> None:
        """Latent-space MMD on pre-sampled latents (no sampling inside)."""
        import time as _time

        from ..evaluation.latent_mmd import compute_latent_mmd

        feature_types = list(lm_cfg.get("feature_types", ["frame", "clip_pooled"]))
        scale_factor = float(lm_cfg.get("scale_factor", 100.0))
        device = self.device

        t0 = _time.time()
        records: dict[str, dict[str, Any]] = {}
        for ft in feature_types:
            try:
                ref = self._get_reference_features(ft, lm_cfg)
            except Exception as exc:
                warn(f"[val] latent_mmd: failed to load reference features for {ft}: {exc}")
                continue
            res = compute_latent_mmd(
                latents,
                ref.to(device),
                feature_type=ft,  # type: ignore[arg-type]
                frame_subsample_per_clip=int(lm_cfg.get("frame_subsample_per_clip", 32)),
                clip_pool_num_chunks=int(lm_cfg.get("clip_pool_num_chunks", 4)),
                kernel=str(lm_cfg.get("kernel", "gaussian")),
                bandwidth=None,
                bandwidth_source=str(lm_cfg.get("bandwidth_source", "reference")),  # type: ignore[arg-type]
                bandwidth_subsample=int(lm_cfg.get("bandwidth_subsample", 10_000)),
                scale_factor=scale_factor,
                seed=int(lm_cfg.get("seed", 0)),
                normalized=bool(lm_cfg.get("normalized", True)),
                device=device,
            )
            self.log(f"val/latent_mmd_{ft}", res.scaled, prog_bar=False, rank_zero_only=True)
            self.log(f"val/latent_mmd_{ft}_raw", res.mmd2, prog_bar=False, rank_zero_only=True)
            records[ft] = res.__dict__

        dt_mmd = _time.time() - t0
        if records:
            self._write_metric_json(
                name="latent_mmd",
                payload={
                    "epoch": int(self.current_epoch + 1),
                    "global_step": int(self.global_step),
                    "n_generated": int(latents.shape[0]),
                    "wall_time_mmd_s": dt_mmd,
                    "wall_time_sample_s": t_sample,
                    "config": dict(lm_cfg),
                    "results": records,
                },
            )
            summary = ", ".join(
                f"{ft}={records[ft]['scaled']:.4f}" for ft in feature_types if ft in records
            )
            ok(f"[val] latent_mmd (scaled×{scale_factor:g}): {summary} | mmd={dt_mmd:.2f}s")

    def _get_reference_features(self, feature_type: str, lm_cfg: dict) -> torch.Tensor:
        """Lazy-load and memoise reference features for the given feature_type."""
        if feature_type in self._latent_mmd_ref_cache:
            return self._latent_mmd_ref_cache[feature_type]

        from ..evaluation.latent_mmd import ReferenceFeatureSpec, load_reference_features

        h5_path = self.hparams.get("h5_path", None) if hasattr(self.hparams, "get") else None
        if h5_path is None:
            h5_path = getattr(self.hparams, "h5_path", None)
        if not h5_path:
            raise RuntimeError(
                "latent_mmd requires h5_path — pass it via FlashDiTModule(h5_path=...)"
            )

        cache_dir = lm_cfg.get("cache_dir")
        if not cache_dir:
            cache_dir = str(Path(self.hparams.output_dir).parent / ".cache" / "latent_mmd")

        spec = ReferenceFeatureSpec(
            h5_path=str(h5_path),
            split=str(lm_cfg.get("split", "val")),
            feature_type=feature_type,  # type: ignore[arg-type]
            n_reference=int(lm_cfg.get("n_reference", 1024)),
            seq_len=int(self.hparams.val_seq_len),
            frame_subsample_per_clip=(
                int(lm_cfg.get("frame_subsample_per_clip", 32))
                if feature_type == "frame" else None
            ),
            clip_pool_num_chunks=int(lm_cfg.get("clip_pool_num_chunks", 4)),
            normalized=bool(lm_cfg.get("normalized", True)),
            seed=int(lm_cfg.get("seed", 0)),
        )
        feats = load_reference_features(spec, cache_dir=cache_dir)
        self._latent_mmd_ref_cache[feature_type] = feats
        return feats

    def _compute_kad(self, gen_dir: Path, kad_cfg: dict) -> None:
        """KAD on an existing WAV directory (decoded once by the orchestrator)."""
        import time as _time

        from ..evaluation.kad import compute_kad

        ref_dir = kad_cfg.get("reference_dir")
        ref_manifest = kad_cfg.get("reference_manifest")
        ref_cache_dir = kad_cfg.get("reference_cache_dir")
        if not ref_dir and not ref_manifest:
            warn(
                "[val] kad: neither evaluation.kad.reference_dir nor "
                "reference_manifest set — skipping"
            )
            return
        if ref_dir and ref_manifest:
            warn(
                "[val] kad: both reference_dir and reference_manifest set — "
                "skipping (set exactly one)"
            )
            return
        if not gen_dir.is_dir() or not any(gen_dir.glob("*.wav")):
            warn(f"[val] kad: no generated WAVs at {gen_dir} — skipping")
            return

        out_json = self._metrics_dir() / f"kad_epoch_{self.current_epoch + 1:06d}.json"
        t0 = _time.time()
        try:
            result = compute_kad(
                generated_dir=gen_dir,
                reference_dir=ref_dir,
                reference_manifest=ref_manifest,
                reference_cache_dir=ref_cache_dir,
                model_name=str(kad_cfg.get("model_name", "panns-wavegram-logmel")),
                device=str(kad_cfg.get("device", "cuda")),
                workers=int(kad_cfg.get("workers", 8)),
                bandwidth_source=str(kad_cfg.get("bandwidth_source", "reference")),
                scale_factor=float(kad_cfg.get("scale_factor", 100.0)),
                output_json=out_json,
            )
        except Exception as exc:
            warn(f"[val] kad: failed — {exc}")
            return

        dt = _time.time() - t0
        self.log("val/kad", result.score, prog_bar=False, rank_zero_only=True)
        ok(f"[val] kad ({result.model_name}): {result.score:.4f} | {dt:.2f}s")

    def _write_metric_json(self, name: str, payload: dict) -> None:
        md = self._metrics_dir()
        md.mkdir(parents=True, exist_ok=True)
        out = md / f"{name}_epoch_{self.current_epoch + 1:06d}.json"
        with open(out, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True, default=str)

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        hp = self.hparams
        optimizers = build_optimizer(
            self.model,
            optimizer_type=hp.optimizer_type,
            lr_muon=hp.lr_muon,
            lr_adamw=hp.lr_adamw,
            weight_decay=hp.weight_decay,
            momentum=hp.muon_momentum,
            ns_steps=hp.muon_ns_steps,
        )

        schedulers = [
            {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=hp.total_steps, eta_min=0
                ),
                "interval": "step",
            }
            for opt in optimizers
        ]

        return optimizers, schedulers

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])
