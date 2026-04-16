"""Lightning training module for the audio DiT."""
from __future__ import annotations

import math
import os
from pathlib import Path

import lightning as L
import torch

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
        ema_beta: float = 0.9999,
        val_generate_every: int = 5,
        val_n_samples: int = 32,
        val_seq_len: int = 172,
        val_cfg_scale: float = 4.0,
        val_sampler: str = "heun",
        val_n_steps: int = 50,
        output_dir: str = "outputs",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.ema = EMA(model, beta=ema_beta)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch  # (B, C, T), (B,)
        loss = flow_matching_loss(self.model, x, y)
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

    def on_validation_epoch_end(self) -> None:
        hp = self.hparams
        if (self.current_epoch + 1) % hp.val_generate_every != 0:
            return

        info(f"[val] generating {hp.val_n_samples} samples (epoch {self.current_epoch + 1})")
        self._generate_and_log(hp.val_n_samples, hp.val_cfg_scale, hp.val_sampler, hp.val_n_steps)

    def _generate_and_log(
        self, n: int, cfg_scale: float, sampler: str, n_steps: int
    ) -> None:
        from ..diffusion.samplers import euler_sample, heun_sample

        device = self.device
        hp = self.hparams

        # Rock (class 13) — largest FMA Medium class (35.9%), best-conditioned
        y = torch.full((n,), 13, dtype=torch.long, device=device)
        noise = torch.randn(n, self.model.in_channels, hp.val_seq_len, device=device)

        fn = heun_sample if sampler == "heun" else euler_sample
        with torch.no_grad():
            latents = fn(self.ema.ema_model, noise, y, n_steps=n_steps, cfg_scale=cfg_scale)

        # Decode to WAV if VAE decoder is available
        if hasattr(self, "vae"):
            wav_paths = self._decode_and_save(latents, self.current_epoch)
            self._score_audiobox(wav_paths)
            self._log_audio_to_wandb(wav_paths[:2])
        else:
            warn("[val] VAE not attached to module — skipping WAV generation")

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

    def _log_audio_to_wandb(self, wav_paths: list[Path]) -> None:
        """Upload up to 2 WAVs to WandB as audio samples."""
        import wandb

        if wandb.run is None:
            return
        clips = [
            wandb.Audio(str(p), sample_rate=44100, caption=f"epoch {self.current_epoch + 1} — {p.name}")
            for p in wav_paths
        ]
        wandb.log({"val/audio": clips})

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
        )

        warmup = hp.warmup_steps
        total  = hp.total_steps

        def _lr_lambda(step: int) -> float:
            """Linear warmup → cosine decay, independent of base lr."""
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        schedulers = [
            {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda),
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
