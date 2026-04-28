#!/usr/bin/env python
"""Training entry point for flash-dit.

Usage:
    uv run python scripts/train.py model=flash_dit data.batch_size=32

Override any config value via Hydra on the command line.
"""
from __future__ import annotations

import os

import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from flash_dit.data.datamodule import LatentDataModule
from flash_dit.training.jsonl_logger import JSONLMetricsCallback
from flash_dit.training.lit_module import FlashDiTModule
from flash_dit.utils.console import info, ok, warn


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    L.seed_everything(cfg.get("seed", 42), workers=True)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = instantiate(cfg.model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ok(f"Model: {cfg.model._target_} — {n_params / 1e6:.1f}M trainable params")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    datamodule = LatentDataModule(
        h5_path=cfg.data.h5_path,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        seq_len=cfg.data.get("seq_len", None),
    )
    # Need to call setup to read normalisation stats
    datamodule.setup()

    # ------------------------------------------------------------------
    # Compute total steps for scheduler
    # ------------------------------------------------------------------
    n_train = len(datamodule.train_ds)
    steps_per_epoch = n_train // cfg.data.batch_size
    total_steps = steps_per_epoch * cfg.trainer.max_epochs
    info(f"Total training steps: {total_steps} ({steps_per_epoch} steps/epoch × {cfg.trainer.max_epochs} epochs)")

    # ------------------------------------------------------------------
    # Lightning module
    # ------------------------------------------------------------------
    evaluation_cfg = (
        OmegaConf.to_container(cfg.evaluation, resolve=True) if "evaluation" in cfg else None
    )

    lit = FlashDiTModule(
        model=model,
        optimizer_type=cfg.optimizer.type,
        lr_muon=cfg.optimizer.get("lr_muon", 0.02),
        lr_adamw=cfg.optimizer.lr_adamw,
        weight_decay=cfg.optimizer.weight_decay,
        warmup_steps=cfg.optimizer.warmup_steps,
        total_steps=total_steps,
        muon_momentum=cfg.optimizer.get("momentum", 0.95),
        muon_ns_steps=cfg.optimizer.get("ns_steps", 5),
        ema_beta=cfg.get("ema_beta", 0.9999),
        val_generate_every=cfg.val.generate_every_n_epochs,
        val_n_samples=cfg.val.n_samples,
        val_seq_len=cfg.val.seq_len,
        val_cfg_scale=cfg.val.cfg_scale,
        val_sampler=cfg.val.sampler,
        val_n_steps=cfg.val.n_steps,
        val_keep_n_wavs=cfg.val.get("keep_n_wavs", 2),
        output_dir=os.path.join(cfg.output_dir, "generated"),
        metrics_dir=os.path.join(cfg.output_dir, "metrics"),
        evaluation_cfg=evaluation_cfg,
        h5_path=cfg.data.h5_path,
    )

    # Attach autoencoder decoder + normalisation stats for validation generation
    autoencoder = _load_autoencoder(cfg)
    if autoencoder is not None:
        lit.vae = autoencoder.to("cuda").eval()
        mean, std = _load_norm_stats(cfg.data.h5_path)
        lit.latent_mean = mean
        lit.latent_std  = std
        ok(f"Autoencoder {autoencoder.metadata.encoder_id!r} attached for validation generation.")
    else:
        warn("Autoencoder not loaded — generation during validation disabled.")

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Two ModelCheckpoint callbacks: top-k stores weights only (small, for
    # downstream eval/inference), while `last.ckpt` keeps the full optimizer
    # state for resumability. This avoids storing optimizer state 4× when
    # only one resumable snapshot is ever needed.
    ckpt_dir = os.path.join(cfg.output_dir, "checkpoints")
    callbacks = [
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="{epoch:04d}-{val/loss:.4f}",
            save_top_k=3,
            monitor="val/loss",
            mode="min",
            save_last=False,
            save_weights_only=True,
        ),
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="last",
            save_top_k=0,
            save_last=True,
            save_weights_only=False,
        ),
        L.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
        JSONLMetricsCallback(cfg.output_dir, every_n_train_steps=50),
    ]

    loggers = [L.pytorch.loggers.CSVLogger(cfg.output_dir, name="csv")]
    if cfg.get("use_wandb", False):
        wandb_logger = L.pytorch.loggers.WandbLogger(
            entity="ar_spectra",
            project="flash_dit",
            name=cfg.run_name,
            save_dir=".",
            config=dict(OmegaConf.to_container(cfg, resolve=True)),
        )
        wandb_logger.watch(lit.model, log="gradients", log_freq=500)
        loggers.append(wandb_logger)

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator="gpu",
        devices=cfg.trainer.get("devices", 1),
        strategy=cfg.trainer.get("strategy", "auto"),
        precision=cfg.trainer.get("precision", "bf16-mixed"),
        log_every_n_steps=50,
        check_val_every_n_epoch=cfg.trainer.get("check_val_every_n_epoch", 1),
        num_sanity_val_steps=cfg.trainer.get("num_sanity_val_steps", 0),
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val=cfg.trainer.get("gradient_clip_val", None),
    )

    info(f"Saving outputs to: {cfg.output_dir}")
    trainer.fit(lit, datamodule=datamodule, ckpt_path=cfg.get("resume_from", None))
    ok("Training complete.")


def _load_autoencoder(cfg: DictConfig):
    """Build the configured autoencoder. Returns None on failure (no HF token, etc.).

    Single point of construction — every encoder-specific quirk lives in the
    wrapper registered under ``cfg.autoencoder.kind``.
    """
    from flash_dit.autoencoder import build_autoencoder

    if "autoencoder" not in cfg:
        warn("cfg.autoencoder missing — generation during validation disabled.")
        return None
    try:
        spec = OmegaConf.to_container(cfg.autoencoder, resolve=True)
        return build_autoencoder(spec)
    except Exception as exc:
        warn(f"Could not build autoencoder: {exc}")
        return None


def _load_norm_stats(h5_path: str) -> tuple:
    import h5py
    import torch

    with h5py.File(h5_path, "r") as f:
        mean = torch.tensor(f.attrs["mean"], dtype=torch.float32)
        std  = torch.tensor(f.attrs["std"],  dtype=torch.float32)
    return mean, std


if __name__ == "__main__":
    main()
