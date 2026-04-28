"""Stable Audio Open 1.0 autoencoder wrapper.

Consolidates the three pre-existing ``_load_vae()`` / ``load_vae()``
copies under a single :class:`AutoencoderBackbone` implementation. The
reference behaviour is the one in ``scripts/precompute_latents.py``
(the version that actually produced our HDF5 files):

- download ``model_config.json`` and ``model.safetensors`` from HF
- build the model with ``stable_audio_tools.create_model_from_config``
- load only ``pretransform.model.*`` keys (strip the prefix), in strict
  mode, into ``model.pretransform.model``
- ``vae.iterate_batch = False`` so encode runs the full batch on GPU
  (default ``True`` defeats batching at the caller level)
- ``encode()`` may return either a tensor or a ``(latent, info_dict)``
  tuple where ``info_dict["mean"]`` is the deterministic latent

The same reference behaviour fixes the bug in ``scripts/sample.py``,
which used to strip the wrong prefix (``pretransform.``) — verified
against the SAO checkpoint's actual key layout.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
import torchaudio

from .base import AutoencoderBackbone, AutoencoderMetadata
from .registry import register_autoencoder

_DEFAULT_HF_REPO = "stabilityai/stable-audio-open-1.0"


@register_autoencoder("stable_audio_open")
class StableAudioOpenAutoencoder(AutoencoderBackbone):
    """Stable Audio Open 1.0 (Oobleck VAE).

    44.1 kHz stereo input, 2048× temporal compression, 64-D continuous
    latent. Mono inputs are duplicated to stereo at encode time;
    non-44.1 kHz inputs are resampled.

    Args:
        hf_repo: HuggingFace repo id. Default ``stabilityai/stable-audio-open-1.0``.
        cache_dir: directory for HF downloads. Defaults to ``$FLASH_DIT_CACHE``
            then to ``~/.cache/flash-dit``.
        hf_token: HF API token. Defaults to ``$HF_TOKEN``.
        device: torch device for the loaded weights.
    """

    def __init__(
        self,
        hf_repo: str = _DEFAULT_HF_REPO,
        cache_dir: str | os.PathLike | None = None,
        hf_token: str | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from stable_audio_tools.models.factory import create_model_from_config

        token = hf_token if hf_token is not None else os.environ.get("HF_TOKEN", "")
        if not token:
            raise RuntimeError(
                "StableAudioOpenAutoencoder requires HF_TOKEN: pass hf_token="
                "..., or export HF_TOKEN, or login via `huggingface-cli login`."
            )

        if cache_dir is None:
            cache_dir = os.environ.get(
                "FLASH_DIT_CACHE", str(Path.home() / ".cache" / "flash-dit")
            )
        cache_dir = str(cache_dir)

        config_path = self._hf_get(hf_hub_download, hf_repo, "model_config.json", token, cache_dir)
        ckpt_path = self._hf_get(hf_hub_download, hf_repo, "model.safetensors", token, cache_dir)

        with open(config_path) as f:
            upstream_config = json.load(f)

        full_model = create_model_from_config(upstream_config)
        state = load_file(ckpt_path)

        # Strict load only the pretransform (VAE) weights, stripping the
        # ``pretransform.model.`` prefix — the inner autoencoder expects
        # keys relative to itself.
        pretransform_state = {
            k.removeprefix("pretransform.model."): v
            for k, v in state.items()
            if k.startswith("pretransform.model.")
        }
        full_model.pretransform.load_state_dict(pretransform_state, strict=True)

        self._model = full_model.pretransform
        # SAO sets iterate_batch=True in its pretransform config which makes
        # encode() loop over the batch dim internally — defeating GPU
        # batching. Force False so we get throughput from caller-level batching.
        self._model.iterate_batch = False
        self._model = self._model.to(device).eval()

        self.metadata = AutoencoderMetadata(
            encoder_id="stable_audio_open",
            encoder_version=hf_repo,
            encoder_config={
                "hf_repo": hf_repo,
                "kind": "stable_audio_open",
            },
            domain="audio",
            input_format="BCT",
            latent_format="BCT",
            input_channels=2,
            latent_channels=64,
            compression_ratios=(2048,),
            sample_rate=44100,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hf_get(
        downloader: Any,
        repo: str,
        filename: str,
        token: str,
        cache_dir: str,
    ) -> str:
        """Fetch a file from HF, preferring the local cache (offline-friendly)."""
        try:
            return downloader(
                repo, filename,
                token=token, cache_dir=cache_dir, local_files_only=True,
            )
        except Exception:
            return downloader(
                repo, filename,
                token=token, cache_dir=cache_dir,
            )

    def _coerce_input(self, audio: torch.Tensor) -> torch.Tensor:
        """Mono→stereo, dtype to float32, send to encoder device."""
        if audio.dim() != 3:
            raise ValueError(
                f"expected (B, C, T) audio, got shape {tuple(audio.shape)}"
            )
        c = audio.shape[1]
        if c == 1:
            audio = audio.repeat(1, 2, 1)
        elif c > 2:
            audio = audio[:, :2]
        return audio.to(self.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # AutoencoderBackbone interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """``(B, audio_channels, T_audio) → (B, 64, T_lat)``.

        Returns the deterministic mean of the VAE posterior — matching
        the convention used by ``precompute_latents.py``.
        """
        x = self._coerce_input(audio)
        result = self._model.encode(x)
        # SAO encode may return either a tensor or a (latent, info) tuple
        # where info["mean"] is the deterministic latent we want.
        if isinstance(result, tuple):
            latents, info = result
            if isinstance(info, dict) and "mean" in info:
                latents = info["mean"]
        else:
            latents = result
        return latents

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """``(B, 64, T_lat) → (B, 2, T_audio)``."""
        return self._model.decode(latent)

    # ------------------------------------------------------------------
    # Convenience: resample arbitrary-rate mono/stereo to the SAO sr
    # ------------------------------------------------------------------

    def resample_to_native_sr(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Resample ``audio`` from ``sr`` to the encoder's native sample rate.

        No-op if ``sr`` already matches. Useful in ``precompute_latents.py``
        when ingesting MP3/FLAC at heterogeneous sample rates.
        """
        if sr == self.metadata.sample_rate:
            return audio
        return torchaudio.functional.resample(audio, sr, self.metadata.sample_rate)

    # ------------------------------------------------------------------
    # Device / mode
    # ------------------------------------------------------------------

    def to(self, device: torch.device | str) -> "StableAudioOpenAutoencoder":
        self._model = self._model.to(device)
        return self

    def eval(self) -> "StableAudioOpenAutoencoder":
        self._model.eval()
        return self

    @property
    def device(self) -> torch.device:
        # The pretransform is an nn.Module; pull the device from its first param.
        return next(self._model.parameters()).device
