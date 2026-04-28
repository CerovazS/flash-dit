"""Abstract base class for autoencoder backbones.

See ``plan.md`` §4 for the full design rationale. The summary:

- One uniform interface across audio (`(B, C, T)`) and vision (`(B, C, H, W)`)
  autoencoders. The DiT and dataset code dispatches on ``metadata.domain``
  and ``metadata.latent_format`` rather than on the encoder identity.
- Wrappers absorb every backend-specific quirk (mono/stereo handling,
  resampling/resize, prefix stripping, RVQ pre-quantizer extraction,
  tuple/dict return unwrapping). The ABC stays narrow.
- ``encode()`` always returns continuous tensors. RVQ codecs bind to
  the pre-quantizer output; integer codes never leak out of the wrapper.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import torch

Domain = Literal["audio", "vision"]


@dataclass(frozen=True)
class AutoencoderMetadata:
    """Identifying + shape metadata for an autoencoder backbone.

    The fields are exactly the ones we need to (a) make the DiT shape-
    agnostic via Hydra interpolation, (b) round-trip latents through HDF5
    with full reproducibility, and (c) fail fast at load time when the
    encoder used to produce a latent file disagrees with the encoder
    configured for the run.

    Attributes:
        encoder_id: stable identifier (also the HDF5 attr value).
        encoder_version: upstream library version, HF revision, or commit SHA.
        encoder_config: serialisable snapshot stored as JSON in HDF5.
        domain: ``"audio"`` or ``"vision"``.
        input_format: ``"BCT"`` (audio) or ``"BCHW"`` (vision).
        latent_format: same vocabulary; almost always equals input_format.
        input_channels: 1 / 2 (audio) or 3 (RGB vision).
        latent_channels: ``C`` of the latent tensor.
        compression_ratios: per-axis stride. Audio: ``(stride,)``. Vision:
            ``(stride_h, stride_w)``. Always a tuple — readers never branch
            on ``domain`` to interpret it.
        sample_rate: only meaningful for audio; ``None`` for vision.
    """

    encoder_id: str
    encoder_version: str
    encoder_config: dict[str, Any]
    domain: Domain
    input_format: str
    latent_format: str
    input_channels: int
    latent_channels: int
    compression_ratios: tuple[int, ...]
    sample_rate: int | None = None

    def to_h5_attrs(self) -> dict[str, Any]:
        """Return a dict suitable for HDF5 attrs.

        ``encoder_config`` is serialised to a JSON string;
        ``compression_ratios`` is converted to a plain list since h5py's
        attribute writer is happier with a sequence than a tuple-of-int.
        """
        import json

        return {
            "encoder_id": self.encoder_id,
            "encoder_version": self.encoder_version,
            "encoder_config_json": json.dumps(self.encoder_config, sort_keys=True),
            "domain": self.domain,
            "input_format": self.input_format,
            "latent_format": self.latent_format,
            "input_channels": int(self.input_channels),
            "latent_channels": int(self.latent_channels),
            "compression_ratios": list(self.compression_ratios),
            "sample_rate": -1 if self.sample_rate is None else int(self.sample_rate),
        }


class AutoencoderBackbone(ABC):
    """Uniform interface every audio / vision autoencoder wrapper exposes.

    Subclasses set ``self.metadata`` in ``__init__`` and implement
    :meth:`encode` and :meth:`decode`. Everything else (mono/stereo
    coercion, resampling, RVQ pre-quantizer plumbing, library-specific
    return-type unwrapping) lives inside the subclass.
    """

    metadata: AutoencoderMetadata

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw input to a continuous latent.

        Audio:
            ``(B, input_channels, T_audio) → (B, latent_channels, T_lat)``.
        Vision:
            ``(B, input_channels, H, W) → (B, latent_channels, H_lat, W_lat)``.

        For RVQ codecs this MUST return the continuous pre-quantizer
        embedding, not integer codes.
        """

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`encode`. Round-trip preserves shape modulo
        stride padding."""

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------

    def latent_shape_for(self, **kwargs: Any) -> tuple[int, ...]:
        """Compute the latent spatial extent for a given input horizon.

        Audio:  ``latent_shape_for(seconds=8.0)`` → ``(T_lat,)``.
        Vision: ``latent_shape_for(image_size=256)`` → ``(H_lat, W_lat)``.
        """
        if self.metadata.domain == "audio":
            seconds = float(kwargs["seconds"])
            sr = self.metadata.sample_rate
            if sr is None:
                raise ValueError(
                    f"audio metadata for {self.metadata.encoder_id!r} has no sample_rate"
                )
            stride = self.metadata.compression_ratios[0]
            return (int(round(seconds * sr / stride)),)
        if self.metadata.domain == "vision":
            size = int(kwargs["image_size"])
            sh, sw = self.metadata.compression_ratios
            return (size // sh, size // sw)
        raise ValueError(f"unknown domain: {self.metadata.domain!r}")

    # ------------------------------------------------------------------
    # Device / mode management
    # ------------------------------------------------------------------

    @abstractmethod
    def to(self, device: torch.device | str) -> "AutoencoderBackbone":
        """Move the underlying model to ``device``. Returns self for chaining."""

    @abstractmethod
    def eval(self) -> "AutoencoderBackbone":
        """Switch the underlying model to eval mode. Returns self for chaining."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Current device of the underlying model."""
