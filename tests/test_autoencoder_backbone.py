"""Tests for the AutoencoderBackbone ABC, registry, and metadata helpers.

Covers everything that doesn't require downloading the actual SAO weights
— so these tests run on CI / local without HF_TOKEN. The SAO wrapper is
only imported (not instantiated) to verify it self-registered correctly.

Real-weight smoke tests live separately and are skipped without HF_TOKEN.
"""
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest
import torch

from flash_dit.autoencoder import (
    AutoencoderBackbone,
    AutoencoderMetadata,
    build_autoencoder,
    list_backends,
    register_autoencoder,
)
from flash_dit.autoencoder.base import Domain


def _make_audio_metadata(**overrides) -> AutoencoderMetadata:
    base = dict(
        encoder_id="fake_audio",
        encoder_version="0.0.0",
        encoder_config={"kind": "fake_audio"},
        domain="audio",
        input_format="BCT",
        latent_format="BCT",
        input_channels=2,
        latent_channels=64,
        compression_ratios=(2048,),
        sample_rate=44100,
    )
    base.update(overrides)
    return AutoencoderMetadata(**base)


def _make_vision_metadata(**overrides) -> AutoencoderMetadata:
    base = dict(
        encoder_id="fake_vision",
        encoder_version="0.0.0",
        encoder_config={"kind": "fake_vision"},
        domain="vision",
        input_format="BCHW",
        latent_format="BCHW",
        input_channels=3,
        latent_channels=4,
        compression_ratios=(8, 8),
        sample_rate=None,
    )
    base.update(overrides)
    return AutoencoderMetadata(**base)


class _FakeAudio(AutoencoderBackbone):
    """Minimum-viable audio backbone for ABC + registry tests."""

    def __init__(self, latent_channels: int = 64) -> None:
        self.metadata = _make_audio_metadata(latent_channels=latent_channels)
        self._device = torch.device("cpu")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        b, _, t = x.shape
        return torch.zeros(b, self.metadata.latent_channels, t // 2048,
                           device=self._device)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        b, _, t_lat = z.shape
        return torch.zeros(b, self.metadata.input_channels, t_lat * 2048,
                           device=self._device)

    def to(self, device):
        self._device = torch.device(device)
        return self

    def eval(self):
        return self

    @property
    def device(self) -> torch.device:
        return self._device


class _FakeVision(AutoencoderBackbone):
    def __init__(self) -> None:
        self.metadata = _make_vision_metadata()
        self._device = torch.device("cpu")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        return torch.zeros(b, self.metadata.latent_channels, h // 8, w // 8,
                           device=self._device)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        b, _, h_lat, w_lat = z.shape
        return torch.zeros(b, self.metadata.input_channels, h_lat * 8, w_lat * 8,
                           device=self._device)

    def to(self, device):
        self._device = torch.device(device)
        return self

    def eval(self):
        return self

    @property
    def device(self) -> torch.device:
        return self._device


# ----------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------


def test_metadata_is_frozen():
    md = _make_audio_metadata()
    with pytest.raises(FrozenInstanceError):
        md.encoder_id = "other"  # type: ignore[misc]


def test_metadata_to_h5_attrs_audio():
    md = _make_audio_metadata()
    attrs = md.to_h5_attrs()
    assert attrs["encoder_id"] == "fake_audio"
    assert attrs["domain"] == "audio"
    assert attrs["input_format"] == "BCT"
    assert attrs["latent_format"] == "BCT"
    assert attrs["latent_channels"] == 64
    # compression_ratios is a list (h5py-friendly), not a tuple
    assert attrs["compression_ratios"] == [2048]
    assert attrs["sample_rate"] == 44100
    # encoder_config_json is parseable
    assert json.loads(attrs["encoder_config_json"]) == {"kind": "fake_audio"}


def test_metadata_to_h5_attrs_vision():
    md = _make_vision_metadata()
    attrs = md.to_h5_attrs()
    assert attrs["domain"] == "vision"
    assert attrs["input_format"] == "BCHW"
    assert attrs["compression_ratios"] == [8, 8]
    # sample_rate=None becomes -1 in HDF5 (h5py rejects None)
    assert attrs["sample_rate"] == -1


# ----------------------------------------------------------------------
# latent_shape_for
# ----------------------------------------------------------------------


def test_latent_shape_for_audio_8s():
    """8 s @ 44100 / 2048 → 172 (matches the FMA seq_len in conf/data/fma_medium.yaml)."""
    backbone = _FakeAudio()
    assert backbone.latent_shape_for(seconds=8.0) == (172,)


def test_latent_shape_for_audio_other_horizons():
    backbone = _FakeAudio()
    # 10 s should round to ceil(10 * 44100 / 2048) → 215
    assert backbone.latent_shape_for(seconds=10.0) == (215,)
    # 1 s → 22 frames (round of 21.5)
    assert backbone.latent_shape_for(seconds=1.0) == (22,)


def test_latent_shape_for_vision_256():
    backbone = _FakeVision()
    assert backbone.latent_shape_for(image_size=256) == (32, 32)


def test_latent_shape_for_vision_non_square_strides():
    md = _make_vision_metadata(compression_ratios=(8, 16))
    backbone = _FakeVision()
    object.__setattr__(backbone, "metadata", md)
    assert backbone.latent_shape_for(image_size=256) == (32, 16)


def test_latent_shape_for_audio_missing_kwarg():
    backbone = _FakeAudio()
    with pytest.raises(KeyError):
        backbone.latent_shape_for(image_size=256)


def test_latent_shape_for_audio_no_sample_rate():
    md = _make_audio_metadata(sample_rate=None)
    backbone = _FakeAudio()
    object.__setattr__(backbone, "metadata", md)
    with pytest.raises(ValueError, match="no sample_rate"):
        backbone.latent_shape_for(seconds=8.0)


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------


def test_sao_is_registered():
    """Importing the package self-registers the SAO wrapper."""
    assert "stable_audio_open" in list_backends()


def test_register_autoencoder_rejects_duplicate():
    @register_autoencoder("dup_test")
    class _A(_FakeAudio):
        pass

    with pytest.raises(ValueError, match="already registered"):
        @register_autoencoder("dup_test")
        class _B(_FakeAudio):
            pass


def test_register_autoencoder_idempotent_on_same_class():
    @register_autoencoder("idem_test")
    class A(_FakeAudio):
        pass

    # Re-applying the same decorator to the same class should not raise.
    register_autoencoder("idem_test")(A)


def test_build_autoencoder_unknown_kind():
    with pytest.raises(ValueError, match="unknown autoencoder kind"):
        build_autoencoder({"kind": "definitely_not_a_real_encoder"})


def test_build_autoencoder_missing_kind():
    with pytest.raises(KeyError, match="missing required field 'kind'"):
        build_autoencoder({"hf_repo": "..."})


def test_build_autoencoder_returns_correct_class():
    @register_autoencoder("build_test")
    class A(_FakeAudio):
        def __init__(self, foo: str = "bar") -> None:  # pragma: no cover - trivial
            super().__init__()
            self.foo = foo

    instance = build_autoencoder({"kind": "build_test", "foo": "baz"})
    assert isinstance(instance, A)
    assert instance.foo == "baz"


def test_build_autoencoder_accepts_omegaconf():
    """DictConfig with .to_container should work without a manual cast."""
    pytest.importorskip("omegaconf")
    from omegaconf import OmegaConf

    @register_autoencoder("oc_test")
    class A(_FakeAudio):
        def __init__(self, latent_channels: int = 64) -> None:
            super().__init__(latent_channels=latent_channels)

    cfg = OmegaConf.create({"kind": "oc_test", "latent_channels": 32})
    instance = build_autoencoder(cfg)
    assert isinstance(instance, A)
    assert instance.metadata.latent_channels == 32


# ----------------------------------------------------------------------
# Fake backbone round-trip (smoke for ABC contract)
# ----------------------------------------------------------------------


def test_fake_audio_roundtrip_shape():
    backbone = _FakeAudio()
    audio = torch.randn(2, 2, 44100)
    z = backbone.encode(audio)
    assert z.shape == (2, 64, 44100 // 2048)
    audio_back = backbone.decode(z)
    assert audio_back.shape[0] == audio.shape[0]
    assert audio_back.shape[1] == audio.shape[1]


def test_fake_vision_roundtrip_shape():
    backbone = _FakeVision()
    img = torch.randn(2, 3, 256, 256)
    z = backbone.encode(img)
    assert z.shape == (2, 4, 32, 32)
    img_back = backbone.decode(z)
    assert img_back.shape == (2, 3, 256, 256)


def test_domain_type_alias():
    """Domain literal accepts only the two declared values."""
    # Type-checked at static time; smoke-check the actual values still work.
    md_a = _make_audio_metadata()
    md_v = _make_vision_metadata()
    assert md_a.domain == "audio"
    assert md_v.domain == "vision"
    # The Domain alias is exported so callers can annotate function args.
    domains: list[Domain] = ["audio", "vision"]
    assert domains == ["audio", "vision"]
