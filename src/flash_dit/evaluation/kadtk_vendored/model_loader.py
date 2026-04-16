"""Pretrained audio embedding model loaders (trimmed vendor of kadtk.model_loader).

Keeps only ``PANNsModel`` — the default KAD paper backbone, PyTorch-only,
no TensorFlow / kapre / hear21passt baggage. Add more ``ModelLoader``
subclasses here if additional backbones are needed later; structure follows
upstream kadtk so lifting extra classes is mechanical.

The full upstream file pulls in ~15 model classes (CLAP, MERT, Whisper,
W2V2, HuBERT, WavLM, OpenL3, PaSST, ...), each with its own heavy deps.
None of those are needed for the KAD default — keeping the scope tight
avoids dependency conflicts (notably ``transformers<4.47`` required by
upstream kadtk but incompatible with the main project).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import soundfile
import torch
from torch import nn

from . import panns
from .utils import download_file

log = logging.getLogger(__name__)


class ModelLoader(ABC):
    """Abstract contract: load a pretrained audio encoder and produce embeddings.

    Subclasses implement ``load_model`` (populate ``self.model``) and
    ``_get_embedding`` (map mono audio → ``(n_frames, num_features)``
    torch.Tensor). Output shape is enforced at the channel axis by
    ``get_embedding``.
    """

    def __init__(
        self,
        name: str,
        num_features: int,
        sr: int,
        audio_len: Optional[Union[float, int]] = None,
    ) -> None:
        self.audio_len = audio_len
        self.model: nn.Module | None = None
        self.sr = sr
        self.num_features = num_features
        self.name = name
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    @torch.no_grad()
    def get_embedding(self, audio: np.ndarray) -> np.ndarray:
        embd = self._get_embedding(audio)
        if self.device.type == "cuda":
            embd = embd.cpu()
        embd = embd.detach().numpy()
        if embd.shape[-1] != self.num_features:
            raise RuntimeError(
                f"[{self.name}]: expected {self.num_features} features, got {embd.shape[-1]}"
            )
        if embd.dtype == np.float32:
            embd = embd.astype(np.float16)
        return embd

    @abstractmethod
    def load_model(self) -> None: ...

    @abstractmethod
    def _get_embedding(self, audio: np.ndarray) -> torch.Tensor: ...

    def load_wav(self, wav_file: Path) -> np.ndarray:
        """Read a mono PCM-16 wav at ``self.sr`` and scale to [-1, 1]."""
        wav_data, _ = soundfile.read(wav_file, dtype="int16")
        wav_data = wav_data / 32768.0
        if self.audio_len is not None and wav_data.shape[0] != int(self.audio_len * self.sr):
            raise RuntimeError(
                f"Audio length mismatch ({wav_data.shape[0] / self.sr:.2f}s != "
                f"{self.audio_len}s) for {wav_file}"
            )
        return wav_data


class PANNsModel(ModelLoader):
    """PANNs embedding model (Kong et al., IEEE/ACM TASLP 2020).

    Three variants; ``wavegram-logmel`` is the KAD paper default for
    general-purpose audio. Weights are downloaded once to
    ``<this_file_parent>/.model-checkpoints/`` and reused.
    """

    _URLS = {
        "cnn14-32k": "https://zenodo.org/record/3576403/files/Cnn14_mAP%3D0.431.pth",
        "cnn14-16k": "https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth",
        "wavegram-logmel": "https://zenodo.org/records/3987831/files/Wavegram_Logmel_Cnn14_mAP%3D0.439.pth",
    }

    def __init__(
        self,
        variant: Literal["cnn14-32k", "cnn14-16k", "wavegram-logmel"],
        audio_len: Optional[Union[float, int]] = None,
    ) -> None:
        super().__init__(
            f"panns-{variant}",
            num_features=2048,
            sr=16000 if variant == "cnn14-16k" else 32000,
            audio_len=audio_len,
        )
        self.variant = variant

    def load_model(self) -> None:
        url = self._URLS[self.variant]
        ckpt = (
            Path(__file__).parent
            / ".model-checkpoints"
            / url.split("/")[-1].replace("%3D", "=")
        )
        if not ckpt.exists():
            log.info(f"Downloading PANNs weights for {self.variant}")
            download_file(url, ckpt)
        self.model_file = ckpt

        features_list = ["2048", "logits"]
        if self.variant == "cnn14-16k":
            self.model = panns.Cnn14(
                features_list=features_list,
                sample_rate=16000, window_size=512, hop_size=160,
                mel_bins=64, fmin=50, fmax=8000, classes_num=527,
            )
        elif self.variant == "cnn14-32k":
            self.model = panns.Cnn14(
                features_list=features_list,
                sample_rate=32000, window_size=1024, hop_size=320,
                mel_bins=64, fmin=50, fmax=14000, classes_num=527,
            )
        elif self.variant == "wavegram-logmel":
            self.model = panns.Wavegram_Logmel_Cnn14(
                sample_rate=32000, window_size=1024, hop_size=320,
                mel_bins=64, fmin=50, fmax=14000, classes_num=527,
            )
        else:
            raise ValueError(f"Unknown PANNs variant: {self.variant}")

        state = torch.load(self.model_file, weights_only=False, map_location="cpu")
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.model.to(self.device)

    def _get_embedding(self, audio: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(audio).float().to(self.device)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        if "cnn14" in self.variant:
            emb = self.model.forward(t)["2048"]
        else:
            emb = self.model.forward(t)["embedding"]
        return emb


_MODEL_REGISTRY: dict[str, type[ModelLoader]] = {
    "panns-cnn14-32k": lambda **kw: PANNsModel("cnn14-32k", **kw),
    "panns-cnn14-16k": lambda **kw: PANNsModel("cnn14-16k", **kw),
    "panns-wavegram-logmel": lambda **kw: PANNsModel("wavegram-logmel", **kw),
}


def get_model(name: str, **kwargs) -> ModelLoader:
    """Factory for vendored audio embedding models.

    Current registry: ``panns-cnn14-32k``, ``panns-cnn14-16k``,
    ``panns-wavegram-logmel`` (paper default). Unknown names raise with a
    list of supported identifiers.
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model {name!r}. Supported: {sorted(_MODEL_REGISTRY)}"
        )
    return _MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    return sorted(_MODEL_REGISTRY)
