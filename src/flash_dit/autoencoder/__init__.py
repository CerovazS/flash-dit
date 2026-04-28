"""Cross-modality autoencoder backbones (audio + vision).

See ``plan.md`` for the design. Public entry points:

- :class:`AutoencoderBackbone` — abstract interface every wrapper implements.
- :class:`AutoencoderMetadata` — frozen dataclass with shape / id metadata.
- :func:`build_autoencoder` — factory from a Hydra-style config dict.
- :func:`list_backends` — registered backbone kinds.

Concrete wrappers live in sibling modules and self-register via the
``@register_autoencoder`` decorator at import time.
"""
from .base import AutoencoderBackbone, AutoencoderMetadata, Domain
from .registry import build_autoencoder, list_backends, register_autoencoder

# Re-exporting the concrete wrapper class triggers registration via the
# @register_autoencoder decorator. Keep this at the bottom so the registry
# is already importable from .registry by the time the decorator fires.
from .stable_audio_open import StableAudioOpenAutoencoder

__all__ = [
    "AutoencoderBackbone",
    "AutoencoderMetadata",
    "Domain",
    "build_autoencoder",
    "list_backends",
    "register_autoencoder",
    "StableAudioOpenAutoencoder",
]
