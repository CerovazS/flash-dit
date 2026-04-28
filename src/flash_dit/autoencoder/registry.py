"""Registry for autoencoder backbones.

Decorator-based registration. ``build_autoencoder`` is the single entry
point used by ``scripts/train.py``, ``scripts/sample.py``, and
``scripts/precompute_latents.py``.

Each backbone subclass registers itself with ``@register_autoencoder("kind")``;
``build_autoencoder`` looks up the kind, strips it from the spec dict, and
forwards the remaining keys as kwargs to the subclass constructor.

Example::

    @register_autoencoder("stable_audio_open")
    class StableAudioOpenAutoencoder(AutoencoderBackbone):
        def __init__(self, hf_repo: str, cache_dir: str | None = None, ...): ...

    spec = {"kind": "stable_audio_open", "hf_repo": "stabilityai/...", ...}
    ae = build_autoencoder(spec)
"""
from __future__ import annotations

from typing import Any, Callable, Type

from .base import AutoencoderBackbone

_REGISTRY: dict[str, Type[AutoencoderBackbone]] = {}


def register_autoencoder(name: str) -> Callable[[Type[AutoencoderBackbone]], Type[AutoencoderBackbone]]:
    """Decorator: register a class under ``name`` so ``build_autoencoder`` can find it.

    Names must be unique. Re-registering a name raises ``ValueError`` —
    catch import-order bugs early rather than letting the second-imported
    backend silently override the first.
    """

    def _decorator(cls: Type[AutoencoderBackbone]) -> Type[AutoencoderBackbone]:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(
                f"autoencoder kind {name!r} already registered to "
                f"{_REGISTRY[name].__name__}, refusing to override with {cls.__name__}"
            )
        _REGISTRY[name] = cls
        return cls

    return _decorator


def list_backends() -> list[str]:
    """Return the registered backbone kinds, sorted alphabetically."""
    return sorted(_REGISTRY.keys())


def build_autoencoder(spec: dict[str, Any]) -> AutoencoderBackbone:
    """Instantiate a backbone from a Hydra-style config dict.

    ``spec`` must contain a ``kind`` key naming a registered backbone. All
    other keys are forwarded as keyword arguments to the backbone's
    ``__init__``.

    Args:
        spec: e.g. ``{"kind": "stable_audio_open", "hf_repo": "...", ...}``.
            Plain dicts and ``omegaconf.DictConfig`` both work — the function
            converts to ``dict`` internally.

    Returns:
        The constructed :class:`AutoencoderBackbone` subclass instance.

    Raises:
        KeyError: if ``spec`` has no ``kind`` field.
        ValueError: if the kind is not registered.
    """
    # Coerce DictConfig / OmegaConf containers to plain dict so we can pop().
    if hasattr(spec, "to_container"):
        cfg = dict(spec.to_container(resolve=True))  # type: ignore[arg-type]
    elif hasattr(spec, "__iter__"):
        cfg = dict(spec)
    else:
        raise TypeError(f"build_autoencoder expects a mapping, got {type(spec).__name__}")

    if "kind" not in cfg:
        raise KeyError("autoencoder spec missing required field 'kind'")
    kind = cfg.pop("kind")

    if kind not in _REGISTRY:
        raise ValueError(
            f"unknown autoencoder kind {kind!r}. "
            f"Registered: {list_backends()}"
        )

    cls = _REGISTRY[kind]
    return cls(**cfg)
