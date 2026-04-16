"""Vendored subset of YoonjinXD/kadtk (MIT License).

Keeps only the PANNs embedding pipeline; the MMD computation is provided
by ``flash_dit.evaluation.mmd`` (chunked, paper-correct default bandwidth).
Upstream source: https://github.com/YoonjinXD/kadtk
"""
from .emb_loader import EmbeddingLoader, cache_embedding_files
from .model_loader import ModelLoader, PANNsModel, get_model, list_models
from .utils import get_cache_embedding_path

__all__ = [
    "EmbeddingLoader",
    "ModelLoader",
    "PANNsModel",
    "cache_embedding_files",
    "get_cache_embedding_path",
    "get_model",
    "list_models",
]
