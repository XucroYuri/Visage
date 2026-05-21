"""Embedding service — pluggable backends, request batching, GPU detection."""

from .backend import EmbeddingBackend
from .batcher import EmbeddingRequest, RequestBatcher
from .gpu import DeviceInfo, detect_device

__all__ = [
    "EmbeddingBackend",
    "DeviceInfo",
    "detect_device",
    "EmbeddingRequest",
    "RequestBatcher",
]
