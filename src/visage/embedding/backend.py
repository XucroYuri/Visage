"""Embedding backend abstraction — re-exports from backends.py with service extensions."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

# Re-export the concrete backends from the legacy module for backward compatibility.
from visage.backends import DlibBackend, InsightFaceBackend  # noqa: F401
from visage.models import FaceBox


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Protocol for embedding backends.

    Moved here as the canonical location; backends.py re-exports for compatibility.
    """

    name: str
    embedding_dim: int

    def generate(self, image: np.ndarray, face_box: FaceBox) -> np.ndarray | None: ...
    def is_available(self) -> bool: ...


def create_backend(
    backend_name: str = "insightface",
    **kwargs: object,
) -> EmbeddingBackend:
    """Factory: create an embedding backend by name.

    Args:
        backend_name: "dlib" or "insightface".
        **kwargs: Passed to the backend constructor.

    Returns:
        An EmbeddingBackend instance.

    Raises:
        ValueError: Unknown backend name.
    """
    if backend_name == "insightface":
        from visage.backends import InsightFaceBackend
        return InsightFaceBackend(**kwargs)  # type: ignore[arg-type]
    elif backend_name == "dlib":
        from visage.backends import DlibBackend
        return DlibBackend(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"Unknown embedding backend: {backend_name!r}")
