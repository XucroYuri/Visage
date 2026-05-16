from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

from .cache import EmbeddingCache
from .heic import load_image_as_numpy
from .models import DetectedFace, FaceBox, ImageResult

logger = logging.getLogger(__name__)

try:
    import face_recognition

    _FR_AVAILABLE = True
except ImportError:
    _FR_AVAILABLE = False


def _check_face_recognition() -> None:
    """Raise if face_recognition library is not available."""
    if not _FR_AVAILABLE:
        raise RuntimeError(
            "face_recognition library not available. "
            "Install it: pip install face-recognition"
        )


def generate_embedding(
    image: np.ndarray,
    face_box: FaceBox,
    model: str = "small",
    num_jitters: int = 1,
) -> Optional[np.ndarray]:
    """Generate a 128-dim face embedding for a single detected face.

    Args:
        image: RGB numpy array of the full image, shape (H, W, 3).
        face_box: Bounding box of the face.
        model: "small" (fast) or "large" (accurate).
        num_jitters: Number of re-samples for the embedding.

    Returns:
        128-dimensional numpy array, or None if embedding fails.
    """
    _check_face_recognition()

    # face_recognition expects face locations as (top, right, bottom, left)
    location = face_box.to_face_recognition_format()

    try:
        encodings = face_recognition.face_encodings(
            image,
            known_face_locations=[location],
            num_jitters=num_jitters,
            model=model,
        )
        if encodings and len(encodings) > 0:
            return encodings[0]
    except Exception:
        logger.warning("Embedding generation failed for face at %s", face_box, exc_info=True)

    return None


def generate_embeddings_for_image(
    image_result: ImageResult,
    model: str = "small",
    num_jitters: int = 1,
) -> ImageResult:
    """Generate embeddings for all detected faces in a single image.

    Updates face embeddings in-place. Faces that fail to encode get None.

    Args:
        image_result: ImageResult with detected faces (no embeddings yet).
        model: Embedding model size.
        num_jitters: Re-sample count.

    Returns:
        The same ImageResult with embeddings populated.
    """
    if image_result.error or image_result.skipped or not image_result.faces:
        return image_result

    try:
        image_array = load_image_as_numpy(image_result.path)
    except Exception as exc:
        image_result.error = f"Cannot load image for embedding: {exc}"
        logger.warning("Failed to load %s for embedding: %s", image_result.path, exc)
        return image_result

    for face in image_result.faces:
        face.embedding = generate_embedding(
            image_array, face.face_box, model=model, num_jitters=num_jitters
        )

    # Filter out faces without embeddings
    image_result.faces = [f for f in image_result.faces if f.embedding is not None]

    return image_result


def generate_embeddings_batch(
    image_results: list[ImageResult],
    model: str = "small",
    num_jitters: int = 1,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cache: Optional[EmbeddingCache] = None,
) -> tuple[list[ImageResult], int]:
    """Generate embeddings for all detected faces across multiple images.

    Processes sequentially (face_recognition/dlib is not thread-safe).
    Skips images with no detected faces or errors from detection phase.
    Uses cache to skip unchanged images on re-runs.

    Args:
        image_results: List of ImageResult from detection phase.
        model: Embedding model size.
        num_jitters: Re-sample count.
        progress_callback: Called with (completed, total) after each image.
        cache: Optional EmbeddingCache for storing/retrieving results.

    Returns:
        Tuple of (updated ImageResult list, number of cache hits).
    """
    _check_face_recognition()

    # Only process images that have faces and no errors
    to_process = [
        (i, r) for i, r in enumerate(image_results)
        if r.faces and not r.error
    ]
    total = len(to_process)
    completed = 0
    cache_hits = 0

    for idx, result in to_process:
        # Try cache first
        if cache is not None:
            cached = cache.lookup(result.path, model=model, num_jitters=num_jitters)
            if cached is not None:
                result.faces = cached
                cache_hits += 1
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)
                continue

        # Compute embeddings
        generate_embeddings_for_image(result, model=model, num_jitters=num_jitters)

        # Store in cache
        if cache is not None and result.faces:
            cache.store(result.path, result.faces, model=model, num_jitters=num_jitters)

        completed += 1
        if progress_callback:
            progress_callback(completed, total)

    return image_results, cache_hits
