from __future__ import annotations

import gc
import logging
from collections.abc import Callable

import numpy as np

from .align import align_face
from .backends import EmbeddingBackend
from .cache import EmbeddingCache
from .head_features import extract_head_features
from .heic import load_image_as_numpy
from .models import FaceBox, ImageResult
from .quality import compute_combined_quality

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
    backend: EmbeddingBackend | None = None,
) -> np.ndarray | None:
    """Generate a face embedding for a single detected face.

    Args:
        image: RGB numpy array of the full image, shape (H, W, 3).
        face_box: Bounding box of the face.
        model: "small" (fast) or "large" (accurate) — dlib only.
        num_jitters: Number of re-samples for the embedding — dlib only.
        backend: Optional EmbeddingBackend instance. If provided, model/num_jitters
                 are ignored in favor of backend's own settings.

    Returns:
        Embedding vector, or None if embedding fails.
    """
    if backend is not None:
        return backend.generate(image, face_box)

    # Legacy path: use face_recognition directly
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
    backend: EmbeddingBackend | None = None,
    min_face_quality: float = 0.0,
    max_image_dimension: int = 0,
) -> ImageResult:
    """Generate embeddings for all detected faces in a single image.

    Updates face embeddings in-place. Faces that fail to encode or fall below
    the quality threshold get filtered out.

    Args:
        image_result: ImageResult with detected faces (no embeddings yet).
        model: Embedding model size — dlib only.
        num_jitters: Re-sample count — dlib only.
        backend: Optional EmbeddingBackend instance.
        min_face_quality: Minimum quality score [0, 1]; 0 = no filtering.
        max_image_dimension: If > 0, downscale image if its longest side exceeds
                             this value (saves memory for large photos).

    Returns:
        The same ImageResult with embeddings populated.
    """
    if image_result.error or image_result.skipped or not image_result.faces:
        return image_result

    try:
        image_array = load_image_as_numpy(
            image_result.path, max_dimension=max_image_dimension,
        )
    except Exception as exc:
        image_result.error = f"Cannot load image for embedding: {exc}"
        logger.warning("Failed to load %s for embedding: %s", image_result.path, exc)
        return image_result

    for face in image_result.faces:
        # Compute quality score (FIQA + legacy fusion)
        face.quality = compute_combined_quality(
            image_array, face.face_box, landmarks_5=face.landmarks_5,
        )

        # Skip faces below quality threshold
        if min_face_quality > 0 and face.quality < min_face_quality:
            logger.debug(
                "Face in %s below quality threshold (%.3f < %.3f)",
                image_result.path, face.quality, min_face_quality,
            )
            continue

        # Try face alignment for more stable embeddings
        aligned_image = None
        aligned_box = None
        if face.landmarks_5 is not None:
            aligned = align_face(image_array, face)
            if aligned is not None:
                aligned_image = aligned
                # Face fills the aligned image — create a full-image bbox
                sz = aligned.shape[0]
                aligned_box = FaceBox(top=0, right=sz, bottom=sz, left=0)

        # Use aligned image if available, otherwise use full image with face box
        src_image = aligned_image if aligned_image is not None else image_array
        src_box = aligned_box if aligned_box is not None else face.face_box

        face.embedding = generate_embedding(
            src_image, src_box, model=model, num_jitters=num_jitters,
            backend=backend,
        )

        # Extract head features for improved clustering
        if face.embedding is not None:
            face.head_features = extract_head_features(image_array, face.face_box)

    # Filter out faces without embeddings (low quality or failed encoding)
    image_result.faces = [f for f in image_result.faces if f.embedding is not None]

    return image_result


def generate_embeddings_batch(
    image_results: list[ImageResult],
    model: str = "small",
    num_jitters: int = 1,
    progress_callback: Callable[[int, int], None] | None = None,
    cache: EmbeddingCache | None = None,
    backend: EmbeddingBackend | None = None,
    min_face_quality: float = 0.0,
    max_image_dimension: int = 0,
) -> tuple[list[ImageResult], int]:
    """Generate embeddings for all detected faces across multiple images.

    Processes sequentially (face_recognition/dlib is not thread-safe).
    Skips images with no detected faces or errors from detection phase.
    Uses cache to skip unchanged images on re-runs.

    Args:
        image_results: List of ImageResult from detection phase.
        model: Embedding model size — dlib only.
        num_jitters: Re-sample count — dlib only.
        progress_callback: Called with (completed, total) after each image.
        cache: Optional EmbeddingCache for storing/retrieving results.
        backend: Optional EmbeddingBackend instance.
        min_face_quality: Minimum quality score [0, 1]; 0 = no filtering.
        max_image_dimension: If > 0, downscale images exceeding this dimension.

    Returns:
        Tuple of (updated ImageResult list, number of cache hits).
    """
    # Only check face_recognition if using the legacy path (no backend)
    if backend is None:
        _check_face_recognition()

    # Only process images that have faces and no errors
    to_process = [
        (i, r) for i, r in enumerate(image_results)
        if r.faces and not r.error
    ]
    total = len(to_process)
    completed = 0
    cache_hits = 0

    # Determine cache key model name
    cache_model = backend.name if backend else model

    for _idx, result in to_process:
        # Try cache first
        if cache is not None:
            cached = cache.lookup(result.path, model=cache_model, num_jitters=num_jitters)
            if cached is not None:
                result.faces = cached
                cache_hits += 1
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)
                continue

        # Compute embeddings
        generate_embeddings_for_image(
            result, model=model, num_jitters=num_jitters,
            backend=backend, min_face_quality=min_face_quality,
            max_image_dimension=max_image_dimension,
        )

        # Store in cache
        if cache is not None and result.faces:
            cache.store(result.path, result.faces, model=cache_model, num_jitters=num_jitters)

        # Free image memory after each encoding (InsightFace can be heavy)
        gc.collect()

        completed += 1
        if progress_callback:
            progress_callback(completed, total)

    return image_results, cache_hits
