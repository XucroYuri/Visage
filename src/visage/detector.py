from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from .models import DetectedFace, FaceBox, ImageResult

logger = logging.getLogger(__name__)

# pyobjc Vision framework imports
try:
    from Foundation import NSURL
    from Vision import VNDetectFaceRectanglesRequest, VNImageRequestHandler

    _VISION_AVAILABLE = True
except ImportError:
    _VISION_AVAILABLE = False


def _check_vision() -> None:
    """Raise if Vision framework is not available."""
    if not _VISION_AVAILABLE:
        raise RuntimeError(
            "macOS Vision framework not available. "
            "Install pyobjc: pip install pyobjc-framework-Vision"
        )


def detect_faces(
    image_path: str,
    min_confidence: float = 0.5,
    min_face_size: int = 40,
) -> list[tuple[FaceBox, float]]:
    """Detect faces in an image using macOS Vision framework.

    Args:
        image_path: Path to the image file.
        min_confidence: Minimum detection confidence threshold (0-1).
        min_face_size: Minimum face bounding box dimension in pixels.

    Returns:
        List of (FaceBox, confidence) tuples for detected faces.
    """
    _check_vision()

    url = NSURL.fileURLWithPath_(image_path)
    handler = VNImageRequestHandler.alloc().initWithURL_options_(url, None)

    request = VNDetectFaceRectanglesRequest.alloc().init()

    success = handler.performRequests_error_([request], None)
    if not success:
        return []

    observations = request.results()
    if not observations:
        return []

    # Collect (normalized_coords, confidence) tuples from Vision observations
    # Vision uses normalized coordinates (0-1), origin bottom-left
    raw_faces: list[tuple[float, float, float, float, float]] = []
    for obs in observations:
        confidence = float(obs.confidence())
        if confidence < min_confidence:
            continue

        bbox = obs.boundingBox()
        raw_faces.append((
            float(bbox.origin.x),
            float(bbox.origin.y),
            float(bbox.size.width),
            float(bbox.size.height),
            confidence,
        ))

    if not raw_faces:
        return []

    # Get image pixel dimensions via PIL
    pixel_width, pixel_height = _get_image_dimensions(image_path)
    if pixel_width == 0 or pixel_height == 0:
        return []

    result: list[tuple[FaceBox, float]] = []
    for x, y, w, h, conf in raw_faces:
        # Convert normalized bottom-left origin to pixel top-left origin
        left = int(x * pixel_width)
        bottom = int(y * pixel_height)
        right = int((x + w) * pixel_width)
        top = int((1.0 - y - h) * pixel_height)

        face_box = FaceBox(top=top, right=right, bottom=bottom, left=left)

        # Filter by minimum face size
        if face_box.width < min_face_size or face_box.height < min_face_size:
            continue

        result.append((face_box, conf))

    return result


def _get_image_dimensions(image_path: str) -> tuple[int, int]:
    """Get image pixel dimensions using PIL (fast, no full decode).

    Returns:
        (width, height) in pixels.
    """
    from PIL import Image

    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        logger.warning("Failed to get image dimensions: %s", image_path, exc_info=True)
        return (0, 0)


def detect_faces_single(
    image_path: str,
    min_confidence: float = 0.5,
    min_face_size: int = 40,
) -> ImageResult:
    """Detect faces in a single image, returning an ImageResult.

    Args:
        image_path: Path to the image file.
        min_confidence: Minimum detection confidence.
        min_face_size: Minimum face dimension in pixels.

    Returns:
        ImageResult with detected faces or error info.
    """
    try:
        face_results = detect_faces(image_path, min_confidence, min_face_size)
        faces = [
            DetectedFace(
                face_box=fb,
                confidence=conf,
                image_path=image_path,
                face_index=i,
            )
            for i, (fb, conf) in enumerate(face_results)
        ]
        if not faces:
            return ImageResult(path=image_path, skipped=True)
        return ImageResult(path=image_path, faces=faces)
    except Exception as exc:
        logger.warning("Face detection failed for %s: %s", image_path, exc)
        return ImageResult(path=image_path, error=str(exc))


def detect_faces_batch(
    image_paths: list[str],
    min_confidence: float = 0.5,
    min_face_size: int = 40,
    max_workers: int = 4,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[ImageResult]:
    """Detect faces across multiple images with parallel processing.

    Args:
        image_paths: List of image file paths.
        min_confidence: Minimum detection confidence.
        min_face_size: Minimum face dimension.
        max_workers: Number of parallel threads.
        progress_callback: Called with (completed, total) after each image.

    Returns:
        List of ImageResult, one per image path.
    """
    _check_vision()

    results: dict[int, ImageResult] = {}
    completed = 0
    total = len(image_paths)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                detect_faces_single, path, min_confidence, min_face_size
            ): idx
            for idx, path in enumerate(image_paths)
        }

        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
            completed += 1
            if progress_callback:
                progress_callback(completed, total)

    return [results[i] for i in range(total)]
