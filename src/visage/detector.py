"""Face detection — delegates to configured backend.

This module provides backward-compatible facades (detect_faces_single,
detect_faces_batch) that internally use DetectorBackend instances.

Backward-compat re-exports from detectors sub-packages are provided
so existing test mocks and callers continue to work.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from .detectors import DetectorBackend
from .detectors.nms import _nms as _nms
from .detectors.vision import (
    _LANDMARKS_CRASH_BUG as _LANDMARKS_CRASH_BUG,
)
from .detectors.vision import (
    _VISION_AVAILABLE as _VISION_AVAILABLE,
)
from .detectors.vision import (
    NSURL as NSURL,
)
from .detectors.vision import (
    VisionDetector,
)
from .detectors.vision import (
    VNDetectFaceLandmarksRequest as VNDetectFaceLandmarksRequest,
)
from .detectors.vision import (
    VNDetectFaceRectanglesRequest as VNDetectFaceRectanglesRequest,
)
from .detectors.vision import (
    VNImageRequestHandler as VNImageRequestHandler,
)
from .detectors.vision import (
    _bbox_from_contour as _bbox_from_contour,
)
from .detectors.vision import (
    _bbox_from_median as _bbox_from_median,
)
from .detectors.vision import (
    _check_vision as _check_vision,
)
from .detectors.vision import (
    _extract_5_landmarks as _extract_5_landmarks,
)
from .detectors.vision import (
    _get_image_dimensions as _get_image_dimensions,
)
from .detectors.vision import (
    _safe_points as _safe_points,
)
from .detectors.vision import (
    _shrink_body_bbox as _shrink_body_bbox,
)
from .heic import load_image_as_numpy
from .models import DetectedFace, FaceBox, ImageResult

logger = logging.getLogger(__name__)


# ── Shared post-processing ────────────────────────────────────────


def _mark_primary_faces(
    faces: list[DetectedFace],
    image_width: int,
    image_height: int,
) -> None:
    """Mark the primary (main) face in each image.

    Uses a weighted scoring formula:
    - 50%: face bounding box area relative to image area
    - 30%: face center proximity to image center (normalized 0-1)
    - 20%: detection confidence

    Only one face per image is marked as primary. Single-face images
    always have their only face marked as primary.

    Mutates DetectedFace objects in place.
    """
    if not faces:
        return
    if len(faces) == 1:
        faces[0].is_primary = True
        return

    img_area = image_width * image_height
    if img_area <= 0:
        faces[0].is_primary = True
        return

    img_cx = image_width / 2.0
    img_cy = image_height / 2.0
    max_dist = ((image_width / 2.0) ** 2 + (image_height / 2.0) ** 2) ** 0.5

    best_idx = 0
    best_score = -1.0

    for i, face in enumerate(faces):
        bbox = face.face_box
        face_area = bbox.area / img_area

        face_cx = (bbox.left + bbox.right) / 2.0
        face_cy = (bbox.top + bbox.bottom) / 2.0
        dist = ((face_cx - img_cx) ** 2 + (face_cy - img_cy) ** 2) ** 0.5
        center_prox = 1.0 - min(dist / max_dist, 1.0) if max_dist > 0 else 1.0

        score = 0.5 * face_area + 0.3 * center_prox + 0.2 * face.confidence
        if score > best_score:
            best_score = score
            best_idx = i

    faces[best_idx].is_primary = True


# ── Legacy Vision entry point ─────────────────────────────────────


def detect_faces(
    image_path: str,
    min_confidence: float = 0.5,
    min_face_size: int = 40,
) -> tuple[
    list[tuple[FaceBox, float, list[tuple[float, float]] | None]],
    dict[str, int | float],
]:
    """Detect faces using macOS Vision framework (legacy entry point).

    Delegates to VisionDetector. Kept for backward compatibility.

    Args:
        image_path: Path to the image file.
        min_confidence: Minimum detection confidence threshold (0-1).
        min_face_size: Minimum face bounding box dimension in pixels.

    Returns:
        Tuple of (detection list, stats dict).
    """
    _check_vision()
    detector = VisionDetector(
        min_confidence=min_confidence, min_face_size=min_face_size,
    )
    return detector.detect_from_path(image_path)


# ── Single image detection ────────────────────────────────────────


def detect_faces_single(
    image_path: str,
    min_confidence: float = 0.5,
    min_face_size: int = 40,
    detector: DetectorBackend | None = None,
) -> ImageResult:
    """Detect faces in a single image, returning an ImageResult.

    Args:
        image_path: Path to the image file.
        min_confidence: Minimum detection confidence (Vision-only).
        min_face_size: Minimum face dimension in pixels.
        detector: Optional detector backend. If None, uses Vision on macOS.

    Returns:
        ImageResult with detected faces or error info.
    """
    try:
        if detector is None:
            # Legacy Vision path
            img_w, img_h = _get_image_dimensions(image_path)
            face_results, stats = detect_faces(
                image_path, min_confidence, min_face_size,
            )
        else:
            # Generic detector path — load image and run detector
            try:
                image_array = load_image_as_numpy(image_path)
            except Exception as exc:
                logger.warning("Cannot load %s for detection: %s", image_path, exc)
                return ImageResult(path=image_path, error=f"Cannot load image: {exc}")

            img_h, img_w = image_array.shape[:2]
            raw_results = detector.detect(image_array)
            face_results = raw_results  # already a list
            stats = None

        # Apply NMS to remove duplicate detections
        face_results = _nms(face_results)

        faces = [
            DetectedFace(
                face_box=fb,
                confidence=conf,
                landmarks_5=lm5,
                image_path=image_path,
                face_index=i,
            )
            for i, (fb, conf, lm5) in enumerate(face_results)
        ]
        _mark_primary_faces(faces, img_w, img_h)

        if not faces:
            return ImageResult(
                path=image_path, skipped=True,
                image_width=img_w, image_height=img_h,
            )
        return ImageResult(
            path=image_path, faces=faces, detection_stats=stats,
            image_width=img_w, image_height=img_h,
        )
    except Exception as exc:
        logger.warning("Face detection failed for %s: %s", image_path, exc)
        return ImageResult(path=image_path, error=str(exc))


# ── Batch detection ───────────────────────────────────────────────


def detect_faces_batch(
    image_paths: list[str],
    min_confidence: float = 0.5,
    min_face_size: int = 40,
    max_workers: int = 4,
    progress_callback: Callable[[int, int], None] | None = None,
    detector: DetectorBackend | None = None,
) -> tuple[list[ImageResult], dict[str, int | float]]:
    """Detect faces across multiple images with parallel processing.

    Args:
        image_paths: List of image file paths.
        min_confidence: Minimum detection confidence (Vision-only).
        min_face_size: Minimum face dimension (Vision-only).
        max_workers: Number of parallel threads.
        progress_callback: Called with (completed, total) after each image.
        detector: Optional detector backend. If None, uses Vision on macOS.

    Returns:
        Tuple of (list of ImageResult, aggregated stats dict).
    """
    if detector is None:
        _check_vision()

    results: dict[int, ImageResult] = {}
    completed = 0
    total = len(image_paths)
    agg_stats: dict[str, int | float] = {
        "total": 0, "contour": 0, "median": 0, "default": 0,
        "shrunk": 0, "aspect_ratio_sum": 0.0,
    }

    # Build a partial for passing detector to each call
    from functools import partial

    detect_fn = partial(
        detect_faces_single,
        min_confidence=min_confidence,
        min_face_size=min_face_size,
        detector=detector,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(detect_fn, path): idx
            for idx, path in enumerate(image_paths)
        }

        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            results[idx] = result
            if result.detection_stats:
                for key in agg_stats:
                    agg_stats[key] = agg_stats.get(key, 0) + result.detection_stats.get(key, 0)
            completed += 1
            if progress_callback:
                progress_callback(completed, total)

    return [results[i] for i in range(total)], agg_stats
