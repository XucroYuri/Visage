from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from .models import DetectedFace, FaceBox, ImageResult

logger = logging.getLogger(__name__)

# pyobjc Vision framework imports
try:
    from Foundation import NSURL
    from Vision import (
        VNDetectFaceLandmarksRequest,
        VNDetectFaceRectanglesRequest,
        VNImageRequestHandler,
    )

    _VISION_AVAILABLE = True
except ImportError:
    _VISION_AVAILABLE = False

# macOS 26+ ships a VNFaceBBoxAligner that triggers intermittent SIGBUS
# crashes in the pyobjc bridge when processing certain AI-generated images
# via VNDetectFaceLandmarksRequest. Detect this and prefer the stable
# rectangles-only request on affected OS versions.
def _macos_major_version() -> int:
    import platform
    try:
        # macOS version string like "26.5" (or "15.2" on older releases)
        return int(platform.mac_ver()[0].split(".")[0])
    except (ValueError, IndexError):
        return 0

_LANDMARKS_CRASH_BUG = _macos_major_version() >= 26

# Padding ratios for expanding the tight face contour bbox.
# Upward padding captures hair, sideways capture ears.
_CONTOUR_PAD_UP = 0.45     # 45% of face height upward (hair)
_CONTOUR_PAD_SIDE = 0.20   # 20% of face width each side (ears)
_CONTOUR_PAD_DOWN = 0.10   # 10% of face height downward (chin margin)

# Body-shrink threshold: when bbox aspect ratio (height/width) exceeds this,
# the box likely includes shoulders/body below the head.
_BODY_ASPECT_RATIO_THRESHOLD = 1.8
# Target head aspect ratio — the head is roughly circular, so ~1.0-1.3
_TARGET_HEAD_ASPECT = 1.25
# Minimum fraction of the original box height to keep (safety floor)
_MIN_HEAD_FRACTION = 0.50


def _check_vision() -> None:
    """Raise if Vision framework is not available."""
    if not _VISION_AVAILABLE:
        raise RuntimeError(
            "macOS Vision framework not available. "
            "Install pyobjc: pip install pyobjc-framework-Vision"
        )


def _safe_points(region) -> list[tuple[float, float]]:
    """Extract (x, y) tuples from a landmark region via normalizedPoints.

    Converts the pyobjc varlist to a plain Python list in one shot,
    minimizing bridge calls that can trigger SIGBUS on macOS 26.5
    with the VNFaceBBoxAligner.
    """
    pts = region.normalizedPoints()
    try:
        return [(float(p.x), float(p.y)) for p in pts]
    except (TypeError, AttributeError):
        return []


def _bbox_from_contour(
    face_contour,
    pixel_width: int,
    pixel_height: int,
) -> FaceBox | None:
    """Compute a tight bounding box from face contour landmark points.

    Uses normalizedPoints (0-1, bottom-left origin) from the faceContour
    landmark region. Applies padding to capture hair and ears.

    Returns None if contour has no points.
    """
    if face_contour is None or face_contour.pointCount() == 0:
        return None

    pts = _safe_points(face_contour)
    if not pts:
        return None

    # Convert normalized bottom-left origin to pixel top-left origin
    xs = [x * pixel_width for x, _ in pts]
    ys = [(1.0 - y) * pixel_height for _, y in pts]

    tight_left = min(xs)
    tight_right = max(xs)
    tight_top = min(ys)
    tight_bottom = max(ys)

    # Apply padding to capture hair/ears
    face_w = tight_right - tight_left
    face_h = tight_bottom - tight_top

    left = max(0, int(tight_left - face_w * _CONTOUR_PAD_SIDE))
    right = min(pixel_width, int(tight_right + face_w * _CONTOUR_PAD_SIDE))
    top = max(0, int(tight_top - face_h * _CONTOUR_PAD_UP))
    bottom = min(pixel_height, int(tight_bottom + face_h * _CONTOUR_PAD_DOWN))

    return FaceBox(top=top, right=right, bottom=bottom, left=left)


def _bbox_from_median(
    median_line,
    left_eye,
    right_eye,
    pixel_width: int,
    pixel_height: int,
) -> FaceBox | None:
    """Compute a bounding box from median line and eye landmarks.

    Used as a fallback when face contour is unavailable. The median line
    provides forehead-to-chin extent, and the eye separation provides
    a width estimate.

    Returns None if required landmarks are missing or have insufficient points.
    """
    if median_line is None or median_line.pointCount() == 0:
        return None

    # Get median line points (forehead → chin)
    median_pts = _safe_points(median_line)
    if not median_pts:
        return None

    ys = [(1.0 - y) * pixel_height for _, y in median_pts]
    xs = [x * pixel_width for x, _ in median_pts]

    top = min(ys)
    bottom = max(ys)
    center_x = sum(xs) / len(xs)

    # Estimate face width from eye separation (if available) or
    # fall back to a fraction of face height
    if (
        left_eye is not None and left_eye.pointCount() > 0
        and right_eye is not None and right_eye.pointCount() > 0
    ):
        left_eye_pts = _safe_points(left_eye)
        right_eye_pts = _safe_points(right_eye)
        if not left_eye_pts or not right_eye_pts:
            face_h = bottom - top
            half_w = face_h * 0.5
        else:
            left_eye_center_x = sum(x for x, _ in left_eye_pts) / len(left_eye_pts)
            right_eye_center_x = sum(x for x, _ in right_eye_pts) / len(right_eye_pts)
            eye_dist = abs(right_eye_center_x - left_eye_center_x) * pixel_width
            # Face width ≈ 2.5x inter-eye distance
            face_w = int(eye_dist * 2.5)
            half_w = face_w / 2
    else:
        # No eye data — estimate width from face height (head ≈ circular)
        face_h = bottom - top
        half_w = face_h * 0.5

    # Apply padding for hair/ears
    face_h = bottom - top
    left = max(0, int(center_x - half_w - face_h * _CONTOUR_PAD_SIDE))
    right = min(pixel_width, int(center_x + half_w + face_h * _CONTOUR_PAD_SIDE))
    top = max(0, int(top - face_h * _CONTOUR_PAD_UP))
    bottom = min(pixel_height, int(bottom + face_h * _CONTOUR_PAD_DOWN))

    return FaceBox(top=top, right=right, bottom=bottom, left=left)


def _shrink_body_bbox(face_box: FaceBox) -> FaceBox:
    """Shrink a bbox that likely includes body below the head.

    When the aspect ratio (height/width) exceeds _BODY_ASPECT_RATIO_THRESHOLD,
    the bbox probably includes shoulders/body. Trim from the bottom to reach
    a target head aspect ratio.
    """
    fw = face_box.width
    fh = face_box.height

    if fw <= 0 or fh <= 0:
        return face_box

    aspect = fh / fw
    if aspect <= _BODY_ASPECT_RATIO_THRESHOLD:
        return face_box  # already reasonable

    # Calculate target head height
    target_h = int(fw * _TARGET_HEAD_ASPECT)
    # Don't shrink below the safety floor
    min_h = int(fh * _MIN_HEAD_FRACTION)
    target_h = max(target_h, min_h)

    # Shrink from the bottom (keep the top where the head is)
    return FaceBox(
        top=face_box.top,
        right=face_box.right,
        bottom=face_box.top + target_h,
        left=face_box.left,
    )


def detect_faces(
    image_path: str,
    min_confidence: float = 0.5,
    min_face_size: int = 40,
) -> tuple[
    list[tuple[FaceBox, float, list[tuple[float, float]] | None]],
    dict[str, int],
]:
    """Detect faces in an image using macOS Vision framework.

    Uses VNDetectFaceLandmarksRequest which provides both face bounding
    boxes and facial landmarks. When face contour landmarks are available,
    computes a tighter bounding box from the contour (with padding for
    hair/ears) instead of the overly large default Vision bbox.

    Also extracts 5 facial landmarks for alignment:
    (left_eye, right_eye, nose_tip, left_mouth, right_mouth).

    Args:
        image_path: Path to the image file.
        min_confidence: Minimum detection confidence threshold (0-1).
        min_face_size: Minimum face bounding box dimension in pixels.

    Returns:
        Tuple of:
        - List of (FaceBox, confidence, landmarks_5) tuples
        - Stats dict with keys: total, contour, median, default, shrunk, aspect_ratio_sum
    """
    _check_vision()

    url = NSURL.fileURLWithPath_(image_path)
    handler = VNImageRequestHandler.alloc().initWithURL_options_(url, None)

    # On macOS 26+, VNDetectFaceLandmarksRequest can trigger SIGBUS in the
    # pyobjc bridge (VNFaceBBoxAligner bug). Fall back to the stable
    # rectangles-only request. We lose contour-based bbox refinement and
    # 5-point landmarks, but detection succeeds reliably.
    use_simple_detection = _LANDMARKS_CRASH_BUG
    if use_simple_detection:
        request = VNDetectFaceRectanglesRequest.alloc().init()
    else:
        request = VNDetectFaceLandmarksRequest.alloc().init()

    result: list[tuple[FaceBox, float, list[tuple[float, float]] | None]] = []
    stats = {
        "total": 0, "contour": 0, "median": 0, "default": 0,
        "shrunk": 0, "aspect_ratio_sum": 0.0,
    }

    success = handler.performRequests_error_([request], None)
    if not success:
        return [], stats

    observations = request.results()
    if not observations:
        return [], stats

    # Get image pixel dimensions via PIL
    pixel_width, pixel_height = _get_image_dimensions(image_path)
    if pixel_width == 0 or pixel_height == 0:
        return [], stats

    for obs in observations:
        confidence = float(obs.confidence())
        if confidence < min_confidence:
            continue

        # Try to compute tight bbox from face contour landmarks (preferred)
        face_box = None
        bbox_source = "default"
        landmarks = obs.landmarks()
        if landmarks is not None:
            face_box = _bbox_from_contour(
                landmarks.faceContour(), pixel_width, pixel_height,
            )
            if face_box is not None:
                bbox_source = "contour"
            else:
                # Fallback: median line + eye landmarks
                face_box = _bbox_from_median(
                    landmarks.medianLine(),
                    landmarks.leftEye(),
                    landmarks.rightEye(),
                    pixel_width,
                    pixel_height,
                )
                if face_box is not None:
                    bbox_source = "median"

        # Final fallback: Vision's default bounding box
        if face_box is None:
            bbox = obs.boundingBox()
            x, y = float(bbox.origin.x), float(bbox.origin.y)
            w, h = float(bbox.size.width), float(bbox.size.height)
            left = int(x * pixel_width)
            right = int((x + w) * pixel_width)
            top = int((1.0 - y - h) * pixel_height)
            bottom = int((1.0 - y) * pixel_height)
            face_box = FaceBox(top=top, right=right, bottom=bottom, left=left)

        # Shrink bbox that includes excessive body area (aspect ratio > 1.8)
        shrunk = False
        fw, fh = face_box.width, face_box.height
        if fw > 0 and fh / fw > _BODY_ASPECT_RATIO_THRESHOLD:
            shrunk = True
            face_box = _shrink_body_bbox(face_box)

        # Extract 5-point landmarks for face alignment
        lm5 = _extract_5_landmarks(landmarks, pixel_width, pixel_height)

        # Filter by minimum face size
        if face_box.width < min_face_size or face_box.height < min_face_size:
            continue

        result.append((face_box, confidence, lm5))

        # Track metrics
        stats["total"] += 1
        stats[bbox_source] += 1
        if shrunk:
            stats["shrunk"] += 1
        if face_box.width > 0:
            stats["aspect_ratio_sum"] += face_box.height / face_box.width

    return result, stats


def _extract_5_landmarks(
    landmarks, pixel_width: int, pixel_height: int,
) -> list[tuple[float, float]] | None:
    """Extract 5 facial landmarks for alignment from Vision face landmarks.

    Returns (left_eye, right_eye, nose_tip, left_mouth, right_mouth) as pixel
    coordinates, or None if any required landmark region is missing.
    """
    if landmarks is None:
        return None

    # Collect required landmark regions
    left_eye_lm = landmarks.leftEye()
    right_eye_lm = landmarks.rightEye()
    nose_lm = landmarks.nose()
    outer_lips = landmarks.outerLips()

    if any(x is None or x.pointCount() == 0
           for x in [left_eye_lm, right_eye_lm, nose_lm, outer_lips]):
        return None

    def _centroid(region) -> tuple[float, float]:
        pts = _safe_points(region)
        if not pts:
            return (0.0, 0.0)
        cx = sum(x for x, _ in pts) / len(pts)
        cy = sum(1.0 - y for _, y in pts) / len(pts)
        return (cx * pixel_width, cy * pixel_height)

    def _nose_tip(region) -> tuple[float, float]:
        """Nose tip is the lowest point in the nose region (top-left coords)."""
        pts = _safe_points(region)
        best = None
        best_y = -1.0
        for x, y in pts:
            py = 1.0 - y
            if py > best_y:
                best_y = py
                best = (x * pixel_width, py * pixel_height)
        return best

    def _mouth_corners(region) -> tuple[tuple[float, float], tuple[float, float]]:
        """Leftmost and rightmost points of the outer lip contour."""
        pts = _safe_points(region)
        leftmost = None
        rightmost = None
        min_x = float("inf")
        max_x = float("-inf")
        for x, y in pts:
            py = 1.0 - y
            if x < min_x:
                min_x = x
                leftmost = (x * pixel_width, py * pixel_height)
            if x > max_x:
                max_x = x
                rightmost = (x * pixel_width, py * pixel_height)
        return leftmost, rightmost

    left_eye = _centroid(left_eye_lm)
    right_eye = _centroid(right_eye_lm)
    nose = _nose_tip(nose_lm)
    left_mouth, right_mouth = _mouth_corners(outer_lips)

    return [left_eye, right_eye, nose, left_mouth, right_mouth]


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
        img_w, img_h = _get_image_dimensions(image_path)
        face_results, stats = detect_faces(image_path, min_confidence, min_face_size)
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
        if not faces:
            return ImageResult(path=image_path, skipped=True, image_width=img_w, image_height=img_h)
        return ImageResult(
            path=image_path, faces=faces, detection_stats=stats,
            image_width=img_w, image_height=img_h,
        )
    except Exception as exc:
        logger.warning("Face detection failed for %s: %s", image_path, exc)
        return ImageResult(path=image_path, error=str(exc))


def detect_faces_batch(
    image_paths: list[str],
    min_confidence: float = 0.5,
    min_face_size: int = 40,
    max_workers: int = 4,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[ImageResult], dict[str, int]]:
    """Detect faces across multiple images with parallel processing.

    Args:
        image_paths: List of image file paths.
        min_confidence: Minimum detection confidence.
        min_face_size: Minimum face dimension.
        max_workers: Number of parallel threads.
        progress_callback: Called with (completed, total) after each image.

    Returns:
        Tuple of (list of ImageResult, aggregated stats dict).
    """
    _check_vision()

    results: dict[int, ImageResult] = {}
    completed = 0
    total = len(image_paths)
    agg_stats = {
        "total": 0, "contour": 0, "median": 0, "default": 0,
        "shrunk": 0, "aspect_ratio_sum": 0.0,
    }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                detect_faces_single, path, min_confidence, min_face_size
            ): idx
            for idx, path in enumerate(image_paths)
        }

        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            results[idx] = result
            # Aggregate per-image detection stats
            if result.detection_stats:
                for key in agg_stats:
                    agg_stats[key] = agg_stats.get(key, 0) + result.detection_stats.get(key, 0)
            completed += 1
            if progress_callback:
                progress_callback(completed, total)

    return [results[i] for i in range(total)], agg_stats
