"""macOS Vision framework face detection backend.

Wraps VNDetectFaceLandmarksRequest + VNDetectFaceRectanglesRequest
in a DetectorBackend-compatible class. Only available on macOS with pyobjc.
"""

from __future__ import annotations

import logging
import os
import tempfile

import numpy as np

from visage.models import FaceBox

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
        return int(platform.mac_ver()[0].split(".")[0])
    except (ValueError, IndexError):
        return 0


_LANDMARKS_CRASH_BUG = _macos_major_version() >= 26

# Padding ratios for expanding the tight face contour bbox.
_CONTOUR_PAD_UP = 0.45
_CONTOUR_PAD_SIDE = 0.20
_CONTOUR_PAD_DOWN = 0.10

# Body-shrink threshold
_BODY_ASPECT_RATIO_THRESHOLD = 1.8
_TARGET_HEAD_ASPECT = 1.25
_MIN_HEAD_FRACTION = 0.50


# ── Internal helpers (same as original detector.py) ───────────────


def _check_vision() -> None:
    """Raise if Vision framework is not available."""
    if not _VISION_AVAILABLE:
        raise RuntimeError(
            "macOS Vision framework not available. "
            "Install pyobjc: pip install pyobjc-framework-Vision"
        )


def _safe_points(region) -> list[tuple[float, float]]:
    """Extract (x, y) tuples from a landmark region via normalizedPoints."""
    pts = region.normalizedPoints()
    try:
        return [(float(p.x), float(p.y)) for p in pts]
    except (TypeError, AttributeError):
        return []


def _bbox_from_contour(
    face_contour, pixel_width: int, pixel_height: int,
) -> FaceBox | None:
    """Compute a tight bounding box from face contour landmark points."""
    if face_contour is None or face_contour.pointCount() == 0:
        return None
    pts = _safe_points(face_contour)
    if not pts:
        return None
    xs = [x * pixel_width for x, _ in pts]
    ys = [(1.0 - y) * pixel_height for _, y in pts]
    tight_left = min(xs)
    tight_right = max(xs)
    tight_top = min(ys)
    tight_bottom = max(ys)
    face_w = tight_right - tight_left
    face_h = tight_bottom - tight_top
    left = max(0, int(tight_left - face_w * _CONTOUR_PAD_SIDE))
    right = min(pixel_width, int(tight_right + face_w * _CONTOUR_PAD_SIDE))
    top = max(0, int(tight_top - face_h * _CONTOUR_PAD_UP))
    bottom = min(pixel_height, int(tight_bottom + face_h * _CONTOUR_PAD_DOWN))
    return FaceBox(top=top, right=right, bottom=bottom, left=left)


def _bbox_from_median(
    median_line, left_eye, right_eye,
    pixel_width: int, pixel_height: int,
) -> FaceBox | None:
    """Compute bbox from median line and eye landmarks (fallback)."""
    if median_line is None or median_line.pointCount() == 0:
        return None
    median_pts = _safe_points(median_line)
    if not median_pts:
        return None
    ys = [(1.0 - y) * pixel_height for _, y in median_pts]
    xs = [x * pixel_width for x, _ in median_pts]
    top = min(ys)
    bottom = max(ys)
    center_x = sum(xs) / len(xs)

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
            face_w = int(eye_dist * 2.5)
            half_w = face_w / 2
    else:
        face_h = bottom - top
        half_w = face_h * 0.5

    face_h = bottom - top
    left = max(0, int(center_x - half_w - face_h * _CONTOUR_PAD_SIDE))
    right = min(pixel_width, int(center_x + half_w + face_h * _CONTOUR_PAD_SIDE))
    top = max(0, int(top - face_h * _CONTOUR_PAD_UP))
    bottom = min(pixel_height, int(bottom + face_h * _CONTOUR_PAD_DOWN))
    return FaceBox(top=top, right=right, bottom=bottom, left=left)


def _shrink_body_bbox(face_box: FaceBox) -> FaceBox:
    """Shrink a bbox that likely includes body below the head."""
    fw = face_box.width
    fh = face_box.height
    if fw <= 0 or fh <= 0:
        return face_box
    aspect = fh / fw
    if aspect <= _BODY_ASPECT_RATIO_THRESHOLD:
        return face_box
    target_h = int(fw * _TARGET_HEAD_ASPECT)
    min_h = int(fh * _MIN_HEAD_FRACTION)
    target_h = max(target_h, min_h)
    return FaceBox(
        top=face_box.top,
        right=face_box.right,
        bottom=face_box.top + target_h,
        left=face_box.left,
    )


def _match_landmarks_to_rect(rect_obs, lm_observations: list) -> object | None:
    """Match a rect observation to its nearest landmarks observation."""
    rb = rect_obs.boundingBox()
    rx, ry = float(rb.origin.x), float(rb.origin.y)
    rw, rh = float(rb.size.width), float(rb.size.height)
    rcx, rcy = rx + rw / 2, ry + rh / 2
    best_dist = 0.05
    best = None
    for lo in lm_observations:
        lb = lo.boundingBox()
        lcx = float(lb.origin.x) + float(lb.size.width) / 2
        lcy = float(lb.origin.y) + float(lb.size.height) / 2
        dist = ((rcx - lcx) ** 2 + (rcy - lcy) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best = lo
    return best


def _bbox_from_vision_observation(obs, pixel_width: int, pixel_height: int) -> FaceBox:
    """Extract default bounding box from a VNFaceObservation."""
    bbox = obs.boundingBox()
    x, y = float(bbox.origin.x), float(bbox.origin.y)
    w, h = float(bbox.size.width), float(bbox.size.height)
    left = int(x * pixel_width)
    right = int((x + w) * pixel_width)
    top = int((1.0 - y - h) * pixel_height)
    bottom = int((1.0 - y) * pixel_height)
    return FaceBox(top=top, right=right, bottom=bottom, left=left)


def _extract_5_landmarks(
    landmarks, pixel_width: int, pixel_height: int,
) -> list[tuple[float, float]] | None:
    """Extract 5 facial landmarks for alignment from Vision face landmarks."""
    if landmarks is None:
        return None
    left_eye_lm = landmarks.leftEye()
    right_eye_lm = landmarks.rightEye()
    nose_lm = landmarks.nose()
    outer_lips = landmarks.outerLips()
    if any(
        x is None or x.pointCount() == 0
        for x in [left_eye_lm, right_eye_lm, nose_lm, outer_lips]
    ):
        return None

    def _centroid(region) -> tuple[float, float]:
        pts = _safe_points(region)
        if not pts:
            return (0.0, 0.0)
        cx = sum(x for x, _ in pts) / len(pts)
        cy = sum(1.0 - y for _, y in pts) / len(pts)
        return (cx * pixel_width, cy * pixel_height)

    def _nose_tip(region) -> tuple[float, float]:
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
    """Get image pixel dimensions using PIL (fast, no full decode)."""
    from PIL import Image
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        logger.warning("Failed to get image dimensions: %s", image_path, exc_info=True)
        return (0, 0)


# ── VisionDetector class ───────────────────────────────────────────


class VisionDetector:
    """macOS Vision framework face detection backend.

    Uses dual-request strategy: VNDetectFaceRectanglesRequest for all faces,
    VNDetectFaceLandmarksRequest for bbox refinement and alignment landmarks.
    """

    name = "vision"

    def __init__(
        self,
        min_confidence: float = 0.5,
        min_face_size: int = 40,
    ) -> None:
        self.min_confidence = min_confidence
        self.min_face_size = min_face_size
        self._last_stats: dict[str, int | float] | None = None

    @property
    def last_stats(self) -> dict[str, int | float] | None:
        """Stats dict from the most recent detect() call, if any."""
        return self._last_stats

    @staticmethod
    def is_available() -> bool:
        return _VISION_AVAILABLE

    def detect(
        self, image: np.ndarray,
    ) -> list[tuple[FaceBox, float, list[tuple[float, float]] | None]]:
        """Detect faces in an RGB image using the Vision framework.

        Internal implementation converts the numpy array to a temp PNG file
        for Vision framework compatibility.

        Args:
            image: RGB numpy array, shape (H, W, 3), dtype uint8.

        Returns:
            List of (FaceBox, confidence, landmarks_5) tuples.
        """
        _check_vision()
        h, w = image.shape[:2]

        # Save numpy array to temp PNG file for Vision framework
        from PIL import Image as PILImage

        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            pil_img = PILImage.fromarray(image)
            pil_img.save(tmp_path, "PNG")
            results, stats = self._detect_from_path(tmp_path, w, h)
            self._last_stats = stats
            return results
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def detect_from_path(
        self, image_path: str,
    ) -> tuple[
        list[tuple[FaceBox, float, list[tuple[float, float]] | None]],
        dict[str, int | float],
    ]:
        """Detect faces from a file path (more efficient, no temp file).

        This is the file-path-based entry point used by the legacy
        detector.py facade. Returns stats dict alongside results.

        Args:
            image_path: Path to an image file.

        Returns:
            Tuple of (detection list, stats dict).
        """
        _check_vision()
        w, h = _get_image_dimensions(image_path)
        return self._detect_from_path(image_path, w, h)

    def _detect_from_path(
        self, image_path: str, pixel_width: int, pixel_height: int,
    ) -> tuple[
        list[tuple[FaceBox, float, list[tuple[float, float]] | None]],
        dict[str, int | float],
    ]:
        """Core Vision detection logic — shared by detect() and detect_from_path()."""
        url = NSURL.fileURLWithPath_(image_path)
        handler = VNImageRequestHandler.alloc().initWithURL_options_(url, None)

        use_simple_detection = _LANDMARKS_CRASH_BUG

        rect_request = VNDetectFaceRectanglesRequest.alloc().init()
        requests = [rect_request]
        lm_request = None
        if not use_simple_detection:
            lm_request = VNDetectFaceLandmarksRequest.alloc().init()
            requests.append(lm_request)

        result: list[tuple[FaceBox, float, list[tuple[float, float]] | None]] = []
        stats: dict[str, int | float] = {
            "total": 0, "contour": 0, "median": 0, "default": 0,
            "shrunk": 0, "aspect_ratio_sum": 0.0,
        }

        success = handler.performRequests_error_(requests, None)
        if not success:
            return [], stats

        rect_observations = list(rect_request.results() or [])
        if not rect_observations:
            return [], stats

        lm_observations: list = []
        if not use_simple_detection and lm_request is not None:
            lm_observations = list(lm_request.results() or [])

        for obs in rect_observations:
            confidence = float(obs.confidence())
            if confidence < self.min_confidence:
                continue

            face_box = _bbox_from_vision_observation(obs, pixel_width, pixel_height)
            bbox_source = "default"
            landmarks = None

            if lm_observations:
                lm_obs = _match_landmarks_to_rect(obs, lm_observations)
                if lm_obs is not None:
                    lm_landmarks = lm_obs.landmarks()
                    if lm_landmarks is not None:
                        landmarks = lm_landmarks
                        contour_box = _bbox_from_contour(
                            lm_landmarks.faceContour(), pixel_width, pixel_height,
                        )
                        if contour_box is not None:
                            face_box = contour_box
                            bbox_source = "contour"
                        else:
                            median_box = _bbox_from_median(
                                lm_landmarks.medianLine(),
                                lm_landmarks.leftEye(),
                                lm_landmarks.rightEye(),
                                pixel_width, pixel_height,
                            )
                            if median_box is not None:
                                face_box = median_box
                                bbox_source = "median"

            shrunk = False
            fw, fh = face_box.width, face_box.height
            if fw > 0 and fh / fw > _BODY_ASPECT_RATIO_THRESHOLD:
                shrunk = True
                face_box = _shrink_body_bbox(face_box)

            lm5 = _extract_5_landmarks(landmarks, pixel_width, pixel_height)

            if face_box.width < self.min_face_size or face_box.height < self.min_face_size:
                continue

            result.append((face_box, confidence, lm5))

            stats["total"] = int(stats["total"]) + 1  # type: ignore[assignment]
            stats[bbox_source] = int(stats[bbox_source]) + 1  # type: ignore[assignment]
            if shrunk:
                stats["shrunk"] = int(stats["shrunk"]) + 1  # type: ignore[assignment]
            if face_box.width > 0:
                stats["aspect_ratio_sum"] = (  # type: ignore[assignment]
                    stats["aspect_ratio_sum"] + face_box.height / face_box.width
                )

        return result, stats
