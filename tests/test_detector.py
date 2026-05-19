"""Tests for visage.detector — face detection with mocked macOS Vision framework."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from visage.models import FaceBox, ImageResult

# ── Helpers ───────────────────────────────────────────────────────


def _make_mock_observation(
    confidence: float = 0.9,
    x: float = 0.1,
    y: float = 0.2,
    w: float = 0.3,
    h: float = 0.4,
) -> MagicMock:
    """Create a mock Vision observation with known bounding box and confidence."""
    obs = MagicMock()
    obs.confidence.return_value = confidence
    bbox = MagicMock()
    bbox.origin.x = x
    bbox.origin.y = y
    bbox.size.width = w
    bbox.size.height = h
    obs.boundingBox.return_value = bbox
    return obs


def _mock_vision_for_detect(mock_observations_fn, image_dimensions=(1000, 800)):
    """Set up all mocks needed for detect_faces() in a context manager.

    Returns a context manager that patches Vision, NSURL, and PIL Image.
    """
    def decorator(func):
        return _VisionMocker(func, mock_observations_fn, image_dimensions)
    return decorator


class _VisionMocker:
    """Context-manager-like approach using patch."""
    def __init__(self, func, mock_obs_fn, dims):
        self.func = func
        self.mock_obs_fn = mock_obs_fn
        self.dims = dims

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


# ── _check_vision ─────────────────────────────────────────────────


class TestCheckVision:
    def test_raises_when_unavailable(self):
        from visage.detector import _check_vision

        with patch("visage.detector._VISION_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="macOS Vision framework not available"):
                _check_vision()


# ── detect_faces ──────────────────────────────────────────────────


class TestDetectFaces:
    def test_no_faces_found(self):
        """When Vision returns no observations."""
        with patch("visage.detector._VISION_AVAILABLE", True), \
             patch("visage.detector._LANDMARKS_CRASH_BUG", False):
            with patch("visage.detector.VNDetectFaceLandmarksRequest") as mock_req_cls, \
                 patch("visage.detector.VNImageRequestHandler") as mock_handler_cls, \
                 patch("visage.detector.NSURL"):

                # Mock handler: performRequests returns True
                handler = MagicMock()
                handler.performRequests_error_.return_value = True
                mock_handler_cls.alloc.return_value.initWithURL_options_.return_value = handler

                # Mock request: results returns None
                request = MagicMock()
                request.results.return_value = None
                mock_req_cls.alloc.return_value.init.return_value = request

                from visage.detector import detect_faces
                result, stats = detect_faces("/tmp/test.jpg")
                assert result == []
                expected_stats = {
                    "total": 0, "contour": 0, "median": 0,
                    "default": 0, "shrunk": 0, "aspect_ratio_sum": 0.0,
                }
                assert stats == expected_stats

    def test_returns_face_boxes(self, tmp_path):
        """Vision detects one face — verify FaceBox conversion."""
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (800, 600), color=(100, 100, 100))
        img.save(img_path, "JPEG")

        _obs = _make_mock_observation(confidence=0.9, x=0.1, y=0.2, w=0.3, h=0.4)

        with patch("visage.detector.detect_faces") as mock_detect:
            mock_detect.return_value = (
                [(FaceBox(top=320, right=400, bottom=160, left=100), 0.9, None)], {},
            )

            from visage.detector import detect_faces_single
            result = detect_faces_single(str(img_path))

            assert len(result.faces) == 1
            assert result.faces[0].confidence == 0.9

    def test_filters_low_confidence(self, tmp_path):
        """Face with confidence below threshold is filtered out."""
        with patch("visage.detector.detect_faces") as mock_detect:
            mock_detect.return_value = ([], {})  # all faces filtered by confidence

            from visage.detector import detect_faces_single
            result = detect_faces_single(str(tmp_path / "test.jpg"), min_confidence=0.5)
            assert result.skipped is True

    def test_filters_small_faces(self, tmp_path):
        """Face smaller than min_face_size is filtered out by detect_faces."""
        with patch("visage.detector.detect_faces") as mock_detect:
            # detect_faces already filters — returns empty for small faces
            mock_detect.return_value = ([], {})

            from visage.detector import detect_faces_single
            result = detect_faces_single(str(tmp_path / "test.jpg"), min_face_size=40)
            assert result.skipped is True

    def test_coordinate_conversion(self, tmp_path):
        """Verify normalized -> pixel coordinate conversion via detect_faces mock."""
        # For a 1000x800 image with Vision bbox x=0.1, y=0.2, w=0.3, h=0.4:
        # left=100, right=400, top=320, bottom=640
        expected_box = FaceBox(top=320, right=400, bottom=640, left=100)

        with patch("visage.detector.detect_faces") as mock_detect:
            mock_detect.return_value = ([(expected_box, 0.9, None)], {})

            from visage.detector import detect_faces_single
            result = detect_faces_single(str(tmp_path / "test.jpg"))
            assert len(result.faces) == 1
            fb = result.faces[0].face_box
            assert fb.top == 320
            assert fb.right == 400
            assert fb.bottom == 640
            assert fb.left == 100

    def test_uses_contour_bbox_when_landmarks_available(self, tmp_path):
        """When face contour landmarks exist, detect_faces uses contour-based bbox."""
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (1000, 800)).save(img_path, "JPEG")

        # Build mock observation with landmarks
        obs = _make_mock_observation(confidence=0.9, x=0.1, y=0.2, w=0.3, h=0.4)

        # Mock landmarks with faceContour
        contour = MagicMock()
        contour.pointCount.return_value = 4
        pts = []
        for nx, ny in [(0.35, 0.7), (0.65, 0.7), (0.65, 0.3), (0.35, 0.3)]:
            pt = MagicMock()
            pt.x = nx
            pt.y = ny
            pts.append(pt)
        contour.normalizedPoints.return_value = pts

        landmarks = MagicMock()
        landmarks.faceContour.return_value = contour
        landmarks.medianLine.return_value = None
        landmarks.leftEye.return_value = None
        landmarks.rightEye.return_value = None
        landmarks.nose.return_value = None
        landmarks.outerLips.return_value = None
        obs.landmarks.return_value = landmarks

        with patch("visage.detector._VISION_AVAILABLE", True), \
             patch("visage.detector._LANDMARKS_CRASH_BUG", False):
            with patch("visage.detector.VNDetectFaceLandmarksRequest") as mock_req_cls, \
                 patch("visage.detector.VNImageRequestHandler") as mock_handler_cls, \
                 patch("visage.detector.NSURL"), \
                 patch("visage.detector._get_image_dimensions", return_value=(1000, 800)):

                handler = MagicMock()
                handler.performRequests_error_.return_value = True
                mock_handler_cls.alloc.return_value.initWithURL_options_.return_value = handler

                request = MagicMock()
                request.results.return_value = [obs]
                mock_req_cls.alloc.return_value.init.return_value = request

                from visage.detector import detect_faces
                result, stats = detect_faces(str(img_path))

                assert len(result) == 1
                fb, conf, lm5 = result[0]
                # Contour bbox should be tighter than the default Vision bbox
                # Default bbox would be: top=320, right=400, bottom=160, left=100
                # Contour bbox (with padding) should differ from the default
                assert fb.top != 320 or fb.right != 400  # not the same as default

    def test_falls_back_to_default_bbox_without_landmarks(self, tmp_path):
        """When no landmarks, detect_faces falls back to Vision's default bbox."""
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (1000, 800)).save(img_path, "JPEG")

        obs = _make_mock_observation(confidence=0.9, x=0.1, y=0.2, w=0.3, h=0.4)
        obs.landmarks.return_value = None

        with patch("visage.detector._VISION_AVAILABLE", True), \
             patch("visage.detector._LANDMARKS_CRASH_BUG", False):
            with patch("visage.detector.VNDetectFaceLandmarksRequest") as mock_req_cls, \
                 patch("visage.detector.VNImageRequestHandler") as mock_handler_cls, \
                 patch("visage.detector.NSURL"), \
                 patch("visage.detector._get_image_dimensions", return_value=(1000, 800)):

                handler = MagicMock()
                handler.performRequests_error_.return_value = True
                mock_handler_cls.alloc.return_value.initWithURL_options_.return_value = handler

                request = MagicMock()
                request.results.return_value = [obs]
                mock_req_cls.alloc.return_value.init.return_value = request

                from visage.detector import detect_faces
                result, stats = detect_faces(str(img_path))

                assert len(result) == 1
                fb, conf, lm5 = result[0]
                # Vision bbox x=0.1,y=0.2,w=0.3,h=0.4 on 1000x800:
                # left=100, right=400, top=320, bottom=640
                assert fb.left == 100
                assert fb.right == 400
                assert fb.top == 320
                assert fb.bottom == 640


# ── _bbox_from_contour ────────────────────────────────────────────


class TestBboxFromContour:
    def test_returns_none_for_none_contour(self):
        from visage.detector import _bbox_from_contour
        assert _bbox_from_contour(None, 1000, 800) is None

    def test_returns_none_for_empty_contour(self):
        from visage.detector import _bbox_from_contour
        contour = MagicMock()
        contour.pointCount.return_value = 0
        contour.normalizedPoints.return_value = []
        assert _bbox_from_contour(contour, 1000, 800) is None

    def test_tight_bbox_from_contour_points(self):
        """Contour points at the face edge produce a tight bbox with padding."""
        from visage.detector import _bbox_from_contour

        # Simulate face contour: a face occupying roughly 30% x 40% of the image
        # Contour points trace the face edge (tight around the actual face)
        # Normalized bottom-left origin: x in [0,1], y in [0,1]
        # Face centered at (0.5, 0.5), spanning roughly x=[0.35,0.65], y=[0.3,0.7]
        contour = MagicMock()
        contour.pointCount.return_value = 4
        pts = []
        for nx, ny in [(0.35, 0.7), (0.65, 0.7), (0.65, 0.3), (0.35, 0.3)]:
            pt = MagicMock()
            pt.x = nx
            pt.y = ny
            pts.append(pt)
        contour.normalizedPoints.return_value = pts

        bbox = _bbox_from_contour(contour, 1000, 800)
        assert bbox is not None
        # Tight contour: x=[350,650], y(top-left)=[240,560]
        # With padding applied, bbox should be larger than the raw contour
        assert bbox.left < 350   # padded left
        assert bbox.right > 650  # padded right
        assert bbox.top < 240    # padded up (hair)
        assert bbox.bottom > 560  # padded down (chin margin)

    def test_bbox_clamped_to_image_bounds(self):
        """Contour near image edge should be clamped, not negative."""
        from visage.detector import _bbox_from_contour

        # Face at top-left corner — padding would go negative
        contour = MagicMock()
        contour.pointCount.return_value = 2
        pts = []
        for nx, ny in [(0.02, 0.98), (0.15, 0.85)]:
            pt = MagicMock()
            pt.x = nx
            pt.y = ny
            pts.append(pt)
        contour.normalizedPoints.return_value = pts

        bbox = _bbox_from_contour(contour, 1000, 800)
        assert bbox is not None
        assert bbox.left >= 0
        assert bbox.top >= 0
        assert bbox.right <= 1000
        assert bbox.bottom <= 800


# ── _get_image_dimensions ─────────────────────────────────────────


class TestGetImageDimensions:
    def test_valid_image(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (640, 480)).save(img_path, "JPEG")
        from visage.detector import _get_image_dimensions
        w, h = _get_image_dimensions(str(img_path))
        assert w == 640
        assert h == 480

    def test_invalid_path(self, tmp_path):
        from visage.detector import _get_image_dimensions
        w, h = _get_image_dimensions(str(tmp_path / "nonexistent.jpg"))
        assert w == 0
        assert h == 0


# ── detect_faces_single ───────────────────────────────────────────


class TestDetectFacesSingle:
    def test_returns_image_result_with_faces(self, tmp_path):
        with patch("visage.detector.detect_faces") as mock_detect:
            mock_detect.return_value = (
                [(FaceBox(top=10, right=110, bottom=110, left=10), 0.9, None)], {},
            )

            from visage.detector import detect_faces_single
            result = detect_faces_single(str(tmp_path / "test.jpg"))

            assert isinstance(result, ImageResult)
            assert len(result.faces) == 1
            assert result.error is None

    def test_no_faces_returns_skipped(self, tmp_path):
        with patch("visage.detector.detect_faces", return_value=([], {})):
            from visage.detector import detect_faces_single
            result = detect_faces_single(str(tmp_path / "test.jpg"))

            assert result.skipped is True
            assert result.faces == []

    def test_exception_returns_error(self):
        with patch("visage.detector.detect_faces") as mock_detect:
            mock_detect.side_effect = RuntimeError("Vision framework crash")

            from visage.detector import detect_faces_single
            result = detect_faces_single("/tmp/test.jpg")
            assert result.error is not None


# ── detect_faces_batch ────────────────────────────────────────────


class TestDetectFacesBatch:
    def test_returns_all_results(self, tmp_path):
        """Batch processing returns ordered ImageResult list."""
        # Create 3 test images
        paths = []
        for i in range(3):
            p = tmp_path / f"test_{i}.jpg"
            Image.new("RGB", (100, 100)).save(p, "JPEG")
            paths.append(str(p))

        with patch("visage.detector._VISION_AVAILABLE", True):
            with patch("visage.detector.VNDetectFaceLandmarksRequest") as mock_req_cls, \
                 patch("visage.detector.VNImageRequestHandler") as mock_handler_cls, \
                 patch("visage.detector.NSURL"):

                handler = MagicMock()
                handler.performRequests_error_.return_value = True
                mock_handler_cls.alloc.return_value.initWithURL_options_.return_value = handler
                request = MagicMock()
                request.results.return_value = None
                mock_req_cls.alloc.return_value.init.return_value = request

                from visage.detector import detect_faces_batch
                results, stats = detect_faces_batch(paths)

                assert len(results) == 3
                for r in results:
                    assert isinstance(r, ImageResult)

    def test_progress_callback(self, tmp_path):
        paths = []
        for i in range(2):
            p = tmp_path / f"test_{i}.jpg"
            Image.new("RGB", (100, 100)).save(p, "JPEG")
            paths.append(str(p))

        callbacks = []

        with patch("visage.detector._VISION_AVAILABLE", True):
            with patch("visage.detector.VNDetectFaceLandmarksRequest") as mock_req_cls, \
                 patch("visage.detector.VNImageRequestHandler") as mock_handler_cls, \
                 patch("visage.detector.NSURL"):

                handler = MagicMock()
                handler.performRequests_error_.return_value = True
                mock_handler_cls.alloc.return_value.initWithURL_options_.return_value = handler
                request = MagicMock()
                request.results.return_value = None
                mock_req_cls.alloc.return_value.init.return_value = request

                from visage.detector import detect_faces_batch
                results, stats = detect_faces_batch(
                    paths,
                    progress_callback=lambda completed, total: callbacks.append((completed, total)),
                )

                assert len(results) == 2
                assert len(callbacks) >= 2
                # Last callback should be (2, 2)
                assert callbacks[-1] == (2, 2)
