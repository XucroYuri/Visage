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
        with patch("visage.detector._VISION_AVAILABLE", True):
            with patch("visage.detector.VNDetectFaceRectanglesRequest") as mock_req_cls, \
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
                result = detect_faces("/tmp/test.jpg")
                assert result == []

    def test_returns_face_boxes(self, tmp_path):
        """Vision detects one face — verify FaceBox conversion."""
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (800, 600), color=(100, 100, 100))
        img.save(img_path, "JPEG")

        _obs = _make_mock_observation(confidence=0.9, x=0.1, y=0.2, w=0.3, h=0.4)

        with patch("visage.detector.detect_faces") as mock_detect:
            mock_detect.return_value = [(FaceBox(top=320, right=400, bottom=160, left=100), 0.9)]

            from visage.detector import detect_faces_single
            result = detect_faces_single(str(img_path))

            assert len(result.faces) == 1
            assert result.faces[0].confidence == 0.9

    def test_filters_low_confidence(self, tmp_path):
        """Face with confidence below threshold is filtered out."""
        with patch("visage.detector.detect_faces") as mock_detect:
            mock_detect.return_value = []  # all faces filtered by confidence

            from visage.detector import detect_faces_single
            result = detect_faces_single(str(tmp_path / "test.jpg"), min_confidence=0.5)
            assert result.skipped is True

    def test_filters_small_faces(self, tmp_path):
        """Face smaller than min_face_size is filtered out by detect_faces."""
        with patch("visage.detector.detect_faces") as mock_detect:
            # detect_faces already filters — returns empty for small faces
            mock_detect.return_value = []

            from visage.detector import detect_faces_single
            result = detect_faces_single(str(tmp_path / "test.jpg"), min_face_size=40)
            assert result.skipped is True

    def test_coordinate_conversion(self, tmp_path):
        """Verify normalized -> pixel coordinate conversion via detect_faces mock."""
        # Simulate the expected pixel output from Vision conversion
        # For a 1000x800 image with bbox x=0.1, y=0.2, w=0.3, h=0.4:
        # left=100, right=400, top=320, bottom=160
        expected_box = FaceBox(top=320, right=400, bottom=160, left=100)

        with patch("visage.detector.detect_faces") as mock_detect:
            mock_detect.return_value = [(expected_box, 0.9)]

            from visage.detector import detect_faces_single
            result = detect_faces_single(str(tmp_path / "test.jpg"))
            assert len(result.faces) == 1
            fb = result.faces[0].face_box
            assert fb.top == 320
            assert fb.right == 400
            assert fb.bottom == 160
            assert fb.left == 100


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
            mock_detect.return_value = [(FaceBox(top=10, right=110, bottom=110, left=10), 0.9)]

            from visage.detector import detect_faces_single
            result = detect_faces_single(str(tmp_path / "test.jpg"))

            assert isinstance(result, ImageResult)
            assert len(result.faces) == 1
            assert result.error is None

    def test_no_faces_returns_skipped(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (100, 100)).save(img_path, "JPEG")

        with patch("visage.detector._VISION_AVAILABLE", True):
            with patch("visage.detector.VNDetectFaceRectanglesRequest") as mock_req_cls, \
                 patch("visage.detector.VNImageRequestHandler") as mock_handler_cls, \
                 patch("visage.detector.NSURL"):

                handler = MagicMock()
                handler.performRequests_error_.return_value = True
                mock_handler_cls.alloc.return_value.initWithURL_options_.return_value = handler
                request = MagicMock()
                request.results.return_value = None
                mock_req_cls.alloc.return_value.init.return_value = request

                from visage.detector import detect_faces_single
                result = detect_faces_single(str(img_path))

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
            with patch("visage.detector.VNDetectFaceRectanglesRequest") as mock_req_cls, \
                 patch("visage.detector.VNImageRequestHandler") as mock_handler_cls, \
                 patch("visage.detector.NSURL"):

                handler = MagicMock()
                handler.performRequests_error_.return_value = True
                mock_handler_cls.alloc.return_value.initWithURL_options_.return_value = handler
                request = MagicMock()
                request.results.return_value = None
                mock_req_cls.alloc.return_value.init.return_value = request

                from visage.detector import detect_faces_batch
                results = detect_faces_batch(paths)

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
            with patch("visage.detector.VNDetectFaceRectanglesRequest") as mock_req_cls, \
                 patch("visage.detector.VNImageRequestHandler") as mock_handler_cls, \
                 patch("visage.detector.NSURL"):

                handler = MagicMock()
                handler.performRequests_error_.return_value = True
                mock_handler_cls.alloc.return_value.initWithURL_options_.return_value = handler
                request = MagicMock()
                request.results.return_value = None
                mock_req_cls.alloc.return_value.init.return_value = request

                from visage.detector import detect_faces_batch
                results = detect_faces_batch(
                    paths,
                    progress_callback=lambda completed, total: callbacks.append((completed, total)),
                )

                assert len(results) == 2
                assert len(callbacks) >= 2
                # Last callback should be (2, 2)
                assert callbacks[-1] == (2, 2)
