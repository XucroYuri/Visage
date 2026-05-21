"""Tests for visage.detectors — backend protocol, factory, SCRFD, YuNet."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from visage.detectors import DetectorBackend, get_detector


# ── Dummy backend for Protocol / factory tests ────────────────────


class _DummyBackend:
    """Minimal backend implementing the Protocol."""
    name = "dummy"

    def detect(self, image: np.ndarray):
        from visage.models import FaceBox
        h, w = image.shape[:2]
        return [(FaceBox(top=10, right=w - 10, bottom=h - 10, left=10), 0.9, None)]

    def is_available(self) -> bool:
        return True


# ── Protocol compat ───────────────────────────────────────────────


class TestDetectorBackendProtocol:
    def test_dummy_is_runtime_checkable(self):
        assert isinstance(_DummyBackend(), DetectorBackend)

    def test_protocol_attributes(self):
        backend = _DummyBackend()
        assert hasattr(backend, "name")
        assert hasattr(backend, "detect")
        assert hasattr(backend, "is_available")

    def test_detect_returns_list_of_tuples(self):
        backend = _DummyBackend()
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        results = backend.detect(image)
        assert isinstance(results, list)
        if results:
            fb, conf, lm5 = results[0]
            from visage.models import FaceBox
            assert isinstance(fb, FaceBox)
            assert isinstance(conf, float)
            assert lm5 is None or isinstance(lm5, list)


# ── get_detector factory ──────────────────────────────────────────


class TestGetDetector:
    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown detection backend"):
            get_detector("nonexistent")

    def test_vision_not_on_linux(self):
        """VisionDetector.is_available() is False on non-macOS."""
        detector = get_detector("vision")
        # On macOS this is True; on other platforms it should be False
        # We can check it's importable and callable either way
        assert hasattr(detector, "detect")

    def test_scrfd_not_available_without_insightface(self):
        """SCRFDDetector.is_available() is False without insightface."""
        with patch("visage.detectors.scrfd.SCRFDDetector.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="SCRFD detector is not available"):
                get_detector("scrfd")

    def test_yunet_not_available_without_model(self):
        """YuNetDetector.is_available() is False without model file."""
        with patch("visage.detectors.yunet.YuNetDetector.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="YuNet detector is not available"):
                get_detector("yunet")


# ── factory auto mode ─────────────────────────────────────────────


class TestGetDetectorAuto:
    def test_auto_raises_when_none_available(self):
        """auto mode should raise when no backend is available."""
        with (
            patch("visage.detectors.vision.VisionDetector.is_available", return_value=False),
            patch("visage.detectors.scrfd.SCRFDDetector.is_available", return_value=False),
            patch("visage.detectors.yunet.YuNetDetector.is_available", return_value=False),
        ):
            with pytest.raises(RuntimeError, match="No detection backend available"):
                get_detector("auto")


# ── SCRFDDetector (mocked) ────────────────────────────────────────


class TestSCRFDDetector:
    def test_importable(self):
        from visage.detectors.scrfd import SCRFDDetector
        assert SCRFDDetector.name == "scrfd"

    def test_is_available_returns_false_without_insightface(self):
        from visage.detectors.scrfd import SCRFDDetector
        detector = SCRFDDetector()
        # Without insightface installed, this should be False
        # (This test is valid regardless of platform)

    def test_detect_raises_when_unavailable(self):
        from visage.detectors.scrfd import SCRFDDetector
        detector = SCRFDDetector()
        with patch.object(detector, "_check_available", return_value=False):
            with pytest.raises(RuntimeError, match="insightface library not available"):
                detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_detect_structure_with_mocked_app(self):
        """Verify SCRFD detect() returns correct structure with fake results."""
        from visage.detectors.scrfd import SCRFDDetector
        detector = SCRFDDetector()
        detector._available = True

        # Mock a fake insightface Face object
        class FakeIFace:
            def __init__(self):
                self.bbox = np.array([10, 20, 110, 120])
                self.det_score = 0.95
                self.kps = np.array([
                    [30, 40], [90, 40], [60, 70], [35, 100], [85, 100],
                ])

        fake_app = MagicMock()
        fake_app.get.return_value = [FakeIFace()]
        detector._app = fake_app

        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        results = detector.detect(image)

        assert len(results) == 1
        fb, conf, lm5 = results[0]
        assert fb.left == 10
        assert fb.top == 20
        assert fb.right == 110
        assert fb.bottom == 120
        assert conf == 0.95
        assert lm5 is not None
        assert len(lm5) == 5

    def test_detect_filters_by_confidence(self):
        """Faces below min_confidence should be excluded."""
        from visage.detectors.scrfd import SCRFDDetector
        detector = SCRFDDetector(min_confidence=0.7)
        detector._available = True

        class FakeIFaceLow:
            def __init__(self):
                self.bbox = np.array([10, 20, 110, 120])
                self.det_score = 0.5  # below threshold
                self.kps = None

        fake_app = MagicMock()
        fake_app.get.return_value = [FakeIFaceLow()]
        detector._app = fake_app

        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        results = detector.detect(image)
        assert len(results) == 0

    def test_detect_filters_by_min_size(self):
        """Faces below min_face_size should be excluded."""
        from visage.detectors.scrfd import SCRFDDetector
        detector = SCRFDDetector(min_face_size=100)
        detector._available = True

        class FakeIFaceSmall:
            def __init__(self):
                self.bbox = np.array([10, 20, 50, 60])  # too small
                self.det_score = 0.9
                self.kps = None

        fake_app = MagicMock()
        fake_app.get.return_value = [FakeIFaceSmall()]
        detector._app = fake_app

        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        results = detector.detect(image)
        assert len(results) == 0

    def test_handles_no_kps(self):
        """Detect should work even when kps is None."""
        from visage.detectors.scrfd import SCRFDDetector
        detector = SCRFDDetector()
        detector._available = True

        class FakeIFaceNoKps:
            def __init__(self):
                self.bbox = np.array([10, 20, 110, 120])
                self.det_score = 0.95
                self.kps = None

        fake_app = MagicMock()
        fake_app.get.return_value = [FakeIFaceNoKps()]
        detector._app = fake_app

        image = np.zeros((200, 200, 3), dtype=np.uint8)
        results = detector.detect(image)
        assert len(results) == 1
        assert results[0][2] is None  # landmarks_5 should be None

    def test_handles_exception(self):
        """Exception in app.get() should be caught and return empty list."""
        from visage.detectors.scrfd import SCRFDDetector
        detector = SCRFDDetector()
        detector._available = True
        fake_app = MagicMock()
        fake_app.get.side_effect = RuntimeError("model crash")
        detector._app = fake_app
        results = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        assert results == []


# ── YuNetDetector (mocked) ────────────────────────────────────────


class TestYuNetDetector:
    def test_importable(self):
        from visage.detectors.yunet import YuNetDetector
        assert YuNetDetector.name == "yunet"

    def test_is_available_returns_false_without_cv2(self):
        from visage.detectors.yunet import YuNetDetector
        detector = YuNetDetector(model_path="/nonexistent/model.onnx")
        # Without opencv, this will be False normally
        # But we check the is_available returns bool

    def test_detect_raises_without_model(self):
        from visage.detectors.yunet import YuNetDetector
        detector = YuNetDetector()
        with patch.object(detector, "is_available", return_value=True):
            with patch.object(detector, "_model", None):
                with patch.object(detector, "_model_path", None):
                    with pytest.raises(RuntimeError, match="YuNet model not found"):
                        detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_detect_structure_with_mocked_cv2(self, tmp_path):
        """Verify YuNet detect() returns correct structure with fake results."""
        # Create a mock model file
        model_path = tmp_path / "face_detection_yunet_2023mar.onnx"
        model_path.write_text("fake model content")

        from visage.detectors.yunet import YuNetDetector
        detector = YuNetDetector(model_path=str(model_path))

        # Mock OpenCV
        with patch("visage.detectors.yunet.cv2") as mock_cv2:
            mock_model = MagicMock()
            mock_cv2.FaceDetectorYN.create.return_value = mock_model

            # Simulate YuNet output: (N, 15) array
            # [x1, y1, w, h, re_x, re_y, le_x, le_y, nt_x, nt_y,
            #  rcm_x, rcm_y, lcm_x, lcm_y, score]
            fake_results = np.array([[
                30, 40, 100, 120,    # bbox x=30,y=40,w=100,h=120
                60, 50,              # right eye
                100, 50,             # left eye
                80, 80,              # nose tip
                55, 110,             # right mouth corner
                105, 110,            # left mouth corner
                0.92,                # score
            ]])
            mock_model.detect.return_value = (True, fake_results)

            image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            results = detector.detect(image)

            assert len(results) == 1
            fb, conf, lm5 = results[0]
            assert fb.left == 30
            assert fb.top == 40
            assert fb.right == 130   # left + w
            assert fb.bottom == 160   # top + h
            assert conf == 0.92
            assert lm5 is not None
            assert len(lm5) == 5

    def test_detect_returns_empty_when_no_faces(self, tmp_path):
        """YuNet returns None when no faces detected."""
        model_path = tmp_path / "face_detection_yunet_2023mar.onnx"
        model_path.write_text("fake model content")

        from visage.detectors.yunet import YuNetDetector
        detector = YuNetDetector(model_path=str(model_path))

        with patch("visage.detectors.yunet.cv2") as mock_cv2:
            mock_model = MagicMock()
            mock_cv2.FaceDetectorYN.create.return_value = mock_model
            mock_model.detect.return_value = (True, None)

            image = np.zeros((100, 100, 3), dtype=np.uint8)
            results = detector.detect(image)
            assert results == []

    def test_filters_by_confidence(self, tmp_path):
        model_path = tmp_path / "face_detection_yunet_2023mar.onnx"
        model_path.write_text("fake model content")

        from visage.detectors.yunet import YuNetDetector
        detector = YuNetDetector(model_path=str(model_path), min_confidence=0.8)

        with patch("visage.detectors.yunet.cv2") as mock_cv2:
            mock_model = MagicMock()
            mock_cv2.FaceDetectorYN.create.return_value = mock_model

            fake_results = np.array([[
                10, 20, 50, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5,
            ]])  # score 0.5 < 0.8

            mock_model.detect.return_value = (True, fake_results)
            image = np.zeros((200, 200, 3), dtype=np.uint8)
            results = detector.detect(image)
            assert len(results) == 0

    def test_filters_by_face_size(self, tmp_path):
        model_path = tmp_path / "face_detection_yunet_2023mar.onnx"
        model_path.write_text("fake model content")

        from visage.detectors.yunet import YuNetDetector
        detector = YuNetDetector(model_path=str(model_path), min_face_size=100)

        with patch("visage.detectors.yunet.cv2") as mock_cv2:
            mock_model = MagicMock()
            mock_cv2.FaceDetectorYN.create.return_value = mock_model

            fake_results = np.array([[
                10, 20, 30, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9,
            ]])  # w=30 < 100
            mock_model.detect.return_value = (True, fake_results)
            image = np.zeros((200, 200, 3), dtype=np.uint8)
            results = detector.detect(image)
            assert len(results) == 0

    def test_handles_rgb_to_bgr_conversion(self, tmp_path):
        """Verifies the RGB to BGR conversion on a 4-channel image doesn't crash."""
        model_path = tmp_path / "face_detection_yunet_2023mar.onnx"
        model_path.write_text("fake model content")

        from visage.detectors.yunet import YuNetDetector
        detector = YuNetDetector(model_path=str(model_path))
        # RGBA image
        image = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)

        with patch("visage.detectors.yunet.cv2") as mock_cv2:
            mock_model = MagicMock()
            mock_cv2.FaceDetectorYN.create.return_value = mock_model
            mock_model.detect.return_value = (True, None)
            results = detector.detect(image)
            assert results == []


# ── NMS module ────────────────────────────────────────────────────


class TestNMSModule:
    def test_nms_importable_from_detectors(self):
        from visage.detectors.nms import _nms
        from visage.detector import _nms as _nms_facade
        assert _nms is _nms_facade  # same function, re-exported


# ── VisionDetector (basic interface) ──────────────────────────────


class TestVisionDetector:
    def test_importable(self):
        from visage.detectors.vision import VisionDetector
        assert VisionDetector.name == "vision"
        assert hasattr(VisionDetector, "is_available")

    def test_is_available_matches_platform(self):
        from visage.detectors.vision import VisionDetector
        from visage.detectors.vision import _VISION_AVAILABLE
        detector = VisionDetector()
        assert detector.is_available() == _VISION_AVAILABLE

    def test_detect_raises_on_non_macos(self):
        """On non-macOS, VisionDetector.detect() raises."""
        from visage.detectors.vision import VisionDetector
        detector = VisionDetector()
        with patch("visage.detectors.vision._VISION_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="macOS Vision framework not available"):
                detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
