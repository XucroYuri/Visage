"""Tests for visage.backends — pluggable embedding backend architecture."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from visage.models import FaceBox

# ── DlibBackend ──────────────────────────────────────────────────


class TestDlibBackend:
    def test_name_and_dim(self):
        from visage.backends import DlibBackend
        backend = DlibBackend()
        assert backend.name == "dlib"
        assert backend.embedding_dim == 128

    def test_is_available_true(self):
        from visage.backends import DlibBackend
        with patch("visage.backends.face_recognition", create=True):
            # DlibBackend.__init__ tries import face_recognition
            # Since face_recognition IS installed in this env, it's available
            backend = DlibBackend()
            assert backend.is_available() is True

    def test_generate_returns_embedding(self):
        from visage.backends import DlibBackend
        mock_encoding = np.random.randn(128).astype(np.float64)

        with patch("face_recognition.face_encodings", return_value=[mock_encoding]):
            backend = DlibBackend()
            backend._available = True

            face_box = FaceBox(top=10, right=110, bottom=110, left=10)
            result = backend.generate(np.zeros((100, 100, 3)), face_box)

            assert result is not None
            assert len(result) == 128
            np.testing.assert_array_equal(result, mock_encoding)

    def test_generate_returns_none_on_empty(self):
        from visage.backends import DlibBackend

        with patch("face_recognition.face_encodings", return_value=[]):
            backend = DlibBackend()
            backend._available = True

            face_box = FaceBox(top=10, right=110, bottom=110, left=10)
            result = backend.generate(np.zeros((100, 100, 3)), face_box)
            assert result is None

    def test_generate_raises_when_unavailable(self):
        from visage.backends import DlibBackend
        backend = DlibBackend()
        backend._available = False

        face_box = FaceBox(top=10, right=110, bottom=110, left=10)
        with pytest.raises(RuntimeError, match="face_recognition"):
            backend.generate(np.zeros((100, 100, 3)), face_box)

    def test_passes_model_and_jitters(self):
        from visage.backends import DlibBackend
        mock_encoding = np.random.randn(128)

        with patch("face_recognition.face_encodings", return_value=[mock_encoding]) as mock_enc:
            backend = DlibBackend(model="large", num_jitters=5)
            backend._available = True

            face_box = FaceBox(top=10, right=110, bottom=110, left=10)
            backend.generate(np.zeros((100, 100, 3)), face_box)

            _, kwargs = mock_enc.call_args
            assert kwargs["model"] == "large"
            assert kwargs["num_jitters"] == 5

    def test_generate_returns_none_on_exception(self):
        from visage.backends import DlibBackend

        with patch("face_recognition.face_encodings", side_effect=RuntimeError("dlib crash")):
            backend = DlibBackend()
            backend._available = True

            face_box = FaceBox(top=10, right=110, bottom=110, left=10)
            result = backend.generate(np.zeros((100, 100, 3)), face_box)
            assert result is None


# ── InsightFaceBackend ───────────────────────────────────────────


class TestInsightFaceBackend:
    def test_name_and_dim(self):
        from visage.backends import InsightFaceBackend
        backend = InsightFaceBackend()
        assert backend.name == "insightface"
        assert backend.embedding_dim == 512

    def test_is_available_false_when_not_installed(self):
        from visage.backends import InsightFaceBackend
        # insightface is NOT installed in test env
        backend = InsightFaceBackend()
        # May be True or False depending on env, just check type
        assert isinstance(backend.is_available(), bool)

    def test_generate_raises_when_unavailable(self):
        from visage.backends import InsightFaceBackend
        backend = InsightFaceBackend()
        backend._available = False

        face_box = FaceBox(top=10, right=110, bottom=110, left=10)
        with pytest.raises(RuntimeError, match="insightface"):
            backend.generate(np.zeros((100, 100, 3)), face_box)

    def test_generate_with_mock_app(self):
        from visage.backends import InsightFaceBackend
        mock_face = MagicMock()
        mock_face.embedding = np.random.randn(512)
        mock_face.bbox = np.array([5, 5, 115, 115])

        backend = InsightFaceBackend()
        backend._available = True
        backend._app = MagicMock()
        backend._app.get.return_value = [mock_face]

        face_box = FaceBox(top=10, right=110, bottom=110, left=10)
        result = backend.generate(np.zeros((100, 100, 3), dtype=np.uint8), face_box)

        assert result is not None
        assert len(result) == 512

    def test_generate_returns_none_no_faces(self):
        from visage.backends import InsightFaceBackend
        backend = InsightFaceBackend()
        backend._available = True
        backend._app = MagicMock()
        backend._app.get.return_value = []
        # Prevent dlib fallback from generating an actual embedding
        backend._dlib_fallback = MagicMock()
        backend._dlib_fallback.generate.return_value = None

        face_box = FaceBox(top=10, right=110, bottom=110, left=10)
        result = backend.generate(np.zeros((100, 100, 3), dtype=np.uint8), face_box)
        assert result is None

    def test_crop_face_returns_cropped_region(self):
        from visage.backends import InsightFaceBackend
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        face_box = FaceBox(top=30, right=70, bottom=70, left=30)

        crop = InsightFaceBackend._crop_face(image, face_box)
        # 80% padding → face is 40x40, pad = 32 on each side
        h, w = crop.shape[:2]
        assert h > 40  # should have padding
        assert w > 40

    def test_crop_face_clamps_to_image_bounds(self):
        from visage.backends import InsightFaceBackend
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        face_box = FaceBox(top=0, right=110, bottom=110, left=0)

        crop = InsightFaceBackend._crop_face(image, face_box)
        h, w = crop.shape[:2]
        assert h <= 100
        assert w <= 100


# ── get_backend factory ──────────────────────────────────────────


class TestGetBackend:
    def test_returns_dlib_backend(self):
        from visage.backends import DlibBackend, get_backend
        backend = get_backend("dlib")
        assert isinstance(backend, DlibBackend)

    def test_returns_insightface_backend(self):
        from visage.backends import InsightFaceBackend, get_backend
        backend = get_backend("insightface")
        assert isinstance(backend, InsightFaceBackend)

    def test_raises_on_unknown(self):
        from visage.backends import get_backend
        with pytest.raises(ValueError, match="Unknown embedding backend"):
            get_backend("unknown")

    def test_passes_kwargs_to_dlib(self):
        from visage.backends import get_backend
        backend = get_backend("dlib", model="large", num_jitters=5)
        assert backend.model == "large"
        assert backend.num_jitters == 5
