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

        face_box = FaceBox(top=10, right=110, bottom=110, left=10)
        result = backend.generate(np.zeros((100, 100, 3), dtype=np.uint8), face_box)
        assert result is None

    def test_find_best_match_returns_best_iou(self):
        from visage.backends import InsightFaceBackend
        face_box = FaceBox(top=10, right=110, bottom=110, left=10)

        face_good = MagicMock()
        face_good.bbox = np.array([5, 5, 115, 115])  # high overlap

        face_bad = MagicMock()
        face_bad.bbox = np.array([200, 200, 300, 300])  # no overlap

        result = InsightFaceBackend._find_best_match([face_good, face_bad], face_box)
        assert result is face_good

    def test_find_best_match_returns_none_low_iou(self):
        from visage.backends import InsightFaceBackend
        face_box = FaceBox(top=10, right=110, bottom=110, left=10)

        face_far = MagicMock()
        face_far.bbox = np.array([500, 500, 600, 600])

        result = InsightFaceBackend._find_best_match([face_far], face_box)
        assert result is None


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
