"""Tests for visage.embedder — face embedding with mocked face_recognition and image loading."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from visage.models import DetectedFace, FaceBox, ImageResult


def _make_face_result(num_faces: int = 1, path: str = "/tmp/test.jpg") -> ImageResult:
    """Build an ImageResult with N detected faces (no embeddings yet)."""
    faces = []
    for i in range(num_faces):
        faces.append(DetectedFace(
            face_box=FaceBox(top=10, right=110, bottom=110, left=10),
            confidence=0.9,
            image_path=path,
            face_index=i,
        ))
    return ImageResult(path=path, faces=faces)


# ── _check_face_recognition ───────────────────────────────────────


class TestCheckFaceRecognition:
    def test_raises_when_unavailable(self):
        from visage.embedder import _check_face_recognition

        with patch("visage.embedder._FR_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="face_recognition library not available"):
                _check_face_recognition()


# ── generate_embedding ────────────────────────────────────────────


class TestGenerateEmbedding:
    def test_returns_128d_vector(self):
        face_box = FaceBox(top=10, right=110, bottom=110, left=10)
        mock_encoding = np.random.randn(128).astype(np.float64)

        with patch("visage.embedder._FR_AVAILABLE", True):
            with patch("visage.embedder.face_recognition") as mock_fr:
                mock_fr.face_encodings.return_value = [mock_encoding]

                from visage.embedder import generate_embedding
                result = generate_embedding(np.zeros((100, 100, 3)), face_box)

                assert result is not None
                assert len(result) == 128
                np.testing.assert_array_equal(result, mock_encoding)

    def test_returns_none_on_empty_result(self):
        face_box = FaceBox(top=10, right=110, bottom=110, left=10)

        with patch("visage.embedder._FR_AVAILABLE", True):
            with patch("visage.embedder.face_recognition") as mock_fr:
                mock_fr.face_encodings.return_value = []

                from visage.embedder import generate_embedding
                result = generate_embedding(np.zeros((100, 100, 3)), face_box)

                assert result is None

    def test_returns_none_on_exception(self):
        face_box = FaceBox(top=10, right=110, bottom=110, left=10)

        with patch("visage.embedder._FR_AVAILABLE", True):
            with patch("visage.embedder.face_recognition") as mock_fr:
                mock_fr.face_encodings.side_effect = RuntimeError("dlib error")

                from visage.embedder import generate_embedding
                result = generate_embedding(np.zeros((100, 100, 3)), face_box)

                assert result is None

    def test_uses_face_recognition_format(self):
        face_box = FaceBox(top=10, right=110, bottom=110, left=10)
        mock_encoding = np.random.randn(128).astype(np.float64)
        image = np.zeros((100, 100, 3))

        with patch("visage.embedder._FR_AVAILABLE", True):
            with patch("visage.embedder.face_recognition") as mock_fr:
                mock_fr.face_encodings.return_value = [mock_encoding]

                from visage.embedder import generate_embedding
                generate_embedding(image, face_box, model="large", num_jitters=5)

                # Verify face_recognition was called with correct format
                args, kwargs = mock_fr.face_encodings.call_args
                assert kwargs.get("known_face_locations") == [(10, 110, 110, 10)]
                assert kwargs.get("model") == "large"
                assert kwargs.get("num_jitters") == 5


# ── generate_embeddings_for_image ─────────────────────────────────


class TestGenerateEmbeddingsForImage:
    def test_populates_embeddings(self):
        result = _make_face_result(num_faces=2, path="/tmp/test.jpg")
        mock_emb = np.random.randn(128).astype(np.float64)

        with patch("visage.embedder._FR_AVAILABLE", True):
            with patch("visage.embedder.face_recognition") as mock_fr, \
                 patch("visage.embedder.load_image_as_numpy") as mock_load:

                mock_load.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
                mock_fr.face_encodings.return_value = [mock_emb]

                from visage.embedder import generate_embeddings_for_image
                result = generate_embeddings_for_image(result)

                assert len(result.faces) == 2
                for face in result.faces:
                    assert face.embedding is not None

    def test_skips_error_result(self):
        result = ImageResult(path="/tmp/test.jpg", error="detection failed")
        from visage.embedder import generate_embeddings_for_image
        result = generate_embeddings_for_image(result)
        assert result.error == "detection failed"
        assert result.faces == []

    def test_skips_skipped_result(self):
        result = ImageResult(path="/tmp/test.jpg", skipped=True)
        from visage.embedder import generate_embeddings_for_image
        result = generate_embeddings_for_image(result)
        assert result.faces == []

    def test_skips_no_faces(self):
        result = ImageResult(path="/tmp/test.jpg", faces=[])
        from visage.embedder import generate_embeddings_for_image
        result = generate_embeddings_for_image(result)
        assert result.faces == []

    def test_filters_none_embeddings(self):
        """Second face returns no encoding — only the successful one is kept."""
        result = _make_face_result(num_faces=2, path="/tmp/test.jpg")

        call_count = 0

        def face_encodings_side_effect(image, known_face_locations=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [np.random.randn(128)]
            else:
                return []  # Second face fails

        with patch("visage.embedder._FR_AVAILABLE", True):
            with patch("visage.embedder.face_recognition") as mock_fr, \
                 patch("visage.embedder.load_image_as_numpy") as mock_load:

                mock_load.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
                mock_fr.face_encodings.side_effect = face_encodings_side_effect

                from visage.embedder import generate_embeddings_for_image
                result = generate_embeddings_for_image(result)

                assert len(result.faces) == 1  # Only 1 got an embedding

    def test_load_error(self):
        result = _make_face_result(path="/tmp/bad.jpg")

        with patch("visage.embedder.load_image_as_numpy") as mock_load:
            mock_load.side_effect = ValueError("Cannot load image")
            from visage.embedder import generate_embeddings_for_image
            result = generate_embeddings_for_image(result)

            assert result.error is not None
            assert "Cannot load image" in result.error


# ── generate_embeddings_batch ─────────────────────────────────────


class TestGenerateEmbeddingsBatch:
    def test_processes_all_faces(self):
        results = [
            _make_face_result(num_faces=1, path="/tmp/a.jpg"),
            _make_face_result(num_faces=2, path="/tmp/b.jpg"),
        ]

        with patch("visage.embedder._FR_AVAILABLE", True):
            with patch("visage.embedder.face_recognition") as mock_fr, \
                 patch("visage.embedder.load_image_as_numpy") as mock_load:

                mock_load.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
                mock_fr.face_encodings.return_value = [np.random.randn(128)]

                from visage.embedder import generate_embeddings_batch
                updated, cache_hits = generate_embeddings_batch(results)

                assert len(updated) == 2
                assert cache_hits == 0

    def test_progress_callback(self):
        results = [
            _make_face_result(num_faces=1, path="/tmp/a.jpg"),
            _make_face_result(num_faces=1, path="/tmp/b.jpg"),
        ]
        callbacks = []

        with patch("visage.embedder._FR_AVAILABLE", True):
            with patch("visage.embedder.face_recognition") as mock_fr, \
                 patch("visage.embedder.load_image_as_numpy") as mock_load:

                mock_load.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
                mock_fr.face_encodings.return_value = [np.random.randn(128)]

                from visage.embedder import generate_embeddings_batch
                generate_embeddings_batch(
                    results,
                    progress_callback=lambda c, t: callbacks.append((c, t)),
                )

                assert len(callbacks) >= 1
                # Final callback should show completed == total
                assert callbacks[-1][0] == callbacks[-1][1]

    def test_cache_hit(self):
        results = [_make_face_result(num_faces=1, path="/tmp/a.jpg")]

        mock_cache = MagicMock()
        mock_cache.lookup.return_value = [
            DetectedFace(
                face_box=FaceBox(top=10, right=110, bottom=110, left=10),
                confidence=0.9,
                embedding=np.random.randn(128),
                image_path="/tmp/a.jpg",
                face_index=0,
            ),
        ]

        from visage.embedder import generate_embeddings_batch
        updated, cache_hits = generate_embeddings_batch(results, cache=mock_cache)

        assert cache_hits == 1
        mock_cache.lookup.assert_called()
        # Should NOT have called store since cache hit
        mock_cache.store.assert_not_called()

    def test_cache_store_on_compute(self):
        results = [_make_face_result(num_faces=1, path="/tmp/a.jpg")]

        mock_cache = MagicMock()
        mock_cache.lookup.return_value = None  # cache miss

        with patch("visage.embedder._FR_AVAILABLE", True):
            with patch("visage.embedder.face_recognition") as mock_fr, \
                 patch("visage.embedder.load_image_as_numpy") as mock_load:

                mock_load.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
                mock_fr.face_encodings.return_value = [np.random.randn(128)]

                from visage.embedder import generate_embeddings_batch
                generate_embeddings_batch(results, cache=mock_cache)

                mock_cache.store.assert_called()

    def test_skips_error_images(self):
        results = [
            ImageResult(path="/tmp/bad.jpg", error="oops"),
            _make_face_result(num_faces=1, path="/tmp/good.jpg"),
        ]

        with patch("visage.embedder._FR_AVAILABLE", True):
            with patch("visage.embedder.face_recognition") as mock_fr, \
                 patch("visage.embedder.load_image_as_numpy") as mock_load:

                mock_load.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
                mock_fr.face_encodings.return_value = [np.random.randn(128)]

                from visage.embedder import generate_embeddings_batch
                updated, cache_hits = generate_embeddings_batch(results)

                # Error image stays as-is, good image gets embeddings
                assert updated[0].error == "oops"
                assert len(updated[1].faces) == 1

    def test_returns_tuple(self):
        results = [_make_face_result(num_faces=1, path="/tmp/a.jpg")]

        with patch("visage.embedder._FR_AVAILABLE", True):
            with patch("visage.embedder.face_recognition") as mock_fr, \
                 patch("visage.embedder.load_image_as_numpy") as mock_load:

                mock_load.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
                mock_fr.face_encodings.return_value = [np.random.randn(128)]

                from visage.embedder import generate_embeddings_batch
                result = generate_embeddings_batch(results)

                assert isinstance(result, tuple)
                assert len(result) == 2
                assert isinstance(result[0], list)
                assert isinstance(result[1], int)
