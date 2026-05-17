"""Tests for visage.cache — SQLite embedding cache with tmp_path."""

from __future__ import annotations

import time

import numpy as np
import pytest

from visage.cache import EmbeddingCache, _file_fingerprint
from visage.models import DetectedFace, FaceBox


def _make_face(
    path: str = "/tmp/test.jpg",
    index: int = 0,
    embedding: np.ndarray | None = None,
) -> DetectedFace:
    """Create a DetectedFace for cache tests."""
    if embedding is None:
        embedding = np.random.randn(128).astype(np.float64)
    return DetectedFace(
        face_box=FaceBox(top=10, right=110, bottom=110, left=10),
        confidence=0.9,
        embedding=embedding,
        image_path=path,
        face_index=index,
    )


# ── _file_fingerprint ─────────────────────────────────────────────


class TestFileFingerprint:
    def test_existing_file(self, tmp_path):
        f = tmp_path / "test.jpg"
        f.write_text("hello")
        fp = _file_fingerprint(str(f))
        assert fp is not None
        assert ":" in fp  # "size:mtime" format

    def test_missing_file(self):
        fp = _file_fingerprint("/nonexistent/file.jpg")
        assert fp is None


# ── EmbeddingCache initialization ─────────────────────────────────


class TestEmbeddingCacheInit:
    def test_creates_cache_directory(self, tmp_path):
        EmbeddingCache(str(tmp_path))
        assert (tmp_path / ".visage_cache").is_dir()

    def test_creates_database(self, tmp_path):
        EmbeddingCache(str(tmp_path))
        assert (tmp_path / ".visage_cache" / "embeddings.db").exists()


# ── store and lookup ──────────────────────────────────────────────


class TestStoreAndLookup:
    def test_store_and_lookup(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path))
        # Create a real file so fingerprint works
        f = tmp_path / "photo.jpg"
        f.write_text("img")

        faces = [_make_face(str(f), index=0)]
        cache.store(str(f), faces)
        result = cache.lookup(str(f))
        assert result is not None
        assert len(result) == 1
        assert result[0].face_index == 0
        cache.close()

    def test_lookup_miss(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path))
        f = tmp_path / "missing.jpg"
        f.write_text("img")
        result = cache.lookup(str(f))
        assert result is None
        cache.close()

    def test_different_model(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path))
        f = tmp_path / "photo.jpg"
        f.write_text("img")

        faces = [_make_face(str(f))]
        cache.store(str(f), faces, model="small")
        result = cache.lookup(str(f), model="large")
        assert result is None
        cache.close()

    def test_different_jitters(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path))
        f = tmp_path / "photo.jpg"
        f.write_text("img")

        faces = [_make_face(str(f))]
        cache.store(str(f), faces, num_jitters=1)
        result = cache.lookup(str(f), num_jitters=2)
        assert result is None
        cache.close()

    def test_store_overwrites_old(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path))
        f = tmp_path / "photo.jpg"
        f.write_text("img")

        old_faces = [_make_face(str(f), index=0)]
        cache.store(str(f), old_faces)

        new_emb = np.random.randn(128).astype(np.float64)
        new_faces = [_make_face(str(f), index=0, embedding=new_emb)]
        cache.store(str(f), new_faces)

        result = cache.lookup(str(f))
        assert result is not None
        np.testing.assert_array_equal(result[0].embedding, new_emb)
        cache.close()

    def test_store_multiple_faces(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path))
        f = tmp_path / "multi.jpg"
        f.write_text("img")

        faces = [_make_face(str(f), index=i) for i in range(3)]
        cache.store(str(f), faces)
        result = cache.lookup(str(f))
        assert len(result) == 3
        cache.close()

    def test_store_skips_none_embedding(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path))
        f = tmp_path / "photo.jpg"
        f.write_text("img")

        face_with = _make_face(str(f), index=0)
        face_without = DetectedFace(
            face_box=FaceBox(top=0, right=50, bottom=50, left=0),
            confidence=0.5,
            embedding=None,
            image_path=str(f),
            face_index=1,
        )
        cache.store(str(f), [face_with, face_without])
        result = cache.lookup(str(f))
        assert len(result) == 1  # Only the face with embedding
        cache.close()


# ── embedding value preservation ──────────────────────────────────


class TestEmbeddingPreservation:
    def test_values_preserved(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path))
        f = tmp_path / "photo.jpg"
        f.write_text("img")

        original_emb = np.array([1.0, 2.0, 3.0] + [0.0] * 125, dtype=np.float64)
        faces = [_make_face(str(f), embedding=original_emb)]
        cache.store(str(f), faces)
        result = cache.lookup(str(f))
        assert result is not None
        np.testing.assert_array_almost_equal(result[0].embedding, original_emb)
        cache.close()


# ── get_stats ─────────────────────────────────────────────────────


class TestGetStats:
    def test_empty(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path))
        stats = cache.get_stats()
        assert stats == {"cached_images": 0, "cached_faces": 0}
        cache.close()

    def test_after_store(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path))
        f1 = tmp_path / "a.jpg"
        f1.write_text("img1")
        f2 = tmp_path / "b.jpg"
        f2.write_text("img2")

        cache.store(str(f1), [_make_face(str(f1), index=0)])
        cache.store(str(f2), [_make_face(str(f2), index=0), _make_face(str(f2), index=1)])

        stats = cache.get_stats()
        assert stats["cached_images"] == 2
        assert stats["cached_faces"] == 3
        cache.close()


# ── fingerprint invalidation ──────────────────────────────────────


class TestFingerprintInvalidation:
    def test_fingerprint_changes_on_file_edit(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path))
        f = tmp_path / "photo.jpg"
        f.write_text("original")
        time.sleep(0.01)

        faces = [_make_face(str(f))]
        cache.store(str(f), faces)

        # Modify the file (change size)
        time.sleep(0.01)
        f.write_text("modified content — much longer")

        result = cache.lookup(str(f))
        assert result is None  # fingerprint mismatch
        cache.close()


# ── checkpoint ────────────────────────────────────────────────────


class TestCheckpoint:
    def test_save_and_load(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path))
        cache.save_checkpoint(3, "100 embeddings")
        cp = cache.load_checkpoint()
        assert cp is not None
        assert cp["phase"] == 3
        assert cp["message"] == "100 embeddings"
        assert "timestamp" in cp
        cache.close()

    def test_load_missing(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path))
        assert cache.load_checkpoint() is None
        cache.close()

    def test_clear(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path))
        cache.save_checkpoint(2, "test")
        cache.clear_checkpoint()
        assert cache.load_checkpoint() is None
        cache.close()

    def test_save_updates_stats(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path))
        f = tmp_path / "photo.jpg"
        f.write_text("img")
        cache.store(str(f), [_make_face(str(f))])

        cache.save_checkpoint(3, "test")
        cp = cache.load_checkpoint()
        assert cp["cached_images"] == 1
        assert cp["cached_faces"] == 1
        cache.close()


# ── close ─────────────────────────────────────────────────────────


class TestClose:
    def test_close(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path))
        cache.close()  # Should not raise

    def test_close_twice(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path))
        cache.close()
        cache.close()  # Should not raise


# ── quality persistence ────────────────────────────────────────────


class TestQualityPersistence:
    def test_quality_stored_and_restored(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path))
        f = tmp_path / "photo.jpg"
        f.write_text("img")

        face = _make_face(str(f))
        face.quality = 0.85
        cache.store(str(f), [face])
        result = cache.lookup(str(f))
        assert result is not None
        assert len(result) == 1
        assert result[0].quality == pytest.approx(0.85)
        cache.close()

    def test_quality_none_stored_and_restored(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path))
        f = tmp_path / "photo.jpg"
        f.write_text("img")

        face = _make_face(str(f))
        face.quality = None
        cache.store(str(f), [face])
        result = cache.lookup(str(f))
        assert result is not None
        assert result[0].quality is None
        cache.close()

    def test_migration_adds_quality_column(self, tmp_path):
        """Verify that an old DB without quality column is migrated."""
        cache = EmbeddingCache(str(tmp_path))
        # Manually drop and recreate without quality to simulate old schema
        cache._conn.execute("DROP TABLE face_embeddings")
        cache._conn.execute("""
            CREATE TABLE face_embeddings (
                image_path TEXT NOT NULL,
                file_fingerprint TEXT NOT NULL,
                face_index INTEGER NOT NULL,
                face_box TEXT NOT NULL,
                confidence REAL NOT NULL,
                embedding BLOB NOT NULL,
                model TEXT NOT NULL,
                num_jitters INTEGER NOT NULL,
                PRIMARY KEY (image_path, face_index)
            )
        """)
        cache._conn.commit()
        cache.close()

        # Re-open — migration should add quality column
        cache2 = EmbeddingCache(str(tmp_path))
        f = tmp_path / "photo.jpg"
        f.write_text("img")
        face = _make_face(str(f))
        face.quality = 0.42
        cache2.store(str(f), [face])
        result = cache2.lookup(str(f))
        assert result is not None
        assert result[0].quality == pytest.approx(0.42)
        cache2.close()
