"""Tests for FAISS vector index and metadata store."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from visage.vector.index import VectorIndex
from visage.vector.metadata import MetadataStore

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def vectors_128d():
    """20 random 128-dim vectors, L2-normalized."""
    rng = np.random.RandomState(42)
    vecs = rng.randn(20, 128).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


@pytest.fixture
def index_128(vectors_128d):
    """VectorIndex with 20 vectors of dim 128, IDs: face_000..face_019."""
    idx = VectorIndex(dim=128)
    ids = [f"face_{i:03d}" for i in range(20)]
    idx.add_batch(ids, vectors_128d)
    return idx


@pytest.fixture
def metadata_db(tmp_path):
    """MetadataStore backed by a temp SQLite file."""
    return MetadataStore(tmp_path / "test_meta.db")


# ── VectorIndex: Add / Search ─────────────────────────────────────


class TestVectorIndexAdd:
    def test_add_single(self):
        idx = VectorIndex(dim=128)
        vec = np.random.randn(128).astype(np.float32)
        idx.add("face_0", vec)
        assert idx.total == 1
        assert idx.active == 1

    def test_add_batch(self, vectors_128d):
        idx = VectorIndex(dim=128)
        ids = [f"face_{i}" for i in range(20)]
        idx.add_batch(ids, vectors_128d)
        assert idx.total == 20
        assert idx.active == 20

    def test_add_duplicate_id_overwrites(self):
        idx = VectorIndex(dim=128)
        vec1 = np.random.randn(128).astype(np.float32)
        vec2 = np.random.randn(128).astype(np.float32)
        idx.add("face_0", vec1)
        idx.add("face_0", vec2)
        assert idx.total == 2


class TestVectorIndexSearch:
    def test_search_returns_results(self, index_128):
        query = np.random.randn(128).astype(np.float32)
        results = index_128.search(query, top_k=5)
        assert 0 < len(results) <= 5
        for fid, score in results:
            assert fid.startswith("face_")
            assert 0 <= score <= 1.01

    def test_search_by_id(self, index_128):
        results = index_128.search_by_id("face_000", top_k=3)
        assert len(results) <= 3
        assert all(fid != "face_000" for fid, _ in results)

    def test_search_by_nonexistent_id(self, index_128):
        results = index_128.search_by_id("nonexistent", top_k=3)
        assert results == []

    def test_search_empty_index(self):
        idx = VectorIndex(dim=128)
        query = np.random.randn(128).astype(np.float32)
        assert idx.search(query) == []

    def test_self_is_most_similar(self, vectors_128d):
        idx = VectorIndex(dim=128)
        ids = [f"face_{i:03d}" for i in range(20)]
        idx.add_batch(ids, vectors_128d)
        query = vectors_128d[0]
        results = idx.search(query, top_k=1)
        assert results[0][0] == "face_000"
        assert results[0][1] > 0.99


# ── VectorIndex: Soft Delete ──────────────────────────────────────


class TestVectorIndexDelete:
    def test_soft_delete(self, index_128):
        assert index_128.soft_delete("face_000")
        assert index_128.active == 19
        assert index_128.deleted_count == 1

    def test_soft_delete_twice(self, index_128):
        assert index_128.soft_delete("face_000")
        assert not index_128.soft_delete("face_000")

    def test_soft_delete_nonexistent(self, index_128):
        assert not index_128.soft_delete("nonexistent")

    def test_search_excludes_deleted(self, index_128):
        index_128.soft_delete("face_000")
        query = np.random.randn(128).astype(np.float32)
        results = index_128.search(query, top_k=20)
        ids = [fid for fid, _ in results]
        assert "face_000" not in ids

    def test_needs_rebuild_threshold(self, vectors_128d):
        idx = VectorIndex(dim=128)
        ids = [f"face_{i:03d}" for i in range(20)]
        idx.add_batch(ids, vectors_128d)

        # Delete 1/20 = 5% — should not need rebuild
        idx.soft_delete("face_000")
        assert not idx.needs_rebuild

        # Delete 4/20 = 20% — should need rebuild
        for i in range(1, 4):
            idx.soft_delete(f"face_{i:03d}")
        assert idx.needs_rebuild


class TestVectorIndexRebuild:
    def test_rebuild_removes_deleted(self, vectors_128d):
        idx = VectorIndex(dim=128)
        ids = [f"face_{i:03d}" for i in range(20)]
        idx.add_batch(ids, vectors_128d)

        idx.soft_delete("face_000")
        idx.soft_delete("face_005")
        idx.soft_delete("face_010")
        assert idx.deleted_count == 3

        idx.rebuild()
        assert idx.active == 17
        assert idx.deleted_count == 0
        assert idx.total == 17

    def test_rebuild_preserves_search(self, vectors_128d):
        idx = VectorIndex(dim=128)
        ids = [f"face_{i:03d}" for i in range(20)]
        idx.add_batch(ids, vectors_128d)

        query = vectors_128d[0]

        idx.soft_delete("face_010")
        idx.rebuild()

        after = idx.search(query, top_k=5)
        after_ids = {fid for fid, _ in after}
        assert "face_010" not in after_ids
        assert idx.search_by_id("face_000", top_k=3) is not None


# ── VectorIndex: Persistence ──────────────────────────────────────


class TestVectorIndexPersistence:
    def test_save_and_load(self, index_128, tmp_path):
        path = tmp_path / "test.faiss"
        index_128.save(path)

        assert path.exists()
        assert Path(str(path) + ".meta").exists()

        loaded = VectorIndex.load(path)
        assert loaded.total == 20
        assert loaded.dim == 128

    def test_loaded_index_searches_correctly(self, index_128, tmp_path):
        path = tmp_path / "test.faiss"
        index_128.save(path)

        loaded = VectorIndex.load(path)
        query = np.random.randn(128).astype(np.float32)

        original_results = index_128.search(query, top_k=5)
        loaded_results = loaded.search(query, top_k=5)

        orig_ids = [fid for fid, _ in original_results]
        load_ids = [fid for fid, _ in loaded_results]
        assert orig_ids == load_ids

    def test_save_load_with_deleted(self, index_128, tmp_path):
        index_128.soft_delete("face_000")
        index_128.soft_delete("face_005")

        path = tmp_path / "test.faiss"
        index_128.save(path)

        loaded = VectorIndex.load(path)
        assert loaded.deleted_count == 2
        assert loaded.active == 18

    def test_metadata_file_format(self, index_128, tmp_path):
        path = tmp_path / "test.faiss"
        index_128.save(path)

        with open(str(path) + ".meta") as f:
            meta = json.load(f)

        assert meta["dim"] == 128
        assert len(meta["ids"]) == 20
        assert meta["version"] == 1


# ── MetadataStore ─────────────────────────────────────────────────


class TestMetadataStore:
    def test_add_and_get(self, metadata_db):
        metadata_db.add_face("face_0", "/photo/a.jpg", cluster_id="c1", quality_score=0.9)
        face = metadata_db.get_face("face_0")
        assert face is not None
        assert face["image_path"] == "/photo/a.jpg"
        assert face["cluster_id"] == "c1"
        assert face["quality_score"] == 0.9

    def test_get_nonexistent(self, metadata_db):
        assert metadata_db.get_face("nope") is None

    def test_update_cluster(self, metadata_db):
        metadata_db.add_face("face_0", "/photo/a.jpg", cluster_id="c1")
        metadata_db.update_cluster("face_0", "c2")
        face = metadata_db.get_face("face_0")
        assert face["cluster_id"] == "c2"

    def test_delete_face(self, metadata_db):
        metadata_db.add_face("face_0", "/photo/a.jpg")
        assert metadata_db.delete_face("face_0")
        assert metadata_db.get_face("face_0") is None

    def test_delete_nonexistent(self, metadata_db):
        assert not metadata_db.delete_face("nope")

    def test_get_faces_by_cluster(self, metadata_db):
        metadata_db.add_face("f1", "/a.jpg", cluster_id="c1")
        metadata_db.add_face("f2", "/b.jpg", cluster_id="c1")
        metadata_db.add_face("f3", "/c.jpg", cluster_id="c2")
        faces = metadata_db.get_faces_by_cluster("c1")
        assert len(faces) == 2

    def test_get_faces_by_image(self, metadata_db):
        metadata_db.add_face("f1", "/photo/same.jpg")
        metadata_db.add_face("f2", "/photo/same.jpg")
        metadata_db.add_face("f3", "/photo/other.jpg")
        faces = metadata_db.get_faces_by_image("/photo/same.jpg")
        assert len(faces) == 2

    def test_count_faces(self, metadata_db):
        metadata_db.add_face("f1", "/a.jpg")
        metadata_db.add_face("f2", "/b.jpg")
        assert metadata_db.count_faces() == 2

    def test_count_clusters(self, metadata_db):
        metadata_db.add_face("f1", "/a.jpg", cluster_id="c1")
        metadata_db.add_face("f2", "/b.jpg", cluster_id="c1")
        metadata_db.add_face("f3", "/c.jpg", cluster_id="c2")
        assert metadata_db.count_clusters() == 2

    def test_batch_update_clusters(self, metadata_db):
        metadata_db.add_face("f1", "/a.jpg", cluster_id="c1")
        metadata_db.add_face("f2", "/b.jpg", cluster_id="c1")
        metadata_db.batch_update_clusters([("f1", "c2"), ("f2", None)])
        assert metadata_db.get_face("f1")["cluster_id"] == "c2"
        assert metadata_db.get_face("f2")["cluster_id"] is None

    def test_bbox_serialization(self, metadata_db):
        metadata_db.add_face("f1", "/a.jpg", bbox=(10, 110, 110, 10))
        face = metadata_db.get_face("f1")
        assert face["bbox"] == (10, 110, 110, 10)

    def test_extra_json(self, metadata_db):
        metadata_db.add_face("f1", "/a.jpg", extra={"landmarks": [[1, 2], [3, 4]]})
        face = metadata_db.get_face("f1")
        assert face["extra"]["landmarks"] == [[1, 2], [3, 4]]

    def test_get_cluster_ids(self, metadata_db):
        metadata_db.add_face("f1", "/a.jpg", cluster_id="c1")
        metadata_db.add_face("f2", "/b.jpg", cluster_id="c2")
        metadata_db.add_face("f3", "/c.jpg")
        ids = metadata_db.get_cluster_ids()
        assert ids == ["c1", "c2"]

    def test_get_all_face_ids(self, metadata_db):
        metadata_db.add_face("f1", "/a.jpg")
        metadata_db.add_face("f2", "/b.jpg")
        assert metadata_db.get_all_face_ids() == ["f1", "f2"]

    def test_add_or_replace(self, metadata_db):
        metadata_db.add_face("f1", "/a.jpg", cluster_id="c1")
        metadata_db.add_face("f1", "/b.jpg", cluster_id="c2")
        assert metadata_db.count_faces() == 1
        face = metadata_db.get_face("f1")
        assert face["image_path"] == "/b.jpg"
