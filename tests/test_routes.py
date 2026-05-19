"""Tests for the FastAPI routes (path validation, thumbnail cache)."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from visage.config import VisageConfig
from visage.models import ClusterResult, DetectedFace, FaceBox, ImageResult
from visage.server.app import create_app
from visage.server.workspace import Workspace


def _make_workspace(input_dir: str = "/photos") -> Workspace:
    """Build a minimal Workspace for testing."""
    config = VisageConfig()
    image_results: list[ImageResult] = []
    face_to_image: list[tuple[str, int]] = []
    labels: list[int] = []
    embeddings = []

    for cid in range(2):
        for i in range(3):
            path = f"{input_dir}/cluster{cid}_img{i}.jpg"
            face = DetectedFace(
                face_box=FaceBox(top=10, right=100, bottom=100, left=10),
                confidence=0.95,
            )
            image_results.append(ImageResult(path=path, faces=[face], error=None))
            face_to_image.append((path, 0))
            labels.append(cid)
            embeddings.append(np.random.randn(128))

    for i in range(2):
        path = f"{input_dir}/noise_{i}.jpg"
        face = DetectedFace(
            face_box=FaceBox(top=5, right=50, bottom=50, left=5),
            confidence=0.6,
        )
        image_results.append(ImageResult(path=path, faces=[face], error=None))
        face_to_image.append((path, 0))
        labels.append(-1)
        embeddings.append(np.random.randn(128))

    emb_array = np.array(embeddings)
    norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
    emb_array = emb_array / (norms + 1e-10)

    cluster_result = ClusterResult(
        labels=np.array(labels),
        embeddings=emb_array,
        num_clusters=2,
        num_noise=2,
    )

    return Workspace(
        input_dir=input_dir,
        config=config,
        image_results=image_results,
        cluster_result=cluster_result,
        face_to_image=face_to_image,
    )


@pytest.fixture
def client():
    """Create a test client with a pre-loaded workspace."""
    app = create_app("/photos", config=VisageConfig())
    # Override the workspace immediately (bypass background pipeline)
    ws = _make_workspace()
    app.state.workspace = ws
    app.state.input_dir = "/photos"
    return TestClient(app)


class TestWorkspaceEndpoint:
    def test_get_workspace(self, client):
        resp = client.get("/api/workspace")
        assert resp.status_code == 200
        data = resp.json()
        assert "clusters" in data
        assert "noise_photos" in data
        assert data["stats"]["num_clusters"] == 2

    def test_workspace_not_loaded(self):
        """Returns 503 when workspace is None."""
        app = create_app("/photos", config=VisageConfig())
        app.state.workspace = None
        tc = TestClient(app)
        resp = tc.get("/api/workspace")
        assert resp.status_code == 503


class TestMergeEndpoint:
    def test_merge_clusters(self, client):
        resp = client.post("/api/clusters/merge", json={"from_id": 1, "to_id": 0})
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["workspace"]["stats"]["num_clusters"] == 1

    def test_merge_missing_params(self, client):
        resp = client.post("/api/clusters/merge", json={"from_id": 1})
        assert resp.status_code == 400

    def test_merge_nonexistent_cluster(self, client):
        resp = client.post("/api/clusters/merge", json={"from_id": 99, "to_id": 0})
        assert resp.status_code == 400


class TestRemoveEndpoint:
    def test_remove_face(self, client):
        resp = client.post("/api/clusters/0/remove", json={
            "image_path": "/photos/cluster0_img0.jpg",
        })
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_remove_missing_path(self, client):
        resp = client.post("/api/clusters/0/remove", json={})
        assert resp.status_code == 400


class TestMoveEndpoint:
    def test_move_face(self, client):
        resp = client.post("/api/clusters/move", json={
            "image_path": "/photos/cluster0_img0.jpg",
            "from_id": 0, "to_id": 1,
        })
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_move_missing_params(self, client):
        resp = client.post("/api/clusters/move", json={"image_path": "/photos/a.jpg"})
        assert resp.status_code == 400


class TestAssignEndpoint:
    def test_assign_noise(self, client):
        resp = client.post("/api/clusters/assign", json={
            "image_path": "/photos/noise_0.jpg", "to_id": 0,
        })
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_assign_non_noise_raises(self, client):
        resp = client.post("/api/clusters/assign", json={
            "image_path": "/photos/cluster0_img0.jpg", "to_id": 1,
        })
        assert resp.status_code == 400


class TestRenameEndpoint:
    def test_rename_cluster(self, client):
        resp = client.put("/api/clusters/0", json={"name": "Alice"})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True


class TestUndoEndpoint:
    def test_undo_merge(self, client):
        client.post("/api/clusters/merge", json={"from_id": 1, "to_id": 0})
        resp = client.post("/api/clusters/undo")
        assert resp.status_code == 200
        assert resp.json()["undo"]["kind"] == "merge"

    def test_undo_empty(self, client):
        resp = client.post("/api/clusters/undo")
        assert resp.status_code == 400


class TestConfigEndpoint:
    def test_get_config(self, client):
        resp = client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["copy_mode"] is True
        assert "embedding_backend" in data


class TestImageEndpoint:
    def test_path_traversal_blocked(self, client):
        resp = client.get("/api/image", params={"path": "/etc/passwd", "size": "full"})
        assert resp.status_code == 403

    def test_nonexistent_image_404(self, client):
        resp = client.get(
            "/api/image",
            params={"path": "/photos/nonexistent.jpg", "size": "full"},
        )
        assert resp.status_code == 404


class TestThumbnailCache:
    def test_lru_eviction(self):
        from visage.server.routes import _THUMB_CACHE_MAX, _thumbnail_cache

        # Simulate cache filling up
        _thumbnail_cache.clear()
        for i in range(_THUMB_CACHE_MAX + 10):
            _thumbnail_cache[f"key_{i}"] = b"data"
            if len(_thumbnail_cache) > _THUMB_CACHE_MAX:
                _thumbnail_cache.popitem(last=False)

        assert len(_thumbnail_cache) <= _THUMB_CACHE_MAX
        # Oldest keys should be evicted
        assert "key_0" not in _thumbnail_cache
        # Newest key should still be present
        assert f"key_{_THUMB_CACHE_MAX + 9}" in _thumbnail_cache
        _thumbnail_cache.clear()

    def test_move_to_end_on_hit(self):
        from visage.server.routes import _thumbnail_cache

        _thumbnail_cache.clear()
        _thumbnail_cache["old"] = b"old_data"
        _thumbnail_cache["new"] = b"new_data"

        # Access "old" to move it to end (most recently used)
        _thumbnail_cache.move_to_end("old")

        # FIFO eviction should remove "new" first now
        evicted_key, _ = _thumbnail_cache.popitem(last=False)
        assert evicted_key == "new"
        _thumbnail_cache.clear()
