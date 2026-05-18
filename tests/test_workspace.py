"""Tests for the Workspace class (in-memory mutable review state)."""

from __future__ import annotations

import numpy as np
import pytest

from visage.config import VisageConfig
from visage.models import ClusterResult, DetectedFace, FaceBox, ImageResult
from visage.server.workspace import Workspace


def _make_workspace(num_clusters: int = 3, photos_per_cluster: int = 5) -> Workspace:
    """Build a minimal Workspace for testing."""
    config = VisageConfig()
    image_results: list[ImageResult] = []
    face_to_image: list[tuple[str, int]] = []
    labels: list[int] = []
    embeddings = []

    for cid in range(num_clusters):
        for i in range(photos_per_cluster):
            path = f"/photos/cluster{cid}_img{i}.jpg"
            face = DetectedFace(
                face_box=FaceBox(top=10, right=100, bottom=100, left=10),
                confidence=0.95,
            )
            image_results.append(ImageResult(path=path, faces=[face], error=None))
            face_to_image.append((path, 0))
            labels.append(cid)
            embeddings.append(np.random.randn(128))

    # Add noise faces
    for i in range(3):
        path = f"/photos/noise_{i}.jpg"
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
        num_clusters=num_clusters,
        num_noise=3,
    )

    return Workspace(
        input_dir="/photos",
        config=config,
        image_results=image_results,
        cluster_result=cluster_result,
        face_to_image=face_to_image,
    )


class TestToApiDict:
    """Test Workspace.to_api_dict() serialization."""

    def test_all_numeric_types_are_python_native(self):
        ws = _make_workspace()
        d = ws.to_api_dict()

        # Stats should be Python ints, not numpy types
        for key, val in d["stats"].items():
            assert isinstance(val, int), f"stats.{key} is {type(val).__name__}, expected int"

        # can_undo should be bool
        assert isinstance(d["can_undo"], bool)

    def test_cluster_ids_are_python_int(self):
        ws = _make_workspace()
        d = ws.to_api_dict()

        for cluster in d["clusters"]:
            assert isinstance(cluster["id"], int), f"cluster id is {type(cluster['id']).__name__}"
            assert isinstance(cluster["photo_count"], int)
            assert isinstance(cluster["confidence"], float)

    def test_photos_have_face_boxes(self):
        ws = _make_workspace()
        d = ws.to_api_dict()

        for cluster in d["clusters"]:
            for photo in cluster["photos"]:
                assert "path" in photo
                assert "faces" in photo
                assert isinstance(photo["faces"], list)

    def test_face_box_values_are_python_int(self):
        ws = _make_workspace()
        d = ws.to_api_dict()

        for cluster in d["clusters"]:
            for photo in cluster["photos"]:
                for face in photo["faces"]:
                    for key in ("top", "right", "bottom", "left"):
                        assert isinstance(face[key], int), (
                            f"face.{key} is {type(face[key]).__name__}"
                        )

    def test_all_photos_field(self):
        ws = _make_workspace()
        d = ws.to_api_dict()

        assert "all_photos" in d
        assert isinstance(d["all_photos"], list)
        # all_photos should have photos from all clusters
        total_in_clusters = sum(c["photo_count"] for c in d["clusters"])
        assert len(d["all_photos"]) == total_in_clusters

    def test_noise_photos_field(self):
        ws = _make_workspace()
        d = ws.to_api_dict()

        assert "noise_photos" in d
        assert isinstance(d["noise_photos"], list)
        assert len(d["noise_photos"]) == 3

    def test_next_cluster_id_field(self):
        ws = _make_workspace()
        d = ws.to_api_dict()

        assert "next_cluster_id" in d
        assert isinstance(d["next_cluster_id"], int)
        assert d["next_cluster_id"] == 3  # clusters 0,1,2 → next is 3

    def test_thumbnail_is_string_path(self):
        ws = _make_workspace()
        d = ws.to_api_dict()

        for cluster in d["clusters"]:
            if cluster["photo_count"] > 0:
                assert isinstance(cluster["thumbnail"], str)


class TestMergeClusters:
    """Test Workspace.merge_clusters()."""

    def test_merge_combines_photos(self):
        ws = _make_workspace(num_clusters=3)
        c0_count = ws.cluster_count(0)
        c1_count = ws.cluster_count(1)

        ws.merge_clusters(1, 0)

        assert ws.cluster_count(0) == c0_count + c1_count
        assert 1 not in ws._cluster_mapping

    def test_merge_same_id_is_noop(self):
        ws = _make_workspace()
        ids_before = ws.cluster_ids

        ws.merge_clusters(0, 0)

        assert ws.cluster_ids == ids_before

    def test_merge_nonexistent_raises(self):
        ws = _make_workspace()

        with pytest.raises(ValueError, match="Cluster not found"):
            ws.merge_clusters(99, 0)

    def test_merge_enables_undo(self):
        ws = _make_workspace()
        assert not ws.can_undo()

        ws.merge_clusters(1, 0)
        assert ws.can_undo()


class TestRemoveFace:
    """Test Workspace.remove_face()."""

    def test_remove_face_from_cluster(self):
        ws = _make_workspace(num_clusters=2, photos_per_cluster=3)
        photos = ws.cluster_photos(0)
        count_before = ws.cluster_count(0)

        ws.remove_face(photos[0], 0)

        assert ws.cluster_count(0) == count_before - 1
        assert photos[0] not in ws.cluster_photos(0)

    def test_remove_last_face_deletes_cluster(self):
        ws = _make_workspace(num_clusters=1, photos_per_cluster=1)
        photos = ws.cluster_photos(0)

        ws.remove_face(photos[0], 0)

        assert 0 not in ws._cluster_mapping

    def test_remove_nonexistent_raises(self):
        ws = _make_workspace()

        with pytest.raises(ValueError, match="not in cluster"):
            ws.remove_face("/nonexistent.jpg", 0)

    def test_remove_from_nonexistent_cluster_raises(self):
        ws = _make_workspace()

        with pytest.raises(ValueError, match="Cluster not found"):
            ws.remove_face("/some/photo.jpg", 99)


class TestRenameCluster:
    """Test Workspace.rename_cluster()."""

    def test_rename_sets_name(self):
        ws = _make_workspace()
        ws.rename_cluster(0, "Alice")

        assert ws.cluster_name(0) == "Alice"

    def test_rename_nonexistent_raises(self):
        ws = _make_workspace()

        with pytest.raises(ValueError, match="Cluster not found"):
            ws.rename_cluster(99, "Bob")


class TestMoveFace:
    """Test Workspace.move_face()."""

    def test_move_between_clusters(self):
        ws = _make_workspace(num_clusters=2, photos_per_cluster=3)
        c0_count = ws.cluster_count(0)
        c1_count = ws.cluster_count(1)
        photo = ws.cluster_photos(0)[0]

        ws.move_face(photo, 0, 1)

        assert ws.cluster_count(0) == c0_count - 1
        assert ws.cluster_count(1) == c1_count + 1
        assert photo in ws.cluster_photos(1)

    def test_move_same_cluster_is_noop(self):
        ws = _make_workspace()
        ids_before = ws.cluster_ids
        photo = ws.cluster_photos(0)[0]

        ws.move_face(photo, 0, 0)

        assert ws.cluster_ids == ids_before

    def test_move_to_new_cluster(self):
        ws = _make_workspace(num_clusters=2, photos_per_cluster=3)
        photo = ws.cluster_photos(0)[0]

        ws.move_face(photo, 0, 99)

        assert 99 in ws._cluster_mapping
        assert photo in ws.cluster_photos(99)

    def test_move_empties_source_cluster(self):
        ws = _make_workspace(num_clusters=2, photos_per_cluster=1)
        photo = ws.cluster_photos(0)[0]

        ws.move_face(photo, 0, 1)

        assert 0 not in ws._cluster_mapping
        assert photo in ws.cluster_photos(1)

    def test_move_from_noise(self):
        ws = _make_workspace(num_clusters=2, photos_per_cluster=3)
        noise = ws.noise_photos
        assert len(noise) > 0

        ws.move_face(noise[0], -1, 0)

        assert noise[0] in ws.cluster_photos(0)
        assert noise[0] not in ws.noise_photos

    def test_move_non_noise_from_noise_raises(self):
        ws = _make_workspace()

        with pytest.raises(ValueError, match="not in noise"):
            ws.move_face(ws.cluster_photos(0)[0], -1, 1)

    def test_undo_move(self):
        ws = _make_workspace(num_clusters=2, photos_per_cluster=3)
        c0_before = set(ws.cluster_photos(0))
        c1_before = set(ws.cluster_photos(1))

        ws.move_face(list(c0_before)[0], 0, 1)
        ws.undo()

        assert set(ws.cluster_photos(0)) == c0_before
        assert set(ws.cluster_photos(1)) == c1_before

    def test_undo_move_from_noise(self):
        ws = _make_workspace(num_clusters=2, photos_per_cluster=3)
        noise_before = set(ws.noise_photos)
        c0_before = set(ws.cluster_photos(0))

        ws.move_face(list(noise_before)[0], -1, 0)
        ws.undo()

        assert set(ws.noise_photos) == noise_before
        assert set(ws.cluster_photos(0)) == c0_before


class TestUndo:
    """Test Workspace.undo()."""

    def test_undo_merge(self):
        ws = _make_workspace(num_clusters=3)
        c0_photos = ws.cluster_photos(0)
        c1_photos = ws.cluster_photos(1)

        ws.merge_clusters(1, 0)
        assert 1 not in ws._cluster_mapping

        result = ws.undo()
        assert result is not None
        assert result["kind"] == "merge"
        assert ws.cluster_photos(0) == c0_photos
        assert ws.cluster_photos(1) == c1_photos

    def test_undo_remove(self):
        ws = _make_workspace(num_clusters=2, photos_per_cluster=3)
        photos = ws.cluster_photos(0)
        count_before = ws.cluster_count(0)

        ws.remove_face(photos[0], 0)
        assert ws.cluster_count(0) == count_before - 1

        result = ws.undo()
        assert result is not None
        assert result["kind"] == "remove"
        assert photos[0] in ws.cluster_photos(0)
        assert ws.cluster_count(0) == count_before

    def test_undo_rename(self):
        ws = _make_workspace()
        ws.rename_cluster(0, "Alice")
        assert ws.cluster_name(0) == "Alice"

        result = ws.undo()
        assert result is not None
        assert result["kind"] == "rename"
        assert ws.cluster_name(0) == ""

    def test_undo_remove_restores_deleted_cluster(self):
        ws = _make_workspace(num_clusters=1, photos_per_cluster=1)
        photos = ws.cluster_photos(0)

        ws.remove_face(photos[0], 0)
        assert 0 not in ws._cluster_mapping

        ws.undo()
        assert 0 in ws._cluster_mapping
        assert photos[0] in ws.cluster_photos(0)

    def test_multiple_undo_lifo(self):
        ws = _make_workspace(num_clusters=3, photos_per_cluster=2)

        # Operation 1: rename cluster 0
        ws.rename_cluster(0, "Alice")
        # Operation 2: merge cluster 2 into 1
        ws.merge_clusters(2, 1)
        # Operation 3: remove a face
        photos = ws.cluster_photos(0)
        ws.remove_face(photos[0], 0)

        # Undo in reverse order
        r3 = ws.undo()
        assert r3["kind"] == "remove"

        r2 = ws.undo()
        assert r2["kind"] == "merge"

        r1 = ws.undo()
        assert r1["kind"] == "rename"

        assert not ws.can_undo()

    def test_undo_empty_returns_none(self):
        ws = _make_workspace()
        assert ws.undo() is None


class TestProperties:
    """Test Workspace property methods."""

    def test_cluster_ids_sorted(self):
        ws = _make_workspace(num_clusters=5)
        assert ws.cluster_ids == [0, 1, 2, 3, 4]

    def test_cluster_photos(self):
        ws = _make_workspace(num_clusters=2, photos_per_cluster=3)
        photos = ws.cluster_photos(0)
        assert len(photos) == 3
        assert all("/photos/cluster0_" in p for p in photos)

    def test_cluster_confidence(self):
        ws = _make_workspace()
        conf = ws.cluster_confidence(0)
        assert 0.0 <= conf <= 1.0

    def test_num_noise_faces(self):
        ws = _make_workspace()
        # num_noise_faces is now dynamically computed from noise_photos
        assert ws.num_noise_faces == len(ws.noise_photos)

    def test_next_cluster_id(self):
        ws = _make_workspace(num_clusters=3)
        assert ws.next_cluster_id() == 3

    def test_noise_photos_lists_unclustered(self):
        ws = _make_workspace(num_clusters=2, photos_per_cluster=3)
        noise = ws.noise_photos
        assert len(noise) == 3
        assert all("/photos/noise_" in p for p in noise)
