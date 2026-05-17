"""Tests for visage.cluster — clustering algorithms, no mocking needed."""

from __future__ import annotations

import numpy as np

from visage.cluster import (
    _normalize_embeddings,
    build_cluster_mapping,
    cluster_faces,
    compute_cluster_confidences,
    estimate_eps,
    extract_embeddings,
)
from visage.models import ClusterResult, DetectedFace, FaceBox, ImageResult

# ── _normalize_embeddings ─────────────────────────────────────────


class TestNormalizeEmbeddings:
    def test_unit_vectors(self):
        emb = np.random.randn(10, 128).astype(np.float64)
        normalized = _normalize_embeddings(emb)
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_preserves_shape(self):
        emb = np.random.randn(5, 128)
        normalized = _normalize_embeddings(emb)
        assert normalized.shape == (5, 128)

    def test_zero_vector_no_crash(self):
        emb = np.zeros((3, 128))
        emb[0] = np.random.randn(128)  # one non-zero row
        normalized = _normalize_embeddings(emb)
        assert not np.isnan(normalized).any()
        # Zero vector should become near-zero after normalization
        assert np.linalg.norm(normalized[1]) < 1e-8

    def test_single_vector(self):
        # Pad to 128 dim for consistency — but function works on any dim
        emb_padded = np.random.randn(1, 128)
        normalized = _normalize_embeddings(emb_padded)
        assert abs(np.linalg.norm(normalized[0]) - 1.0) < 1e-10


# ── extract_embeddings ────────────────────────────────────────────


class TestExtractEmbeddings:
    def test_from_single_result(self, sample_image_result: ImageResult):
        embeddings, mapping = extract_embeddings([sample_image_result])
        assert embeddings.shape == (1, 128)
        assert len(mapping) == 1
        assert mapping[0] == ("/tmp/test.jpg", 0)

    def test_from_multi_face_result(self, multi_face_image_result: ImageResult):
        embeddings, mapping = extract_embeddings([multi_face_image_result])
        assert embeddings.shape == (2, 128)
        assert mapping[0] == ("/tmp/multi.jpg", 0)
        assert mapping[1] == ("/tmp/multi.jpg", 1)

    def test_skips_error_results(self, error_image_result: ImageResult):
        embeddings, mapping = extract_embeddings([error_image_result])
        assert embeddings.shape == (0, 128)
        assert mapping == []

    def test_skips_none_embedding(self, sample_face_box: FaceBox):
        face_no_emb = DetectedFace(
            face_box=sample_face_box, confidence=0.9, embedding=None,
        )
        result = ImageResult(path="/tmp/test.jpg", faces=[face_no_emb])
        embeddings, mapping = extract_embeddings([result])
        assert embeddings.shape == (0, 128)
        assert mapping == []

    def test_mixed_results(
        self, sample_image_result: ImageResult, error_image_result: ImageResult
    ):
        embeddings, mapping = extract_embeddings([sample_image_result, error_image_result])
        assert embeddings.shape == (1, 128)
        assert len(mapping) == 1

    def test_empty_input(self):
        embeddings, mapping = extract_embeddings([])
        assert embeddings.shape == (0, 128)
        assert mapping == []

    def test_custom_embedding_dim(self):
        embeddings, mapping = extract_embeddings([], embedding_dim=512)
        assert embeddings.shape == (0, 512)
        assert mapping == []

    def test_custom_embedding_dim_with_results(self):
        face = DetectedFace(
            face_box=FaceBox(top=10, right=110, bottom=110, left=10),
            confidence=0.9,
            embedding=np.random.randn(512).astype(np.float64),
            image_path="/tmp/test.jpg",
            face_index=0,
        )
        result = ImageResult(path="/tmp/test.jpg", faces=[face])
        embeddings, mapping = extract_embeddings([result], embedding_dim=512)
        assert embeddings.shape == (1, 512)
        assert len(mapping) == 1


# ── estimate_eps ──────────────────────────────────────────────────


class TestEstimateEps:
    def test_returns_float(self, clusterable_embeddings):
        embeddings, _ = clusterable_embeddings
        eps = estimate_eps(embeddings)
        assert isinstance(eps, float)
        assert eps > 0.0

    def test_small_dataset_returns_default(self):
        embeddings = np.random.randn(3, 128)
        eps = estimate_eps(embeddings, k=5)
        assert eps == 0.5

    def test_with_normalized(self, clusterable_embeddings):
        embeddings, _ = clusterable_embeddings
        normalized = _normalize_embeddings(embeddings)
        eps = estimate_eps(normalized)
        assert isinstance(eps, float)
        assert 0.0 < eps < 2.0


# ── cluster_faces ─────────────────────────────────────────────────


class TestClusterFaces:
    def test_two_clusters(self, clusterable_embeddings):
        embeddings, _ = clusterable_embeddings
        result = cluster_faces(embeddings, eps=1.0, min_samples=2)
        assert result.num_clusters == 2
        assert result.num_noise == 0
        assert len(result.labels) == 20

    def test_empty_input(self):
        result = cluster_faces(np.empty((0, 128)))
        assert result.num_clusters == 0
        assert result.num_noise == 0
        assert len(result.labels) == 0

    def test_all_noise(self):
        # Spread-out embeddings with small eps
        np.random.seed(99)
        embeddings = np.random.randn(10, 128) * 5.0
        result = cluster_faces(embeddings, eps=0.1, min_samples=2)
        assert result.num_clusters == 0
        assert result.num_noise == 10

    def test_auto_eps(self, clusterable_embeddings):
        embeddings, _ = clusterable_embeddings
        result = cluster_faces(embeddings, auto_eps=True, min_samples=2)
        assert result.num_clusters >= 1
        assert len(result.labels) == 20

    def test_single_face(self):
        emb = np.random.randn(1, 128)
        result = cluster_faces(emb, eps=0.5, min_samples=2)
        # Single face is always noise with min_samples=2
        assert result.num_clusters == 0
        assert result.num_noise == 1

    def test_labels_match_embeddings_count(self, clusterable_embeddings):
        embeddings, _ = clusterable_embeddings
        result = cluster_faces(embeddings, eps=0.5, min_samples=2)
        assert len(result.labels) == len(embeddings)

    def test_result_contains_normalized_embeddings(self, clusterable_embeddings):
        embeddings, _ = clusterable_embeddings
        result = cluster_faces(embeddings, eps=0.5, min_samples=2)
        # Result embeddings should be L2-normalized
        norms = np.linalg.norm(result.embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)


# ── cluster_faces with HDBSCAN ─────────────────────────────────────


class TestClusterFacesHDBSCAN:
    def test_hdbscan_two_clusters(self, clusterable_embeddings):
        embeddings, _ = clusterable_embeddings
        result = cluster_faces(embeddings, min_samples=2, cluster_method="hdbscan")
        assert result.num_clusters >= 1
        assert len(result.labels) == 20

    def test_hdbscan_empty_input(self):
        result = cluster_faces(np.empty((0, 128)), cluster_method="hdbscan")
        assert result.num_clusters == 0
        assert result.num_noise == 0
        assert len(result.labels) == 0

    def test_hdbscan_has_probabilities(self, clusterable_embeddings):
        embeddings, _ = clusterable_embeddings
        result = cluster_faces(embeddings, min_samples=2, cluster_method="hdbscan")
        if result.num_clusters > 0:
            assert result.probabilities is not None
            assert len(result.probabilities) == len(embeddings)
            # Non-noise points should have probability > 0
            non_noise = result.labels != -1
            if non_noise.any():
                assert all(result.probabilities[non_noise] > 0)

    def test_hdbscan_default_method_is_dbscan(self, clusterable_embeddings):
        embeddings, _ = clusterable_embeddings
        result = cluster_faces(embeddings, eps=1.0, min_samples=2)
        assert result.probabilities is None  # DBSCAN has no probabilities


# ── build_cluster_mapping ─────────────────────────────────────────


class TestBuildClusterMapping:
    def test_basic_mapping(self, clusterable_embeddings):
        embeddings, face_to_image = clusterable_embeddings
        result = cluster_faces(embeddings, eps=1.0, min_samples=2)
        mapping = build_cluster_mapping(result, face_to_image)
        assert len(mapping) == result.num_clusters

    def test_noise_excluded(self):
        labels = np.array([0, 0, -1, 1, 1])
        embeddings = np.random.randn(5, 128)
        result = ClusterResult(labels=labels, embeddings=embeddings, num_clusters=2, num_noise=1)
        face_to_image = [
            ("/a.jpg", 0), ("/a.jpg", 0), ("/noise.jpg", 0),
            ("/b.jpg", 0), ("/b.jpg", 0),
        ]
        mapping = build_cluster_mapping(result, face_to_image)
        # Noise excluded; unique paths per cluster
        assert set(mapping[0]) == {"/a.jpg"}
        assert set(mapping[1]) == {"/b.jpg"}
        assert "/noise.jpg" not in {p for ps in mapping.values() for p in ps}

    def test_image_in_multiple_clusters(self):
        # Same image appears in cluster 0 and 1 (multi-face photo)
        labels = np.array([0, 1])
        embeddings = np.random.randn(2, 128)
        result = ClusterResult(labels=labels, embeddings=embeddings, num_clusters=2, num_noise=0)
        face_to_image = [("/same.jpg", 0), ("/same.jpg", 1)]
        mapping = build_cluster_mapping(result, face_to_image)
        assert "/same.jpg" in mapping[0]
        assert "/same.jpg" in mapping[1]

    def test_sorted_output(self):
        labels = np.array([2, 1, 0])
        embeddings = np.random.randn(3, 128)
        result = ClusterResult(labels=labels, embeddings=embeddings, num_clusters=3, num_noise=0)
        face_to_image = [("/c.jpg", 0), ("/b.jpg", 0), ("/a.jpg", 0)]
        mapping = build_cluster_mapping(result, face_to_image)
        # Keys sorted
        assert list(mapping.keys()) == [0, 1, 2]
        # Values sorted
        for paths in mapping.values():
            assert paths == sorted(paths)

    def test_out_of_range_labels(self):
        labels = np.array([0, 0])
        embeddings = np.random.randn(2, 128)
        result = ClusterResult(labels=labels, embeddings=embeddings, num_clusters=1, num_noise=0)
        # face_to_image shorter than labels
        face_to_image = [("/a.jpg", 0)]
        mapping = build_cluster_mapping(result, face_to_image)
        assert len(mapping[0]) == 1


# ── compute_cluster_confidences ───────────────────────────────────


class TestComputeClusterConfidences:
    def test_tight_cluster_high_confidence(self):
        np.random.seed(42)
        centroid = np.random.randn(128).astype(np.float64)
        centroid /= np.linalg.norm(centroid)
        faces = np.array([centroid + 0.01 * np.random.randn(128) for _ in range(5)])
        faces = _normalize_embeddings(faces)
        labels = np.array([0, 0, 0, 0, 0])
        result = ClusterResult(labels=labels, embeddings=faces, num_clusters=1, num_noise=0)
        confidences = compute_cluster_confidences(result)
        assert len(confidences) == 1
        assert confidences[0] > 0.9

    def test_multiple_clusters(self, clusterable_embeddings):
        embeddings, _ = clusterable_embeddings
        result = cluster_faces(embeddings, eps=1.0, min_samples=2)
        confidences = compute_cluster_confidences(result)
        assert len(confidences) == result.num_clusters
        for _cid, conf in confidences.items():
            assert 0.0 <= conf <= 1.0

    def test_all_noise(self):
        labels = np.array([-1, -1, -1])
        embeddings = np.random.randn(3, 128)
        result = ClusterResult(labels=labels, embeddings=embeddings, num_clusters=0, num_noise=3)
        confidences = compute_cluster_confidences(result)
        assert confidences == {}

    def test_confidence_range(self, clusterable_embeddings):
        embeddings, _ = clusterable_embeddings
        result = cluster_faces(embeddings, eps=1.0, min_samples=2)
        confidences = compute_cluster_confidences(result)
        for conf in confidences.values():
            assert 0.0 <= conf <= 1.0
