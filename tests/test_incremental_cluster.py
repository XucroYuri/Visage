"""Tests for incremental clustering engine."""

from __future__ import annotations

import numpy as np
import pytest

from visage.cluster.engine import ClusterEngine
from visage.cluster.incremental import AssignmentResult, IncrementalAssigner
from visage.cluster.optimizer import GlobalOptimizer, OptimizationResult

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def cluster_a_center():
    """Center of cluster A, 128-dim."""
    rng = np.random.RandomState(42)
    c = rng.randn(128).astype(np.float32)
    return c / np.linalg.norm(c)


@pytest.fixture
def cluster_b_center():
    """Center of cluster B, far from A, 128-dim."""
    rng = np.random.RandomState(99)
    c = rng.randn(128).astype(np.float32)
    return c / np.linalg.norm(c)


@pytest.fixture
def mock_search_fn(cluster_a_center):
    """Mock search function that returns neighbors near cluster_a_center."""
    # Store known faces with their cluster assignments
    known = {
        "known_0": ("cluster_a", cluster_a_center + 0.05 * np.random.randn(128).astype(np.float32)),
        "known_1": ("cluster_a", cluster_a_center + 0.05 * np.random.randn(128).astype(np.float32)),
        "known_2": ("cluster_a", cluster_a_center + 0.05 * np.random.randn(128).astype(np.float32)),
        "known_3": ("cluster_b", np.random.randn(128).astype(np.float32)),
    }
    # Normalize
    for fid in known:
        _, vec = known[fid]
        known[fid] = (known[fid][0], vec / np.linalg.norm(vec))

    def search(query, top_k=3):
        results = []
        q = query / np.linalg.norm(query)
        for fid, (_cid, vec) in known.items():
            score = float(np.dot(q, vec))
            results.append((fid, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def lookup(fid):
        return known.get(fid, (None, None))[0]

    return search, lookup, known


# ── IncrementalAssigner ───────────────────────────────────────────


class TestIncrementalAssigner:
    def test_high_confidence_assignment(self, cluster_a_center, mock_search_fn):
        search_fn, lookup_fn, _ = mock_search_fn
        assigner = IncrementalAssigner(high_threshold=0.5)

        # Query very close to cluster_a center
        query = cluster_a_center + 0.01 * np.random.randn(128).astype(np.float32)
        result = assigner.assign("new_face", query, search_fn, lookup_fn)

        assert result.face_id == "new_face"
        assert result.cluster_id == "cluster_a"
        assert result.method == "high"
        assert result.confidence > 0

    def test_medium_confidence_assignment(self, mock_search_fn):
        search_fn, lookup_fn, _ = mock_search_fn
        assigner = IncrementalAssigner(high_threshold=0.01, medium_threshold=0.9)

        # Query a bit further away — won't hit high threshold
        query = np.random.randn(128).astype(np.float32)
        query = query / np.linalg.norm(query)
        result = assigner.assign("new_face", query, search_fn, lookup_fn)

        # Should get some assignment (medium, pending, or new_cluster)
        assert result.face_id == "new_face"
        assert result.method in ("medium", "pending", "new_cluster")

    def test_no_neighbors_creates_new_cluster(self):
        assigner = IncrementalAssigner()

        def empty_search(query, top_k):
            return []

        def empty_lookup(fid):
            return None

        result = assigner.assign(
            "lonely", np.random.randn(128).astype(np.float32),
            empty_search, empty_lookup,
        )
        assert result.method == "new_cluster"
        assert result.cluster_id is not None
        assert result.cluster_id.startswith("inc_")

    def test_new_cluster_ids_increment(self):
        assigner = IncrementalAssigner()

        def empty_search(query, top_k):
            return []

        def empty_lookup(fid):
            return None

        r1 = assigner.assign(
            "f1", np.random.randn(128).astype(np.float32),
            empty_search, empty_lookup,
        )
        r2 = assigner.assign(
            "f2", np.random.randn(128).astype(np.float32),
            empty_search, empty_lookup,
        )
        assert r1.cluster_id != r2.cluster_id

    def test_assignment_result_fields(self):
        result = AssignmentResult(
            face_id="test",
            cluster_id="c1",
            confidence=0.95,
            method="high",
            neighbor_ids=["n1", "n2"],
            neighbor_scores=[0.9, 0.8],
        )
        assert result.face_id == "test"
        assert len(result.neighbor_ids) == 2


# ── GlobalOptimizer ───────────────────────────────────────────────


class TestGlobalOptimizer:
    def test_compute_drift_no_change(self):
        optimizer = GlobalOptimizer()
        current = {"f1": "c1", "f2": "c1", "f3": "c2"}
        new = {"f1": "c1", "f2": "c1", "f3": "c2"}
        assert optimizer.compute_drift(current, new) == 0.0

    def test_compute_drift_all_changed(self):
        optimizer = GlobalOptimizer()
        current = {"f1": "c1", "f2": "c1"}
        new = {"f1": "c2", "f2": "c2"}
        assert optimizer.compute_drift(current, new) == 1.0

    def test_compute_drift_partial(self):
        optimizer = GlobalOptimizer()
        current = {"f1": "c1", "f2": "c1", "f3": "c2", "f4": "c2"}
        new = {"f1": "c1", "f2": "c3", "f3": "c2", "f4": "c4"}
        assert optimizer.compute_drift(current, new) == 0.5

    def test_compute_drift_empty(self):
        optimizer = GlobalOptimizer()
        assert optimizer.compute_drift({}, {}) == 0.0

    def test_should_optimize_force(self):
        optimizer = GlobalOptimizer()
        assert optimizer.should_optimize(10, 0, force=True)

    def test_should_optimize_too_few_faces(self):
        optimizer = GlobalOptimizer(min_faces_for_optimization=100)
        assert not optimizer.should_optimize(50, 0)

    def test_should_optimize_enough_new_faces(self):
        optimizer = GlobalOptimizer(min_faces_for_optimization=10)
        assert optimizer.should_optimize(100, 5000)

    def test_should_not_optimize_few_new_faces(self):
        optimizer = GlobalOptimizer(min_faces_for_optimization=10)
        assert not optimizer.should_optimize(100, 100)

    def test_optimize_returns_result(self):
        optimizer = GlobalOptimizer()
        rng = np.random.RandomState(42)
        embeddings = rng.randn(20, 128).astype(np.float32)
        face_ids = [f"f{i}" for i in range(20)]
        current = {f"f{i}": "c1" for i in range(20)}

        def mock_cluster_fn(embs):
            # All in one cluster
            return np.zeros(len(embs), dtype=int)

        result = optimizer.optimize(embeddings, face_ids, current, mock_cluster_fn)
        assert isinstance(result, OptimizationResult)
        assert result.faces_optimized == 20
        assert result.elapsed_seconds >= 0


# ── ClusterEngine ─────────────────────────────────────────────────


class TestClusterEngine:
    def test_epoch_increments(self, mock_search_fn):
        search_fn, lookup_fn, _ = mock_search_fn
        engine = ClusterEngine()

        assert engine.epoch == 0
        engine.assign_face("f1", np.random.randn(128).astype(np.float32), search_fn, lookup_fn)
        assert engine.epoch == 1

    def test_batch_assignment(self, mock_search_fn):
        search_fn, lookup_fn, _ = mock_search_fn
        engine = ClusterEngine()

        face_ids = ["f1", "f2", "f3"]
        embeddings = np.random.randn(3, 128).astype(np.float32)
        results = engine.assign_batch(face_ids, embeddings, search_fn, lookup_fn)

        assert len(results) == 3
        assert engine.epoch == 3

    def test_should_optimize_delegates(self):
        engine = ClusterEngine()
        assert engine.should_optimize(10, force=True)

    def test_run_optimization(self):
        engine = ClusterEngine()
        rng = np.random.RandomState(42)
        embeddings = rng.randn(20, 128).astype(np.float32)
        face_ids = [f"f{i}" for i in range(20)]
        current = {f"f{i}": "c1" for i in range(20)}

        def mock_cluster_fn(embs):
            return np.zeros(len(embs), dtype=int)

        result = engine.run_optimization(embeddings, face_ids, current, mock_cluster_fn)
        assert result.faces_optimized == 20
