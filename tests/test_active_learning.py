"""Tests for active learning — prototypes, nearest centroid, corrections, threshold."""

from __future__ import annotations

import numpy as np

from visage.active.correction_store import CorrectionStore
from visage.active.nearest_centroid import NearestCentroidClassifier
from visage.active.prototype import PrototypeManager
from visage.active.threshold_adapter import ThresholdAdapter

# ── Prototype Manager ─────────────────────────────────────────────


class TestPrototypeManager:
    """Test prototype vector management."""

    def _make_embeddings(self, n_per_cluster: int = 10, n_clusters: int = 3):
        """Create well-separated cluster embeddings."""
        np.random.seed(42)
        embeddings = []
        labels = []
        for cid in range(n_clusters):
            centroid = np.random.randn(128).astype(np.float64)
            centroid += cid * 3.0  # Spread apart
            for _ in range(n_per_cluster):
                emb = centroid + 0.1 * np.random.randn(128).astype(np.float64)
                embeddings.append(emb)
                labels.append(cid)
        return np.array(embeddings), np.array(labels)

    def test_build_from_embeddings(self):
        embs, labels = self._make_embeddings(n_clusters=3)
        mgr = PrototypeManager()
        mgr.build_from_embeddings(embs, labels)
        assert len(mgr.cluster_ids) == 3
        assert all(mgr.get_centroid(cid) is not None for cid in mgr.cluster_ids)

    def test_noise_labels_ignored(self):
        embs, labels = self._make_embeddings(n_clusters=2)
        labels[0] = -1  # Add noise
        labels[1] = -1
        mgr = PrototypeManager()
        mgr.build_from_embeddings(embs, labels)
        assert len(mgr.cluster_ids) == 2

    def test_centroids_are_normalized(self):
        embs, labels = self._make_embeddings(n_clusters=2)
        mgr = PrototypeManager()
        mgr.build_from_embeddings(embs, labels)
        for cid in mgr.cluster_ids:
            centroid = mgr.get_centroid(cid)
            norm = np.linalg.norm(centroid)
            assert abs(norm - 1.0) < 1e-6

    def test_update_on_addition(self):
        embs, labels = self._make_embeddings(n_clusters=1)
        mgr = PrototypeManager()
        mgr.build_from_embeddings(embs, labels)

        proto = mgr.get_prototype(0)
        initial_count = proto.member_count

        new_emb = np.random.randn(128).astype(np.float64)
        mgr.update_on_correction(0, new_emb, weight=1.0, is_addition=True)

        proto = mgr.get_prototype(0)
        assert proto.member_count == initial_count + 1

    def test_update_on_removal(self):
        embs, labels = self._make_embeddings(n_clusters=1, n_per_cluster=5)
        mgr = PrototypeManager()
        mgr.build_from_embeddings(embs, labels)

        proto = mgr.get_prototype(0)
        initial_count = proto.member_count
        assert initial_count == 5

        mgr.update_on_correction(0, embs[0], weight=1.0, is_addition=False)
        proto = mgr.get_prototype(0)
        assert proto.member_count == 4

    def test_removal_last_member_deletes_prototype(self):
        embs = np.array([[1.0, 0.0], [0.0, 1.0]])
        labels = np.array([0, 0])
        mgr = PrototypeManager(embedding_dim=2)
        mgr.build_from_embeddings(embs, labels)
        assert 0 in mgr.cluster_ids

        mgr.update_on_correction(0, embs[0], is_addition=False)
        mgr.update_on_correction(0, embs[1], is_addition=False)
        assert 0 not in mgr.cluster_ids

    def test_new_cluster_creation(self):
        mgr = PrototypeManager()
        emb = np.array([1.0, 0.0, 0.0])
        mgr.update_on_correction(99, emb, weight=1.0, is_addition=True)
        assert 99 in mgr.cluster_ids
        assert mgr.get_prototype(99).member_count == 1

    def test_serialization_roundtrip(self):
        embs, labels = self._make_embeddings(n_clusters=2)
        mgr = PrototypeManager()
        mgr.build_from_embeddings(embs, labels)

        data = mgr.to_dict()
        restored = PrototypeManager.from_dict(data)

        assert set(restored.cluster_ids) == set(mgr.cluster_ids)
        for cid in mgr.cluster_ids:
            orig = mgr.get_centroid(cid)
            rest = restored.get_centroid(cid)
            np.testing.assert_allclose(orig, rest, atol=1e-6)

    def test_get_nonexistent_returns_none(self):
        mgr = PrototypeManager()
        assert mgr.get_prototype(999) is None
        assert mgr.get_centroid(999) is None


# ── Nearest Centroid Classifier ───────────────────────────────────


class TestNearestCentroidClassifier:
    """Test nearest-centroid classification."""

    def _make_prototypes(self):
        np.random.seed(42)
        embs, labels = [], []
        for cid in range(3):
            centroid = np.random.randn(64).astype(np.float64)
            centroid += cid * 5.0
            for _ in range(5):
                embs.append(centroid + 0.1 * np.random.randn(64))
                labels.append(cid)
        embs = np.array(embs)
        labels = np.array(labels)
        mgr = PrototypeManager(embedding_dim=64)
        mgr.build_from_embeddings(embs, labels)
        return mgr, embs, labels

    def test_classify_correct_cluster(self):
        mgr, embs, labels = self._make_prototypes()
        clf = NearestCentroidClassifier(mgr)

        # Test with a sample from each cluster
        for cid in range(3):
            mask = labels == cid
            sample = embs[mask][0]
            result = clf.classify(sample)
            assert result is not None
            assert result.predicted_cluster == cid
            assert result.confidence > 0.5

    def test_classify_returns_distances(self):
        mgr, _, _ = self._make_prototypes()
        clf = NearestCentroidClassifier(mgr)
        query = np.random.randn(64)
        result = clf.classify(query)
        assert len(result.distances) == 3

    def test_classify_exclude_clusters(self):
        mgr, _, _ = self._make_prototypes()
        clf = NearestCentroidClassifier(mgr)
        query = np.random.randn(64)
        result = clf.classify(query, exclude_clusters={0})
        assert 0 not in result.distances

    def test_classify_batch(self):
        mgr, embs, _ = self._make_prototypes()
        clf = NearestCentroidClassifier(mgr)
        results = clf.classify_batch(embs[:5])
        assert len(results) == 5
        assert all(r is not None for r in results)

    def test_classify_empty_prototypes_returns_none(self):
        mgr = PrototypeManager()
        clf = NearestCentroidClassifier(mgr)
        result = clf.classify(np.random.randn(64))
        assert result is None

    def test_classify_zero_embedding_returns_none(self):
        mgr, _, _ = self._make_prototypes()
        clf = NearestCentroidClassifier(mgr)
        result = clf.classify(np.zeros(64))
        assert result is None

    def test_find_misclassified(self):
        mgr, embs, labels = self._make_prototypes()
        clf = NearestCentroidClassifier(mgr)
        misclassified = clf.find_misclassified(embs, labels, threshold=0.5)
        # With well-separated clusters, should have few misclassifications
        assert isinstance(misclassified, list)

    def test_find_misclassified_ignores_noise(self):
        mgr, embs, labels = self._make_prototypes()
        labels[0] = -1  # noise
        clf = NearestCentroidClassifier(mgr)
        misclassified = clf.find_misclassified(embs, labels)
        # First entry should not appear since it's noise
        indices = [m[0] for m in misclassified]
        assert 0 not in indices


# ── Correction Store ──────────────────────────────────────────────


class TestCorrectionStore:
    """Test user correction persistence."""

    def test_record_and_retrieve(self, tmp_path):
        store = CorrectionStore(str(tmp_path))
        cid = store.record_correction(
            "merge",
            ["face_1", "face_2"],
            source_cluster=0,
            target_cluster=1,
        )
        assert cid > 0

        corrections = store.get_corrections()
        assert len(corrections) == 1
        assert corrections[0]["action"] == "merge"
        assert corrections[0]["face_ids"] == ["face_1", "face_2"]
        store.close()

    def test_filter_by_action(self, tmp_path):
        store = CorrectionStore(str(tmp_path))
        store.record_correction("merge", ["f1"], source_cluster=0, target_cluster=1)
        store.record_correction("split", ["f2"], source_cluster=1, target_cluster=2)
        store.record_correction("merge", ["f3"], source_cluster=2, target_cluster=0)

        merges = store.get_corrections(action="merge")
        assert len(merges) == 2
        splits = store.get_corrections(action="split")
        assert len(splits) == 1
        store.close()

    def test_correction_count(self, tmp_path):
        store = CorrectionStore(str(tmp_path))
        assert store.get_correction_count() == 0
        store.record_correction("merge", ["f1"])
        store.record_correction("split", ["f2"])
        assert store.get_correction_count() == 2
        store.close()

    def test_correction_stats(self, tmp_path):
        store = CorrectionStore(str(tmp_path))
        store.record_correction("merge", ["f1"])
        store.record_correction("merge", ["f2"])
        store.record_correction("split", ["f3"])

        stats = store.get_correction_stats()
        assert stats["merge"] == 2
        assert stats["split"] == 1
        store.close()

    def test_with_details(self, tmp_path):
        store = CorrectionStore(str(tmp_path))
        store.record_correction(
            "reassign",
            ["f1"],
            source_cluster=0,
            target_cluster=1,
            details={"reason": "wrong cluster", "confidence": 0.85},
        )
        corrections = store.get_corrections()
        assert corrections[0]["details"]["reason"] == "wrong cluster"
        store.close()

    def test_limit_respected(self, tmp_path):
        store = CorrectionStore(str(tmp_path))
        for i in range(10):
            store.record_correction("merge", [f"f{i}"])
        results = store.get_corrections(limit=3)
        assert len(results) == 3
        store.close()


# ── Threshold Adapter ─────────────────────────────────────────────


class TestThresholdAdapter:
    """Test adaptive threshold adjustment."""

    def test_initial_threshold(self):
        adapter = ThresholdAdapter()
        assert adapter.threshold == 0.70

    def test_custom_initial_threshold(self):
        adapter = ThresholdAdapter(initial_threshold=0.80)
        assert adapter.threshold == 0.80

    def test_merge_lowers_threshold(self):
        adapter = ThresholdAdapter(initial_threshold=0.70)
        new_t = adapter.record_merge()
        assert new_t < 0.70
        assert new_t >= 0.40  # Min bound

    def test_split_raises_threshold(self):
        adapter = ThresholdAdapter(initial_threshold=0.70)
        new_t = adapter.record_split()
        assert new_t > 0.70
        assert new_t <= 0.95  # Max bound

    def test_threshold_bounded_min(self):
        adapter = ThresholdAdapter(initial_threshold=0.41, adaptation_rate=0.05)
        adapter.record_merge()
        assert adapter.threshold == 0.40

    def test_threshold_bounded_max(self):
        adapter = ThresholdAdapter(initial_threshold=0.94, adaptation_rate=0.05)
        adapter.record_split()
        assert adapter.threshold == 0.95

    def test_reassign_to_existing_like_merge(self):
        adapter = ThresholdAdapter(initial_threshold=0.70)
        adapter.record_reassign(0, 1)
        assert adapter.threshold < 0.70

    def test_reassign_to_noise_like_split(self):
        adapter = ThresholdAdapter(initial_threshold=0.70)
        adapter.record_reassign(0, -1)
        assert adapter.threshold > 0.70

    def test_stats(self):
        adapter = ThresholdAdapter()
        adapter.record_merge()
        adapter.record_merge()
        adapter.record_split()
        stats = adapter.stats
        assert stats["merge_count"] == 2
        assert stats["split_count"] == 1
        assert stats["total_corrections"] == 3

    def test_serialization_roundtrip(self):
        adapter = ThresholdAdapter(initial_threshold=0.65)
        adapter.record_merge()
        adapter.record_split()

        data = adapter.to_dict()
        restored = ThresholdAdapter.from_dict(data)

        assert restored.threshold == adapter.threshold
        assert restored.stats["merge_count"] == 1
        assert restored.stats["split_count"] == 1

    def test_reset(self):
        adapter = ThresholdAdapter(initial_threshold=0.80)
        adapter.record_merge()
        adapter.record_split()
        adapter.reset()
        assert adapter.threshold == 0.70
        assert adapter.stats["total_corrections"] == 0

    def test_many_merges_converge(self):
        adapter = ThresholdAdapter(initial_threshold=0.70, adaptation_rate=0.01)
        for _ in range(100):
            adapter.record_merge()
        assert adapter.threshold == 0.40  # Hit min

    def test_many_splits_converge(self):
        adapter = ThresholdAdapter(initial_threshold=0.70, adaptation_rate=0.01)
        for _ in range(100):
            adapter.record_split()
        assert adapter.threshold == 0.95  # Hit max
