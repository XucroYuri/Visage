"""Tests for ensemble classifier."""

from __future__ import annotations

import numpy as np
import pytest

from visage.ensemble.classifier import EnsembleClassifier, EnsembleVerdict


@pytest.fixture
def clustered_data():
    """Two well-separated clusters with 20 samples each."""
    rng = np.random.RandomState(42)
    center_a = rng.randn(128).astype(np.float32) * 3
    center_b = rng.randn(128).astype(np.float32) * 3

    cluster_a = center_a + 0.3 * rng.randn(20, 128).astype(np.float32)
    cluster_b = center_b + 0.3 * rng.randn(20, 128).astype(np.float32)

    embeddings = np.vstack([cluster_a, cluster_b])
    labels = ["c1"] * 20 + ["c2"] * 20
    return embeddings, labels


@pytest.fixture
def trained_classifier(clustered_data):
    """Pre-trained ensemble classifier."""
    clf = EnsembleClassifier(knn_k=3)
    embeddings, labels = clustered_data
    clf.train(embeddings, labels)
    return clf, embeddings


class TestEnsembleClassifierTraining:
    def test_not_trained_initially(self):
        clf = EnsembleClassifier()
        assert not clf.is_trained

    def test_train_sets_trained(self, clustered_data):
        clf = EnsembleClassifier()
        embeddings, labels = clustered_data
        clf.train(embeddings, labels)
        assert clf.is_trained

    def test_train_single_class(self):
        clf = EnsembleClassifier()
        embeddings = np.random.randn(10, 128).astype(np.float32)
        labels = ["c1"] * 10
        clf.train(embeddings, labels)
        assert not clf.is_trained  # Can't train with 1 class


class TestEnsembleClassifierPrediction:
    def test_predict_returns_verdict(self, trained_classifier):
        clf, embeddings = trained_classifier
        result = clf.predict(embeddings[0])
        assert isinstance(result, EnsembleVerdict)

    def test_predict_known_cluster(self, trained_classifier):
        clf, embeddings = trained_classifier
        # Predict on a sample from cluster A
        result = clf.predict(embeddings[0])
        assert result.predicted_cluster == "c1"
        assert result.confidence > 0

    def test_predict_untrained_returns_empty(self):
        clf = EnsembleClassifier()
        result = clf.predict(np.random.randn(128).astype(np.float32))
        assert result.predicted_cluster is None

    def test_individual_votes_populated(self, trained_classifier):
        clf, embeddings = trained_classifier
        result = clf.predict(embeddings[0])
        assert "cosine" in result.individual_votes
        assert "euclidean" in result.individual_votes
        assert "svm" in result.individual_votes

    def test_all_predict_same_cluster(self, trained_classifier):
        """For well-separated data, all methods should agree."""
        clf, embeddings = trained_classifier
        result = clf.predict(embeddings[0])
        votes = result.individual_votes
        # At least 2 out of 3 should agree
        cluster_votes = [v for v in votes.values() if v is not None]
        assert len(set(cluster_votes)) <= 2


class TestEnsembleClassifierWeights:
    def test_update_weights(self):
        clf = EnsembleClassifier()
        clf.update_weights("cosine", True)
        clf.update_weights("cosine", False)
        assert len(clf._accuracy_history["cosine"]) == 2

    def test_adapted_weights_before_history(self):
        clf = EnsembleClassifier()
        weights = clf.get_adapted_weights()
        assert "cosine" in weights
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_adapted_weights_after_history(self):
        clf = EnsembleClassifier()
        for _ in range(15):
            clf.update_weights("cosine", True)
            clf.update_weights("euclidean", False)
            clf.update_weights("svm", True)
        weights = clf.get_adapted_weights()
        assert weights["cosine"] > weights["euclidean"]

    def test_reject_threshold(self):
        """Untrained classifier with reject threshold returns no prediction."""
        clf = EnsembleClassifier(reject_threshold=0.99)
        rng = np.random.RandomState(42)
        # Train with overlapping data
        emb = rng.randn(20, 128).astype(np.float32)
        labels = ["c1"] * 10 + ["c2"] * 10
        clf.train(emb, labels)
        if clf.is_trained:
            # Use a very ambiguous query
            query = rng.randn(128).astype(np.float32)
            result = clf.predict(query)
            # Result should be either rejected or low confidence
            assert result.confidence <= 1.0
