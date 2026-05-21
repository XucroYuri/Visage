"""Ensemble classifier combining KNN (cosine + euclidean) and SVM with weighted voting."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


@dataclass
class EnsembleVerdict:
    """Result of ensemble classification."""

    predicted_cluster: str | None = None
    confidence: float = 0.0
    method: str = "ensemble"
    individual_votes: dict[str, str | None] = field(default_factory=dict)
    individual_confidences: dict[str, float] = field(default_factory=dict)


class EnsembleClassifier:
    """Three-classifier ensemble: Cosine KNN + Euclidean KNN + SVM.

    Uses weighted voting where weights adapt based on recent accuracy.
    Only activated for low-confidence predictions from the incremental assigner.
    """

    def __init__(
        self,
        cosine_weight: float = 0.5,
        euclidean_weight: float = 0.2,
        svm_weight: float = 0.3,
        reject_threshold: float = 0.4,
        knn_k: int = 5,
    ) -> None:
        self.cosine_weight = cosine_weight
        self.euclidean_weight = euclidean_weight
        self.svm_weight = svm_weight
        self.reject_threshold = reject_threshold
        self.knn_k = knn_k

        self._cosine_knn = KNeighborsClassifier(
            n_neighbors=knn_k, metric="cosine"
        )
        self._euclidean_knn = KNeighborsClassifier(
            n_neighbors=knn_k, metric="euclidean"
        )
        self._svm = SVC(kernel="rbf", probability=True)
        self._trained = False
        self._accuracy_history: dict[str, list[bool]] = {
            "cosine": [],
            "euclidean": [],
            "svm": [],
        }

    @property
    def is_trained(self) -> bool:
        return self._trained

    def train(
        self,
        embeddings: np.ndarray,
        labels: list[str],
    ) -> None:
        """Train all three classifiers on labeled embeddings.

        Args:
            embeddings: (N, D) matrix of face embeddings.
            labels: Cluster ID for each embedding.
        """
        if len(set(labels)) < 2:
            logger.warning("Need at least 2 classes for ensemble training")
            return

        X = embeddings.astype(np.float32)
        # L2-normalize for cosine KNN
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        X_norm = X / norms

        self._cosine_knn.fit(X_norm, labels)
        self._euclidean_knn.fit(X, labels)
        self._svm.fit(X, labels)
        self._trained = True

    def predict(
        self,
        embedding: np.ndarray,
    ) -> EnsembleVerdict:
        """Predict cluster for a single embedding using ensemble voting.

        Args:
            embedding: Face embedding vector.

        Returns:
            EnsembleVerdict with weighted prediction.
        """
        if not self._trained:
            return EnsembleVerdict()

        vec = embedding.reshape(1, -1).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec_norm = vec / norm
        else:
            vec_norm = vec

        # Individual predictions
        votes: dict[str, str | None] = {}
        confidences: dict[str, float] = {}

        # Cosine KNN
        try:
            cos_pred = self._cosine_knn.predict(vec_norm)[0]
            cos_proba = self._cosine_knn.predict_proba(vec_norm)[0]
            cos_conf = float(cos_proba.max())
            votes["cosine"] = cos_pred
            confidences["cosine"] = cos_conf
        except Exception:
            votes["cosine"] = None
            confidences["cosine"] = 0.0

        # Euclidean KNN
        try:
            euc_pred = self._euclidean_knn.predict(vec)[0]
            euc_proba = self._euclidean_knn.predict_proba(vec)[0]
            euc_conf = float(euc_proba.max())
            votes["euclidean"] = euc_pred
            confidences["euclidean"] = euc_conf
        except Exception:
            votes["euclidean"] = None
            confidences["euclidean"] = 0.0

        # SVM
        try:
            svm_pred = self._svm.predict(vec)[0]
            svm_proba = self._svm.predict_proba(vec)[0]
            svm_conf = float(svm_proba.max())
            votes["svm"] = svm_pred
            confidences["svm"] = svm_conf
        except Exception:
            votes["svm"] = None
            confidences["svm"] = 0.0

        # Weighted voting
        cluster_scores: dict[str, float] = {}
        weight_map = {
            "cosine": self.cosine_weight,
            "euclidean": self.euclidean_weight,
            "svm": self.svm_weight,
        }

        for method, predicted_cluster in votes.items():
            if predicted_cluster is not None:
                w = weight_map[method] * confidences.get(method, 0.0)
                cluster_scores[predicted_cluster] = (
                    cluster_scores.get(predicted_cluster, 0.0) + w
                )

        if not cluster_scores:
            return EnsembleVerdict(
                confidence=0.0,
                individual_votes=votes,
                individual_confidences=confidences,
            )

        best_cluster = max(cluster_scores, key=cluster_scores.get)  # type: ignore[arg-type]
        total_weight = sum(cluster_scores.values())
        confidence = cluster_scores[best_cluster] / max(total_weight, 1e-10)

        if confidence < self.reject_threshold:
            return EnsembleVerdict(
                predicted_cluster=None,
                confidence=confidence,
                method="rejected",
                individual_votes=votes,
                individual_confidences=confidences,
            )

        return EnsembleVerdict(
            predicted_cluster=best_cluster,
            confidence=confidence,
            method="ensemble",
            individual_votes=votes,
            individual_confidences=confidences,
        )

    def update_weights(self, method: str, correct: bool) -> None:
        """Update dynamic weights based on prediction accuracy.

        Args:
            method: "cosine", "euclidean", or "svm".
            correct: Whether the prediction was correct.
        """
        if method not in self._accuracy_history:
            return

        self._accuracy_history[method].append(correct)
        # Keep last 100 predictions
        if len(self._accuracy_history[method]) > 100:
            self._accuracy_history[method] = self._accuracy_history[method][-100:]

    def get_adapted_weights(self) -> dict[str, float]:
        """Get weights adapted based on recent accuracy."""
        adapted: dict[str, float] = {}
        for method, history in self._accuracy_history.items():
            if len(history) >= 10:
                accuracy = sum(history) / len(history)
                adapted[method] = accuracy
            else:
                adapted[method] = {
                    "cosine": self.cosine_weight,
                    "euclidean": self.euclidean_weight,
                    "svm": self.svm_weight,
                }[method]

        # Normalize
        total = sum(adapted.values())
        if total > 0:
            adapted = {k: v / total for k, v in adapted.items()}
        return adapted
