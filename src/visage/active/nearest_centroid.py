"""Nearest centroid classifier using prototype vectors."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from visage.active.prototype import PrototypeManager

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result from nearest-centroid classification."""

    face_id: str
    predicted_cluster: int
    confidence: float
    distances: dict[int, float]  # cluster_id → distance


class NearestCentroidClassifier:
    """Classify faces by finding the nearest cluster prototype.

    Uses cosine distance to prototype centroids. More stable than
    single-image nearest-neighbor because centroids average out noise.
    """

    def __init__(self, prototype_manager: PrototypeManager) -> None:
        self._prototypes = prototype_manager

    def classify(
        self,
        embedding: np.ndarray,
        face_id: str = "",
        exclude_clusters: set[int] | None = None,
    ) -> ClassificationResult | None:
        """Classify a face embedding by nearest prototype centroid.

        Args:
            embedding: Face embedding vector.
            face_id: Optional face identifier.
            exclude_clusters: Cluster IDs to exclude from consideration.

        Returns:
            ClassificationResult or None if no prototypes exist.
        """
        prototypes = self._prototypes.prototypes
        if not prototypes:
            return None

        # Normalize query
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return None
        query = embedding / norm

        exclude = exclude_clusters or set()
        distances: dict[int, float] = {}
        for cid, proto in prototypes.items():
            if cid in exclude:
                continue
            # Cosine distance = 1 - cosine_similarity
            sim = float(np.dot(query, proto.centroid))
            dist = 1.0 - sim
            distances[cid] = dist

        if not distances:
            return None

        best_cluster = min(distances, key=distances.get)
        best_dist = distances[best_cluster]
        confidence = max(0.0, 1.0 - best_dist)

        return ClassificationResult(
            face_id=face_id,
            predicted_cluster=best_cluster,
            confidence=confidence,
            distances=distances,
        )

    def classify_batch(
        self,
        embeddings: np.ndarray,
        face_ids: list[str] | None = None,
    ) -> list[ClassificationResult | None]:
        """Classify a batch of embeddings.

        Args:
            embeddings: (N, D) array of face embeddings.
            face_ids: Optional list of face identifiers.

        Returns:
            List of ClassificationResult (or None for each input).
        """
        ids = face_ids or [""] * len(embeddings)
        return [
            self.classify(emb, fid)
            for emb, fid in zip(embeddings, ids, strict=True)
        ]

    def find_misclassified(
        self,
        embeddings: np.ndarray,
        true_labels: np.ndarray,
        threshold: float = 0.5,
    ) -> list[tuple[int, int, int, float]]:
        """Find faces where predicted cluster != true cluster.

        Args:
            embeddings: (N, D) face embeddings.
            true_labels: (N,) true cluster assignments.
            threshold: Minimum confidence to flag as misclassified.

        Returns:
            List of (index, true_cluster, predicted_cluster, confidence).
        """
        misclassified: list[tuple[int, int, int, float]] = []
        for i, (emb, true_cid) in enumerate(zip(embeddings, true_labels, strict=True)):
            if true_cid == -1:
                continue
            result = self.classify(emb, exclude_clusters=set())
            if result is None:
                continue
            if result.predicted_cluster != int(true_cid) and result.confidence >= threshold:
                misclassified.append((
                    i,
                    int(true_cid),
                    result.predicted_cluster,
                    result.confidence,
                ))
        return misclassified
