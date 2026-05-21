"""Incremental face assignment via ANN search and majority voting."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AssignmentResult:
    """Result of assigning a single face to a cluster."""

    face_id: str
    cluster_id: str | None = None
    confidence: float = 0.0
    method: str = "none"  # "high", "medium", "new_cluster", "pending"
    neighbor_ids: list[str] = field(default_factory=list)
    neighbor_scores: list[float] = field(default_factory=list)


class IncrementalAssigner:
    """Assigns new face embeddings to existing clusters using ANN majority vote.

    Strategy:
    - top-1 distance < high_threshold → direct assignment (high confidence)
    - top-3 majority ≥ 2 from same cluster → assignment (medium confidence)
    - top-3 all different clusters → mark as "pending" (low confidence)
    - all distances > new_threshold → create new cluster
    """

    def __init__(
        self,
        high_threshold: float = 0.4,
        medium_threshold: float = 0.55,
        new_threshold: float = 0.65,
        top_k: int = 3,
    ) -> None:
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.new_threshold = new_threshold
        self.top_k = top_k
        self._next_cluster_id = 0

    def assign(
        self,
        face_id: str,
        embedding: np.ndarray,
        search_fn,
        cluster_lookup,
    ) -> AssignmentResult:
        """Assign a face to a cluster using ANN search + majority vote.

        Args:
            face_id: ID of the face to assign.
            embedding: Face embedding vector.
            search_fn: Callable(vector, top_k) -> list[(face_id, score)].
            cluster_lookup: Callable(face_id) -> cluster_id | None.

        Returns:
            AssignmentResult with cluster assignment and confidence.
        """
        neighbors = search_fn(embedding, self.top_k)

        if not neighbors:
            return self._new_cluster(face_id, "no_neighbors")

        # Check high-confidence top-1
        top_id, top_score = neighbors[0]
        top_cluster = cluster_lookup(top_id)

        # Inner product similarity (higher = more similar)
        if top_score > (1.0 - self.high_threshold) and top_cluster is not None:
            return AssignmentResult(
                face_id=face_id,
                cluster_id=top_cluster,
                confidence=top_score,
                method="high",
                neighbor_ids=[n[0] for n in neighbors],
                neighbor_scores=[n[1] for n in neighbors],
            )

        # Majority vote among top-k
        cluster_votes: dict[str, int] = {}
        cluster_best_score: dict[str, float] = {}
        for nid, score in neighbors:
            cid = cluster_lookup(nid)
            if cid is not None:
                cluster_votes[cid] = cluster_votes.get(cid, 0) + 1
                if cid not in cluster_best_score or score > cluster_best_score[cid]:
                    cluster_best_score[cid] = score

        if cluster_votes:
            best_cluster = max(cluster_votes, key=cluster_votes.get)  # type: ignore[arg-type]
            votes = cluster_votes[best_cluster]

            if votes >= 2 and cluster_best_score.get(best_cluster, 0) > (
                    1.0 - self.medium_threshold
                ):
                return AssignmentResult(
                    face_id=face_id,
                    cluster_id=best_cluster,
                    confidence=cluster_best_score[best_cluster],
                    method="medium",
                    neighbor_ids=[n[0] for n in neighbors],
                    neighbor_scores=[n[1] for n in neighbors],
                )

        # Check if all distances exceed new_threshold → new cluster
        if top_score < (1.0 - self.new_threshold):
            return self._new_cluster(face_id, "distant")

        # Otherwise mark as pending
        return AssignmentResult(
            face_id=face_id,
            cluster_id=None,
            confidence=top_score,
            method="pending",
            neighbor_ids=[n[0] for n in neighbors],
            neighbor_scores=[n[1] for n in neighbors],
        )

    def _new_cluster(self, face_id: str, reason: str) -> AssignmentResult:
        """Create a new cluster for a face."""
        cluster_id = f"inc_{self._next_cluster_id:06d}"
        self._next_cluster_id += 1
        return AssignmentResult(
            face_id=face_id,
            cluster_id=cluster_id,
            confidence=1.0,
            method="new_cluster",
        )
