"""Cluster engine — orchestrates incremental assignment and global optimization."""

from __future__ import annotations

import logging

import numpy as np

from .incremental import AssignmentResult, IncrementalAssigner
from .optimizer import GlobalOptimizer

logger = logging.getLogger(__name__)


class ClusterEngine:
    """Orchestrates incremental and global clustering.

    Manages epoch tracking, delegates to IncrementalAssigner for new faces,
    and triggers GlobalOptimizer periodically.
    """

    def __init__(
        self,
        assigner: IncrementalAssigner | None = None,
        optimizer: GlobalOptimizer | None = None,
    ) -> None:
        self.assigner = assigner or IncrementalAssigner()
        self.optimizer = optimizer or GlobalOptimizer()
        self._epoch = 0
        self._faces_since_optimization = 0

    @property
    def epoch(self) -> int:
        """Current epoch counter — incremented on each assignment batch."""
        return self._epoch

    def assign_face(
        self,
        face_id: str,
        embedding: np.ndarray,
        search_fn,
        cluster_lookup,
    ) -> AssignmentResult:
        """Assign a single face using incremental assigner.

        Args:
            face_id: Face to assign.
            embedding: Face embedding vector.
            search_fn: Callable(vector, top_k) -> list[(face_id, score)].
            cluster_lookup: Callable(face_id) -> cluster_id | None.

        Returns:
            AssignmentResult with cluster assignment.
        """
        result = self.assigner.assign(face_id, embedding, search_fn, cluster_lookup)
        self._epoch += 1
        self._faces_since_optimization += 1
        return result

    def assign_batch(
        self,
        face_ids: list[str],
        embeddings: np.ndarray,
        search_fn,
        cluster_lookup,
    ) -> list[AssignmentResult]:
        """Assign a batch of faces.

        Args:
            face_ids: List of face IDs.
            embeddings: Matrix of embeddings, shape (N, D).
            search_fn: Callable(vector, top_k) -> list[(face_id, score)].
            cluster_lookup: Callable(face_id) -> cluster_id | None.

        Returns:
            List of AssignmentResults.
        """
        results = []
        for i, fid in enumerate(face_ids):
            result = self.assigner.assign(fid, embeddings[i], search_fn, cluster_lookup)
            results.append(result)
        self._epoch += len(face_ids)
        self._faces_since_optimization += len(face_ids)
        return results

    def should_optimize(self, total_faces: int, force: bool = False) -> bool:
        """Check if global optimization should run."""
        return self.optimizer.should_optimize(
            total_faces, self._faces_since_optimization, force=force
        )

    def run_optimization(
        self,
        embeddings: np.ndarray,
        face_ids: list[str],
        current_assignments: dict[str, str],
        cluster_faces_fn,
    ):
        """Run global optimization and reset counter."""
        result = self.optimizer.optimize(
            embeddings, face_ids, current_assignments, cluster_faces_fn
        )
        self._faces_since_optimization = 0
        return result
