"""Global clustering optimizer — periodic HDBSCAN with drift detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of a global optimization run."""

    faces_optimized: int = 0
    assignments_changed: int = 0
    drift_pct: float = 0.0
    clusters_merged: int = 0
    new_clusters: int = 0
    elapsed_seconds: float = 0.0


class GlobalOptimizer:
    """Runs periodic full HDBSCAN clustering and detects drift from incremental assignments.

    Compares incremental assignments against full HDBSCAN results.
    If drift > threshold, triggers a global re-assignment.
    """

    def __init__(
        self,
        drift_threshold: float = 0.05,
        min_faces_for_optimization: int = 100,
    ) -> None:
        self.drift_threshold = drift_threshold
        self.min_faces_for_optimization = min_faces_for_optimization
        self._last_optimization_faces: int = 0

    def should_optimize(
        self,
        total_faces: int,
        faces_since_last: int,
        force: bool = False,
    ) -> bool:
        """Determine if global optimization should run.

        Args:
            total_faces: Total faces in the index.
            faces_since_last: New faces since last optimization.
            force: Force optimization regardless of thresholds.
        """
        if force:
            return True
        if total_faces < self.min_faces_for_optimization:
            return False
        if faces_since_last >= 5000:
            return True
        return False

    def compute_drift(
        self,
        current_assignments: dict[str, str],
        new_assignments: dict[str, str],
    ) -> float:
        """Compute the fraction of faces that changed cluster assignment.

        Args:
            current_assignments: face_id -> cluster_id (current incremental).
            new_assignments: face_id -> cluster_id (from full HDBSCAN).

        Returns:
            Drift fraction [0, 1].
        """
        common = set(current_assignments) & set(new_assignments)
        if not common:
            return 0.0

        changed = sum(
            1 for fid in common
            if current_assignments[fid] != new_assignments[fid]
        )
        return changed / len(common)

    def optimize(
        self,
        embeddings: np.ndarray,
        face_ids: list[str],
        current_assignments: dict[str, str],
        cluster_faces_fn,
    ) -> OptimizationResult:
        """Run global optimization using full HDBSCAN.

        Args:
            embeddings: All face embeddings, shape (N, D).
            face_ids: Face IDs corresponding to each row.
            current_assignments: Current face_id -> cluster_id mapping.
            cluster_faces_fn: Callable(embeddings, ...) -> labels array.

        Returns:
            OptimizationResult with drift and change statistics.
        """
        import time

        t0 = time.time()

        if len(face_ids) == 0:
            return OptimizationResult()

        # Run full HDBSCAN
        labels = cluster_faces_fn(embeddings)

        # Build new assignments from HDBSCAN labels
        new_assignments: dict[str, str] = {}
        for i, label in enumerate(labels):
            if label == -1:
                new_assignments[face_ids[i]] = f"noise_{i}"
            else:
                new_assignments[face_ids[i]] = f"cluster_{label}"

        # Compute drift
        drift = self.compute_drift(current_assignments, new_assignments)

        # Count changes
        changed = sum(
            1 for fid in set(current_assignments) & set(new_assignments)
            if current_assignments[fid] != new_assignments[fid]
        )

        # Count new clusters
        new_cluster_ids = set(new_assignments.values())
        old_cluster_ids = set(current_assignments.values())
        new_clusters = len(new_cluster_ids - old_cluster_ids)

        elapsed = time.time() - t0
        self._last_optimization_faces = len(face_ids)

        logger.info(
            "Global optimization: %d faces, %d changed (%.1f%% drift), %.1fs",
            len(face_ids), changed, drift * 100, elapsed,
        )

        return OptimizationResult(
            faces_optimized=len(face_ids),
            assignments_changed=changed,
            drift_pct=drift,
            new_clusters=new_clusters,
            elapsed_seconds=elapsed,
        )
