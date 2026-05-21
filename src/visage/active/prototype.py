"""Prototype vector management — weighted centroids per cluster.

Maintains a mean embedding vector per cluster that is incrementally
updated as users provide corrections. Prototypes are more stable than
single-image embeddings and improve classification accuracy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Prototype:
    """A cluster prototype — weighted centroid of member embeddings."""

    cluster_id: int
    centroid: np.ndarray
    member_count: int = 0
    total_weight: float = 0.0
    updated_at: float = 0.0


class PrototypeManager:
    """Manages prototype vectors for face clusters.

    Prototypes are weighted centroids that improve over time as users
    correct cluster assignments. Each correction updates the affected
    prototypes incrementally without recomputing from scratch.
    """

    def __init__(self, embedding_dim: int = 128) -> None:
        self._prototypes: dict[int, Prototype] = {}
        self._embedding_dim = embedding_dim

    def get_prototype(self, cluster_id: int) -> Prototype | None:
        """Get the prototype for a cluster, or None if not built."""
        return self._prototypes.get(cluster_id)

    def get_centroid(self, cluster_id: int) -> np.ndarray | None:
        """Get the centroid vector for a cluster."""
        proto = self._prototypes.get(cluster_id)
        return proto.centroid if proto else None

    def build_from_embeddings(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> None:
        """Build prototypes from a full set of embeddings and cluster labels.

        Args:
            embeddings: (N, D) array of face embeddings.
            labels: (N,) array of cluster IDs (-1 for noise).
            weights: Optional (N,) array of sample weights.
        """
        unique_labels = set(labels.tolist()) - {-1}
        for cid in unique_labels:
            mask = labels == cid
            cluster_embs = embeddings[mask]
            w = weights[mask] if weights is not None else np.ones(len(cluster_embs))

            centroid = np.average(cluster_embs, axis=0, weights=w)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm

            self._prototypes[int(cid)] = Prototype(
                cluster_id=int(cid),
                centroid=centroid,
                member_count=int(len(cluster_embs)),
                total_weight=float(w.sum()),
            )

        logger.info(
            "Built %d prototypes from %d embeddings",
            len(self._prototypes), len(embeddings),
        )

    def update_on_correction(
        self,
        cluster_id: int,
        embedding: np.ndarray,
        weight: float = 1.0,
        is_addition: bool = True,
    ) -> None:
        """Incrementally update a prototype after a user correction.

        Args:
            cluster_id: The cluster being modified.
            embedding: The face embedding being added or removed.
            weight: Weight for this embedding (corrections get higher weight).
            is_addition: True if adding to cluster, False if removing.
        """
        proto = self._prototypes.get(cluster_id)
        if proto is None:
            # Create new prototype
            norm = np.linalg.norm(embedding)
            centroid = embedding / norm if norm > 0 else embedding
            self._prototypes[cluster_id] = Prototype(
                cluster_id=cluster_id,
                centroid=centroid,
                member_count=1,
                total_weight=weight,
            )
            return

        old_centroid = proto.centroid
        old_weight = proto.total_weight

        if is_addition:
            new_weight = old_weight + weight
            new_centroid = (old_centroid * old_weight + embedding * weight) / new_weight
            proto.member_count += 1
        else:
            if proto.member_count <= 1:
                # Removing last member — delete prototype
                del self._prototypes[cluster_id]
                return
            new_weight = max(old_weight - weight, 0.001)
            new_centroid = (old_centroid * old_weight - embedding * weight) / new_weight
            proto.member_count -= 1

        # Renormalize
        norm = np.linalg.norm(new_centroid)
        if norm > 0:
            new_centroid = new_centroid / norm

        proto.centroid = new_centroid
        proto.total_weight = new_weight

    @property
    def prototypes(self) -> dict[int, Prototype]:
        return self._prototypes

    @property
    def cluster_ids(self) -> list[int]:
        return sorted(self._prototypes.keys())

    def to_dict(self) -> dict:
        """Serialize prototypes for persistence."""
        return {
            "embedding_dim": self._embedding_dim,
            "prototypes": {
                str(cid): {
                    "centroid": p.centroid.tolist(),
                    "member_count": p.member_count,
                    "total_weight": p.total_weight,
                }
                for cid, p in self._prototypes.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> PrototypeManager:
        """Deserialize prototypes from persistence."""
        mgr = cls(embedding_dim=data.get("embedding_dim", 128))
        for cid_str, pdata in data.get("prototypes", {}).items():
            cid = int(cid_str)
            mgr._prototypes[cid] = Prototype(
                cluster_id=cid,
                centroid=np.array(pdata["centroid"], dtype=np.float64),
                member_count=pdata["member_count"],
                total_weight=pdata["total_weight"],
            )
        return mgr
