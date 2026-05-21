"""FAISS vector index with CRUD, persistence, and soft-delete rebuild."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# FAISS is optional — only required for vector search features
try:
    import faiss

    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


def _check_faiss() -> None:
    if not _FAISS_AVAILABLE:
        raise RuntimeError(
            "faiss-cpu is required for vector search. "
            "Install with: uv sync --extra vector"
        )


class VectorIndex:
    """FAISS IVFFlat index wrapper with add, search, soft-delete, save/load.

    The index uses IVF (Inverted File) partitioning for approximate nearest
    neighbor search. For small collections (<10k), falls back to a flat index.

    Attributes:
        dim: Embedding dimensionality.
        nlist: Number of Voronoi cells for IVF (0 = flat index).
    """

    def __init__(self, dim: int = 512, nlist: int = 0) -> None:
        _check_faiss()
        self.dim = dim
        self.nlist = nlist
        self._ids: list[str] = []  # face_id for each vector
        self._id_to_pos: dict[str, int] = {}  # face_id -> position in index
        self._deleted_ids: set[str] = set()
        self._index: faiss.Index | None = None
        self._trained = False

    @property
    def total(self) -> int:
        """Total number of vectors (including soft-deleted)."""
        return len(self._ids)

    @property
    def active(self) -> int:
        """Number of active (non-deleted) vectors."""
        return len(self._ids) - len(self._deleted_ids)

    @property
    def deleted_count(self) -> int:
        """Number of soft-deleted vectors."""
        return len(self._deleted_ids)

    @property
    def needs_rebuild(self) -> bool:
        """Whether the index should be rebuilt due to accumulated deletions.

        Triggers when >10% of vectors are soft-deleted (or >100 absolute).
        """
        if self.total == 0:
            return False
        return self.deleted_count * 100 > self.total * 10

    def _ensure_index(self, n: int = 1) -> None:
        """Create or grow the index to accommodate n new vectors."""
        if self._index is not None:
            return

        if self.nlist > 0 and self.total + n >= self.nlist * 10:
            # Enough vectors for IVF
            self._index = faiss.IndexIVFFlat(
                faiss.IndexFlatIP(self.dim),  # quantizer: inner product
                self.dim,
                self.nlist,
                faiss.METRIC_INNER_PRODUCT,
            )
        else:
            # Flat index for small collections
            self._index = faiss.IndexFlatIP(self.dim)
            self._trained = True  # Flat index needs no training

    def _ensure_trained(self, vectors: np.ndarray) -> None:
        """Train IVF index if needed."""
        if self._trained or self._index is None:
            return
        if hasattr(self._index, "is_trained") and self._index.is_trained:
            self._trained = True
            return
        if hasattr(self._index, "train"):
            n = vectors.shape[0]
            if n < self.nlist:
                # Not enough data for IVF, fall back to flat
                self._index = faiss.IndexFlatIP(self.dim)
                self._trained = True
                return
            self._index.train(vectors)
            self._trained = True

    def add(self, face_id: str, vector: np.ndarray) -> None:
        """Add a single vector to the index.

        Args:
            face_id: Unique identifier for the face.
            vector: Embedding vector, shape (dim,).
        """
        vec = vector.reshape(1, -1).astype(np.float32)
        # L2-normalize for inner product = cosine similarity
        faiss.normalize_L2(vec)

        self._ensure_index(1)
        self._ensure_trained(vec)

        pos = len(self._ids)
        self._ids.append(face_id)
        self._id_to_pos[face_id] = pos
        self._index.add(vec)  # type: ignore[union-attr]

    def add_batch(self, ids: list[str], vectors: np.ndarray) -> None:
        """Add multiple vectors at once.

        Args:
            ids: List of face IDs, one per vector.
            vectors: Embedding matrix, shape (N, dim).
        """
        if len(ids) == 0:
            return
        vecs = vectors.astype(np.float32)
        faiss.normalize_L2(vecs)

        self._ensure_index(len(ids))
        self._ensure_trained(vecs)

        start = len(self._ids)
        self._ids.extend(ids)
        for i, fid in enumerate(ids):
            self._id_to_pos[fid] = start + i
        self._index.add(vecs)  # type: ignore[union-attr]

    def search(
        self, query: np.ndarray, top_k: int = 10, exclude_deleted: bool = True
    ) -> list[tuple[str, float]]:
        """Search for nearest neighbors.

        Args:
            query: Query vector, shape (dim,).
            top_k: Number of results to return.
            exclude_deleted: Whether to filter out soft-deleted vectors.

        Returns:
            List of (face_id, similarity_score) tuples, sorted by score desc.
        """
        if self._index is None or self.total == 0:
            return []

        vec = query.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(vec)

        # Over-fetch to account for deleted vectors
        k = min(top_k + len(self._deleted_ids), self.total) if exclude_deleted else top_k
        k = max(k, top_k)
        distances, indices = self._index.search(vec, k)  # type: ignore[union-attr]

        results: list[tuple[str, float]] = []
        for dist, idx in zip(distances[0], indices[0], strict=False):
            if idx < 0 or idx >= len(self._ids):
                continue
            fid = self._ids[idx]
            if exclude_deleted and fid in self._deleted_ids:
                continue
            results.append((fid, float(dist)))
            if len(results) >= top_k:
                break

        return results

    def search_by_id(
        self, face_id: str, top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Search for neighbors of a specific face in the index.

        Args:
            face_id: The face to search neighbors for.
            top_k: Number of results.

        Returns:
            List of (face_id, similarity_score) tuples (excludes self).
        """
        if face_id not in self._id_to_pos:
            return []
        pos = self._id_to_pos[face_id]
        if pos >= self._index.ntotal:  # type: ignore[union-attr]
            return []

        # Reconstruct the vector and search
        vec = self._index.reconstruct(int(pos))  # type: ignore[union-attr]
        results = self.search(vec, top_k + 1)  # +1 to exclude self
        return [(fid, score) for fid, score in results if fid != face_id][:top_k]

    def soft_delete(self, face_id: str) -> bool:
        """Soft-delete a vector (marked but not removed).

        Returns True if the face_id existed and was deleted.
        """
        if face_id in self._id_to_pos and face_id not in self._deleted_ids:
            self._deleted_ids.add(face_id)
            return True
        return False

    def rebuild(self) -> None:
        """Rebuild the index, excluding soft-deleted vectors.

        Compacts the index by removing deleted vectors.
        """
        if self._index is None:
            return

        # Collect active vectors
        active_vecs: list[np.ndarray] = []
        active_ids: list[str] = []
        for i, fid in enumerate(self._ids):
            if fid not in self._deleted_ids and i < self._index.ntotal:  # type: ignore[union-attr]
                vec = self._index.reconstruct(int(i))  # type: ignore[union-attr]
                active_vecs.append(vec)
                active_ids.append(fid)

        # Reset and re-add
        self._ids = []
        self._id_to_pos = {}
        self._deleted_ids = set()
        self._index = None
        self._trained = False

        if active_vecs:
            vectors = np.stack(active_vecs)
            self.add_batch(active_ids, vectors)

        logger.info(
            "Index rebuilt: %d active vectors (%d deleted removed)",
            len(active_ids), len(active_vecs),
        )

    def save(self, path: str | Path) -> None:
        """Save index and metadata to disk.

        Creates:
            <path> — FAISS index file
            <path>.meta — JSON metadata (ids, dimension, deleted_ids)
        """
        if self._index is None:
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(path))  # type: ignore[union-attr]

        meta = {
            "dim": self.dim,
            "nlist": self.nlist,
            "ids": self._ids,
            "deleted_ids": sorted(self._deleted_ids),
            "version": 1,
        }
        with open(str(path) + ".meta", "w") as f:
            json.dump(meta, f)

        logger.info("Index saved: %d vectors to %s", self.total, path)

    @classmethod
    def load(cls, path: str | Path) -> VectorIndex:
        """Load index and metadata from disk.

        Args:
            path: Path to the FAISS index file (meta file is <path>.meta).

        Returns:
            Restored VectorIndex.
        """
        _check_faiss()
        path = Path(path)

        with open(str(path) + ".meta") as f:
            meta = json.load(f)

        idx = cls(dim=meta["dim"], nlist=meta.get("nlist", 0))
        idx._index = faiss.read_index(str(path))
        idx._ids = meta["ids"]
        idx._id_to_pos = {fid: i for i, fid in enumerate(idx._ids)}
        idx._deleted_ids = set(meta.get("deleted_ids", []))
        idx._trained = True  # Loaded index is already trained

        logger.info("Index loaded: %d vectors from %s", idx.total, path)
        return idx
