"""Face search business logic — query, ranking, pagination."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single face search result."""

    face_id: str
    image_path: str
    similarity: float
    quality_score: float = 0.0
    cluster_id: str | None = None
    bbox: tuple[int, int, int, int] | None = None


@dataclass
class SearchResponse:
    """Complete search response."""

    query_face_id: str
    results: list[SearchResult] = field(default_factory=list)
    total: int = 0
    elapsed_ms: float = 0.0
    page: int = 0
    page_size: int = 20


def search_faces(
    query_face_id: str,
    query_vector: np.ndarray | None = None,
    top_k: int = 50,
    min_score: float = 0.4,
    cluster_id: str | None = None,
    page: int = 0,
    page_size: int = 20,
    search_fn=None,
    metadata_lookup=None,
) -> SearchResponse:
    """Execute a face similarity search.

    Args:
        query_face_id: ID of the query face.
        query_vector: Optional pre-computed query vector. If None, uses search_fn.
        top_k: Number of candidates to retrieve from the index.
        min_score: Minimum similarity score to include.
        cluster_id: Optional cluster filter.
        page: Page number for pagination.
        page_size: Results per page.
        search_fn: Callable(vector, top_k) -> list[(face_id, score)].
        metadata_lookup: Callable(face_id) -> dict | None.

    Returns:
        SearchResponse with ranked results.
    """
    t0 = time.time()

    if search_fn is None:
        return SearchResponse(query_face_id=query_face_id, elapsed_ms=0)

    # Get search results
    if query_vector is not None:
        candidates = search_fn(query_vector, top_k)
    else:
        # Use face_id based search (search_by_id)
        candidates = search_fn(query_face_id, top_k)

    # Filter by min_score and cluster
    filtered: list[SearchResult] = []
    for fid, score in candidates:
        if fid == query_face_id:
            continue  # Skip self
        if score < min_score:
            continue

        meta = metadata_lookup(fid) if metadata_lookup else {}

        # Optional cluster filter
        if cluster_id and meta.get("cluster_id") != cluster_id:
            continue

        result = SearchResult(
            face_id=fid,
            image_path=meta.get("image_path", ""),
            similarity=round(score, 4),
            quality_score=meta.get("quality_score", 0.0),
            cluster_id=meta.get("cluster_id"),
            bbox=meta.get("bbox"),
        )
        filtered.append(result)

    # Sort by similarity desc, then quality desc
    filtered.sort(key=lambda r: (-r.similarity, -r.quality_score))

    total = len(filtered)

    # Paginate
    start = page * page_size
    end = start + page_size
    page_results = filtered[start:end]

    elapsed = (time.time() - t0) * 1000

    return SearchResponse(
        query_face_id=query_face_id,
        results=page_results,
        total=total,
        elapsed_ms=round(elapsed, 1),
        page=page,
        page_size=page_size,
    )
