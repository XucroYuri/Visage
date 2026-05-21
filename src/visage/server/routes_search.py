"""API routes for semantic image search and tag queries."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/search", tags=["search"])


@router.get("/semantic")
async def semantic_search(
    request: Request,
    q: str = Query(..., description="Natural language search query"),
    top_k: int = Query(20, ge=1, le=100),
    min_score: float = Query(0.3, ge=0.0, le=1.0),
) -> dict:
    """Search images by natural language query using CLIP.

    Args:
        q: Search query (e.g., "sunset at the beach").
        top_k: Maximum number of results.
        min_score: Minimum similarity score.
    """
    pipeline = getattr(request.app.state, "classify_pipeline", None)
    if pipeline is None:
        raise HTTPException(503, "Classification pipeline not initialized")

    results = pipeline.semantic_search(q, top_k=top_k, min_score=min_score)
    return {
        "query": q,
        "results": [
            {"image_path": path, "score": round(score, 4)}
            for path, score in results
        ],
        "total": len(results),
    }


@router.get("/tags")
async def search_by_tags(
    request: Request,
    tags: str = Query(..., description="Comma-separated tags"),
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(100, ge=1, le=500),
) -> dict:
    """Search images by tag names (OR query).

    Args:
        tags: Comma-separated tag list (e.g., "sunset,beach").
        min_score: Minimum tag confidence score.
        limit: Maximum results.
    """
    pipeline = getattr(request.app.state, "classify_pipeline", None)
    if pipeline is None or pipeline.store is None:
        raise HTTPException(503, "Classification pipeline not initialized")

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    if not tag_list:
        return {"tags": [], "results": [], "total": 0}

    results = pipeline.store.search_by_tags(tag_list, min_score=min_score, limit=limit)
    return {
        "tags": tag_list,
        "results": [
            {"image_path": path, "score": round(score, 4)}
            for path, score in results
        ],
        "total": len(results),
    }


@router.get("/tags/counts")
async def tag_counts(request: Request) -> dict:
    """Get count of images per tag."""
    pipeline = getattr(request.app.state, "classify_pipeline", None)
    if pipeline is None or pipeline.store is None:
        raise HTTPException(503, "Classification pipeline not initialized")

    counts = pipeline.store.get_tag_counts()
    return {"counts": counts, "unique_tags": len(counts)}


@router.get("/image/{image_path:path}")
async def image_tags(request: Request, image_path: str) -> dict:
    """Get all tags for a specific image."""
    pipeline = getattr(request.app.state, "classify_pipeline", None)
    if pipeline is None or pipeline.store is None:
        raise HTTPException(503, "Classification pipeline not initialized")

    tags_by_category = pipeline.store.get_tags(image_path)
    if not tags_by_category:
        raise HTTPException(404, f"No tags found for {image_path!r}")

    return {"image_path": image_path, "tags": tags_by_category}
