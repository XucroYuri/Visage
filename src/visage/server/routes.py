"""FastAPI routes for the Visage review UI."""

from __future__ import annotations

import io
import logging
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from PIL import Image

from visage.cluster import (
    _normalize_embeddings,
    cluster_faces,
    compute_composite_distance,
    compute_composite_distance_chunked,
)
from visage.cluster import (
    merge_clusters as cluster_merge,
)
from visage.head_features import FEATURE_DIM
from visage.server.workspace import Workspace

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# Simple in-memory LRU thumbnail cache
_thumbnail_cache: OrderedDict[str, bytes] = OrderedDict()
_THUMB_CACHE_MAX = 200
_THUMB_SIZE = (300, 300)


def _get_workspace(request: Request):
    """Get the workspace from app state."""
    ws = request.app.state.workspace
    if ws is None:
        raise HTTPException(status_code=503, detail="Workspace not loaded")
    return ws


def _validate_image_path(path: str, request: Request) -> str:
    """Validate that the image path is within the input directory.

    Prevents path traversal attacks.
    """
    input_dir = getattr(request.app.state, "input_dir", None)
    if input_dir:
        resolved = Path(path).resolve()
        allowed = Path(input_dir).resolve()
        try:
            resolved.relative_to(allowed)
        except ValueError:
            raise HTTPException(
                status_code=403,
                detail="Access denied: path outside input directory",
            ) from None
    return path


@router.get("/workspace")
def get_workspace(request: Request):
    """Return full workspace state for the frontend."""
    ws = _get_workspace(request)
    return ws.to_api_dict()


@router.get("/image")
def get_image(
    request: Request,
    path: str = Query(..., description="Absolute path to image file"),
    size: str = Query("thumb", description="'thumb' or 'full'"),
):
    """Serve an image file — thumbnail or full resolution."""
    path = _validate_image_path(path, request)

    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"Image not found: {path}")

    if size == "thumb":
        cache_key = path
        if cache_key in _thumbnail_cache:
            _thumbnail_cache.move_to_end(cache_key)
            return Response(
                content=_thumbnail_cache[cache_key],
                media_type="image/jpeg",
            )

        try:
            img = Image.open(path)
            img = img.convert("RGB")
            img.thumbnail(_THUMB_SIZE, Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            data = buf.getvalue()

            # LRU eviction
            if len(_thumbnail_cache) >= _THUMB_CACHE_MAX:
                _thumbnail_cache.popitem(last=False)
            _thumbnail_cache[cache_key] = data

            return Response(content=data, media_type="image/jpeg")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Thumbnail error: {exc}") from exc

    # Full resolution
    return FileResponse(path)


@router.post("/clusters/merge")
async def merge_clusters(request: Request):
    """Merge one cluster into another. Body: {from_id: int, to_id: int}."""
    ws = _get_workspace(request)
    body = await request.json()
    from_id = body.get("from_id")
    to_id = body.get("to_id")

    if from_id is None or to_id is None:
        raise HTTPException(status_code=400, detail="Missing from_id or to_id")

    try:
        ws.merge_clusters(from_id, to_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"ok": True, "workspace": ws.to_api_dict()}


@router.post("/clusters/{cluster_id}/remove")
async def remove_face(cluster_id: int, request: Request):
    """Remove a face from a cluster. Body: {image_path: str}."""
    ws = _get_workspace(request)
    body = await request.json()
    image_path = body.get("image_path")
    if not image_path:
        raise HTTPException(status_code=400, detail="Missing image_path")

    try:
        ws.remove_face(image_path, cluster_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"ok": True, "workspace": ws.to_api_dict()}


@router.post("/clusters/{cluster_id}/remove-batch")
async def remove_faces_batch(cluster_id: int, request: Request):
    """Remove multiple faces from a cluster at once.

    Body: {image_paths: [str, ...]}
    All removed photos go to noise. Pushes a single undo operation.
    """
    ws = _get_workspace(request)
    body = await request.json()
    image_paths = body.get("image_paths")
    if not image_paths:
        raise HTTPException(status_code=400, detail="Missing image_paths")
    if not isinstance(image_paths, list):
        raise HTTPException(
            status_code=400, detail="image_paths must be a list"
        )

    try:
        ws.batch_remove_faces(cluster_id, image_paths)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"ok": True, "workspace": ws.to_api_dict()}


@router.post("/clusters/move")
async def move_face(request: Request):
    """Move a face from one cluster to another. Body: {image_path, from_id, to_id}."""
    ws = _get_workspace(request)
    body = await request.json()
    image_path = body.get("image_path")
    from_id = body.get("from_id")
    to_id = body.get("to_id")

    if not image_path or from_id is None or to_id is None:
        raise HTTPException(status_code=400, detail="Missing image_path, from_id, or to_id")

    try:
        ws.move_face(image_path, from_id, to_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"ok": True, "workspace": ws.to_api_dict()}


@router.post("/clusters/assign")
async def assign_noise(request: Request):
    """Assign a noise/unclustered photo to a cluster. Body: {image_path, to_id}."""
    ws = _get_workspace(request)
    body = await request.json()
    image_path = body.get("image_path")
    to_id = body.get("to_id")

    if not image_path or to_id is None:
        raise HTTPException(status_code=400, detail="Missing image_path or to_id")

    try:
        # Use a virtual "noise" source cluster ID (-1)
        ws.move_face(image_path, -1, to_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"ok": True, "workspace": ws.to_api_dict()}


@router.post("/clusters/assign-batch")
async def assign_noise_batch(request: Request):
    """Assign multiple noise/unclustered photos to a cluster at once.

    Body: {image_paths: [str, ...], to_id: int}
    Pushes a single undo operation for all assignments.
    """
    ws = _get_workspace(request)
    body = await request.json()
    image_paths = body.get("image_paths")
    to_id = body.get("to_id")

    if not image_paths or to_id is None:
        raise HTTPException(
            status_code=400, detail="Missing image_paths or to_id"
        )
    if not isinstance(image_paths, list):
        raise HTTPException(
            status_code=400, detail="image_paths must be a list"
        )

    try:
        ws.batch_assign_noise(image_paths, to_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"ok": True, "workspace": ws.to_api_dict()}


@router.put("/clusters/{cluster_id}")
async def rename_cluster(cluster_id: int, request: Request):
    """Rename a cluster. Body: {name: str}."""
    ws = _get_workspace(request)
    body = await request.json()
    name = body.get("name", "")

    try:
        ws.rename_cluster(cluster_id, name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"ok": True, "workspace": ws.to_api_dict()}


@router.post("/clusters/undo")
def undo(request: Request):
    """Undo the last merge/remove/rename/move operation."""
    ws = _get_workspace(request)
    result = ws.undo()
    if result is None:
        raise HTTPException(status_code=400, detail="Nothing to undo")
    return {"ok": True, "undo": result, "workspace": ws.to_api_dict()}


@router.post("/save")
async def save(request: Request):
    """Save organized photos to disk with optional output settings.

    Body (all fields optional, null = use server defaults):
        output_dir: str | None
        copy_mode: bool | None
        folder_prefix: str | None
        include_unclustered: bool | None
        include_no_faces: bool | None
        cluster_ids: list[int] | None (null = all clusters)
    """
    ws = _get_workspace(request)
    body = await request.json()
    try:
        stats = ws.save_to_disk(
            output_dir=body.get("output_dir"),
            copy_mode=body.get("copy_mode"),
            folder_prefix=body.get("folder_prefix"),
            include_unclustered=body.get("include_unclustered"),
            include_no_faces=body.get("include_no_faces"),
            cluster_ids=body.get("cluster_ids"),
        )
        return {"ok": True, "stats": stats}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/config")
def get_config(request: Request):
    """Return current config (read-only reference)."""
    ws = _get_workspace(request)
    return {
        "copy_mode": ws.config.copy_mode,
        "folder_prefix": ws.config.folder_prefix,
        "embedding_backend": ws.config.embedding_backend,
        "cluster_method": ws.config.cluster_method,
        "min_samples": ws.config.dbscan_min_samples,
        "min_cluster_size": ws.config.hdbscan_min_cluster_size,
        "cluster_selection_epsilon": ws.config.cluster_selection_epsilon,
        "cluster_selection_method": ws.config.cluster_selection_method,
        "merge_threshold": ws.config.merge_threshold,
        "small_merge_threshold": ws.config.small_merge_threshold,
        "min_reliable_size": ws.config.min_reliable_size,
        "head_feature_weight": ws.config.head_feature_weight,
    }


# ── Re-clustering ─────────────────────────────────────────────


def _extract_head_features_for_recluster(
    image_results: list,
    face_to_image: list[tuple[str, int]],
) -> tuple[np.ndarray | None, np.ndarray]:
    """Extract head feature vectors aligned with face_to_image ordering.

    Mirrors _extract_head_features from pipeline.py but takes image_results
    in list form (for use in re-cluster endpoint).
    """
    face_lookup: dict[tuple[str, int], object] = {}
    for result in image_results:
        if getattr(result, "error", None):
            continue
        for face in getattr(result, "faces", []):
            if getattr(face, "embedding", None) is not None:
                face_lookup[(result.path, getattr(face, "face_index", 0))] = face

    feats: list[np.ndarray] = []
    valid_mask: list[bool] = []
    for path, face_idx in face_to_image:
        face = face_lookup.get((path, face_idx))
        if face is not None and getattr(face, "head_features", None) is not None:
            feats.append(face.head_features)
            valid_mask.append(True)
        else:
            feats.append(np.zeros(FEATURE_DIM, dtype=np.float64))
            valid_mask.append(False)

    if not feats:
        return None, np.array([], dtype=bool)
    return np.stack(feats), np.array(valid_mask)


@router.post("/recluster")
async def recluster(request: Request):
    """Re-run clustering with new parameters using existing embeddings.

    Body (optional fields use workspace defaults):
        cluster_method: str ("hdbscan"|"dbscan")
        min_samples: int
        min_cluster_size: int
        cluster_selection_epsilon: float
        cluster_selection_method: str ("eom"|"leaf")
        merge_threshold: float
        small_merge_threshold: float
        min_reliable_size: int
        head_feature_weight: float
    """
    ws = _get_workspace(request)
    body = await request.json()
    cfg = ws.config

    # Get raw data from workspace
    rd = ws.get_recluster_data()
    embeddings: np.ndarray = rd["embeddings"]
    face_to_image: list[tuple[str, int]] = rd["face_to_image"]
    image_results = rd["image_results"]

    if len(embeddings) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 faces to cluster")

    # Merge body params with config defaults
    hf_weight = body.get("head_feature_weight", cfg.head_feature_weight)
    cluster_method = body.get("cluster_method", cfg.cluster_method)
    min_samples = body.get("min_samples", cfg.dbscan_min_samples)
    min_cluster_size = body.get("min_cluster_size", cfg.hdbscan_min_cluster_size)
    cse = body.get("cluster_selection_epsilon", cfg.cluster_selection_epsilon)
    csm = body.get("cluster_selection_method", cfg.cluster_selection_method)
    merge_threshold = body.get("merge_threshold", cfg.merge_threshold)
    small_merge_threshold = body.get("small_merge_threshold", cfg.small_merge_threshold)
    min_reliable_size = body.get("min_reliable_size", cfg.min_reliable_size)

    # Build composite distance matrix if using head features
    distance_matrix = None
    if hf_weight > 0.0 and len(embeddings) > 1:
        head_result = _extract_head_features_for_recluster(image_results, face_to_image)
        head_feats, head_valid = head_result
        if head_feats is not None and head_feats.shape[1] > 0:
            normed = _normalize_embeddings(embeddings)
            cluster_dtype = np.float32 if cfg.use_float32_cluster else np.float64
            if cfg.cluster_chunk_size > 0 and len(embeddings) > cfg.cluster_chunk_size:
                distance_matrix = compute_composite_distance_chunked(
                    normed, head_feats, head_weight=hf_weight,
                    chunk_size=cfg.cluster_chunk_size, dtype=cluster_dtype,
                )
            else:
                distance_matrix = compute_composite_distance(
                    normed, head_feats, head_weight=hf_weight, dtype=cluster_dtype,
                )
            if not head_valid.all():
                face_sim = normed @ normed.T
                face_only_dist = np.clip(1.0 - face_sim, 0.0, 2.0)
                invalid = ~head_valid
                use_face_only = invalid[:, None] | invalid[None, :]
                distance_matrix = np.where(use_face_only, face_only_dist, distance_matrix)

    # Run clustering
    cluster_result = cluster_faces(
        embeddings,
        eps=cfg.dbscan_eps,
        min_samples=min_samples,
        auto_eps=cfg.auto_eps,
        cluster_method=cluster_method,
        min_cluster_size=min_cluster_size,
        cluster_selection_epsilon=cse,
        cluster_selection_method=csm,
        distance_matrix=distance_matrix,
    )

    # Post-clustering merge
    if merge_threshold > 0.0 and cluster_result.num_clusters > 1:
        cluster_result = cluster_merge(
            cluster_result,
            merge_threshold=merge_threshold,
            min_reliable_size=min_reliable_size,
            small_merge_threshold=small_merge_threshold,
        )

    if cluster_result.num_clusters == 0:
        raise HTTPException(
            status_code=400,
            detail="Clustering produced 0 clusters with these parameters. "
                   "Try lowering min_samples or min_cluster_size.",
        )

    # Build new workspace
    new_ws = Workspace(
        input_dir=ws.input_dir,
        config=cfg,
        image_results=image_results,
        cluster_result=cluster_result,
        face_to_image=face_to_image,
    )

    # Replace workspace in app state (preserve user-assigned names where possible)
    old_names = ws.cluster_names
    unique_ids = {
        int(label) for label in set(cluster_result.labels) if int(label) >= 0
    }
    for cid in unique_ids:
        if cid in old_names:
            new_ws.rename_cluster(cid, old_names[cid])

    request.app.state.workspace = new_ws

    # Clear thumbnail cache
    _thumbnail_cache.clear()

    logger.info(
        "Re-clustered: %d → %d clusters (method=%s, min_samples=%d, merge=%.2f)",
        len(set(cluster_result.labels)) - (1 if -1 in cluster_result.labels else 0),
        cluster_result.num_clusters,
        cluster_method, min_samples, merge_threshold,
    )

    return {"ok": True, "workspace": new_ws.to_api_dict()}
