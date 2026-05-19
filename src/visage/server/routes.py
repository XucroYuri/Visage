"""FastAPI routes for the Visage review UI."""

from __future__ import annotations

import io
import os
from collections import OrderedDict
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from PIL import Image

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
    """Write organized files to disk."""
    ws = _get_workspace(request)
    body = await request.json()
    output_dir = body.get("output_dir")
    try:
        stats = ws.save_to_disk(output_dir=output_dir)
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
        "merge_threshold": ws.config.merge_threshold,
    }
