"""API routes for active learning — corrections, prototypes, adaptive threshold."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/active", tags=["active"])


@router.post("/correction")
async def record_correction(request: Request) -> dict:
    """Record a user correction (merge, split, reassign).

    Body:
        action: "merge" | "split" | "reassign"
        face_ids: list of affected face IDs
        source_cluster: original cluster ID
        target_cluster: new cluster ID
    """
    body = await request.json()
    action = body.get("action")
    face_ids = body.get("face_ids", [])
    source = body.get("source_cluster")
    target = body.get("target_cluster")

    if action not in ("merge", "split", "reassign", "rename"):
        raise HTTPException(400, f"Invalid action: {action!r}")
    if not face_ids:
        raise HTTPException(400, "face_ids required")

    ws = getattr(request.app.state, "workspace", None)
    if ws is None:
        raise HTTPException(503, "Workspace not ready")

    # Record in correction store
    store = getattr(ws, "_correction_store", None)
    if store is not None:
        correction_id = store.record_correction(
            action=action,
            face_ids=face_ids,
            source_cluster=source,
            target_cluster=target,
        )
    else:
        correction_id = -1

    # Update adaptive threshold
    adapter = getattr(ws, "_threshold_adapter", None)
    if adapter is not None:
        if action == "merge":
            adapter.record_merge()
        elif action == "split":
            adapter.record_split()
        elif action == "reassign" and source is not None and target is not None:
            adapter.record_reassign(source, target)

    # Update prototypes
    prototypes = getattr(ws, "_prototype_manager", None)
    if prototypes is not None and action == "reassign":
        for fid in face_ids:
            emb = _get_face_embedding(ws, fid)
            if emb is not None and target is not None:
                prototypes.update_on_correction(
                    target, emb, weight=1.5, is_addition=True,
                )
                if source is not None:
                    prototypes.update_on_correction(
                        source, emb, weight=1.0, is_addition=False,
                    )

    return {"correction_id": correction_id, "action": action, "recorded": True}


@router.get("/corrections")
async def list_corrections(
    request: Request,
    action: str | None = None,
    limit: int = Query(100, ge=1, le=500),
) -> dict:
    """List recorded corrections."""
    ws = getattr(request.app.state, "workspace", None)
    if ws is None:
        raise HTTPException(503, "Workspace not ready")

    store = getattr(ws, "_correction_store", None)
    if store is None:
        return {"corrections": [], "total": 0}

    corrections = store.get_corrections(action=action, limit=limit)
    return {"corrections": corrections, "total": len(corrections)}


@router.get("/corrections/stats")
async def correction_stats(request: Request) -> dict:
    """Get correction statistics."""
    ws = getattr(request.app.state, "workspace", None)
    if ws is None:
        raise HTTPException(503, "Workspace not ready")

    store = getattr(ws, "_correction_store", None)
    adapter = getattr(ws, "_threshold_adapter", None)

    result: dict = {}
    if store is not None:
        result["counts"] = store.get_correction_stats()
        result["total"] = store.get_correction_count()
    if adapter is not None:
        result["threshold"] = adapter.stats
    return result


@router.get("/prototypes")
async def list_prototypes(request: Request) -> dict:
    """List all cluster prototypes."""
    ws = getattr(request.app.state, "workspace", None)
    if ws is None:
        raise HTTPException(503, "Workspace not ready")

    prototypes = getattr(ws, "_prototype_manager", None)
    if prototypes is None:
        return {"prototypes": [], "total": 0}

    protos = []
    for cid in prototypes.cluster_ids:
        p = prototypes.get_prototype(cid)
        if p:
            protos.append({
                "cluster_id": cid,
                "member_count": p.member_count,
                "total_weight": round(p.total_weight, 4),
            })
    return {"prototypes": protos, "total": len(protos)}


@router.get("/threshold")
async def get_threshold(request: Request) -> dict:
    """Get current adaptive threshold state."""
    ws = getattr(request.app.state, "workspace", None)
    if ws is None:
        raise HTTPException(503, "Workspace not ready")

    adapter = getattr(ws, "_threshold_adapter", None)
    if adapter is None:
        return {"threshold": 0.70, "active": False}

    return {"active": True, **adapter.stats}


@router.post("/threshold/reset")
async def reset_threshold(request: Request) -> dict:
    """Reset adaptive threshold to defaults."""
    ws = getattr(request.app.state, "workspace", None)
    if ws is None:
        raise HTTPException(503, "Workspace not ready")

    adapter = getattr(ws, "_threshold_adapter", None)
    if adapter is not None:
        adapter.reset()
    return {"reset": True, **(adapter.stats if adapter else {})}


def _get_face_embedding(ws, face_id: str):
    """Look up a face embedding from workspace by face_id."""
    for result in ws.image_results:
        if result.error or not result.faces:
            continue
        for face in result.faces:
            fid = f"{result.path}:{face.face_index}"
            if fid == face_id and face.embedding is not None:
                return face.embedding
    return None
