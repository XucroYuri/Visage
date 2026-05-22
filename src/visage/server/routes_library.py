"""API routes for multi-library management."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from visage.library.manager import LibraryManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/libraries", tags=["libraries"])


def _get_manager(request: Request) -> LibraryManager:
    mgr = getattr(request.app.state, "library_manager", None)
    if mgr is None:
        raise HTTPException(503, "Library manager not initialized")
    return mgr


@router.get("")
async def list_libraries(request: Request) -> dict:
    """List all libraries."""
    mgr = _get_manager(request)
    libraries = mgr.list_libraries()
    return {
        "libraries": [lib.to_dict() for lib in libraries],
        "total": len(libraries),
    }


@router.post("")
async def create_library(request: Request) -> dict:
    """Create a new library.

    Body:
        name: Display name
        input_dir: Path to photo directory
    """
    body = await request.json()
    name = body.get("name", "").strip()
    input_dir = body.get("input_dir", "").strip()

    if not name:
        raise HTTPException(400, "name is required")
    if not input_dir:
        raise HTTPException(400, "input_dir is required")

    mgr = _get_manager(request)
    lib = mgr.create_library(name, input_dir)
    return lib.to_dict()


@router.get("/{library_id}")
async def get_library(library_id: str, request: Request) -> dict:
    """Get a specific library."""
    mgr = _get_manager(request)
    lib = mgr.get_library(library_id)
    if lib is None:
        raise HTTPException(404, f"Library {library_id!r} not found")
    return lib.to_dict()


@router.put("/{library_id}")
async def update_library(library_id: str, request: Request) -> dict:
    """Update library fields."""
    body = await request.json()
    mgr = _get_manager(request)
    lib = mgr.update_library(library_id, **body)
    if lib is None:
        raise HTTPException(404, f"Library {library_id!r} not found")
    return lib.to_dict()


@router.delete("/{library_id}")
async def delete_library(library_id: str, request: Request) -> dict:
    """Delete a library."""
    mgr = _get_manager(request)
    deleted = mgr.delete_library(library_id)
    if not deleted:
        raise HTTPException(404, f"Library {library_id!r} not found")
    return {"deleted": True, "library_id": library_id}
