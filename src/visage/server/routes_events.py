"""API routes for auto-generated albums and events."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from visage.events.cluster import Event, cluster_by_time, extract_timestamps
from visage.events.cover_selector import select_event_cover
from visage.events.naming import generate_event_name

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/events", tags=["events"])


def _serialize_event(event: Event) -> dict:
    """Serialize an Event to a JSON-friendly dict."""
    return {
        "event_id": event.event_id,
        "name": event.name,
        "start_time": event.start_time.isoformat(),
        "end_time": event.end_time.isoformat(),
        "photo_count": event.photo_count,
        "photo_paths": event.photo_paths,
        "cover_path": event.cover_path,
        "location_name": event.location_name,
        "is_multi_day": event.is_multi_day,
    }


@router.get("")
async def list_events(request: Request) -> dict:
    """List all auto-detected events for the current workspace."""
    ws = getattr(request.app.state, "workspace", None)
    if ws is None:
        raise HTTPException(503, "Workspace not ready")

    # Collect all photo paths from workspace
    all_paths: list[str] = []
    for _, photos in ws._cluster_mapping.items():
        all_paths.extend(photos)
    # Add noise photos
    for photo in ws.noise_photos:
        all_paths.append(photo)

    if not all_paths:
        return {"events": [], "total": 0}

    # Extract timestamps and cluster
    timestamps = extract_timestamps(all_paths)
    events = cluster_by_time(timestamps)

    # Generate names and select covers
    face_quality_map = _build_face_quality_map(ws)
    for event in events:
        event.name = generate_event_name(event)
        event.cover_path = select_event_cover(event, face_quality_map)

    return {
        "events": [_serialize_event(e) for e in events],
        "total": len(events),
    }


@router.get("/{event_id}")
async def get_event(event_id: str, request: Request) -> dict:
    """Get details for a specific event."""
    ws = getattr(request.app.state, "workspace", None)
    if ws is None:
        raise HTTPException(503, "Workspace not ready")

    # Recompute events (could be cached in future)
    all_paths: list[str] = []
    for _, photos in ws._cluster_mapping.items():
        all_paths.extend(photos)
    for photo in ws.noise_photos:
        all_paths.append(photo)

    timestamps = extract_timestamps(all_paths)
    events = cluster_by_time(timestamps)

    face_quality_map = _build_face_quality_map(ws)
    for event in events:
        event.name = generate_event_name(event)
        event.cover_path = select_event_cover(event, face_quality_map)

    for event in events:
        if event.event_id == event_id:
            return _serialize_event(event)

    raise HTTPException(404, f"Event {event_id!r} not found")


@router.get("/people-intersection")
async def people_intersection(
    request: Request,
    person_ids: str = "",
) -> dict:
    """Find photos containing ALL specified people (AND query).

    Args:
        person_ids: Comma-separated cluster IDs, e.g. "0,3,5"
    """
    ws = getattr(request.app.state, "workspace", None)
    if ws is None:
        raise HTTPException(503, "Workspace not ready")

    if not person_ids:
        return {"photo_paths": [], "total": 0}

    requested_ids = set()
    for pid in person_ids.split(","):
        try:
            requested_ids.add(int(pid.strip()))
        except ValueError:
            continue

    if not requested_ids:
        return {"photo_paths": [], "total": 0}

    # For each photo, find which people (clusters) appear in it
    photo_to_clusters: dict[str, set[int]] = {}
    for cid, photos in ws._cluster_mapping.items():
        for photo_path in photos:
            if photo_path not in photo_to_clusters:
                photo_to_clusters[photo_path] = set()
            photo_to_clusters[photo_path].add(cid)

    # Find photos containing ALL requested people
    result_paths = [
        path for path, clusters in photo_to_clusters.items()
        if requested_ids.issubset(clusters)
    ]

    return {"photo_paths": result_paths, "total": len(result_paths)}


def _build_face_quality_map(ws) -> dict[str, float]:
    """Build a mapping of photo_path → max face quality score."""
    quality_map: dict[str, float] = {}
    for result in ws.image_results:
        if result.error or not result.faces:
            continue
        max_q = 0.0
        for face in result.faces:
            if face.quality is not None:
                max_q = max(max_q, face.quality)
        if max_q > 0:
            quality_map[result.path] = max_q
    return quality_map
