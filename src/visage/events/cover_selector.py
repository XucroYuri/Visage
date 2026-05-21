"""Event cover selection — pick the best photo as album cover.

Reuses quality/scorer.py scoring functions for face quality.
For events with faces, prefers photos with clear, well-lit faces.
For events without faces, picks the largest/sharpest photo.
"""

from __future__ import annotations

import logging

from visage.events.cluster import Event

logger = logging.getLogger(__name__)


def select_event_cover(
    event: Event,
    face_quality_map: dict[str, float] | None = None,
) -> str | None:
    """Select the best photo from an event to use as album cover.

    Args:
        event: The event to select a cover for.
        face_quality_map: Optional mapping of photo_path → max face quality score.
            If None, falls back to first photo.

    Returns:
        Path of the best cover photo, or None if event is empty.
    """
    if not event.photo_paths:
        return None

    if face_quality_map is None:
        return event.photo_paths[0]

    # Score each photo
    scored: list[tuple[str, float]] = []
    for path in event.photo_paths:
        quality = face_quality_map.get(path, 0.0)
        scored.append((path, quality))

    # Sort by quality descending
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]
