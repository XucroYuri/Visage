"""Shared Non-Maximum Suppression for duplicate detection removal.

Extracted from detector.py so all detection backends can use it.
"""

from __future__ import annotations

import logging

from visage.models import FaceBox

logger = logging.getLogger(__name__)


def _nms(
    detections: list[tuple[FaceBox, float, list[tuple[float, float]] | None]],
    iou_threshold: float = 0.5,
) -> list[tuple[FaceBox, float, list[tuple[float, float]] | None]]:
    """Non-Maximum Suppression: remove duplicate detections for the same face.

    Sorts by confidence (highest first), then greedily selects the highest
    confidence box and suppresses any remaining box with IoU >= threshold.

    Args:
        detections: List of (FaceBox, confidence, landmarks_5) tuples.
        iou_threshold: IoU threshold for suppression (0.0-1.0).

    Returns:
        Filtered list with duplicates removed.
    """
    if not detections:
        return []

    def _iou(a: FaceBox, b: FaceBox) -> float:
        """Compute Intersection-over-Union of two bounding boxes."""
        x_left = max(a.left, b.left)
        y_top = max(a.top, b.top)
        x_right = min(a.right, b.right)
        y_bottom = min(a.bottom, b.bottom)

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area_a = a.area
        area_b = b.area
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    # Sort by confidence descending
    sorted_detections = sorted(detections, key=lambda x: x[1], reverse=True)
    keep: list[tuple[FaceBox, float, list[tuple[float, float]] | None]] = []

    while sorted_detections:
        best = sorted_detections.pop(0)
        keep.append(best)
        # Suppress remaining boxes with IoU >= threshold
        sorted_detections = [
            d for d in sorted_detections if _iou(best[0], d[0]) < iou_threshold
        ]

    logger.debug("NMS: %d → %d detections", len(detections), len(keep))
    return keep
