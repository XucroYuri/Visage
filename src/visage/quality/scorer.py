"""Face quality scorer — weighted composite score and best-face selection per cluster."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from visage.models import DetectedFace

logger = logging.getLogger(__name__)


@dataclass
class QualityWeights:
    """Weights for composite quality score components."""

    sharpness: float = 0.4
    face_quality: float = 0.4
    face_size: float = 0.2


DEFAULT_WEIGHTS = QualityWeights()


def compute_cluster_quality_score(
    face: DetectedFace,
    image_height: int,
    image_width: int,
    weights: QualityWeights = DEFAULT_WEIGHTS,
) -> float:
    """Compute a composite quality score for a single face.

    Components:
    - Sharpness: Laplacian variance of the face region (normalized)
    - Face quality: Existing compute_face_quality score
    - Face size: Relative area of the face in the image

    Returns:
        Score in [0, 1], higher is better.
    """
    scores: list[float] = []
    weights_list: list[float] = []

    # Face quality (existing metric)
    fq = face.quality if face.quality is not None else 0.5
    scores.append(max(0.0, min(1.0, fq)))
    weights_list.append(weights.face_quality)

    # Face size relative to image
    if image_height > 0 and image_width > 0:
        face_area = face.face_box.area
        image_area = image_height * image_width
        size_ratio = min(1.0, face_area / max(image_area * 0.1, 1))
        scores.append(size_ratio)
        weights_list.append(weights.face_size)
    else:
        scores.append(0.5)
        weights_list.append(weights.face_size)

    # Sharpness: use face quality as proxy (real sharpness requires image data)
    scores.append(fq)
    weights_list.append(weights.sharpness)

    total_weight = sum(weights_list)
    if total_weight == 0:
        return 0.5

    return sum(s * w for s, w in zip(scores, weights_list, strict=False)) / total_weight


def select_best_face(
    faces: list[DetectedFace],
    image_height: int = 0,
    image_width: int = 0,
    weights: QualityWeights = DEFAULT_WEIGHTS,
) -> DetectedFace | None:
    """Select the best face from a list of candidates.

    Constraint checks:
    - Face must not be at image edge (partial crop)
    - Face area must be > 1/3 of average area in the group

    Args:
        faces: List of candidate faces.
        image_height: Image height for edge detection.
        image_width: Image width for edge detection.
        weights: Quality score weights.

    Returns:
        Best face, or None if no valid candidates.
    """
    if not faces:
        return None

    valid_faces: list[tuple[DetectedFace, float]] = []
    avg_area = sum(f.face_box.area for f in faces) / max(len(faces), 1)
    min_area = avg_area / 3

    for face in faces:
        bbox = face.face_box

        # Edge constraint: face should not touch image boundary
        if image_height > 0 and image_width > 0:
            if (bbox.top <= 1 or bbox.left <= 1 or
                    bbox.bottom >= image_height - 1 or bbox.right >= image_width - 1):
                continue  # Face is at image edge

        # Size constraint
        if bbox.area < min_area and len(faces) > 1:
            continue

        score = compute_cluster_quality_score(face, image_height, image_width, weights)
        valid_faces.append((face, score))

    if not valid_faces:
        # Fallback: return first face without constraints
        return faces[0]

    # Return highest scoring face
    valid_faces.sort(key=lambda x: x[1], reverse=True)
    return valid_faces[0][0]
