"""Face alignment via affine transformation.

Aligns a detected face to a canonical position using 5 facial landmarks,
improving embedding consistency across varying head poses.
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from .models import DetectedFace

logger = logging.getLogger(__name__)

# Canonical 5-point template (112×112) — standard face alignment positions.
# Coordinates: (left_eye, right_eye, nose_tip, left_mouth, right_mouth)
_ALIGN_SIZE = 112
_CANONICAL_LANDMARKS = np.array([
    [38.0, 42.0],   # left eye
    [74.0, 42.0],   # right eye
    [56.0, 62.0],   # nose tip
    [38.0, 74.0],   # left mouth corner
    [74.0, 74.0],   # right mouth corner
], dtype=np.float64)

# Minimum required eye distance (in pixels) to perform alignment.
# Below this the face is too small for reliable landmark-based alignment.
_MIN_EYE_DISTANCE = 20.0


def _compute_affine_inverse(
    src_pts: np.ndarray, dst_pts: np.ndarray,
) -> np.ndarray | None:
    """Compute the inverse affine matrix (dst → src) via least-squares.

    Uses 5 point pairs to fit a 2×3 affine transformation. Returns the
    inverse matrix coefficients (a,b,c,d,e,f) for use with PIL.Image.transform
    with method=Image.AFFINE.

    The forward transform maps src → dst. PIL needs the inverse transform
    that maps output pixel (dst coordinate) → input pixel (src coordinate).
    """
    if len(src_pts) < 3:
        return None

    # Build the design matrix: each row is [x, y, 1]
    n = len(src_pts)
    M = np.hstack([src_pts, np.ones((n, 1))])  # (n, 3)

    # Solve: M @ A_inv^T = dst_pts where A_inv is the 2×3 inverse matrix.
    # dst_pts[:, 0] gives x-inverse params (a, b, c)
    # dst_pts[:, 1] gives y-inverse params (d, e, f)
    try:
        params_x, _, _, _ = np.linalg.lstsq(M, dst_pts[:, 0], rcond=None)
        params_y, _, _, _ = np.linalg.lstsq(M, dst_pts[:, 1], rcond=None)
    except np.linalg.LinAlgError:
        return None

    # Return PIL order: (a, b, c, d, e, f)
    return np.array([
        params_x[0], params_x[1], params_x[2],
        params_y[0], params_y[1], params_y[2],
    ], dtype=np.float64)


def align_face(
    image: np.ndarray,
    face: DetectedFace,
    align_size: int = _ALIGN_SIZE,
) -> np.ndarray | None:
    """Align a detected face to the canonical template.

    Extracts the face region from the full image, computes the affine
    transform using the 5 facial landmarks, and warps the face to a
    canonical 112×112 position.

    Args:
        image: RGB numpy array of the full image, shape (H, W, 3).
        face: DetectedFace with face_box and landmarks_5 set.
        align_size: Size of the output aligned image (square).

    Returns:
        Aligned face as (align_size, align_size, 3) RGB numpy array,
        or None if landmarks are unavailable or alignment fails.
    """
    if face.landmarks_5 is None:
        return None

    src_pts = np.array(face.landmarks_5, dtype=np.float64)

    # Check minimum eye distance
    eye_dist = np.linalg.norm(src_pts[1] - src_pts[0])
    if eye_dist < _MIN_EYE_DISTANCE:
        return None

    # Compute inverse affine matrix: canonical template → source pixels
    affine_inv = _compute_affine_inverse(src_pts, _CANONICAL_LANDMARKS)
    if affine_inv is None:
        return None

    # Crop face region with generous padding for the warp
    h, w = image.shape[:2]
    fw = face.face_box.width
    fh = face.face_box.height
    pad_x = int(fw * 0.5)
    pad_y = int(fh * 0.5)
    x1 = max(0, face.face_box.left - pad_x)
    y1 = max(0, face.face_box.top - pad_y)
    x2 = min(w, face.face_box.right + pad_x)
    y2 = min(h, face.face_box.bottom + pad_y)

    crop = image[y1:y2, x1:x2]

    # Adjust the inverse transform to account for the crop offset.
    # The inverse matrix maps canonical coords → full-image coords.
    # We need it to map canonical coords → crop coords.
    affine_inv[2] -= x1  # c (x-offset)
    affine_inv[5] -= y1  # f (y-offset)

    # Validate crop is non-empty
    if crop.shape[0] < 2 or crop.shape[1] < 2:
        return None

    try:
        pil_crop = Image.fromarray(crop)
        # PIL AFFINE expects (a, b, c, d, e, f) where:
        #   src_x = a * dst_x + b * dst_y + c
        #   src_y = d * dst_x + e * dst_y + f
        # With our inverse matrix, (dst_x,dst_y) = canonical coords,
        # (src_x,src_y) = crop coords.
        aligned = pil_crop.transform(
            (align_size, align_size),
            Image.AFFINE,
            data=tuple(affine_inv.tolist()),
            resample=Image.BICUBIC,
        )
        return np.array(aligned)
    except Exception:
        logger.debug("Face alignment failed", exc_info=True)
        return None
