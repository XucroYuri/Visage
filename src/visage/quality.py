"""Face quality assessment — blur detection via Laplacian variance, size ratio, and FIQA."""

from __future__ import annotations

import logging

import numpy as np

from .models import FaceBox

logger = logging.getLogger(__name__)

# ── FIQA: landmark-based heuristic scoring ──────────────────────
# Evaluates face quality from the spatial distribution of 5 facial
# landmarks (eyes, nose, mouth corners). A well-formed face has:
# - Symmetrical left/right eye positions (similar y coordinates)
# - Nose below eyes and above mouth
# - Mouth corners below nose
# - Reasonable aspect ratios


def compute_landmark_quality(
    landmarks_5: list[tuple[float, float]] | None,
) -> float:
    """Compute a quality score [0, 1] based on facial landmark geometry.

    A well-structured face produces landmarks with predictable spatial
    relationships. This function checks:
    - All 5 landmarks present (penalty for missing)
    - Left and right eyes at similar height (symmetry)
    - Nose below eyes, mouth below nose (vertical ordering)
    - Reasonable inter-landmark distances

    Args:
        landmarks_5: List of 5 (x, y) pixel-coordinate landmarks:
            (left_eye, right_eye, nose_tip, left_mouth, right_mouth).

    Returns:
        Quality score from 0.0 (poor) to 1.0 (excellent).
    """
    if landmarks_5 is None or len(landmarks_5) < 5:
        return 0.4  # partial credit for detection-only (no landmarks)

    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks_5

    # 1. Check all landmarks have valid coordinates
    for pt in landmarks_5:
        if pt is None:
            return 0.4

    # 2. Eye symmetry: left and right eyes should be at similar y
    eye_y_diff = abs(left_eye[1] - right_eye[1])
    eye_dist = ((left_eye[0] - right_eye[0]) ** 2 + (left_eye[1] - right_eye[1]) ** 2) ** 0.5
    if eye_dist < 1.0:
        return 0.3  # eyes too close — likely bad detection

    # Normalize eye y-diff by eye distance
    eye_symmetry = max(0.0, 1.0 - min(eye_y_diff / (eye_dist + 1e-6), 1.0))

    # 3. Vertical ordering: nose below eyes, mouth below nose
    eye_y = (left_eye[1] + right_eye[1]) / 2.0
    mouth_y = (left_mouth[1] + right_mouth[1]) / 2.0

    ordering_ok = (nose[1] > eye_y) and (mouth_y > nose[1])
    if not ordering_ok:
        return 0.3  # severely disordered — likely poor detection

    # 4. Nose-to-mouth vs eye distance ratio (should be ~0.5-1.0)
    nose_mouth_dist = abs(mouth_y - nose[1])
    ratio = nose_mouth_dist / (eye_dist + 1e-6)

    if ratio < 0.2 or ratio > 2.0:
        ratio_score = 0.5
    elif 0.3 <= ratio <= 1.5:
        ratio_score = 1.0
    else:
        ratio_score = 0.7

    # Combine: symmetry 40%, ratio 60%
    return float(0.4 * eye_symmetry + 0.6 * ratio_score)


def compute_face_quality(
    image: np.ndarray,
    face_box: FaceBox,
) -> float:
    """Compute a quality score [0, 1] for a detected face region.

    Based on:
    - Laplacian variance of the face region (blur detection)
    - Face size relative to image size

    Args:
        image: RGB numpy array of the full image, shape (H, W, 3).
        face_box: Bounding box of the face in pixel coordinates.

    Returns:
        Quality score between 0.0 (poor) and 1.0 (excellent).
    """
    img_h, img_w = image.shape[:2]

    # Clamp face box to image bounds
    x1 = max(0, face_box.left)
    y1 = max(0, face_box.top)
    x2 = min(img_w, face_box.right)
    y2 = min(img_h, face_box.bottom)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    # Extract face region and convert to grayscale
    face_region = image[y1:y2, x1:x2]
    if face_region.size == 0:
        return 0.0

    gray = _to_grayscale(face_region)

    # 1. Blur score: Laplacian variance
    blur_score = _laplacian_variance(gray)

    # 2. Size score: face area / image area ratio
    face_area = (x2 - x1) * (y2 - y1)
    img_area = img_h * img_w
    size_ratio = face_area / img_area if img_area > 0 else 0.0

    # Normalize blur score to [0, 1] using sigmoid-like mapping
    # Typical sharp face: laplacian variance > 100-500
    # Typical blurry face: laplacian variance < 20-50
    blur_quality = min(1.0, blur_score / 200.0)

    # Normalize size ratio to [0, 1]
    # A face covering >5% of the image is large; <0.5% is tiny
    size_quality = min(1.0, size_ratio / 0.05) if size_ratio > 0.001 else 0.0

    # Weighted combination (blur is more important)
    return float(0.7 * blur_quality + 0.3 * size_quality)


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to grayscale using luminosity method."""
    if image.ndim == 2:
        return image.astype(np.float64)
    # RGB -> gray: 0.299*R + 0.587*G + 0.114*B
    return (
        0.299 * image[:, :, 0].astype(np.float64)
        + 0.587 * image[:, :, 1].astype(np.float64)
        + 0.114 * image[:, :, 2].astype(np.float64)
    )


def compute_combined_quality(
    image: np.ndarray,
    face_box: FaceBox,
    landmarks_5: list[tuple[float, float]] | None = None,
    fiqa_weight: float = 0.4,
) -> float:
    """Compute a fused quality score combining legacy metrics and FIQA.

    Legacy (Laplacian + size) assesses image quality (blur, resolution).
    FIQA (landmark geometry) assesses face structure quality.
    These are complementary: a sharp image of a poorly-detected face
    scores well on legacy but poorly on FIQA.

    Args:
        image: RGB numpy array of the full image.
        face_box: Bounding box of the face.
        landmarks_5: Optional 5-point facial landmarks for FIQA scoring.
        fiqa_weight: Weight for FIQA score (0.0 = legacy only, 1.0 = FIQA only).

    Returns:
        Quality score between 0.0 (poor) and 1.0 (excellent).
    """
    legacy_score = compute_face_quality(image, face_box)
    fiqa_score = compute_landmark_quality(landmarks_5)

    # Weighted fusion: FIQA gets configurable weight (default 0.4)
    combined = (1.0 - fiqa_weight) * legacy_score + fiqa_weight * fiqa_score
    return float(np.clip(combined, 0.0, 1.0))


def _laplacian_variance(gray: np.ndarray) -> float:
    """Compute Laplacian variance as a blur measure.

    Higher values indicate sharper images. Uses a pure numpy
    implementation (no OpenCV dependency).
    """
    # Laplacian kernel: detects edges/gradient magnitude
    kernel = np.array(
        [
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0],
        ],
        dtype=np.float64,
    )

    # Pad image for convolution
    padded = np.pad(gray, 1, mode="edge")

    # Manual 2D convolution (kernel is small, no need for FFT)
    h, w = gray.shape
    laplacian = np.zeros_like(gray)
    for dy in range(3):
        for dx in range(3):
            laplacian += kernel[dy, dx] * padded[dy : dy + h, dx : dx + w]

    return float(laplacian.var())
