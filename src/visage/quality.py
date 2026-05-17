"""Face quality assessment — blur detection via Laplacian variance and size ratio."""

from __future__ import annotations

import logging

import numpy as np

from .models import FaceBox

logger = logging.getLogger(__name__)


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
