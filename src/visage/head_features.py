"""Head crop perceptual features — color histogram, texture, and shape cues.

Extracts lightweight features from an expanded head crop (including hair and
some background) to distinguish people with similar facial geometry but
different hairstyles, accessories, and head shape.
"""

from __future__ import annotations

import numpy as np

from .models import FaceBox

# Number of bins for the hue channel histogram
_HUE_BINS = 16
# Number of bins for the saturation channel histogram
_SAT_BINS = 8
# Expected feature vector length: hue histogram + sat histogram + mean hsv + edge density
FEATURE_DIM = _HUE_BINS + _SAT_BINS + 3 + 1  # = 28


def extract_head_features(
    image: np.ndarray,
    face_box: FaceBox,
    expand_ratio: float = 2.0,
) -> np.ndarray:
    """Extract perceptual features from the head region around a detected face.

    Expands the face bounding box to capture hair, ears, and upper body,
    then extracts color and texture features from the upper portion
    (hair region) and the full head crop.

    Args:
        image: RGB numpy array of the full image, shape (H, W, 3).
        face_box: Detected face bounding box in pixel coordinates.
        expand_ratio: How much to expand the box in each direction (default 2x).

    Returns:
        Feature vector of shape (FEATURE_DIM,).
    """
    img_h, img_w = image.shape[:2]

    # Compute expanded bounding box
    face_w = face_box.right - face_box.left
    face_h = face_box.bottom - face_box.top

    # Expand outward, with extra expansion upward to capture hair
    pad_x = int(face_w * (expand_ratio - 1.0) / 2)
    pad_y_top = int(face_h * (expand_ratio - 1.0) * 0.7)  # more upward
    pad_y_bottom = int(face_h * (expand_ratio - 1.0) * 0.3)

    x1 = max(0, face_box.left - pad_x)
    y1 = max(0, face_box.top - pad_y_top)
    x2 = min(img_w, face_box.right + pad_x)
    y2 = min(img_h, face_box.bottom + pad_y_bottom)

    if x2 <= x1 or y2 <= y1:
        return np.zeros(FEATURE_DIM, dtype=np.float64)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(FEATURE_DIM, dtype=np.float64)

    # Convert to HSV
    hsv = _rgb_to_hsv(crop)

    # Focus on the upper 40% of the crop (hair region)
    hair_end = max(1, int(hsv.shape[0] * 0.4))
    hair_region = hsv[:hair_end]

    # 1. Hue histogram from hair region (16 bins)
    hue_hist = _histogram(hair_region[:, :, 0], _HUE_BINS, 0.0, 360.0)

    # 2. Saturation histogram from hair region (8 bins)
    sat_hist = _histogram(hair_region[:, :, 1], _SAT_BINS, 0.0, 1.0)

    # 3. Mean HSV of hair region
    mean_h = float(hair_region[:, :, 0].mean()) / 360.0  # normalize to [0,1]
    mean_s = float(hair_region[:, :, 1].mean())
    mean_v = float(hair_region[:, :, 2].mean())

    # 4. Edge density of full head crop (texture proxy)
    gray = _to_grayscale(crop)
    edge_density = _edge_density(gray)

    return np.concatenate([
        hue_hist,
        sat_hist,
        np.array([mean_h, mean_s, mean_v]),
        np.array([edge_density]),
    ])


def _rgb_to_hsv(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to HSV. Returns float64 array with H in [0,360], S,V in [0,1]."""
    rgb = image.astype(np.float64) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    c_max = np.maximum(np.maximum(r, g), b)
    c_min = np.minimum(np.minimum(r, g), b)
    delta = c_max - c_min

    # Hue
    h = np.zeros_like(c_max)
    mask_r = (c_max == r) & (delta > 0)
    mask_g = (c_max == g) & (delta > 0)
    mask_b = (c_max == b) & (delta > 0)
    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)

    # Saturation
    s = np.where(c_max > 0, delta / np.maximum(c_max, 1e-10), 0.0)

    hsv = np.stack([h, s, c_max], axis=-1)
    return hsv


def _histogram(channel: np.ndarray, bins: int, lo: float, hi: float) -> np.ndarray:
    """Compute a normalized histogram of a single channel."""
    flat = channel.ravel()
    counts, _ = np.histogram(flat, bins=bins, range=(lo, hi))
    total = counts.sum()
    if total > 0:
        counts = counts.astype(np.float64) / total
    return counts.astype(np.float64)


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB to grayscale."""
    return (
        0.299 * image[:, :, 0].astype(np.float64)
        + 0.587 * image[:, :, 1].astype(np.float64)
        + 0.114 * image[:, :, 2].astype(np.float64)
    )


def _edge_density(gray: np.ndarray) -> float:
    """Compute edge density using gradient magnitude (Sobel-like).

    Returns a value in [0, 1] indicating how much edge content exists.
    """
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0

    # Simple horizontal and vertical gradient
    gx = np.abs(gray[1:, :] - gray[:-1, :])
    gy = np.abs(gray[:, 1:] - gray[:, :-1])

    # Average gradient magnitude
    gx_mean = float(gx.mean())
    gy_mean = float(gy.mean())

    # Normalize to roughly [0, 1] — typical gradients are 0-50 for natural images
    return min(1.0, (gx_mean + gy_mean) / 100.0)
