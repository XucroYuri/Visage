"""Tests for visage.quality — face quality assessment."""

from __future__ import annotations

import numpy as np

from visage.models import FaceBox
from visage.quality import (
    _laplacian_variance,
    _to_grayscale,
    compute_combined_quality,
    compute_face_quality,
    compute_landmark_quality,
)

# ── compute_face_quality ─────────────────────────────────────────


class TestComputeFaceQuality:
    def test_returns_float_between_0_and_1(self):
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        face_box = FaceBox(top=20, right=180, bottom=180, left=20)
        score = compute_face_quality(image, face_box)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_sharp_face_high_score(self):
        # Sharp edges → high Laplacian variance
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[50:150, 50:150] = 255  # sharp rectangle
        face_box = FaceBox(top=40, right=160, bottom=160, left=40)
        score = compute_face_quality(image, face_box)
        assert score > 0.3

    def test_uniform_region_low_score(self):
        # Uniform color → low Laplacian variance
        image = np.full((200, 200, 3), 128, dtype=np.uint8)
        face_box = FaceBox(top=50, right=150, bottom=150, left=50)
        score = compute_face_quality(image, face_box)
        assert score < 0.5

    def test_zero_area_returns_zero(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        face_box = FaceBox(top=50, right=50, bottom=50, left=50)
        score = compute_face_quality(image, face_box)
        assert score == 0.0

    def test_out_of_bounds_clamped(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Box extends outside image
        face_box = FaceBox(top=-10, right=120, bottom=120, left=-10)
        score = compute_face_quality(image, face_box)
        assert isinstance(score, float)

    def test_small_face_lower_score_than_large(self):
        # Create a sharp image
        image = np.zeros((500, 500, 3), dtype=np.uint8)
        image[100:400, 100:400] = 255

        # Large face
        large_box = FaceBox(top=80, right=420, bottom=420, left=80)
        large_score = compute_face_quality(image, large_box)

        # Tiny face (same content, but smaller portion of image)
        tiny_box = FaceBox(top=100, right=120, bottom=120, left=100)
        tiny_score = compute_face_quality(image, tiny_box)

        # Large face should have higher size contribution
        assert large_score >= tiny_score


# ── _to_grayscale ────────────────────────────────────────────────


class TestToGrayscale:
    def test_converts_rgb(self):
        image = np.array([[[100, 150, 200]]], dtype=np.uint8)
        gray = _to_grayscale(image)
        expected = 0.299 * 100 + 0.587 * 150 + 0.114 * 200
        assert abs(gray[0, 0] - expected) < 1.0

    def test_already_gray(self):
        gray_in = np.array([[50.0, 100.0]])
        gray_out = _to_grayscale(gray_in)
        np.testing.assert_array_equal(gray_out, gray_in)


# ── _laplacian_variance ──────────────────────────────────────────


class TestLaplacianVariance:
    def test_uniform_image_low_variance(self):
        uniform = np.full((50, 50), 128.0)
        var = _laplacian_variance(uniform)
        assert var < 1.0

    def test_edge_image_high_variance(self):
        edge = np.zeros((50, 50))
        edge[:, 25:] = 255.0
        var = _laplacian_variance(edge)
        assert var > 10.0

    def test_returns_float(self):
        gray = np.random.randn(30, 30)
        var = _laplacian_variance(gray)
        assert isinstance(var, float)
        assert var >= 0.0


# ── compute_landmark_quality ──────────────────────────────────────


class TestComputeLandmarkQuality:
    def test_none_landmarks_returns_partial(self):
        score = compute_landmark_quality(None)
        assert score == 0.4

    def test_partial_landmarks_returns_partial(self):
        score = compute_landmark_quality([(10, 10), (20, 10)])
        assert score == 0.4

    def test_well_formed_landmarks_high_score(self):
        """Symmetrical eyes, vertical ordering correct."""
        landmarks = [
            (50, 50),   # left eye
            (150, 50),  # right eye
            (100, 100),  # nose
            (70, 180),   # left mouth
            (130, 180),  # right mouth
        ]
        score = compute_landmark_quality(landmarks)
        assert 0.5 <= score <= 1.0

    def test_poor_vertical_ordering_low_score(self):
        """Nose above eyes — not anatomically plausible."""
        landmarks = [
            (50, 100),  # left eye
            (150, 100), # right eye
            (100, 50),  # nose above eyes
            (70, 180),  # left mouth
            (130, 180), # right mouth
        ]
        score = compute_landmark_quality(landmarks)
        assert score < 0.5

    def test_eyes_too_close_low_score(self):
        """Identical eye positions — not anatomically plausible."""
        landmarks = [
            (100, 50),  # left eye
            (100, 50),  # right eye at same position
            (100, 100), # nose
            (70, 180),  # left mouth
            (130, 180), # right mouth
        ]
        score = compute_landmark_quality(landmarks)
        assert score < 0.5

    def test_valid_but_asymmetric(self):
        """Eyes at different y heights."""
        landmarks = [
            (50, 50),   # left eye
            (150, 80),  # right eye lower
            (100, 100), # nose
            (70, 180),  # left mouth
            (130, 180), # right mouth
        ]
        score = compute_landmark_quality(landmarks)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ── compute_combined_quality ──────────────────────────────────────


class TestComputeCombinedQuality:
    def test_combines_legacy_and_fiqa(self):
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        face_box = FaceBox(top=40, right=160, bottom=160, left=40)
        landmarks = [
            (50, 50),
            (150, 50),
            (100, 100),
            (70, 180),
            (130, 180),
        ]
        score = compute_combined_quality(image, face_box, landmarks_5=landmarks)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_no_landmarks_still_works(self):
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        face_box = FaceBox(top=40, right=160, bottom=160, left=40)
        score = compute_combined_quality(image, face_box, landmarks_5=None)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_pure_legacy_weight(self):
        """fiqa_weight=0 means legacy only."""
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        face_box = FaceBox(top=40, right=160, bottom=160, left=40)
        legacy_only = compute_combined_quality(image, face_box, fiqa_weight=0.0)
        legacy = compute_face_quality(image, face_box)
        assert abs(legacy_only - legacy) < 1e-6
