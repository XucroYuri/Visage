"""Tests for face quality scorer."""

from __future__ import annotations

from visage.models import DetectedFace, FaceBox
from visage.quality.scorer import (
    QualityWeights,
    compute_cluster_quality_score,
    select_best_face,
)


def _make_face(
    top: int = 10,
    right: int = 110,
    bottom: int = 110,
    left: int = 10,
    quality: float = 0.8,
) -> DetectedFace:
    return DetectedFace(
        face_box=FaceBox(top=top, right=right, bottom=bottom, left=left),
        confidence=0.9,
        quality=quality,
        image_path="/tmp/test.jpg",
        face_index=0,
    )


class TestComputeClusterQualityScore:
    def test_high_quality_face(self):
        face = _make_face(quality=0.9)
        score = compute_cluster_quality_score(face, 480, 640)
        assert 0.0 <= score <= 1.0
        assert score > 0.5

    def test_low_quality_face(self):
        face = _make_face(quality=0.1)
        score = compute_cluster_quality_score(face, 480, 640)
        assert score < 0.6

    def test_no_quality_defaults(self):
        face = _make_face()
        face.quality = None
        score = compute_cluster_quality_score(face, 480, 640)
        assert 0.0 <= score <= 1.0

    def test_custom_weights(self):
        face = _make_face(quality=0.9)
        weights = QualityWeights(sharpness=1.0, face_quality=0.0, face_size=0.0)
        score = compute_cluster_quality_score(face, 480, 640, weights=weights)
        assert 0.0 <= score <= 1.0


class TestSelectBestFace:
    def test_single_face(self):
        face = _make_face()
        result = select_best_face([face], 480, 640)
        assert result is face

    def test_empty_list(self):
        assert select_best_face([], 480, 640) is None

    def test_selects_higher_quality(self):
        low = _make_face(quality=0.3)
        high = _make_face(top=200, right=300, bottom=300, left=200, quality=0.9)
        result = select_best_face([low, high], 480, 640)
        assert result is high

    def test_edge_face_skipped(self):
        """Face touching image boundary should be skipped."""
        edge_face = _make_face(top=0, right=100, bottom=100, left=0, quality=0.99)
        good_face = _make_face(top=200, right=300, bottom=300, left=200, quality=0.5)
        result = select_best_face([edge_face, good_face], 480, 640)
        assert result is good_face

    def test_tiny_face_skipped(self):
        """Very small face should be skipped when better options exist."""
        tiny = _make_face(top=200, right=210, bottom=210, left=200, quality=0.99)
        normal = _make_face(top=200, right=300, bottom=300, left=200, quality=0.5)
        result = select_best_face([tiny, normal], 480, 640)
        assert result is normal

    def test_fallback_when_all_edge(self):
        """If all faces are at edge, fallback to first."""
        edge1 = _make_face(top=0, right=100, bottom=100, left=0, quality=0.9)
        edge2 = _make_face(top=0, right=200, bottom=100, left=100, quality=0.3)
        result = select_best_face([edge1, edge2], 100, 200)
        assert result is edge1

    def test_without_image_dims(self):
        """Should work without image dimensions (no edge check)."""
        face = _make_face(quality=0.8)
        result = select_best_face([face])
        assert result is face
