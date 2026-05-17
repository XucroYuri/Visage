"""Tests for visage.models — pure dataclass tests, no mocking needed."""

from __future__ import annotations

import numpy as np
import pytest

from visage.models import (
    ClusterResult,
    DetectedFace,
    FaceBox,
    ImageResult,
    OrganizePlan,
    PipelineResult,
)

# ── FaceBox ───────────────────────────────────────────────────────


class TestFaceBox:
    def test_width(self, sample_face_box: FaceBox):
        assert sample_face_box.width == 100

    def test_height(self, sample_face_box: FaceBox):
        assert sample_face_box.height == 100

    def test_area(self, sample_face_box: FaceBox):
        assert sample_face_box.area == 10000

    def test_wide_box(self, wide_face_box: FaceBox):
        assert wide_face_box.width == 200
        assert wide_face_box.height == 50
        assert wide_face_box.area == 10000

    def test_narrow_box(self, narrow_face_box: FaceBox):
        assert narrow_face_box.width == 50
        assert narrow_face_box.height == 200

    def test_zero_dimensions(self):
        box = FaceBox(top=50, right=50, bottom=50, left=50)
        assert box.width == 0
        assert box.height == 0
        assert box.area == 0

    def test_negative_dimensions(self):
        # Inverted box — no validation, just verify property behavior
        box = FaceBox(top=110, right=10, bottom=10, left=110)
        assert box.width == -100
        assert box.height == -100

    def test_to_face_recognition_format(self, sample_face_box: FaceBox):
        result = sample_face_box.to_face_recognition_format()
        assert result == (10, 110, 110, 10)
        assert isinstance(result, tuple)

    def test_frozen(self, sample_face_box: FaceBox):
        with pytest.raises(AttributeError):
            sample_face_box.top = 999  # type: ignore[misc]

    def test_frozen_hashable(self, sample_face_box: FaceBox):
        # Frozen dataclasses are hashable
        s = {sample_face_box}
        assert len(s) == 1


# ── DetectedFace ──────────────────────────────────────────────────


class TestDetectedFace:
    def test_defaults(self, sample_face_box: FaceBox):
        face = DetectedFace(face_box=sample_face_box, confidence=0.9)
        assert face.embedding is None
        assert face.image_path == ""
        assert face.face_index == 0

    def test_with_embedding(self, sample_face_box: FaceBox):
        emb = np.zeros(128)
        face = DetectedFace(face_box=sample_face_box, confidence=0.8, embedding=emb)
        assert face.embedding is not None
        assert len(face.embedding) == 128

    def test_all_fields(self, sample_detected_face: DetectedFace):
        assert sample_detected_face.confidence == 0.95
        assert sample_detected_face.image_path == "/tmp/test.jpg"
        assert sample_detected_face.face_index == 0
        assert sample_detected_face.embedding is not None


# ── ImageResult ───────────────────────────────────────────────────


class TestImageResult:
    def test_defaults(self):
        result = ImageResult(path="/tmp/test.jpg")
        assert result.faces == []
        assert result.error is None
        assert result.skipped is False

    def test_with_error(self, error_image_result: ImageResult):
        assert error_image_result.error == "load failed"
        assert error_image_result.faces == []
        assert error_image_result.skipped is False

    def test_skipped(self, empty_image_result: ImageResult):
        assert empty_image_result.skipped is True
        assert empty_image_result.faces == []
        assert empty_image_result.error is None

    def test_with_faces(self, sample_image_result: ImageResult):
        assert len(sample_image_result.faces) == 1
        assert sample_image_result.path == "/tmp/test.jpg"


# ── ClusterResult ─────────────────────────────────────────────────


class TestClusterResult:
    def test_fields(self):
        labels = np.array([0, 0, 1, 1, -1])
        embeddings = np.random.randn(5, 128)
        result = ClusterResult(labels=labels, embeddings=embeddings, num_clusters=2, num_noise=1)
        assert result.num_clusters == 2
        assert result.num_noise == 1
        assert len(result.labels) == 5

    def test_empty(self):
        result = ClusterResult(
            labels=np.array([], dtype=int),
            embeddings=np.empty((0, 128)),
            num_clusters=0,
            num_noise=0,
        )
        assert result.num_clusters == 0


# ── OrganizePlan ──────────────────────────────────────────────────


class TestOrganizePlan:
    def test_fields(self):
        plan = OrganizePlan(
            person_folders={0: ["/a.jpg", "/b.jpg"]},
            unclustered=["/c.jpg"],
            no_faces=["/d.jpg"],
        )
        assert len(plan.person_folders) == 1
        assert plan.person_folders[0] == ["/a.jpg", "/b.jpg"]
        assert plan.unclustered == ["/c.jpg"]
        assert plan.no_faces == ["/d.jpg"]

    def test_defaults(self):
        plan = OrganizePlan(person_folders={}, unclustered=[], no_faces=[])
        assert plan.person_folders == {}
        assert plan.unclustered == []
        assert plan.no_faces == []


# ── PipelineResult ────────────────────────────────────────────────


class TestPipelineResult:
    def test_required_fields(self):
        result = PipelineResult(
            total_images=10,
            images_with_faces=5,
            total_faces=7,
            num_clusters=2,
            num_noise_faces=1,
        )
        assert result.total_images == 10
        assert result.images_with_faces == 5
        assert result.total_faces == 7
        assert result.num_clusters == 2
        assert result.num_noise_faces == 1

    def test_defaults(self):
        result = PipelineResult(
            total_images=0,
            images_with_faces=0,
            total_faces=0,
            num_clusters=0,
            num_noise_faces=0,
        )
        assert result.organize_plan is None
        assert result.cluster_confidences == {}
        assert result.duration_seconds == 0.0
        assert result.errors == []

    def test_with_all_fields(self):
        plan = OrganizePlan(person_folders={0: ["/a.jpg"]}, unclustered=[], no_faces=[])
        result = PipelineResult(
            total_images=100,
            images_with_faces=80,
            total_faces=120,
            num_clusters=5,
            num_noise_faces=10,
            organize_plan=plan,
            cluster_confidences={0: 0.95, 1: 0.88},
            duration_seconds=12.5,
            errors=["one error"],
        )
        assert result.organize_plan is not None
        assert result.cluster_confidences[1] == 0.88
        assert result.duration_seconds == 12.5
        assert len(result.errors) == 1

    def test_phase_durations_default(self):
        result = PipelineResult(
            total_images=0, images_with_faces=0,
            total_faces=0, num_clusters=0, num_noise_faces=0,
        )
        assert result.phase_durations == {}

    def test_phase_durations_set(self):
        result = PipelineResult(
            total_images=10, images_with_faces=5,
            total_faces=7, num_clusters=2, num_noise_faces=1,
            phase_durations={"scan": 0.1, "detection": 2.3, "embedding": 5.0},
        )
        assert result.phase_durations["scan"] == 0.1
        assert len(result.phase_durations) == 3
