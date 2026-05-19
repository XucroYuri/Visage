from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class FaceBox:
    """A detected face bounding box in pixel coordinates (top, right, bottom, left).

    This format matches face_recognition's convention.
    """

    top: int
    right: int
    bottom: int
    left: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_face_recognition_format(self) -> tuple[int, int, int, int]:
        """Return as (top, right, bottom, left) tuple for face_recognition."""
        return (self.top, self.right, self.bottom, self.left)


@dataclass
class DetectedFace:
    """A single detected face with optional embedding."""

    face_box: FaceBox
    confidence: float
    embedding: np.ndarray | None = None
    quality: float | None = None
    head_features: np.ndarray | None = None
    image_path: str = ""
    face_index: int = 0
    # 5 facial landmarks for alignment: (left_eye, right_eye, nose, left_mouth, right_mouth)
    # Each landmark is (x, y) in pixel coordinates.
    landmarks_5: list[tuple[float, float]] | None = None


@dataclass
class ImageResult:
    """Processing result for a single image."""

    path: str
    faces: list[DetectedFace] = field(default_factory=list)
    error: str | None = None
    skipped: bool = False
    detection_stats: dict[str, int] | None = None  # per-image detection metrics


@dataclass
class ClusterResult:
    """Result of clustering all faces."""

    labels: np.ndarray  # cluster labels, -1 = noise/outlier
    embeddings: np.ndarray  # (N, D) embedding matrix
    num_clusters: int
    num_noise: int
    probabilities: np.ndarray | None = None  # HDBSCAN membership probabilities


@dataclass
class OrganizePlan:
    """Plan for organizing files (dry-run friendly)."""

    person_folders: dict[int, list[str]]  # cluster_id -> list of image paths
    unclustered: list[str]  # images with faces that didn't cluster
    no_faces: list[str]  # images with no detected faces


@dataclass
class DetectionStats:
    """Quality metrics collected during face detection."""

    total_faces: int = 0
    contour_boxes: int = 0  # bboxes from face contour landmarks
    median_boxes: int = 0  # bboxes from median line fallback
    default_boxes: int = 0  # bboxes from Vision default bbox
    body_shrunk: int = 0  # bboxes shrunk (aspect ratio > 1.8)
    aligned_faces: int = 0  # faces that were aligned before embedding
    contour_rate: float = 0.0  # fraction using contour landmarks
    mean_aspect_ratio: float = 0.0  # average height/width ratio

    def compute_rates(self) -> None:
        """Compute derived rates after populating counts."""
        if self.total_faces > 0:
            self.contour_rate = self.contour_boxes / self.total_faces
            if self.total_faces > 0:
                # mean_aspect_ratio is already accumulated — no-op here
                pass


@dataclass
class PipelineResult:
    """Complete result of the face clustering pipeline."""

    total_images: int
    images_with_faces: int
    total_faces: int
    num_clusters: int
    num_noise_faces: int
    organize_plan: OrganizePlan | None = None
    cluster_confidences: dict[int, float] = field(default_factory=dict)
    duration_seconds: float = 0.0
    phase_durations: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
