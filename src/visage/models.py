from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

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
    embedding: Optional[np.ndarray] = None
    quality: Optional[float] = None
    image_path: str = ""
    face_index: int = 0


@dataclass
class ImageResult:
    """Processing result for a single image."""

    path: str
    faces: list[DetectedFace] = field(default_factory=list)
    error: Optional[str] = None
    skipped: bool = False


@dataclass
class ClusterResult:
    """Result of clustering all faces."""

    labels: np.ndarray  # cluster labels, -1 = noise/outlier
    embeddings: np.ndarray  # (N, D) embedding matrix
    num_clusters: int
    num_noise: int
    probabilities: Optional[np.ndarray] = None  # HDBSCAN membership probabilities


@dataclass
class OrganizePlan:
    """Plan for organizing files (dry-run friendly)."""

    person_folders: dict[int, list[str]]  # cluster_id -> list of image paths
    unclustered: list[str]  # images with faces that didn't cluster
    no_faces: list[str]  # images with no detected faces


@dataclass
class PipelineResult:
    """Complete result of the face clustering pipeline."""

    total_images: int
    images_with_faces: int
    total_faces: int
    num_clusters: int
    num_noise_faces: int
    organize_plan: Optional[OrganizePlan] = None
    cluster_confidences: dict[int, float] = field(default_factory=dict)
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
