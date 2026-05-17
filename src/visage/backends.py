"""Pluggable embedding backends — dlib (face_recognition) and InsightFace (ArcFace)."""

from __future__ import annotations

import logging
import threading
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from .models import FaceBox

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Protocol for embedding backends."""

    name: str
    embedding_dim: int

    def generate(self, image: np.ndarray, face_box: FaceBox) -> Optional[np.ndarray]: ...
    def is_available(self) -> bool: ...


class DlibBackend:
    """Embedding backend using face_recognition (dlib)."""

    name = "dlib"
    embedding_dim = 128

    def __init__(self, model: str = "small", num_jitters: int = 1) -> None:
        self.model = model
        self.num_jitters = num_jitters
        self._lock = threading.Lock()

        try:
            import face_recognition  # noqa: F401

            self._available = True
        except ImportError:
            self._available = False

    def is_available(self) -> bool:
        return self._available

    def generate(self, image: np.ndarray, face_box: FaceBox) -> Optional[np.ndarray]:
        if not self._available:
            raise RuntimeError("face_recognition library not available")

        import face_recognition

        location = face_box.to_face_recognition_format()
        with self._lock:
            try:
                encodings = face_recognition.face_encodings(
                    image,
                    known_face_locations=[location],
                    num_jitters=self.num_jitters,
                    model=self.model,
                )
                if encodings and len(encodings) > 0:
                    return encodings[0]
            except Exception:
                logger.warning("Dlib embedding failed for face at %s", face_box, exc_info=True)
        return None


class InsightFaceBackend:
    """Embedding backend using InsightFace (ArcFace)."""

    name = "insightface"
    embedding_dim = 512

    def __init__(self, det_size: tuple[int, int] = (640, 640)) -> None:
        self._det_size = det_size
        self._lock = threading.Lock()
        self._app = None

        try:
            import insightface  # noqa: F401

            self._available = True
        except ImportError:
            self._available = False

    def _init_app(self) -> None:
        """Lazy-initialize the InsightFace model (expensive)."""
        if self._app is not None:
            return

        import insightface

        self._app = insightface.app.FaceAnalysis(name="buffalo_l")
        self._app.prepare(ctx_id=0, det_size=self._det_size)

    def is_available(self) -> bool:
        return self._available

    def generate(self, image: np.ndarray, face_box: FaceBox) -> Optional[np.ndarray]:
        if not self._available:
            raise RuntimeError("insightface library not available")

        with self._lock:
            self._init_app()

            # InsightFace expects BGR input
            bgr_image = image[:, :, ::-1] if image.shape[-1] == 3 else image

            try:
                faces = self._app.get(bgr_image)
                if not faces:
                    return None

                # Find the InsightFace-detected face that best overlaps with face_box
                best_face = self._find_best_match(faces, face_box)
                if best_face is not None:
                    return best_face.embedding

                # Fallback: use the first face's embedding if only one detected
                if len(faces) == 1:
                    return faces[0].embedding

            except Exception:
                logger.warning("InsightFace embedding failed for face at %s", face_box, exc_info=True)

        return None

    @staticmethod
    def _find_best_match(
        insightface_faces: list, face_box: FaceBox
    ) -> Optional[object]:
        """Find the InsightFace face with highest IoU overlap with face_box."""
        best_iou = 0.0
        best_face = None

        for face in insightface_faces:
            bbox = face.bbox.astype(int)
            # InsightFace bbox format: [x1, y1, x2, y2]
            ix1, iy1, ix2, iy2 = bbox[0], bbox[1], bbox[2], bbox[3]

            # Compute IoU
            x1 = max(face_box.left, ix1)
            y1 = max(face_box.top, iy1)
            x2 = min(face_box.right, ix2)
            y2 = min(face_box.bottom, iy2)

            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            area_a = face_box.area
            area_b = (ix2 - ix1) * (iy2 - iy1)
            union = area_a + area_b - intersection

            iou = intersection / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_face = face

        return best_face if best_iou > 0.1 else None


def get_backend(name: str, **kwargs) -> EmbeddingBackend:
    """Factory: create an embedding backend by name.

    Args:
        name: "dlib" or "insightface"
        **kwargs: Backend-specific options (model, num_jitters, det_size).

    Returns:
        An EmbeddingBackend instance.

    Raises:
        ValueError: If name is not recognized.
    """
    if name == "dlib":
        return DlibBackend(
            model=kwargs.get("model", "small"),
            num_jitters=kwargs.get("num_jitters", 1),
        )
    elif name == "insightface":
        return InsightFaceBackend(
            det_size=kwargs.get("det_size", (640, 640)),
        )
    else:
        raise ValueError(f"Unknown embedding backend: {name!r}. Choose 'dlib' or 'insightface'.")
