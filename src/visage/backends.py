"""Pluggable embedding backends — dlib (face_recognition) and InsightFace (ArcFace)."""

from __future__ import annotations

import logging
import threading
from typing import Protocol, runtime_checkable

import numpy as np

from .models import FaceBox

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Protocol for embedding backends."""

    name: str
    embedding_dim: int

    def generate(self, image: np.ndarray, face_box: FaceBox) -> np.ndarray | None: ...
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

    def generate(self, image: np.ndarray, face_box: FaceBox) -> np.ndarray | None:
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
    """Embedding backend using InsightFace (ArcFace).

    Optimised for memory: only loads detection + recognition modules,
    and runs on face crops (not full images) to minimise temporary memory.

    Falls back to dlib embeddings when InsightFace detection fails on a crop.
    """

    name = "insightface"
    embedding_dim = 512

    def __init__(
        self,
        det_size: tuple[int, int] = (640, 640),
        fallback_model: str = "small",
    ) -> None:
        self._det_size = det_size
        self._fallback_model = fallback_model
        self._lock = threading.Lock()
        self._app = None
        self._dlib_fallback = None  # lazy-init on first use

        try:
            import insightface  # noqa: F401

            self._available = True
        except ImportError:
            self._available = False

    def _init_app(self) -> None:
        """Lazy-initialize InsightFace with only detection + recognition modules.

        Skips gender/age and 3D landmark models to reduce memory footprint.
        """
        if self._app is not None:
            return

        import insightface

        self._app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
        )
        self._app.prepare(ctx_id=0, det_size=self._det_size)

    def is_available(self) -> bool:
        return self._available

    def generate(self, image: np.ndarray, face_box: FaceBox) -> np.ndarray | None:
        if not self._available:
            raise RuntimeError("insightface library not available")

        with self._lock:
            self._init_app()

            # Crop face region with padding for detection context.
            # Running on a crop drastically reduces InsightFace temporary memory
            # compared to running detection on the full high-resolution image.
            crop = self._crop_face(image, face_box)
            bgr_crop = crop[:, :, ::-1] if crop.shape[-1] == 3 else crop

            try:
                faces = self._app.get(bgr_crop)
                if faces:
                    # The crop contains one face — return the best detection
                    embedding = faces[0].embedding
                    if embedding is not None:
                        return embedding
            except Exception:
                logger.warning(
                    "InsightFace embedding failed for face at %s",
                    face_box, exc_info=True,
                )

        # Fall back to dlib if InsightFace detection fails
        return self._dlib_generate(image, face_box)

    def _dlib_generate(self, image: np.ndarray, face_box: FaceBox) -> np.ndarray | None:
        """Fallback embedding via dlib when InsightFace fails."""
        if self._dlib_fallback is None:
            try:
                import face_recognition  # noqa: F401
            except ImportError:
                return None
            self._dlib_fallback = DlibBackend(model=self._fallback_model)
        return self._dlib_fallback.generate(image, face_box)

    @staticmethod
    def _crop_face(image: np.ndarray, face_box: FaceBox) -> np.ndarray:
        """Crop face region with 80% padding for landmark detection context."""
        h, w = image.shape[:2]
        fw = face_box.width
        fh = face_box.height
        pad_x = int(fw * 0.8)
        pad_y = int(fh * 0.8)

        x1 = max(0, face_box.left - pad_x)
        y1 = max(0, face_box.top - pad_y)
        x2 = min(w, face_box.right + pad_x)
        y2 = min(h, face_box.bottom + pad_y)

        return image[y1:y2, x1:x2]


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
