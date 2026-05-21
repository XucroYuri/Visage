"""SCRFD face detection backend using InsightFace.

SCRFD (Sample and Computation Redistribution for Face Detection) is a
high-accuracy detection model from the InsightFace team. It supports both
CPU and GPU inference and provides bounding boxes with 5-point landmarks.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from visage.models import FaceBox

logger = logging.getLogger(__name__)


class SCRFDDetector:
    """Face detection using InsightFace's SCRFD model.

    Provides bounding boxes with 5-point facial landmarks. Falls back
    gracefully if insightface is not installed.
    """

    name = "scrfd"

    def __init__(
        self,
        det_size: tuple[int, int] = (640, 640),
        min_confidence: float = 0.5,
        min_face_size: int = 40,
    ) -> None:
        self.det_size = det_size
        self.min_confidence = min_confidence
        self.min_face_size = min_face_size
        self._app = None
        self._available: bool | None = None
        self._lock = threading.Lock()

    def _check_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import insightface  # noqa: F401

            self._available = True
        except ImportError:
            self._available = False
        return self._available

    def is_available(self) -> bool:
        return self._check_available()

    def _init_app(self) -> None:
        """Lazy-initialize InsightFace with detection-only modules."""
        if self._app is not None:
            return
        import insightface

        self._app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection"],
        )
        self._app.prepare(ctx_id=0, det_size=self.det_size)
        logger.info("SCRFD detector initialized (det_size=%s)", self.det_size)

    def detect(
        self, image: np.ndarray,
    ) -> list[tuple[FaceBox, float, list[tuple[float, float]] | None]]:
        """Detect faces using SCRFD.

        Args:
            image: RGB numpy array, shape (H, W, 3), dtype uint8.

        Returns:
            List of (FaceBox, confidence, landmarks_5) tuples.
            landmarks_5 contains 5-point keypoints when available.
        """
        if not self._check_available():
            raise RuntimeError("insightface library not available")

        with self._lock:
            self._init_app()

            # Convert RGB to BGR for InsightFace
            bgr = image[:, :, ::-1] if image.shape[-1] == 3 else image

            try:
                faces = self._app.get(bgr)
            except Exception:
                logger.warning("SCRFD detection failed", exc_info=True)
                return []

            results: list[tuple[FaceBox, float, list[tuple[float, float]] | None]] = []

            for face in faces:
                if face.det_score < self.min_confidence:
                    continue

                # bbox format: [x1, y1, x2, y2]
                bbox = face.bbox.astype(int)
                fb = FaceBox(
                    top=int(bbox[1]),
                    right=int(bbox[2]),
                    bottom=int(bbox[3]),
                    left=int(bbox[0]),
                )

                # Filter by minimum face size
                if fb.width < self.min_face_size or fb.height < self.min_face_size:
                    continue

                # Extract 5-point landmarks (keypoints)
                lm5: list[tuple[float, float]] | None = None
                if hasattr(face, "kps") and face.kps is not None:
                    kps = face.kps
                    if len(kps) == 5:
                        lm5 = [(float(pt[0]), float(pt[1])) for pt in kps]

                results.append((fb, float(face.det_score), lm5))

            logger.debug(
                "SCRFD: %d faces detected (conf>=%.2f)",
                len(results), self.min_confidence,
            )
            return results
