"""Pluggable detection backends — Vision, SCRFD, YuNet.

Each backend implements the DetectorBackend Protocol and returns
a unified list of (FaceBox, confidence, landmarks_5) tuples.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import numpy as np

from visage.models import FaceBox

logger = logging.getLogger(__name__)


@runtime_checkable
class DetectorBackend(Protocol):
    """Protocol for face detection backends.

    Each backend must provide:
      - name: unique string identifier
      - detect(image) -> list of (FaceBox, confidence, landmarks_5)
      - is_available() -> bool
    """

    name: str

    def detect(
        self, image: np.ndarray,
    ) -> list[tuple[FaceBox, float, list[tuple[float, float]] | None]]:
        """Detect faces in an RGB image.

        Args:
            image: RGB numpy array, shape (H, W, 3), dtype uint8.

        Returns:
            List of (FaceBox, confidence, landmarks_5) tuples.
            landmarks_5 is None if the backend does not provide landmarks.
        """
        ...

    def is_available(self) -> bool:
        """Whether this backend can be used on the current platform."""
        ...


def get_detector(name: str = "auto", **kwargs: object) -> DetectorBackend:
    """Factory: create a detector backend by name.

    Args:
        name: One of "auto", "vision", "scrfd", "yunet".
        **kwargs: Backend-specific options passed to the constructor.
            Common options: min_confidence (float), min_face_size (int).

    Returns:
        A DetectorBackend instance.

    Raises:
        ValueError: If name is not recognized or no backend is available.
    """
    _name = name.lower()

    if _name == "vision":
        from .vision import VisionDetector

        inst: DetectorBackend = VisionDetector(**kwargs)  # type: ignore[assignment]
        if not inst.is_available():
            raise RuntimeError(
                "Vision framework is not available on this platform "
                "(requires macOS with pyobjc)"
            )
        return inst

    if _name == "scrfd":
        from .scrfd import SCRFDDetector

        inst = SCRFDDetector(**kwargs)  # type: ignore[assignment]
        if not inst.is_available():
            raise RuntimeError(
                "SCRFD detector is not available. "
                "Install insightface: pip install visage[scrfd]"
            )
        return inst

    if _name == "yunet":
        from .yunet import YuNetDetector

        inst = YuNetDetector(**kwargs)  # type: ignore[assignment]
        if not inst.is_available():
            raise RuntimeError(
                "YuNet detector is not available. "
                "Install OpenCV: pip install opencv-python-headless"
            )
        return inst

    if _name == "auto":
        # Try backends in priority order
        import sys

        if sys.platform == "darwin":
            from .vision import VisionDetector

            vision = VisionDetector(**kwargs)  # type: ignore[arg-type]
            if vision.is_available():
                logger.info("Auto-detection: using Vision backend")
                return vision

            from .scrfd import SCRFDDetector

            scrfd = SCRFDDetector(**kwargs)  # type: ignore[arg-type]
            if scrfd.is_available():
                logger.info("Auto-detection: using SCRFD backend")
                return scrfd

            from .yunet import YuNetDetector

            yunet = YuNetDetector(**kwargs)  # type: ignore[arg-type]
            if yunet.is_available():
                logger.info("Auto-detection: using YuNet backend")
                return yunet
        else:
            from .scrfd import SCRFDDetector

            scrfd = SCRFDDetector(**kwargs)  # type: ignore[arg-type]
            if scrfd.is_available():
                logger.info("Auto-detection: using SCRFD backend")
                return scrfd

            from .yunet import YuNetDetector

            yunet = YuNetDetector(**kwargs)  # type: ignore[arg-type]
            if yunet.is_available():
                logger.info("Auto-detection: using YuNet backend")
                return yunet

        raise RuntimeError(
            "No detection backend available. "
            "On macOS install pyobjc; on other platforms install "
            "insightface or opencv-python-headless."
        )

    raise ValueError(
        f"Unknown detection backend: {name!r}. "
        f"Choose 'vision', 'scrfd', 'yunet', or 'auto'."
    )
