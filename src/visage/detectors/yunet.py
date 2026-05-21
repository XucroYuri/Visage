"""YuNet face detection backend using OpenCV DNN.

YuNet is a lightweight face detection model from OpenCV Zoo. It provides
bounding boxes with 5-point landmarks. The model file must be downloaded
separately from the OpenCV Zoo repository.

Model download URL:
  https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]

from visage.models import FaceBox

logger = logging.getLogger(__name__)

# Default model filename shipped with OpenCV Zoo
_YUNET_MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"


class YuNetDetector:
    """Face detection using OpenCV's YuNet model.

    Lightweight detector suitable as a fallback when SCRFD is unavailable.
    Requires OpenCV (cv2) and the YuNet ONNX model file.
    """

    name = "yunet"

    def __init__(
        self,
        model_path: str | None = None,
        min_confidence: float = 0.5,
        min_face_size: int = 40,
    ) -> None:
        self.min_confidence = min_confidence
        self.min_face_size = min_face_size
        self._model = None
        self._model_path = model_path or self._default_model_path()
        self._available: bool | None = None

    @staticmethod
    def _default_model_path() -> str | None:
        """Look for the YuNet ONNX model in common locations."""
        # Check next to this module
        local_path = Path(__file__).parent.parent / "models" / _YUNET_MODEL_FILENAME
        if local_path.exists():
            return str(local_path)
        # Check environment variable
        import os as _os

        env_path = _os.environ.get("VISAGE_YUNET_MODEL")
        if env_path and Path(env_path).exists():
            return env_path
        return None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import cv2 as _cv2  # noqa: F401

            # Also check that model exists
            if self._model_path and Path(self._model_path).exists():
                self._available = True
            else:
                logger.warning(
                    "YuNet model not found at %s. "
                    "Download from: "
                    "https://github.com/opencv/opencv_zoo/blob/main/"
                    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                    self._model_path,
                )
                self._available = False
        except ImportError:
            self._available = False
        return self._available

    def detect(
        self, image: np.ndarray,
    ) -> list[tuple[FaceBox, float, list[tuple[float, float]] | None]]:
        """Detect faces using YuNet.

        Args:
            image: RGB numpy array, shape (H, W, 3), dtype uint8.

        Returns:
            List of (FaceBox, confidence, landmarks_5) tuples.
            landmarks_5 contains 5-point landmarks when detected.
        """
        h, w = image.shape[:2]

        if self._model is None:
            if not self._model_path or not Path(self._model_path).exists():
                raise RuntimeError(
                    f"YuNet model not found at {self._model_path}. "
                    "Download the ONNX model from OpenCV Zoo."
                )
            self._model = cv2.FaceDetectorYN.create(
                model=self._model_path,
                config="",
                input_size=(320, 320),
                score_threshold=self.min_confidence,
                nms_threshold=0.3,
                top_k=5000,
            )

        # Convert RGB to BGR for OpenCV
        bgr = image[:, :, ::-1] if image.shape[-1] == 3 else image

        # YuNet expects HWC BGR uint8
        self._model.setInputSize((w, h))
        _, raw_results = self._model.detect(bgr)

        if raw_results is None:
            return []

        # raw_results shape: (N, 15) per detection:
        # [x1, y1, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt,
        #  x_rcm, y_rcm, x_lcm, y_lcm, score]
        results: list[tuple[FaceBox, float, list[tuple[float, float]] | None]] = []

        for det in raw_results:
            det = det.astype(np.float64)
            x1, y1, fw, fh = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            score = float(det[14])

            if score < self.min_confidence:
                continue

            fb = FaceBox(
                top=y1,
                right=x1 + fw,
                bottom=y1 + fh,
                left=x1,
            )

            if fb.width < self.min_face_size or fb.height < self.min_face_size:
                continue

            # Extract 5-point landmarks
            lm5: list[tuple[float, float]] | None = [
                (float(det[4]), float(det[5])),   # right eye
                (float(det[6]), float(det[7])),   # left eye
                (float(det[8]), float(det[9])),   # nose tip
                (float(det[10]), float(det[11])),  # right mouth corner
                (float(det[12]), float(det[13])),  # left mouth corner
            ]

            results.append((fb, score, lm5))

        logger.debug(
            "YuNet: %d faces detected (conf>=%.2f)",
            len(results), self.min_confidence,
        )
        return results
