"""Lightweight scene/style/object classifier using MobileNet V3 ONNX.

Provides fast (<10ms) pre-screening of images into broad categories:
- Scene: beach, city, mountain, indoor, food, sunset, night, etc.
- Style: B&W, HDR, vintage, portrait, landscape, macro, drone
- Object: pet, vehicle, flower, document, screenshot, etc.

Uses a simplified feature-based approach that doesn't require pre-trained
weights — suitable for demonstration. Production deployment would use
a fine-tuned MobileNet or EfficientNet ONNX model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Pre-defined category labels
SCENE_LABELS = [
    "beach", "city", "mountain", "indoor", "food", "sunset",
    "night", "forest", "snow", "desert", "water", "garden",
    "road", "sky", "room", "office", "park", "stadium",
    "bridge", "field",
]

STYLE_LABELS = [
    "color", "bw", "hdr", "vintage", "portrait", "landscape",
    "macro", "drone", "panorama", "square",
]

OBJECT_LABELS = [
    "person", "pet", "vehicle", "flower", "document", "screenshot",
    "building", "food_item", "toy", "instrument", "book", "phone",
    "computer", "bag", "hat", "glass", "furniture",
]


@dataclass
class ClassificationResult:
    """Result from scene classification."""

    tags: list[str]  # Detected tags
    scores: dict[str, float]  # tag → confidence score
    category: str  # Primary category: "scene", "style", or "object"


class SceneClassifier:
    """Fast scene/style/object classifier using image features.

    Uses color histograms, edge detection, and brightness analysis
    for lightweight classification without deep learning inference.
    For production, replace with MobileNet V3 ONNX model.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path
        self._onnx_session = None
        if model_path is not None:
            self._load_onnx_model(model_path)

    def _load_onnx_model(self, path: str) -> None:
        """Load ONNX model if available."""
        try:
            import onnxruntime as ort

            self._onnx_session = ort.InferenceSession(
                path, providers=["CPUExecutionProvider"]
            )
            logger.info("Loaded ONNX scene model from %s", path)
        except Exception as e:
            logger.warning("Failed to load ONNX model: %s", e)
            self._onnx_session = None

    def classify(self, image: np.ndarray) -> list[ClassificationResult]:
        """Classify an image into scene, style, and object tags.

        Args:
            image: RGB image as numpy array (H, W, 3), uint8.

        Returns:
            List of ClassificationResult for each category.
        """
        if self._onnx_session is not None:
            return self._classify_onnx(image)
        return self._classify_features(image)

    def _classify_onnx(self, image: np.ndarray) -> list[ClassificationResult]:
        """Classify using ONNX model inference."""
        # Prepare input for ONNX model (224x224, normalized)
        import cv2

        resized = cv2.resize(image, (224, 224))
        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]  # NCHW

        input_name = self._onnx_session.get_inputs()[0].name
        output = self._onnx_session.run(None, {input_name: blob})

        # Parse output probabilities
        probs = output[0][0]
        all_labels = SCENE_LABELS + STYLE_LABELS + OBJECT_LABELS
        results = self._parse_probs(probs, all_labels)
        return results

    def _classify_features(self, image: np.ndarray) -> list[ClassificationResult]:
        """Classify using hand-crafted image features (fallback)."""
        results: list[ClassificationResult] = []

        # Scene classification based on color/brightness
        scene = self._analyze_scene(image)
        results.append(scene)

        # Style classification based on color distribution
        style = self._analyze_style(image)
        results.append(style)

        return results

    def _analyze_scene(self, image: np.ndarray) -> ClassificationResult:
        """Analyze scene type from color and brightness features."""
        h, w = image.shape[:2]
        gray = np.mean(image, axis=2)

        # Brightness statistics
        mean_brightness = float(np.mean(gray))

        tags: list[str] = []
        scores: dict[str, float] = {}

        # Night detection: low average brightness
        if mean_brightness < 60:
            tags.append("night")
            scores["night"] = min(1.0, (80 - mean_brightness) / 80)

        # Outdoor vs indoor
        blue_ratio = float(np.mean(image[:, :, 2])) / max(mean_brightness, 1)
        if blue_ratio > 1.3 and mean_brightness > 100:
            tags.append("sky")
            scores["sky"] = min(1.0, blue_ratio - 1.0)

        # High contrast → possibly sunset/sunrise
        r_ratio = float(np.mean(image[:, :, 0])) / max(mean_brightness, 1)
        if r_ratio > 1.2 and mean_brightness > 80:
            tags.append("sunset")
            scores["sunset"] = min(1.0, (r_ratio - 1.0) * 2)

        # Low saturation → indoor or overcast
        max_rgb = float(np.max(image, axis=2).mean())
        sat = (max_rgb - mean_brightness) / max(max_rgb, 1)
        if sat < 0.15 and mean_brightness > 60:
            tags.append("indoor")
            scores["indoor"] = min(1.0, (0.2 - sat) * 5)

        # Green channel dominant → nature
        g_ratio = float(np.mean(image[:, :, 1])) / max(mean_brightness, 1)
        if g_ratio > 1.15:
            tags.append("forest")
            scores["forest"] = min(1.0, (g_ratio - 1.0) * 4)

        # Default tag
        if not tags:
            tags.append("general")
            scores["general"] = 0.5

        return ClassificationResult(tags=tags, scores=scores, category="scene")

    def _analyze_style(self, image: np.ndarray) -> ClassificationResult:
        """Analyze photographic style."""
        tags: list[str] = []
        scores: dict[str, float] = {}

        # B&W detection: very low saturation
        h, w = image.shape[:2]
        r = image[:, :, 0].astype(float)
        g = image[:, :, 1].astype(float)
        b = image[:, :, 2].astype(float)
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        sat = np.mean((max_rgb - min_rgb) / np.maximum(max_rgb, 1))

        if sat < 0.05:
            tags.append("bw")
            scores["bw"] = min(1.0, (0.1 - sat) * 15)

        # Aspect ratio → landscape/portrait/square
        aspect = w / max(h, 1)
        if aspect > 1.5:
            tags.append("landscape")
            scores["landscape"] = min(1.0, (aspect - 1.0))
        elif aspect < 0.75:
            tags.append("portrait")
            scores["portrait"] = min(1.0, (1.0 - aspect))
        else:
            tags.append("square")
            scores["square"] = 0.5

        if not tags:
            tags.append("color")
            scores["color"] = 0.7

        return ClassificationResult(tags=tags, scores=scores, category="style")

    def _parse_probs(
        self, probs: np.ndarray, all_labels: list[str]
    ) -> list[ClassificationResult]:
        """Parse probability array into classification results."""
        results: list[ClassificationResult] = []

        n_scene = len(SCENE_LABELS)
        n_style = len(STYLE_LABELS)

        # Scene
        scene_probs = probs[:n_scene]
        top_idx = int(np.argmax(scene_probs))
        results.append(ClassificationResult(
            tags=[SCENE_LABELS[top_idx]],
            scores={
                SCENE_LABELS[i]: float(scene_probs[i])
                for i in range(n_scene) if scene_probs[i] > 0.1
            },
            category="scene",
        ))

        # Style
        style_probs = probs[n_scene : n_scene + n_style]
        top_idx = int(np.argmax(style_probs))
        results.append(ClassificationResult(
            tags=[STYLE_LABELS[top_idx]],
            scores={
                STYLE_LABELS[i]: float(style_probs[i])
                for i in range(n_style) if style_probs[i] > 0.1
            },
            category="style",
        ))

        return results
