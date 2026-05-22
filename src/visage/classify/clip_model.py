"""CLIP zero-shot classifier using ONNX text+vision encoders.

Provides natural-language image search by encoding text queries and
image pixels into a shared embedding space. Falls back to a lightweight
feature-based approach when ONNX models are not available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Default candidate labels for zero-shot classification
DEFAULT_LABELS = [
    "a photo of a person", "a photo of a pet", "a photo of food",
    "a photo of a building", "a photo of nature", "a photo of a vehicle",
    "a photo of a document", "a photo of a sunset", "a photo of the beach",
    "a photo of a city", "a photo of a mountain", "a photo of snow",
    "a photo of indoor scene", "a photo of a party", "a photo of a wedding",
    "a photo of a sports event", "a photo of a concert", "a photo of travel",
]


@dataclass
class ZeroShotResult:
    """Result from zero-shot classification."""

    label: str
    score: float
    all_scores: dict[str, float]


class CLIPClassifier:
    """Zero-shot image classifier using CLIP-style embeddings.

    When ONNX models (text encoder + vision encoder) are available,
    performs true zero-shot classification. Otherwise falls back to
    a feature-based similarity approach.
    """

    def __init__(
        self,
        text_model_path: str | None = None,
        vision_model_path: str | None = None,
    ) -> None:
        self._text_session = None
        self._vision_session = None
        self._label_dim = 64  # Feature-based embedding dimension

        if text_model_path:
            self._load_model("text", text_model_path)
        if vision_model_path:
            self._load_model("vision", vision_model_path)

    def _load_model(self, kind: str, path: str) -> None:
        """Load an ONNX encoder model."""
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            if kind == "text":
                self._text_session = session
            else:
                self._vision_session = session
            logger.info("Loaded ONNX %s model from %s", kind, path)
        except Exception as e:
            logger.warning("Failed to load ONNX %s model: %s", kind, e)

    def classify(
        self,
        image: np.ndarray,
        labels: list[str] | None = None,
        top_k: int = 5,
    ) -> list[ZeroShotResult]:
        """Classify an image against candidate text labels.

        Args:
            image: RGB image as numpy array (H, W, 3), uint8.
            labels: Candidate text labels. Uses DEFAULT_LABELS if None.
            top_k: Number of top results to return.

        Returns:
            List of ZeroShotResult sorted by score descending.
        """
        candidate_labels = labels or DEFAULT_LABELS

        if self._vision_session is not None and self._text_session is not None:
            return self._classify_onnx(image, candidate_labels, top_k)
        return self._classify_features(image, candidate_labels, top_k)

    def search(
        self,
        query: str,
        image_features: dict[str, np.ndarray],
        top_k: int = 20,
        min_score: float = 0.3,
    ) -> list[tuple[str, float]]:
        """Search images by natural language query.

        Args:
            query: Natural language search query.
            image_features: Mapping of image_path → feature vector.
            top_k: Number of results to return.
            min_score: Minimum similarity score.

        Returns:
            List of (image_path, score) tuples sorted by score descending.
        """
        if self._text_session is not None:
            query_vec = self._encode_text_onnx(query)
        else:
            query_vec = self._encode_text_features(query)

        results: list[tuple[str, float]] = []
        for path, feat in image_features.items():
            if feat.shape != query_vec.shape:
                continue
            sim = float(np.dot(query_vec, feat) / (
                np.linalg.norm(query_vec) * np.linalg.norm(feat) + 1e-8
            ))
            if sim >= min_score:
                results.append((path, sim))

        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract a feature vector from an image for later search.

        Args:
            image: RGB image as numpy array (H, W, 3), uint8.

        Returns:
            Normalized feature vector.
        """
        if self._vision_session is not None:
            return self._encode_image_onnx(image)
        return self._encode_image_features(image)

    def _classify_onnx(
        self, image: np.ndarray, labels: list[str], top_k: int,
    ) -> list[ZeroShotResult]:
        """Classify using ONNX CLIP encoders."""
        img_vec = self._encode_image_onnx(image)

        # Encode all labels
        all_scores: dict[str, float] = {}
        for label in labels:
            txt_vec = self._encode_text_onnx(label)
            sim = float(np.dot(img_vec, txt_vec))
            all_scores[label] = sim

        # Softmax-like normalization
        max_score = max(all_scores.values()) if all_scores else 0
        exp_scores = {k: np.exp(v - max_score) for k, v in all_scores.items()}
        total = sum(exp_scores.values())
        normalized = {k: v / total for k, v in exp_scores.items()}

        sorted_labels = sorted(normalized.items(), key=lambda x: -x[1])
        return [
            ZeroShotResult(label=label, score=score, all_scores=normalized)
            for label, score in sorted_labels[:top_k]
        ]

    def _classify_features(
        self, image: np.ndarray, labels: list[str], top_k: int,
    ) -> list[ZeroShotResult]:
        """Fallback classification using image feature vectors."""
        img_vec = self._encode_image_features(image)

        all_scores: dict[str, float] = {}
        for label in labels:
            txt_vec = self._encode_text_features(label)
            sim = float(np.dot(img_vec, txt_vec) / (
                np.linalg.norm(img_vec) * np.linalg.norm(txt_vec) + 1e-8
            ))
            all_scores[label] = sim

        sorted_labels = sorted(all_scores.items(), key=lambda x: -x[1])
        return [
            ZeroShotResult(label=label, score=score, all_scores=all_scores)
            for label, score in sorted_labels[:top_k]
        ]

    def _encode_image_onnx(self, image: np.ndarray) -> np.ndarray:
        """Encode image using ONNX vision encoder."""
        import cv2

        resized = cv2.resize(image, (224, 224))
        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]

        input_name = self._vision_session.get_inputs()[0].name
        output = self._vision_session.run(None, {input_name: blob})
        vec = output[0].flatten()
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _encode_text_onnx(self, text: str) -> np.ndarray:
        """Encode text using ONNX text encoder."""
        tokens = np.array([ord(c) for c in text[:77]], dtype=np.int64)
        tokens = np.pad(tokens, (0, max(0, 77 - len(tokens))))

        input_name = self._text_session.get_inputs()[0].name
        output = self._text_session.run(None, {input_name: tokens[np.newaxis, ...]})
        vec = output[0].flatten()
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _encode_image_features(self, image: np.ndarray) -> np.ndarray:
        """Encode image into a feature vector using color/statistical features."""
        h, w = image.shape[:2]
        features: list[float] = []

        # Color histogram (4 bins per channel = 12)
        for c in range(3):
            channel = image[:, :, c].flatten().astype(float)
            hist, _ = np.histogram(channel, bins=4, range=(0, 256))
            hist = hist / (hist.sum() + 1e-8)
            features.extend(hist.tolist())

        # Brightness stats (4)
        gray = np.mean(image, axis=2).flatten()
        features.extend([
            float(np.mean(gray)),
            float(np.std(gray)),
            float(np.percentile(gray, 10)),
            float(np.percentile(gray, 90)),
        ])

        # Aspect ratio (1)
        features.append(w / max(h, 1))

        # Saturation (1)
        r = image[:, :, 0].astype(float)
        g = image[:, :, 1].astype(float)
        b = image[:, :, 2].astype(float)
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        features.append(float(np.mean((max_rgb - min_rgb) / (max_rgb + 1e-8))))

        # Edge density (1)
        gray_img = np.mean(image, axis=2).astype(float)
        dx = np.abs(np.diff(gray_img, axis=1))
        dy = np.abs(np.diff(gray_img, axis=0))
        features.append(float(np.mean(dx) + np.mean(dy)))

        # Color channel ratios (3)
        total = float(np.mean(image))
        features.extend([
            float(np.mean(image[:, :, c])) / max(total, 1)
            for c in range(3)
        ])

        # Spatial: top vs bottom brightness (2)
        top_half = image[:h // 2]
        bot_half = image[h // 2:]
        features.append(float(np.mean(top_half)))
        features.append(float(np.mean(bot_half)))

        # Spatial: left vs right brightness (2)
        left_half = image[:, :w // 2]
        right_half = image[:, w // 2:]
        features.append(float(np.mean(left_half)))
        features.append(float(np.mean(right_half)))

        # Texture: variance of local patches (1)
        patch_size = max(h // 8, 1)
        patches = []
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                patch = gray_img[i:i + patch_size, j:j + patch_size]
                patches.append(float(np.var(patch)))
        features.append(float(np.mean(patches)) if patches else 0.0)

        # Pad or truncate to fixed dimension
        vec = np.array(features[:self._label_dim], dtype=np.float32)
        if len(vec) < self._label_dim:
            vec = np.pad(vec, (0, self._label_dim - len(vec)))
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _encode_text_features(self, text: str) -> np.ndarray:
        """Encode text into a feature vector matching image feature space.

        Uses keyword-to-feature mapping to align text concepts with
        the statistical image features extracted by _encode_image_features.
        """
        vec = np.zeros(self._label_dim, dtype=np.float32)
        text_lower = text.lower()

        # Map keywords to feature vector positions
        # Positions 0-11: color histograms (R, G, B × 4 bins)
        keyword_color_map: dict[str, dict[int, float]] = {
            "sunset": {0: 0.8, 1: 0.3, 2: 0.1},  # Red dominant
            "nature": {0: 0.2, 1: 0.6, 2: 0.2},  # Green dominant
            "beach": {0: 0.3, 1: 0.4, 2: 0.6},   # Blue-ish
            "snow": {0: 0.7, 1: 0.7, 2: 0.7},    # Bright
            "night": {0: 0.1, 1: 0.1, 2: 0.15},  # Dark
            "food": {0: 0.6, 1: 0.4, 2: 0.2},    # Warm tones
            "indoor": {0: 0.4, 1: 0.35, 2: 0.3},  # Neutral warm
            "city": {0: 0.3, 1: 0.3, 2: 0.35},   # Neutral cool
        }

        for keyword, color_shifts in keyword_color_map.items():
            if keyword in text_lower:
                for idx, val in color_shifts.items():
                    vec[idx * 4] += val * 0.3  # First bin of each channel

        # Brightness keywords → positions 12-15 (mean, std, p10, p90)
        if "night" in text_lower or "dark" in text_lower:
            vec[12] -= 0.5  # Low mean brightness
        if "snow" in text_lower or "bright" in text_lower:
            vec[12] += 0.5  # High mean brightness

        # Aspect ratio → position 16
        if "portrait" in text_lower or "person" in text_lower or "people" in text_lower:
            vec[16] += 0.3  # Taller aspect
        if "landscape" in text_lower or "panorama" in text_lower:
            vec[16] -= 0.3  # Wider aspect

        # Saturation → position 17
        if "colorful" in text_lower or "vivid" in text_lower:
            vec[17] += 0.3

        # Person/pet keywords → general similarity boost
        person_words = {"person", "people", "face", "portrait", "party", "wedding"}
        if person_words & set(text_lower.split()):
            vec[:12] += 0.05  # Slight warm shift

        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
