"""Classification pipeline — orchestrates scene, CLIP, and tag storage."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

from visage.classify.clip_model import CLIPClassifier
from visage.classify.scene import SceneClassifier
from visage.classify.tag_store import TagStore

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result from running the classification pipeline on a single image."""

    image_path: str
    tags: list[str]
    scores: dict[str, float]
    categories: dict[str, list[str]]  # category → tags
    elapsed_ms: float


class ClassificationPipeline:
    """Orchestrates scene classification, CLIP zero-shot, and tag persistence.

    Runs a three-layer classification:
    1. Scene/style classifier → broad category tags
    2. CLIP zero-shot → semantic labels
    3. Persist results to TagStore
    """

    def __init__(
        self,
        scene_model_path: str | None = None,
        clip_text_model_path: str | None = None,
        clip_vision_model_path: str | None = None,
        input_path: str | None = None,
    ) -> None:
        self._scene = SceneClassifier(model_path=scene_model_path)
        self._clip = CLIPClassifier(
            text_model_path=clip_text_model_path,
            vision_model_path=clip_vision_model_path,
        )
        self._store: TagStore | None = None
        if input_path is not None:
            self._store = TagStore(input_path)

        # Cache image features for CLIP search
        self._image_features: dict[str, np.ndarray] = {}

    @property
    def store(self) -> TagStore | None:
        return self._store

    def classify_image(self, image_path: str) -> PipelineResult:
        """Run full classification pipeline on a single image.

        Args:
            image_path: Path to the image file.

        Returns:
            PipelineResult with all detected tags and scores.
        """
        t0 = time.time()

        image = np.array(Image.open(image_path).convert("RGB"))
        all_tags: list[str] = []
        all_scores: dict[str, float] = {}
        categories: dict[str, list[str]] = {}

        # Layer 1: Scene classification
        scene_results = self._scene.classify(image)
        for result in scene_results:
            categories[result.category] = result.tags
            for tag in result.tags:
                if tag not in all_tags:
                    all_tags.append(tag)
            all_scores.update(result.scores)

        # Layer 2: CLIP zero-shot (top-3)
        clip_results = self._clip.classify(image, top_k=3)
        clip_tags = []
        for r in clip_results:
            # Extract short label from "a photo of X" format
            short = r.label.replace("a photo of ", "").replace("a ", "")
            clip_tags.append(short)
            if short not in all_tags:
                all_tags.append(short)
            all_scores[short] = r.score
        categories["clip"] = clip_tags

        # Layer 3: Extract and cache features for search
        features = self._clip.extract_features(image)
        self._image_features[image_path] = features

        # Persist to TagStore
        if self._store is not None:
            for cat, cat_tags in categories.items():
                cat_scores = {t: all_scores.get(t, 0.0) for t in cat_tags}
                self._store.store_tags(image_path, cat_tags, cat_scores, category=cat)

        elapsed = (time.time() - t0) * 1000
        return PipelineResult(
            image_path=image_path,
            tags=all_tags,
            scores=all_scores,
            categories=categories,
            elapsed_ms=elapsed,
        )

    def classify_batch(self, image_paths: list[str]) -> list[PipelineResult]:
        """Classify a batch of images.

        Args:
            image_paths: List of image file paths.

        Returns:
            List of PipelineResult for each image.
        """
        results: list[PipelineResult] = []
        for path in image_paths:
            try:
                result = self.classify_image(path)
                results.append(result)
            except Exception as e:
                logger.warning("Failed to classify %s: %s", path, e)
        return results

    def semantic_search(
        self,
        query: str,
        top_k: int = 20,
        min_score: float = 0.3,
    ) -> list[tuple[str, float]]:
        """Search images by natural language query.

        Args:
            query: Natural language search text.
            top_k: Number of results.
            min_score: Minimum similarity threshold.

        Returns:
            List of (image_path, score) sorted by score descending.
        """
        return self._clip.search(
            query,
            self._image_features,
            top_k=top_k,
            min_score=min_score,
        )

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
