"""Tests for the classification pipeline — scene, CLIP, tag store."""

from __future__ import annotations

import numpy as np

from visage.classify.clip_model import CLIPClassifier
from visage.classify.pipeline import ClassificationPipeline
from visage.classify.scene import ClassificationResult, SceneClassifier
from visage.classify.tag_store import TagStore

# ── Scene Classifier ─────────────────────────────────────────────


class TestSceneClassifier:
    """Test scene classification using feature-based fallback."""

    def test_classify_returns_results(self):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        clf = SceneClassifier()
        results = clf.classify(img)
        assert len(results) >= 1
        for r in results:
            assert isinstance(r, ClassificationResult)
            assert r.category in ("scene", "style")
            assert len(r.tags) > 0

    def test_dark_image_detected_as_night(self):
        img = np.full((480, 640, 3), 30, dtype=np.uint8)  # Very dark
        clf = SceneClassifier()
        results = clf.classify(img)
        scene = [r for r in results if r.category == "scene"][0]
        assert "night" in scene.tags

    def test_sky_detected_with_blue_dominant(self):
        # Blue-dominant bright image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:, :, 2] = 200  # Blue channel high
        img[:, :, 0] = 100
        img[:, :, 1] = 100
        clf = SceneClassifier()
        results = clf.classify(img)
        scene = [r for r in results if r.category == "scene"][0]
        assert "sky" in scene.tags

    def test_bw_detection(self):
        # Grayscale image (zero saturation)
        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        clf = SceneClassifier()
        results = clf.classify(img)
        style = [r for r in results if r.category == "style"][0]
        assert "bw" in style.tags

    def test_landscape_aspect_ratio(self):
        img = np.random.randint(0, 255, (480, 960, 3), dtype=np.uint8)
        clf = SceneClassifier()
        results = clf.classify(img)
        style = [r for r in results if r.category == "style"][0]
        assert "landscape" in style.tags

    def test_portrait_aspect_ratio(self):
        img = np.random.randint(0, 255, (960, 480, 3), dtype=np.uint8)
        clf = SceneClassifier()
        results = clf.classify(img)
        style = [r for r in results if r.category == "style"][0]
        assert "portrait" in style.tags

    def test_forest_detected_with_green(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:, :, 1] = 180  # Green dominant
        img[:, :, 0] = 80
        img[:, :, 2] = 80
        clf = SceneClassifier()
        results = clf.classify(img)
        scene = [r for r in results if r.category == "scene"][0]
        assert "forest" in scene.tags

    def test_general_fallback(self):
        # Neutral image that shouldn't match specific scenes
        img = np.full((480, 640, 3), 150, dtype=np.uint8)
        clf = SceneClassifier()
        results = clf.classify(img)
        scene = [r for r in results if r.category == "scene"][0]
        # Should have at least one tag (general as fallback)
        assert len(scene.tags) >= 1

    def test_onnx_model_missing_graceful(self):
        """When model path doesn't exist, falls back to features."""
        clf = SceneClassifier(model_path="/nonexistent/model.onnx")
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = clf.classify(img)
        assert len(results) >= 1  # Still produces results


# ── CLIP Classifier ──────────────────────────────────────────────


class TestCLIPClassifier:
    """Test CLIP zero-shot classification using feature fallback."""

    def test_classify_returns_results(self):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        clf = CLIPClassifier()
        results = clf.classify(img)
        assert len(results) >= 1
        assert all(hasattr(r, "label") for r in results)
        assert all(hasattr(r, "score") for r in results)

    def test_classify_with_custom_labels(self):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        clf = CLIPClassifier()
        custom = ["a photo of cats", "a photo of dogs", "a photo of cars"]
        results = clf.classify(img, labels=custom, top_k=2)
        assert len(results) <= 2
        assert all(r.label in custom for r in results)

    def test_classify_top_k_respected(self):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        clf = CLIPClassifier()
        results = clf.classify(img, top_k=3)
        assert len(results) <= 3

    def test_extract_features_shape(self):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        clf = CLIPClassifier()
        features = clf.extract_features(img)
        assert features.shape == (64,)
        # Should be approximately normalized
        norm = np.linalg.norm(features)
        assert abs(norm - 1.0) < 0.01 or norm < 0.01

    def test_search_returns_ranked(self):
        clf = CLIPClassifier()
        # Create diverse features
        features = {}
        for i in range(10):
            feat = np.random.randn(64).astype(np.float32)
            feat /= np.linalg.norm(feat) + 1e-8
            features[f"/tmp/img_{i}.jpg"] = feat

        results = clf.search("nature outdoor", features, top_k=5, min_score=0.0)
        assert len(results) <= 5
        # Scores should be descending
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_min_score_filter(self):
        clf = CLIPClassifier()
        features = {"/tmp/test.jpg": np.zeros(64, dtype=np.float32)}
        results = clf.search("test", features, min_score=0.99)
        assert len(results) == 0

    def test_search_dimension_mismatch_skipped(self):
        clf = CLIPClassifier()
        features = {"/tmp/test.jpg": np.zeros(32, dtype=np.float32)}  # Wrong dim
        results = clf.search("test", features, min_score=0.0)
        assert len(results) == 0

    def test_onnx_missing_graceful(self):
        clf = CLIPClassifier(
            text_model_path="/nonexistent/text.onnx",
            vision_model_path="/nonexistent/vision.onnx",
        )
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = clf.classify(img)
        assert len(results) >= 1  # Falls back to features


# ── Tag Store ─────────────────────────────────────────────────────


class TestTagStore:
    """Test SQLite-backed tag storage."""

    def test_store_and_retrieve(self, tmp_path):
        store = TagStore(str(tmp_path))
        store.store_tags(
            "/tmp/photo1.jpg",
            ["sunset", "beach"],
            {"sunset": 0.9, "beach": 0.8},
            category="scene",
        )
        tags = store.get_tags("/tmp/photo1.jpg")
        assert "scene" in tags
        assert "sunset" in tags["scene"]
        assert "beach" in tags["scene"]
        store.close()

    def test_search_by_tag(self, tmp_path):
        store = TagStore(str(tmp_path))
        store.store_tags("/tmp/a.jpg", ["sunset"], {"sunset": 0.9}, category="scene")
        store.store_tags("/tmp/b.jpg", ["sunset"], {"sunset": 0.7}, category="scene")
        store.store_tags("/tmp/c.jpg", ["night"], {"night": 0.95}, category="scene")

        results = store.search_by_tag("sunset")
        assert len(results) == 2
        # Should be sorted by score descending
        assert results[0][1] >= results[1][1]
        store.close()

    def test_search_by_tags_or(self, tmp_path):
        store = TagStore(str(tmp_path))
        store.store_tags("/tmp/a.jpg", ["sunset"], {"sunset": 0.9}, category="scene")
        store.store_tags("/tmp/b.jpg", ["night"], {"night": 0.8}, category="scene")
        store.store_tags("/tmp/c.jpg", ["beach"], {"beach": 0.7}, category="scene")

        # OR query: sunset OR night
        results = store.search_by_tags(["sunset", "night"])
        assert len(results) == 2
        store.close()

    def test_search_min_score(self, tmp_path):
        store = TagStore(str(tmp_path))
        store.store_tags("/tmp/a.jpg", ["sunset"], {"sunset": 0.5}, category="scene")
        store.store_tags("/tmp/b.jpg", ["sunset"], {"sunset": 0.9}, category="scene")

        results = store.search_by_tag("sunset", min_score=0.8)
        assert len(results) == 1
        assert results[0][0] == "/tmp/b.jpg"
        store.close()

    def test_get_all_tagged_paths(self, tmp_path):
        store = TagStore(str(tmp_path))
        store.store_tags("/tmp/a.jpg", ["x"], {"x": 0.5}, category="scene")
        store.store_tags("/tmp/b.jpg", ["y"], {"y": 0.5}, category="scene")

        paths = store.get_all_tagged_paths()
        assert paths == {"/tmp/a.jpg", "/tmp/b.jpg"}
        store.close()

    def test_tag_counts(self, tmp_path):
        store = TagStore(str(tmp_path))
        store.store_tags("/tmp/a.jpg", ["sunset"], {"sunset": 0.9}, category="scene")
        store.store_tags("/tmp/b.jpg", ["sunset"], {"sunset": 0.8}, category="scene")
        store.store_tags("/tmp/c.jpg", ["night"], {"night": 0.95}, category="scene")

        counts = store.get_tag_counts()
        assert counts["sunset"] == 2
        assert counts["night"] == 1
        store.close()

    def test_overwrite_existing_tags(self, tmp_path):
        store = TagStore(str(tmp_path))
        store.store_tags("/tmp/a.jpg", ["old"], {"old": 0.5}, category="scene")
        store.store_tags("/tmp/a.jpg", ["new"], {"new": 0.9}, category="scene")

        tags = store.get_tags("/tmp/a.jpg")
        assert tags["scene"] == ["new"]
        store.close()

    def test_empty_search(self, tmp_path):
        store = TagStore(str(tmp_path))
        results = store.search_by_tags([])
        assert results == []
        store.close()

    def test_nonexistent_image_returns_empty(self, tmp_path):
        store = TagStore(str(tmp_path))
        tags = store.get_tags("/nonexistent.jpg")
        assert tags == {}
        store.close()


# ── Classification Pipeline ──────────────────────────────────────


class TestClassificationPipeline:
    """Test the full classification pipeline orchestration."""

    def test_classify_real_image(self, real_image_path):
        pipeline = ClassificationPipeline()
        result = pipeline.classify_image(real_image_path)
        assert result.image_path == real_image_path
        assert len(result.tags) > 0
        assert "scene" in result.categories
        assert result.elapsed_ms > 0

    def test_classify_with_tag_store(self, real_image_path, tmp_path):
        pipeline = ClassificationPipeline(input_path=str(tmp_path))
        pipeline.classify_image(real_image_path)
        assert pipeline.store is not None

        # Verify tags were persisted
        tags = pipeline.store.get_tags(real_image_path)
        assert len(tags) > 0
        pipeline.close()

    def test_classify_batch(self, real_image_path, real_png_path):
        pipeline = ClassificationPipeline()
        results = pipeline.classify_batch([real_image_path, real_png_path])
        assert len(results) == 2
        assert all(r.tags for r in results)

    def test_classify_batch_handles_errors(self, real_image_path, tmp_path):
        pipeline = ClassificationPipeline()
        results = pipeline.classify_batch([
            real_image_path,
            "/nonexistent/image.jpg",
        ])
        # Should have result for valid image, skip invalid
        assert len(results) == 1
        assert results[0].image_path == real_image_path

    def test_semantic_search_after_classify(self, real_image_path):
        pipeline = ClassificationPipeline()
        pipeline.classify_image(real_image_path)

        results = pipeline.semantic_search("outdoor scene", min_score=0.0)
        assert len(results) >= 1
        # Should find our image
        paths = [p for p, _ in results]
        assert real_image_path in paths
