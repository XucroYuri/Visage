"""Universal image classification — scene, style, object, and zero-shot CLIP tags."""

from visage.classify.clip_model import CLIPClassifier
from visage.classify.pipeline import ClassificationPipeline
from visage.classify.scene import SceneClassifier
from visage.classify.tag_store import TagStore

__all__ = [
    "SceneClassifier",
    "CLIPClassifier",
    "TagStore",
    "ClassificationPipeline",
]
