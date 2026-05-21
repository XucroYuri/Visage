"""Active learning — prototype vectors, user corrections, adaptive thresholds."""

from visage.active.correction_store import CorrectionStore
from visage.active.nearest_centroid import NearestCentroidClassifier
from visage.active.prototype import PrototypeManager
from visage.active.threshold_adapter import ThresholdAdapter

__all__ = [
    "PrototypeManager",
    "NearestCentroidClassifier",
    "CorrectionStore",
    "ThresholdAdapter",
]
