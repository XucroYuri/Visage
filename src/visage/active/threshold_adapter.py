"""Adaptive threshold for cluster assignment confidence.

Adjusts the similarity threshold for cluster assignments based on
user correction patterns. If users frequently split clusters, the
threshold increases (stricter). If users frequently merge, it decreases.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Default thresholds
DEFAULT_THRESHOLD = 0.70
MIN_THRESHOLD = 0.40
MAX_THRESHOLD = 0.95
ADAPTATION_RATE = 0.02  # How much to adjust per correction


class ThresholdAdapter:
    """Adapts cluster assignment threshold based on user feedback.

    Tracks merge vs split corrections and adjusts the threshold
    to reduce future corrections needed.
    """

    def __init__(
        self,
        initial_threshold: float = DEFAULT_THRESHOLD,
        adaptation_rate: float = ADAPTATION_RATE,
    ) -> None:
        self._threshold = initial_threshold
        self._adaptation_rate = adaptation_rate
        self._merge_count = 0
        self._split_count = 0

    @property
    def threshold(self) -> float:
        return self._threshold

    def record_merge(self) -> float:
        """Record that a user merged two clusters.

        Lowers the threshold to make future assignments more lenient.
        """
        self._merge_count += 1
        self._threshold = max(
            MIN_THRESHOLD,
            self._threshold - self._adaptation_rate,
        )
        logger.debug(
            "Merge recorded: threshold → %.3f (merges=%d, splits=%d)",
            self._threshold, self._merge_count, self._split_count,
        )
        return self._threshold

    def record_split(self) -> float:
        """Record that a user split a cluster.

        Raises the threshold to make future assignments stricter.
        """
        self._split_count += 1
        self._threshold = min(
            MAX_THRESHOLD,
            self._threshold + self._adaptation_rate,
        )
        logger.debug(
            "Split recorded: threshold → %.3f (merges=%d, splits=%d)",
            self._threshold, self._merge_count, self._split_count,
        )
        return self._threshold

    def record_reassign(self, old_cluster: int, new_cluster: int) -> float:
        """Record a reassignment. Adjusts threshold based on direction."""
        # Reassign to existing cluster → similar to merge
        # Reassign to new cluster → similar to split
        if new_cluster == -1:
            return self.record_split()
        return self.record_merge()

    @property
    def stats(self) -> dict:
        """Get current adapter statistics."""
        return {
            "threshold": round(self._threshold, 4),
            "merge_count": self._merge_count,
            "split_count": self._split_count,
            "total_corrections": self._merge_count + self._split_count,
            "merge_ratio": (
                self._merge_count / max(self._merge_count + self._split_count, 1)
            ),
        }

    def to_dict(self) -> dict:
        """Serialize adapter state."""
        return {
            "threshold": self._threshold,
            "adaptation_rate": self._adaptation_rate,
            "merge_count": self._merge_count,
            "split_count": self._split_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ThresholdAdapter:
        """Restore adapter from serialized state."""
        adapter = cls(
            initial_threshold=data.get("threshold", DEFAULT_THRESHOLD),
            adaptation_rate=data.get("adaptation_rate", ADAPTATION_RATE),
        )
        adapter._merge_count = data.get("merge_count", 0)
        adapter._split_count = data.get("split_count", 0)
        return adapter

    def reset(self) -> None:
        """Reset to defaults."""
        self._threshold = DEFAULT_THRESHOLD
        self._merge_count = 0
        self._split_count = 0
