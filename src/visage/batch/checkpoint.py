"""Crash-recovery checkpoint for batch processing.

Records progress so that interrupted batch operations can resume
from the last checkpoint instead of starting over.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class Checkpoint:
    """Manages batch processing checkpoints for crash recovery.

    Stores which items have been completed in a JSON file.
    On restart, completed items are skipped.
    """

    def __init__(self, checkpoint_dir: str, name: str = "batch") -> None:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self._path = os.path.join(checkpoint_dir, f"{name}_checkpoint.json")
        self._data: dict = self._load()

    def _load(self) -> dict:
        """Load checkpoint from disk."""
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    data = json.load(f)
                logger.info("Loaded checkpoint: %d items completed", len(data.get("completed", {})))
                return data
            except Exception as e:
                logger.warning("Failed to load checkpoint: %s", e)
        return {"completed": {}, "started_at": time.time()}

    def _save(self) -> None:
        """Persist checkpoint to disk."""
        self._data["updated_at"] = time.time()
        tmp_path = self._path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(self._data, f)
        os.replace(tmp_path, self._path)

    def mark_completed(self, item_id: str, result: dict | None = None) -> None:
        """Mark an item as completed.

        Args:
            item_id: Unique item identifier.
            result: Optional result metadata.
        """
        self._data["completed"][item_id] = {
            "completed_at": time.time(),
            "result": result,
        }
        self._save()

    def is_completed(self, item_id: str) -> bool:
        """Check if an item has been completed."""
        return item_id in self._data["completed"]

    def get_pending(self, all_ids: list[str]) -> list[str]:
        """Filter out already-completed items.

        Args:
            all_ids: Full list of item IDs to process.

        Returns:
            List of IDs not yet completed.
        """
        return [id_ for id_ in all_ids if not self.is_completed(id_)]

    @property
    def completed_count(self) -> int:
        return len(self._data.get("completed", {}))

    @property
    def started_at(self) -> float:
        return self._data.get("started_at", 0.0)

    def clear(self) -> None:
        """Remove the checkpoint file."""
        if os.path.exists(self._path):
            os.remove(self._path)
        self._data = {"completed": {}, "started_at": time.time()}

    def summary(self) -> dict:
        """Get checkpoint summary."""
        return {
            "completed": self.completed_count,
            "started_at": self.started_at,
            "updated_at": self._data.get("updated_at"),
        }
