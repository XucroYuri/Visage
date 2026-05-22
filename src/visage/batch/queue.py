"""Priority-based batch processing queue.

Supports ordered processing of image batches with priority levels
so that user-visible operations complete before background tasks.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    """Batch item priority — lower values processed first."""
    USER_ACTION = 0  # User-initiated (merge, move, etc.)
    INTERACTIVE = 1  # Search, classification
    BACKGROUND = 2   # Batch processing, indexing
    MAINTENANCE = 3  # Cache cleanup, optimization


@dataclass(order=True)
class BatchItem:
    """An item in the batch queue."""
    sort_key: tuple[int, float] = field(compare=True)
    item_id: str = field(compare=False)
    payload: Any = field(compare=False, default=None)
    callback: Callable | None = field(compare=False, default=None)


class BatchQueue:
    """Thread-safe priority queue for batch processing.

    Items are processed in priority order. Within the same priority,
    items are processed in FIFO order (by submission time).

    Usage:
        queue = BatchQueue()
        queue.submit("detect_001", image_paths, priority=Priority.BACKGROUND)
        queue.start(worker_fn)
    """

    def __init__(self, max_workers: int = 1) -> None:
        self._max_workers = max_workers
        self._queue: list[BatchItem] = []
        self._lock = threading.Lock()
        self._counter = 0
        self._running = False
        self._workers: list[threading.Thread] = []
        self._completed: dict[str, Any] = {}
        self._stop_event = threading.Event()

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def completed_count(self) -> int:
        with self._lock:
            return len(self._completed)

    def submit(
        self,
        item_id: str,
        payload: Any = None,
        priority: Priority = Priority.BACKGROUND,
        callback: Callable | None = None,
    ) -> None:
        """Submit an item to the batch queue.

        Args:
            item_id: Unique identifier for this batch item.
            payload: Data to process.
            priority: Processing priority.
            callback: Optional callback(processor_result) when done.
        """
        import time

        with self._lock:
            self._counter += 1
            item = BatchItem(
                sort_key=(int(priority), time.time()),
                item_id=item_id,
                payload=payload,
                callback=callback,
            )
            self._queue.append(item)
            # Sort by priority, then by submission time
            self._queue.sort()

        logger.debug("Submitted %s (priority=%s)", item_id, priority.name)

    def start(self, processor: Callable) -> None:
        """Start background workers processing the queue.

        Args:
            processor: Callable(payload) -> result for each item.
        """
        self._running = True
        self._stop_event.clear()

        for i in range(self._max_workers):
            t = threading.Thread(
                target=self._worker_loop,
                args=(processor, f"worker-{i}"),
                daemon=True,
            )
            t.start()
            self._workers.append(t)

        logger.info("Started %d batch workers", self._max_workers)

    def stop(self) -> None:
        """Signal workers to stop and wait for them to finish."""
        self._running = False
        self._stop_event.set()
        for t in self._workers:
            t.join(timeout=5.0)
        self._workers.clear()
        logger.info("Batch workers stopped")

    def get_result(self, item_id: str) -> Any | None:
        """Get the result of a completed item."""
        with self._lock:
            return self._completed.pop(item_id, None)

    def _worker_loop(self, processor: Callable, name: str) -> None:
        """Worker thread main loop."""
        while self._running and not self._stop_event.is_set():
            item = self._dequeue()
            if item is None:
                self._stop_event.wait(0.5)
                continue

            try:
                result = processor(item.payload)
                with self._lock:
                    self._completed[item.item_id] = result
                if item.callback:
                    item.callback(result)
                logger.debug("Completed %s", item.item_id)
            except Exception as e:
                logger.error("Worker %s failed on %s: %s", name, item.item_id, e)
                with self._lock:
                    self._completed[item.item_id] = {"error": str(e)}

    def _dequeue(self) -> BatchItem | None:
        """Pop the highest-priority item from the queue."""
        with self._lock:
            if not self._queue:
                return None
            return self._queue.pop(0)
