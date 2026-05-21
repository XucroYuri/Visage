"""Request batching for embedding inference — merges pending requests into GPU-friendly batches."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from visage.models import FaceBox

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingRequest:
    """A single embedding generation request."""

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    image: np.ndarray | None = None  # RGB array
    face_box: FaceBox | None = None
    priority: str = "low"  # "high" (user-initiated) or "low" (background)
    # Set by the batcher when result is ready (lazy-init on first access)
    _future: asyncio.Future[Any] | None = field(default=None, repr=False)

    def get_future(self) -> asyncio.Future[Any]:
        """Lazily create the asyncio Future on first access."""
        if self._future is None:
            self._future = asyncio.get_event_loop().create_future()
        return self._future


@dataclass
class BatchResult:
    """Result of processing a batch of embedding requests."""

    request_ids: list[str]
    embeddings: list[np.ndarray | None]
    error: str | None = None


class RequestBatcher:
    """Collects embedding requests and merges them into batches for efficient inference.

    Batches accumulate until either:
    - max_batch_size requests are pending, OR
    - max_wait_ms has elapsed since the first pending request

    High-priority requests are processed before low-priority ones.
    """

    def __init__(
        self,
        max_batch_size: int = 16,
        max_wait_ms: int = 200,
        min_batch_size: int = 1,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.min_batch_size = min_batch_size
        self._high_queue: list[EmbeddingRequest] = []
        self._low_queue: list[EmbeddingRequest] = []
        self._lock = asyncio.Lock()

    async def submit(self, request: EmbeddingRequest) -> np.ndarray | None:
        """Submit a request and wait for its result.

        Returns:
            The embedding vector, or None if generation failed.
        """
        async with self._lock:
            if request.priority == "high":
                self._high_queue.append(request)
            else:
                self._low_queue.append(request)
        return await request.get_future()

    def drain(self) -> list[EmbeddingRequest]:
        """Drain all pending requests, high-priority first.

        Returns up to max_batch_size requests.
        """
        batch: list[EmbeddingRequest] = []

        # High priority first
        while self._high_queue and len(batch) < self.max_batch_size:
            batch.append(self._high_queue.pop(0))

        # Fill remaining with low priority
        while self._low_queue and len(batch) < self.max_batch_size:
            batch.append(self._low_queue.pop(0))

        return batch

    @property
    def pending_count(self) -> int:
        """Total number of pending requests."""
        return len(self._high_queue) + len(self._low_queue)

    def should_flush(self) -> bool:
        """Whether the batcher should flush its current queue."""
        total = self.pending_count
        if total >= self.max_batch_size:
            return True
        if total >= self.min_batch_size and self._high_queue:
            return True  # Flush immediately for high-priority requests
        return False
