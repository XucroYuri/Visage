"""Tests for the embedding service package."""

from __future__ import annotations

import pytest

from visage.embedding.backend import create_backend
from visage.embedding.batcher import EmbeddingRequest, RequestBatcher
from visage.embedding.gpu import DeviceInfo, detect_device

# ── GPU detection tests ───────────────────────────────────────────


class TestDeviceInfo:
    def test_cpu_device_is_not_gpu(self):
        info = DeviceInfo(device="cpu", name="Test CPU", supports_half=False)
        assert not info.is_gpu

    def test_cuda_device_is_gpu(self):
        info = DeviceInfo(device="cuda", name="Test GPU", supports_half=True)
        assert info.is_gpu

    def test_mps_device_is_gpu(self):
        info = DeviceInfo(device="mps", name="Apple MPS", supports_half=True)
        assert info.is_gpu


class TestDetectDevice:
    def test_detect_returns_device_info(self):
        info = detect_device()
        assert isinstance(info, DeviceInfo)
        assert info.device in ("cpu", "cuda", "mps")

    def test_force_cpu(self):
        info = detect_device(prefer="cpu")
        assert info.device == "cpu"
        assert not info.is_gpu

    def test_force_cpu_always_succeeds(self):
        """CPU fallback should never fail."""
        info = detect_device(prefer="cpu")
        assert info.name  # Should have a name


# ── Backend factory tests ─────────────────────────────────────────


class TestCreateBackend:
    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            create_backend("nonexistent")

    def test_dlib_backend_type(self):
        backend = create_backend("dlib")
        assert hasattr(backend, "generate")
        assert hasattr(backend, "is_available")
        assert backend.name == "dlib"
        assert backend.embedding_dim == 128

    def test_insightface_backend_type(self):
        backend = create_backend("insightface")
        assert hasattr(backend, "generate")
        assert hasattr(backend, "is_available")
        assert backend.name == "insightface"
        assert backend.embedding_dim == 512


# ── Batcher tests ─────────────────────────────────────────────────


class TestEmbeddingRequest:
    def test_request_has_id(self):
        req = EmbeddingRequest()
        assert req.request_id
        assert len(req.request_id) == 12

    def test_request_default_priority(self):
        req = EmbeddingRequest()
        assert req.priority == "low"

    def test_request_high_priority(self):
        req = EmbeddingRequest(priority="high")
        assert req.priority == "high"


class TestRequestBatcher:
    def test_initial_state(self):
        batcher = RequestBatcher()
        assert batcher.pending_count == 0

    def test_drain_empty(self):
        batcher = RequestBatcher()
        batch = batcher.drain()
        assert batch == []

    def test_drain_respects_max_batch_size(self):
        batcher = RequestBatcher(max_batch_size=3)
        for _ in range(5):
            batcher._low_queue.append(EmbeddingRequest())
        batch = batcher.drain()
        assert len(batch) == 3

    def test_high_priority_drained_first(self):
        batcher = RequestBatcher(max_batch_size=4)
        low_req = EmbeddingRequest(priority="low")
        high_req = EmbeddingRequest(priority="high")
        batcher._low_queue.append(low_req)
        batcher._high_queue.append(high_req)
        batch = batcher.drain()
        assert batch[0] is high_req
        assert batch[1] is low_req

    def test_should_flush_at_max_size(self):
        batcher = RequestBatcher(max_batch_size=2)
        assert not batcher.should_flush()
        batcher._low_queue.append(EmbeddingRequest())
        batcher._low_queue.append(EmbeddingRequest())
        assert batcher.should_flush()

    def test_should_flush_for_high_priority(self):
        batcher = RequestBatcher(max_batch_size=10)
        batcher._high_queue.append(EmbeddingRequest(priority="high"))
        assert batcher.should_flush()

    def test_pending_count(self):
        batcher = RequestBatcher()
        batcher._high_queue.append(EmbeddingRequest())
        batcher._low_queue.append(EmbeddingRequest())
        batcher._low_queue.append(EmbeddingRequest())
        assert batcher.pending_count == 3

    def test_drain_returns_correct_requests(self):
        """Drain pulls requests in priority order and returns the batch."""
        batcher = RequestBatcher(max_batch_size=5)
        low1 = EmbeddingRequest(priority="low")
        low2 = EmbeddingRequest(priority="low")
        high1 = EmbeddingRequest(priority="high")
        batcher._low_queue.extend([low1, low2])
        batcher._high_queue.append(high1)

        batch = batcher.drain()
        assert len(batch) == 3
        assert batch[0] is high1  # high priority first
        assert batch[1] is low1
        assert batch[2] is low2
        assert batcher.pending_count == 0  # all drained


# ── Service model tests ───────────────────────────────────────────


class TestServiceModels:
    def test_embed_request_defaults(self):
        from visage.embedding.service import EmbedRequest
        req = EmbedRequest(bbox=[10, 110, 110, 10])
        assert req.priority == "low"
        assert req.face_id == ""
        assert req.image_b64 == ""

    def test_embed_response_no_embedding(self):
        from visage.embedding.service import EmbedResponse
        resp = EmbedResponse(face_id="test", error="failed")
        assert resp.embedding is None
        assert resp.error == "failed"

    def test_hot_swap_request(self):
        from visage.embedding.service import HotSwapRequest
        req = HotSwapRequest(backend="dlib")
        assert req.backend == "dlib"

    def test_status_response(self):
        from visage.embedding.service import StatusResponse
        resp = StatusResponse(status="ready", backend="insightface", device="cpu")
        assert resp.status == "ready"
        assert resp.uptime_seconds == 0.0
