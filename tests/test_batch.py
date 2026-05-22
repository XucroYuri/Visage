"""Tests for batch processing — queue, checkpoint, three-tier cache."""

from __future__ import annotations

import time

from visage.batch.checkpoint import Checkpoint
from visage.batch.queue import BatchQueue, Priority
from visage.core import ThreeTierCache


class TestBatchQueue:
    """Test priority batch queue."""

    def test_submit_and_process(self):
        queue = BatchQueue()
        results = []
        queue.submit("item_1", payload="hello", callback=lambda r: results.append(r))
        queue.start(processor=lambda p: f"processed_{p}")
        time.sleep(0.5)
        queue.stop()
        assert len(results) == 1
        assert results[0] == "processed_hello"

    def test_priority_ordering(self):
        queue = BatchQueue()
        order: list[str] = []

        queue.submit("bg", payload="bg", priority=Priority.BACKGROUND)
        queue.submit("user", payload="user", priority=Priority.USER_ACTION)
        queue.submit("maint", payload="maint", priority=Priority.MAINTENANCE)

        queue.start(processor=lambda p: order.append(p))
        time.sleep(0.5)
        queue.stop()

        assert order == ["user", "bg", "maint"]

    def test_pending_count(self):
        queue = BatchQueue()
        assert queue.pending_count == 0
        queue.submit("a", payload="a")
        assert queue.pending_count == 1

    def test_completed_count(self):
        queue = BatchQueue()
        queue.submit("a", payload="a")
        queue.start(processor=lambda p: p)
        time.sleep(0.5)
        assert queue.completed_count == 1
        queue.stop()

    def test_get_result(self):
        queue = BatchQueue()
        queue.submit("r1", payload=42)
        queue.start(processor=lambda p: p * 2)
        time.sleep(0.5)
        queue.stop()
        result = queue.get_result("r1")
        assert result == 84

    def test_error_handling(self):
        queue = BatchQueue()
        queue.submit("fail", payload="bad")

        def failing_processor(payload):
            raise ValueError("test error")

        queue.start(processor=failing_processor)
        time.sleep(0.5)
        queue.stop()
        result = queue.get_result("fail")
        assert result is not None
        assert "error" in result

    def test_multiple_workers(self):
        queue = BatchQueue(max_workers=2)
        for i in range(5):
            queue.submit(f"item_{i}", payload=i)
        queue.start(processor=lambda p: p)
        time.sleep(1)
        queue.stop()
        assert queue.completed_count == 5

    def test_stop_without_start(self):
        queue = BatchQueue()
        queue.stop()  # Should not raise


class TestCheckpoint:
    """Test crash-recovery checkpoints."""

    def test_mark_and_check(self, tmp_path):
        cp = Checkpoint(str(tmp_path), name="test")
        assert not cp.is_completed("img_001")
        cp.mark_completed("img_001")
        assert cp.is_completed("img_001")

    def test_get_pending(self, tmp_path):
        cp = Checkpoint(str(tmp_path), name="test")
        cp.mark_completed("img_001")
        cp.mark_completed("img_003")
        pending = cp.get_pending(["img_001", "img_002", "img_003", "img_004"])
        assert pending == ["img_002", "img_004"]

    def test_completed_count(self, tmp_path):
        cp = Checkpoint(str(tmp_path), name="test")
        assert cp.completed_count == 0
        cp.mark_completed("a")
        cp.mark_completed("b")
        assert cp.completed_count == 2

    def test_persistence(self, tmp_path):
        cp1 = Checkpoint(str(tmp_path), name="test")
        cp1.mark_completed("persistent_item")
        cp1._save()  # Ensure saved

        cp2 = Checkpoint(str(tmp_path), name="test")
        assert cp2.is_completed("persistent_item")

    def test_clear(self, tmp_path):
        cp = Checkpoint(str(tmp_path), name="test")
        cp.mark_completed("a")
        cp.clear()
        assert not cp.is_completed("a")

    def test_summary(self, tmp_path):
        cp = Checkpoint(str(tmp_path), name="test")
        cp.mark_completed("a")
        summary = cp.summary()
        assert summary["completed"] == 1
        assert summary["started_at"] > 0

    def test_with_result_metadata(self, tmp_path):
        cp = Checkpoint(str(tmp_path), name="test")
        cp.mark_completed("img_001", result={"faces": 2, "time_ms": 50})
        assert cp.is_completed("img_001")


class TestThreeTierCache:
    """Test three-tier LRU+SQLite cache."""

    def test_put_and_get(self, tmp_path):
        cache = ThreeTierCache(str(tmp_path))
        cache.put("/tmp/photo.jpg", b"image_data", "thumb")
        result = cache.get("/tmp/photo.jpg", "thumb")
        assert result == b"image_data"
        cache.close()

    def test_cache_miss(self, tmp_path):
        cache = ThreeTierCache(str(tmp_path))
        result = cache.get("/nonexistent.jpg", "thumb")
        assert result is None
        cache.close()

    def test_l1_eviction(self, tmp_path):
        cache = ThreeTierCache(str(tmp_path), max_memory_items=3)
        for i in range(5):
            cache.put(f"/img_{i}.jpg", f"data_{i}".encode(), "thumb")

        # Only last 3 should be in L1
        stats = cache.stats()
        assert stats["l1_items"] == 3
        cache.close()

    def test_l2_persistence(self, tmp_path):
        cache1 = ThreeTierCache(str(tmp_path))
        cache1.put("/photo.jpg", b"persistent_data", "thumb")
        cache1.close()

        cache2 = ThreeTierCache(str(tmp_path))
        result = cache2.get("/photo.jpg", "thumb")
        assert result == b"persistent_data"
        cache2.close()

    def test_different_sizes(self, tmp_path):
        cache = ThreeTierCache(str(tmp_path))
        cache.put("/photo.jpg", b"thumb_data", "thumb")
        cache.put("/photo.jpg", b"full_data", "full")
        assert cache.get("/photo.jpg", "thumb") == b"thumb_data"
        assert cache.get("/photo.jpg", "full") == b"full_data"
        cache.close()

    def test_stats(self, tmp_path):
        cache = ThreeTierCache(str(tmp_path))
        cache.put("/a.jpg", b"data_a", "thumb")
        cache.put("/b.jpg", b"data_b", "thumb")
        stats = cache.stats()
        assert stats["l1_items"] == 2
        assert stats["l2_items"] == 2
        assert stats["l2_size_mb"] >= 0  # Small data rounds to 0.00 MB
        cache.close()

    def test_clear(self, tmp_path):
        cache = ThreeTierCache(str(tmp_path))
        cache.put("/a.jpg", b"data", "thumb")
        cache.clear()
        assert cache.get("/a.jpg", "thumb") is None
        assert cache.stats()["l1_items"] == 0
        cache.close()

    def test_overwrite(self, tmp_path):
        cache = ThreeTierCache(str(tmp_path))
        cache.put("/a.jpg", b"old_data", "thumb")
        cache.put("/a.jpg", b"new_data", "thumb")
        assert cache.get("/a.jpg", "thumb") == b"new_data"
        cache.close()
