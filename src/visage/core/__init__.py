"""Three-tier cache — memory LRU → disk SQLite → original file."""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import threading
from collections import OrderedDict
from pathlib import Path

logger = logging.getLogger(__name__)


class ThreeTierCache:
    """Three-tier caching for thumbnail/feature data.

    L1: In-memory LRU cache (fastest, bounded)
    L2: SQLite disk cache (persists across restarts)
    L3: Original file access (slowest, always available)

    Lookups cascade through L1 → L2 → L3.
    """

    def __init__(
        self,
        cache_dir: str,
        max_memory_items: int = 500,
        max_disk_mb: int = 500,
    ) -> None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self._l1: OrderedDict[str, bytes] = OrderedDict()
        self._l1_max = max_memory_items
        self._max_disk_mb = max_disk_mb
        self._db_path = os.path.join(cache_dir, "cache.db")
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                key TEXT PRIMARY KEY,
                data BLOB NOT NULL,
                size INTEGER NOT NULL,
                accessed_at REAL NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_accessed
            ON cache_entries(accessed_at)
        """)
        conn.commit()

    @staticmethod
    def _make_key(path: str, size: str) -> str:
        raw = f"{path}:{size}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, path: str, size: str = "thumb") -> bytes | None:
        """Look up cached data through L1 → L2.

        Args:
            path: Image file path.
            size: Size identifier ("thumb", "full").

        Returns:
            Cached bytes or None if not found.
        """
        import time

        key = self._make_key(path, size)

        # L1: Memory
        with self._lock:
            if key in self._l1:
                self._l1.move_to_end(key)
                return self._l1[key]

        # L2: Disk SQLite
        conn = self._connect()
        row = conn.execute(
            "SELECT data FROM cache_entries WHERE key = ?",
            (key,),
        ).fetchone()
        if row is not None:
            data = row[0]
            # Promote to L1
            with self._lock:
                self._l1[key] = data
                self._l1.move_to_end(key)
                self._evict_l1()
            # Update access time
            conn.execute(
                "UPDATE cache_entries SET accessed_at = ? WHERE key = ?",
                (time.time(), key),
            )
            conn.commit()
            return data

        return None

    def put(self, path: str, data: bytes, size: str = "thumb") -> None:
        """Store data in L1 and L2.

        Args:
            path: Image file path.
            data: Cached bytes.
            size: Size identifier.
        """
        import time

        key = self._make_key(path, size)
        now = time.time()

        # L1: Memory
        with self._lock:
            self._l1[key] = data
            self._l1.move_to_end(key)
            self._evict_l1()

        # L2: Disk
        conn = self._connect()
        conn.execute(
            """INSERT OR REPLACE INTO cache_entries (key, data, size, accessed_at, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (key, data, len(data), now, now),
        )
        conn.commit()
        self._evict_l2()

    def _evict_l1(self) -> None:
        """Evict oldest items from L1 if over capacity."""
        while len(self._l1) > self._l1_max:
            self._l1.popitem(last=False)

    def _evict_l2(self) -> None:
        """Evict oldest items from L2 if over disk size limit."""
        conn = self._connect()
        total = conn.execute("SELECT SUM(size) FROM cache_entries").fetchone()[0]
        if total is None:
            return
        limit = self._max_disk_mb * 1024 * 1024
        if total <= limit:
            return

        # Delete oldest entries until under limit
        rows = conn.execute(
            "SELECT key, size FROM cache_entries ORDER BY accessed_at ASC"
        ).fetchall()
        for key, size in rows:
            conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
            total -= size
            if total <= limit * 0.8:  # Evict to 80% of limit
                break
        conn.commit()

    def stats(self) -> dict:
        """Get cache statistics."""
        conn = self._connect()
        row = conn.execute(
            "SELECT COUNT(*), SUM(size) FROM cache_entries"
        ).fetchone()
        disk_count = row[0] or 0
        disk_size = row[1] or 0
        return {
            "l1_items": len(self._l1),
            "l1_max": self._l1_max,
            "l2_items": disk_count,
            "l2_size_mb": round(disk_size / (1024 * 1024), 2),
            "l2_max_mb": self._max_disk_mb,
        }

    def clear(self) -> None:
        """Clear all cache levels."""
        with self._lock:
            self._l1.clear()
        conn = self._connect()
        conn.execute("DELETE FROM cache_entries")
        conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
