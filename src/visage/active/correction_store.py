"""User correction persistence for active learning.

Stores merge/split/reassign corrections so the system can learn from
user feedback and improve future classifications.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_DIRNAME = ".visage_cache"
_DB_FILENAME = "corrections.db"


class CorrectionStore:
    """SQLite-backed store for user corrections.

    Each correction records:
    - What action was taken (merge, split, reassign, rename)
    - Which face(s) and cluster(s) were affected
    - When it happened
    """

    def __init__(self, input_path: str) -> None:
        cache_dir = os.path.join(input_path, _CACHE_DIRNAME)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self._db_path = os.path.join(cache_dir, _DB_FILENAME)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                face_ids TEXT NOT NULL,
                source_cluster INTEGER,
                target_cluster INTEGER,
                details TEXT,
                created_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_corrections_action
            ON corrections(action)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_corrections_created
            ON corrections(created_at)
        """)
        conn.commit()

    def record_correction(
        self,
        action: str,
        face_ids: list[str],
        source_cluster: int | None = None,
        target_cluster: int | None = None,
        details: dict | None = None,
    ) -> int:
        """Record a user correction.

        Args:
            action: Type of correction ("merge", "split", "reassign", "rename").
            face_ids: List of affected face IDs.
            source_cluster: Original cluster ID.
            target_cluster: New cluster ID.
            details: Optional additional details.

        Returns:
            Row ID of the recorded correction.
        """
        conn = self._connect()
        now = time.time()
        cursor = conn.execute(
            """INSERT INTO corrections
               (action, face_ids, source_cluster, target_cluster, details, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                action,
                json.dumps(face_ids),
                source_cluster,
                target_cluster,
                json.dumps(details) if details else None,
                now,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)

    def get_corrections(
        self,
        action: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get recorded corrections, optionally filtered by action.

        Args:
            action: Filter by action type (None for all).
            limit: Maximum results.

        Returns:
            List of correction dicts sorted by time descending.
        """
        conn = self._connect()
        if action:
            rows = conn.execute(
                """SELECT id, action, face_ids, source_cluster, target_cluster, details, created_at
                   FROM corrections WHERE action = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (action, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT id, action, face_ids, source_cluster, target_cluster, details, created_at
                   FROM corrections
                   ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()

        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "action": row[1],
                "face_ids": json.loads(row[2]),
                "source_cluster": row[3],
                "target_cluster": row[4],
                "details": json.loads(row[5]) if row[5] else None,
                "created_at": row[6],
            })
        return results

    def get_correction_count(self) -> int:
        """Total number of recorded corrections."""
        conn = self._connect()
        row = conn.execute("SELECT COUNT(*) FROM corrections").fetchone()
        return int(row[0]) if row else 0

    def get_correction_stats(self) -> dict[str, int]:
        """Get correction counts by action type."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT action, COUNT(*) FROM corrections GROUP BY action"
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
