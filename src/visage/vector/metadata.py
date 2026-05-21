"""SQLite metadata store for face and cluster metadata."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MetadataStore:
    """SQLite-backed metadata for face embeddings and cluster assignments.

    Stores face_id → (image_path, cluster_id, quality_score, embedding_backend, extra)
    with indexes on cluster_id and image_path for fast lookups.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS faces (
                face_id TEXT PRIMARY KEY,
                image_path TEXT NOT NULL,
                cluster_id TEXT,
                embedding_backend TEXT NOT NULL DEFAULT 'insightface',
                quality_score REAL DEFAULT 0.0,
                bbox TEXT,
                extra TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_cluster ON faces(cluster_id);
            CREATE INDEX IF NOT EXISTS idx_image ON faces(image_path);
        """)
        self._conn.commit()

    def add_face(
        self,
        face_id: str,
        image_path: str,
        cluster_id: str | None = None,
        embedding_backend: str = "insightface",
        quality_score: float = 0.0,
        bbox: tuple[int, int, int, int] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Insert or replace a face metadata record."""
        self._conn.execute(
            """INSERT OR REPLACE INTO faces
               (face_id, image_path, cluster_id, embedding_backend, quality_score, bbox, extra)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                face_id,
                image_path,
                cluster_id,
                embedding_backend,
                quality_score,
                json.dumps(bbox) if bbox else None,
                json.dumps(extra) if extra else None,
            ),
        )
        self._conn.commit()

    def get_face(self, face_id: str) -> dict[str, Any] | None:
        """Get face metadata by ID."""
        row = self._conn.execute(
            "SELECT * FROM faces WHERE face_id = ?", (face_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def get_faces_by_cluster(self, cluster_id: str) -> list[dict[str, Any]]:
        """Get all faces in a cluster."""
        rows = self._conn.execute(
            "SELECT * FROM faces WHERE cluster_id = ?", (cluster_id,)
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_faces_by_image(self, image_path: str) -> list[dict[str, Any]]:
        """Get all faces from a specific image."""
        rows = self._conn.execute(
            "SELECT * FROM faces WHERE image_path = ?", (image_path,)
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def update_cluster(self, face_id: str, cluster_id: str | None) -> None:
        """Update cluster assignment for a face."""
        self._conn.execute(
            "UPDATE faces SET cluster_id = ? WHERE face_id = ?",
            (cluster_id, face_id),
        )
        self._conn.commit()

    def delete_face(self, face_id: str) -> bool:
        """Delete a face record. Returns True if deleted."""
        cursor = self._conn.execute(
            "DELETE FROM faces WHERE face_id = ?", (face_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def get_all_face_ids(self) -> list[str]:
        """Get all face IDs in the store."""
        rows = self._conn.execute("SELECT face_id FROM faces").fetchall()
        return [r[0] for r in rows]

    def count_faces(self) -> int:
        """Total number of faces."""
        row = self._conn.execute("SELECT COUNT(*) FROM faces").fetchone()
        return row[0]

    def count_clusters(self) -> int:
        """Number of distinct non-null cluster IDs."""
        row = self._conn.execute(
            "SELECT COUNT(DISTINCT cluster_id) FROM faces WHERE cluster_id IS NOT NULL"
        ).fetchone()
        return row[0]

    def get_cluster_ids(self) -> list[str]:
        """Get all distinct cluster IDs."""
        rows = self._conn.execute(
            "SELECT DISTINCT cluster_id FROM faces WHERE cluster_id IS NOT NULL ORDER BY cluster_id"
        ).fetchall()
        return [r[0] for r in rows]

    def batch_update_clusters(self, assignments: list[tuple[str, str | None]]) -> None:
        """Update cluster assignments for multiple faces in a single transaction."""
        self._conn.executemany(
            "UPDATE faces SET cluster_id = ? WHERE face_id = ?",
            [(cid, fid) for fid, cid in assignments],
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        if d.get("bbox"):
            d["bbox"] = tuple(json.loads(d["bbox"]))
        if d.get("extra"):
            d["extra"] = json.loads(d["extra"])
        return d
