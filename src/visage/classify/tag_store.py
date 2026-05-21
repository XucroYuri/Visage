"""SQLite-backed tag storage for image classification results.

Persists scene, style, and zero-shot tags per image so they survive
across server restarts and can be queried efficiently.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_DIRNAME = ".visage_cache"
_DB_FILENAME = "tags.db"


class TagStore:
    """Persistent tag storage backed by SQLite.

    Stores tags as a JSON array per image, with a GIN-like tag→image
    reverse index table for fast tag-based queries.
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
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS image_tags (
                image_path TEXT NOT NULL,
                tags TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'auto',
                updated_at REAL NOT NULL,
                PRIMARY KEY (image_path, category)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tag_search
            ON image_tags(tags)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tag_index (
                tag TEXT NOT NULL,
                image_path TEXT NOT NULL,
                score REAL NOT NULL DEFAULT 0.0,
                PRIMARY KEY (tag, image_path)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tag_lookup
            ON tag_index(tag)
        """)
        conn.commit()

    def store_tags(
        self,
        image_path: str,
        tags: list[str],
        scores: dict[str, float],
        category: str = "auto",
    ) -> None:
        """Store classification tags for an image.

        Args:
            image_path: Path to the image.
            tags: List of tag strings.
            scores: Mapping of tag → confidence score.
            category: Tag category (e.g., "scene", "style", "clip").
        """
        import time

        conn = self._connect()
        now = time.time()
        tags_json = json.dumps(tags)

        conn.execute(
            """INSERT OR REPLACE INTO image_tags (image_path, tags, category, updated_at)
               VALUES (?, ?, ?, ?)""",
            (image_path, tags_json, category, now),
        )

        # Update reverse index
        for tag in tags:
            score = scores.get(tag, 0.0)
            conn.execute(
                """INSERT OR REPLACE INTO tag_index (tag, image_path, score)
                   VALUES (?, ?, ?)""",
                (tag, image_path, score),
            )

        conn.commit()

    def get_tags(self, image_path: str) -> dict[str, list[str]]:
        """Get all tags for an image, grouped by category.

        Returns:
            Dict mapping category → list of tags.
        """
        conn = self._connect()
        rows = conn.execute(
            "SELECT category, tags FROM image_tags WHERE image_path = ?",
            (image_path,),
        ).fetchall()
        return {row[0]: json.loads(row[1]) for row in rows}

    def search_by_tag(
        self,
        tag: str,
        min_score: float = 0.0,
        limit: int = 100,
    ) -> list[tuple[str, float]]:
        """Find images matching a tag.

        Args:
            tag: Tag string to search for.
            min_score: Minimum confidence score.
            limit: Maximum results to return.

        Returns:
            List of (image_path, score) sorted by score descending.
        """
        conn = self._connect()
        rows = conn.execute(
            """SELECT image_path, score FROM tag_index
               WHERE tag = ? AND score >= ?
               ORDER BY score DESC LIMIT ?""",
            (tag, min_score, limit),
        ).fetchall()
        return [(row[0], row[1]) for row in rows]

    def search_by_tags(
        self,
        tags: list[str],
        min_score: float = 0.0,
        limit: int = 100,
    ) -> list[tuple[str, float]]:
        """Find images matching ANY of the given tags (OR query).

        Returns combined results with max score across matching tags.
        """
        if not tags:
            return []

        conn = self._connect()
        placeholders = ",".join("?" * len(tags))
        rows = conn.execute(
            f"""SELECT image_path, MAX(score) as max_score
                FROM tag_index
                WHERE tag IN ({placeholders}) AND score >= ?
                GROUP BY image_path
                ORDER BY max_score DESC LIMIT ?""",
            (*tags, min_score, limit),
        ).fetchall()
        return [(row[0], row[1]) for row in rows]

    def get_all_tagged_paths(self) -> set[str]:
        """Get all image paths that have been tagged."""
        conn = self._connect()
        rows = conn.execute("SELECT DISTINCT image_path FROM image_tags").fetchall()
        return {row[0] for row in rows}

    def get_tag_counts(self) -> dict[str, int]:
        """Get count of images per tag."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT tag, COUNT(*) FROM tag_index GROUP BY tag ORDER BY COUNT(*) DESC"
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
