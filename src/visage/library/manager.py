"""Library manager — CRUD operations for photo libraries."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from pathlib import Path

from visage.library.model import Library

logger = logging.getLogger(__name__)

_DB_FILENAME = "libraries.db"


class LibraryManager:
    """Manages multiple photo libraries.

    Each library is an independent photo collection with its own
    workspace, cache, and settings. Libraries are persisted in
    a shared SQLite database.
    """

    def __init__(self, base_path: str | None = None) -> None:
        if base_path is None:
            base_path = os.path.expanduser("~/.visage")
        Path(base_path).mkdir(parents=True, exist_ok=True)
        self._db_path = os.path.join(base_path, _DB_FILENAME)
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
            CREATE TABLE IF NOT EXISTS libraries (
                library_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                input_dir TEXT NOT NULL,
                created_at REAL NOT NULL,
                last_opened_at REAL NOT NULL,
                photo_count INTEGER NOT NULL DEFAULT 0,
                cluster_count INTEGER NOT NULL DEFAULT 0,
                face_count INTEGER NOT NULL DEFAULT 0,
                settings TEXT NOT NULL DEFAULT '{}'
            )
        """)
        conn.commit()

    def create_library(self, name: str, input_dir: str) -> Library:
        """Create a new library.

        Args:
            name: Display name for the library.
            input_dir: Path to the photo directory.

        Returns:
            The created Library object.
        """
        import uuid

        library_id = str(uuid.uuid4())[:8]
        now = time.time()
        lib = Library(
            library_id=library_id,
            name=name,
            input_dir=input_dir,
            created_at=now,
            last_opened_at=now,
        )

        conn = self._connect()
        conn.execute(
            """INSERT INTO libraries
               (library_id, name, input_dir, created_at, last_opened_at, settings)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (library_id, name, input_dir, now, now, "{}"),
        )
        conn.commit()

        logger.info("Created library %s (%s) at %s", name, library_id, input_dir)
        return lib

    def get_library(self, library_id: str) -> Library | None:
        """Get a library by ID."""
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM libraries WHERE library_id = ?",
            (library_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_library(row)

    def list_libraries(self) -> list[Library]:
        """List all libraries sorted by last opened (most recent first)."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM libraries ORDER BY last_opened_at DESC"
        ).fetchall()
        return [self._row_to_library(row) for row in rows]

    def update_library(self, library_id: str, **updates) -> Library | None:
        """Update library fields.

        Args:
            library_id: Library to update.
            **updates: Fields to update (name, photo_count, etc.).

        Returns:
            Updated Library or None if not found.
        """
        lib = self.get_library(library_id)
        if lib is None:
            return None

        conn = self._connect()
        for key, value in updates.items():
            if key == "settings":
                conn.execute(
                    "UPDATE libraries SET settings = ? WHERE library_id = ?",
                    (json.dumps(value), library_id),
                )
            elif key in ("name", "input_dir", "photo_count", "cluster_count", "face_count"):
                conn.execute(
                    f"UPDATE libraries SET {key} = ? WHERE library_id = ?",
                    (value, library_id),
                )

        conn.execute(
            "UPDATE libraries SET last_opened_at = ? WHERE library_id = ?",
            (time.time(), library_id),
        )
        conn.commit()

        return self.get_library(library_id)

    def delete_library(self, library_id: str) -> bool:
        """Delete a library by ID.

        Returns:
            True if deleted, False if not found.
        """
        conn = self._connect()
        cursor = conn.execute(
            "DELETE FROM libraries WHERE library_id = ?",
            (library_id,),
        )
        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("Deleted library %s", library_id)
        return deleted

    def _row_to_library(self, row: tuple) -> Library:
        return Library(
            library_id=row[0],
            name=row[1],
            input_dir=row[2],
            created_at=row[3],
            last_opened_at=row[4],
            photo_count=row[5],
            cluster_count=row[6],
            face_count=row[7],
            settings=json.loads(row[8]) if row[8] else {},
        )

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
