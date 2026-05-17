from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from pathlib import Path

import numpy as np

from .models import DetectedFace, FaceBox

logger = logging.getLogger(__name__)

_CACHE_DIRNAME = ".visage_cache"
_DB_FILENAME = "embeddings.db"
_CHECKPOINT_FILENAME = "checkpoint.json"


def _file_fingerprint(path: str) -> str | None:
    """Compute a fast fingerprint for a file based on size and mtime.

    Avoids reading the full file content. Good enough for detecting changes.
    Returns None if the file does not exist.
    """
    try:
        stat = os.stat(path)
        return f"{stat.st_size}:{int(stat.st_mtime)}"
    except OSError:
        return None


class EmbeddingCache:
    """SQLite-backed cache for face embeddings.

    Stores embeddings keyed by image path + file fingerprint,
    so unchanged images can skip re-computation on subsequent runs.
    """

    def __init__(self, input_path: str) -> None:
        """Initialize the cache, creating the database if needed.

        Args:
            input_path: The input directory being processed. Cache is stored
                        in a .visage_cache subdirectory inside it.
        """
        self._cache_dir = os.path.join(input_path, _CACHE_DIRNAME)
        Path(self._cache_dir).mkdir(parents=True, exist_ok=True)
        self._db_path = os.path.join(self._cache_dir, _DB_FILENAME)
        self._checkpoint_path = os.path.join(self._cache_dir, _CHECKPOINT_FILENAME)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        """Create the database schema if it doesn't exist."""
        conn = self._connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS face_embeddings (
                image_path TEXT NOT NULL,
                file_fingerprint TEXT NOT NULL,
                face_index INTEGER NOT NULL,
                face_box TEXT NOT NULL,
                confidence REAL NOT NULL,
                embedding BLOB NOT NULL,
                quality REAL,
                model TEXT NOT NULL,
                num_jitters INTEGER NOT NULL,
                PRIMARY KEY (image_path, face_index)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_fingerprint
            ON face_embeddings(image_path, file_fingerprint)
        """)
        # Migration: add quality column if upgrading from older schema
        self._migrate_add_column(conn, "quality", "REAL")
        conn.commit()

    @staticmethod
    def _migrate_add_column(conn: sqlite3.Connection, column: str, col_type: str) -> None:
        """Add a column to the table if it doesn't already exist."""
        try:
            conn.execute(f"ALTER TABLE face_embeddings ADD COLUMN {column} {col_type}")
        except sqlite3.OperationalError:
            pass  # Column already exists

    def _connect(self) -> sqlite3.Connection:
        """Get or create a database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
        return self._conn

    def lookup(
        self,
        image_path: str,
        model: str = "small",
        num_jitters: int = 1,
    ) -> list[DetectedFace] | None:
        """Look up cached embeddings for an image.

        Args:
            image_path: Path to the image file.
            model: Embedding model name.
            num_jitters: Jitter count used for embedding.

        Returns:
            List of DetectedFace with embeddings, or None if cache miss.
        """
        fingerprint = _file_fingerprint(image_path)
        if fingerprint is None:
            return None
        conn = self._connect()

        rows = conn.execute(
            """
            SELECT face_index, face_box, confidence, embedding, quality
            FROM face_embeddings
            WHERE image_path = ? AND file_fingerprint = ?
              AND model = ? AND num_jitters = ?
            ORDER BY face_index
            """,
            (image_path, fingerprint, model, num_jitters),
        ).fetchall()

        if not rows:
            return None

        faces: list[DetectedFace] = []
        for face_index, box_json, confidence, emb_blob, quality in rows:
            box_data = json.loads(box_json)
            face_box = FaceBox(**box_data)
            embedding = np.frombuffer(emb_blob, dtype=np.float64).copy()
            faces.append(DetectedFace(
                face_box=face_box,
                confidence=confidence,
                embedding=embedding,
                quality=quality,
                image_path=image_path,
                face_index=face_index,
            ))

        logger.debug("Cache hit for %s (%d faces)", image_path, len(faces))
        return faces

    def store(
        self,
        image_path: str,
        faces: list[DetectedFace],
        model: str = "small",
        num_jitters: int = 1,
    ) -> None:
        """Store computed embeddings for an image.

        Args:
            image_path: Path to the image file.
            faces: List of DetectedFace with embeddings populated.
            model: Embedding model name.
            num_jitters: Jitter count used for embedding.
        """
        fingerprint = _file_fingerprint(image_path)
        if fingerprint is None:
            logger.warning("Cannot cache embeddings for missing file: %s", image_path)
            return
        conn = self._connect()

        # Delete old entries for this image first (stale fingerprint)
        conn.execute(
            "DELETE FROM face_embeddings WHERE image_path = ?",
            (image_path,),
        )

        for face in faces:
            if face.embedding is None:
                continue
            box_json = json.dumps({
                "top": face.face_box.top,
                "right": face.face_box.right,
                "bottom": face.face_box.bottom,
                "left": face.face_box.left,
            })
            emb_blob = face.embedding.astype(np.float64).tobytes()
            conn.execute(
                """
                INSERT INTO face_embeddings
                    (image_path, file_fingerprint, face_index, face_box,
                     confidence, embedding, quality, model, num_jitters)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (image_path, fingerprint, face.face_index, box_json,
                 face.confidence, emb_blob, face.quality, model, num_jitters),
            )

        conn.commit()
        logger.debug("Cached %d faces for %s", len(faces), image_path)

    def get_stats(self) -> dict[str, int]:
        """Get cache statistics.

        Returns:
            Dict with 'images' and 'faces' counts.
        """
        conn = self._connect()
        images = conn.execute(
            "SELECT COUNT(DISTINCT image_path) FROM face_embeddings"
        ).fetchone()[0]
        faces = conn.execute(
            "SELECT COUNT(*) FROM face_embeddings"
        ).fetchone()[0]
        return {"cached_images": images, "cached_faces": faces}

    def save_checkpoint(self, phase: int, message: str = "") -> None:
        """Save a checkpoint after completing a pipeline phase.

        Args:
            phase: Phase number completed (1-5).
            message: Human-readable description of the checkpoint state.
        """
        stats = self.get_stats()
        data = {
            "phase": phase,
            "message": message,
            "timestamp": time.time(),
            "cached_images": stats["cached_images"],
            "cached_faces": stats["cached_faces"],
        }
        with open(self._checkpoint_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug("Checkpoint saved: phase %d", phase)

    def load_checkpoint(self) -> dict | None:
        """Load an existing checkpoint if one exists.

        Returns:
            Checkpoint dict with phase/message/timestamp, or None.
        """
        if not os.path.exists(self._checkpoint_path):
            return None
        try:
            with open(self._checkpoint_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to read checkpoint file")
            return None

    def clear_checkpoint(self) -> None:
        """Delete the checkpoint file after successful completion."""
        if os.path.exists(self._checkpoint_path):
            os.remove(self._checkpoint_path)
            logger.debug("Checkpoint cleared")

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
