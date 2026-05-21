"""Time-based event clustering — groups photos into events by EXIF timestamps.

Algorithm:
1. Sort photos by DateTimeOriginal (fall back to file mtime)
2. Within same day: gap < 4h → same event
3. Consecutive days: check if part of same multi-day event (e.g. trip)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# Maximum gap between photos in the same event (hours)
SAME_EVENT_GAP_HOURS = 4.0
# Maximum gap for consecutive-day merging (hours)
MULTI_DAY_GAP_HOURS = 12.0


@dataclass
class Event:
    """A detected event (album) from time clustering."""

    event_id: str
    start_time: datetime
    end_time: datetime
    photo_paths: list[str] = field(default_factory=list)
    name: str = ""
    cover_path: str | None = None
    location_name: str | None = None

    @property
    def duration(self) -> timedelta:
        return self.end_time - self.start_time

    @property
    def photo_count(self) -> int:
        return len(self.photo_paths)

    @property
    def is_multi_day(self) -> bool:
        return self.start_time.date() != self.end_time.date()


@dataclass
class PhotoTimestamp:
    """Photo with its timestamp for clustering."""

    path: str
    timestamp: datetime


def extract_timestamps(image_paths: list[str]) -> list[PhotoTimestamp]:
    """Extract timestamps from photos using EXIF data.

    Falls back to file modification time when EXIF is unavailable.
    """

    timestamps: list[PhotoTimestamp] = []

    for path in image_paths:
        ts = _read_exif_timestamp(path)
        if ts is None:
            ts = _read_file_mtime(path)
        if ts is not None:
            timestamps.append(PhotoTimestamp(path=path, timestamp=ts))

    # Sort chronologically
    timestamps.sort(key=lambda p: p.timestamp)
    return timestamps


def _read_exif_timestamp(path: str) -> datetime | None:
    """Read DateTimeOriginal from EXIF data."""
    try:
        from PIL import Image
        from PIL.ExifTags import Base as ExifBase

        with Image.open(path) as img:
            exif = img.getexif()
            if exif is None:
                return None
            # DateTimeOriginal = tag 36867
            date_str = (
                exif.get(ExifBase.DateTimeOriginal)
                or exif.get(ExifBase.DateTimeDigitized)
                or exif.get(ExifBase.DateTime)
            )
            if date_str is None:
                return None
            return _parse_exif_date(str(date_str))
    except Exception:
        return None


def _parse_exif_date(date_str: str) -> datetime | None:
    """Parse EXIF date string (YYYY:MM:DD HH:MM:SS or YYYY-MM-DD HH:MM:SS)."""
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def _read_file_mtime(path: str) -> datetime | None:
    """Fall back to file modification time."""
    try:
        mtime = Path(path).stat().st_mtime
        return datetime.fromtimestamp(mtime)
    except OSError:
        return None


def cluster_by_time(
    timestamps: list[PhotoTimestamp],
    gap_hours: float = SAME_EVENT_GAP_HOURS,
) -> list[Event]:
    """Group photos into events based on time gaps.

    Args:
        timestamps: PhotoTimestamps sorted chronologically.
        gap_hours: Maximum gap between consecutive photos in the same event.

    Returns:
        List of Event objects, sorted by start time.
    """
    if not timestamps:
        return []

    events: list[Event] = []
    current_paths: list[str] = [timestamps[0].path]
    event_start = timestamps[0].timestamp
    event_end = timestamps[0].timestamp

    gap_threshold = timedelta(hours=gap_hours)

    for i in range(1, len(timestamps)):
        gap = timestamps[i].timestamp - event_end

        if gap <= gap_threshold:
            # Same event
            current_paths.append(timestamps[i].path)
            event_end = timestamps[i].timestamp
        else:
            # Finalize current event
            events.append(Event(
                event_id=f"event_{len(events):04d}",
                start_time=event_start,
                end_time=event_end,
                photo_paths=current_paths,
            ))
            # Start new event
            current_paths = [timestamps[i].path]
            event_start = timestamps[i].timestamp
            event_end = timestamps[i].timestamp

    # Finalize last event
    events.append(Event(
        event_id=f"event_{len(events):04d}",
        start_time=event_start,
        end_time=event_end,
        photo_paths=current_paths,
    ))

    logger.info("Clustered %d photos into %d events", len(timestamps), len(events))
    return events
