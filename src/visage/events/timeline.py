"""Growth timeline — arrange photos of a person chronologically.

Builds a timeline of a person's photos ordered by date, optionally
detecting growth milestones when photos span >12 months.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TimelineEntry:
    """A single entry in a person's growth timeline."""

    photo_path: str
    timestamp: datetime
    age_label: str | None = None  # e.g. "3-6 岁"


@dataclass
class GrowthTimeline:
    """A person's photo timeline with optional growth milestones."""

    person_id: str
    entries: list[TimelineEntry] = field(default_factory=list)
    span_months: float = 0.0
    has_growth_milestone: bool = False

    @property
    def entry_count(self) -> int:
        return len(self.entries)


def build_growth_timeline(
    person_id: str,
    photo_paths: list[str],
    timestamps: list[datetime],
    min_photos: int = 5,
    min_span_months: int = 12,
) -> GrowthTimeline | None:
    """Build a growth timeline for a person.

    Only returns a timeline if the person has enough photos spanning
    enough time to make a meaningful timeline.

    Args:
        person_id: Identifier for the person (cluster ID or name).
        photo_paths: Paths of photos containing this person.
        timestamps: Corresponding timestamps for each photo.
        min_photos: Minimum number of photos required.
        min_span_months: Minimum time span (months) to trigger growth timeline.

    Returns:
        GrowthTimeline if criteria met, None otherwise.
    """
    if len(photo_paths) < min_photos or len(timestamps) < min_photos:
        return None

    # Sort by timestamp
    paired = sorted(zip(photo_paths, timestamps, strict=False), key=lambda x: x[1])
    earliest = paired[0][1]
    latest = paired[-1][1]

    span_months = (latest - earliest).days / 30.44  # average month length

    entries: list[TimelineEntry] = []
    for path, ts in paired:
        age_label = _estimate_age_label(ts, earliest)
        entries.append(TimelineEntry(photo_path=path, timestamp=ts, age_label=age_label))

    has_milestone = span_months >= min_span_months

    if not has_milestone and len(photo_paths) < min_photos * 2:
        return None  # Not enough growth data

    timeline = GrowthTimeline(
        person_id=person_id,
        entries=entries,
        span_months=round(span_months, 1),
        has_growth_milestone=has_milestone,
    )

    logger.info(
        "Growth timeline for %s: %d photos, %.1f months, milestone=%s",
        person_id, len(entries), span_months, has_milestone,
    )
    return timeline


def _estimate_age_label(photo_time: datetime, reference_time: datetime) -> str | None:
    """Estimate an age label based on time elapsed since reference.

    This is a rough placeholder — real age estimation would use a facial
    age regression model. For now, returns the time elapsed as a label.
    """
    return None  # Age estimation requires ML model — placeholder for Phase 3 M2
