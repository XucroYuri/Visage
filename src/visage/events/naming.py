"""Smart event naming — generate human-readable names for detected events.

Priority:
1. Calendar holiday match → "2024 年圣诞节"
2. Multi-day with location → "云南之旅 (2024-07)"
3. Single day → "周末随拍 · 2024-06-15"
4. Fallback → "未命名事件 2024-06-15"
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from visage.events.cluster import Event

logger = logging.getLogger(__name__)

# Known holidays: (month, day) → name (Chinese)
HOLIDAYS: dict[tuple[int, int], str] = {
    (1, 1): "元旦",
    (2, 14): "情人节",
    (3, 8): "妇女节",
    (5, 1): "劳动节",
    (6, 1): "儿童节",
    (10, 1): "国庆节",
    (10, 31): "万圣节",
    (12, 25): "圣诞节",
    (12, 31): "跨年夜",
}

# Lunar holidays are approximate — we match by proximity (±3 days)
LUNAR_HOLIDAYS: list[tuple[int, int, str]] = [
    # (approx_month, approx_day, name) — these are rough solar equivalents
    (1, 22, "春节"),  # varies Jan 21 - Feb 20
    (2, 5, "元宵节"),  # varies
    (4, 5, "清明节"),
    (6, 10, "端午节"),  # varies
    (8, 15, "中秋节"),  # varies
]


def generate_event_name(event: Event, location: str | None = None) -> str:
    """Generate a human-readable name for an event.

    Args:
        event: The event to name.
        location: Optional location name (from GPS geocoding).

    Returns:
        A descriptive event name string.
    """
    start = event.start_time
    end = event.end_time

    # 1. Calendar holiday match
    holiday = _match_holiday(start, end)
    if holiday:
        return f"{start.year} 年{holiday}"

    # 2. Multi-day with location
    if event.is_multi_day and location:
        return f"{location} ({start.strftime('%Y-%m')})"

    # 3. Multi-day without location
    if event.is_multi_day:
        days = (end.date() - start.date()).days + 1
        return f"{start.strftime('%Y-%m-%d')} ~ {end.strftime('%m-%d')} ({days}天)"

    # 4. Single day — check weekend
    day_name = _day_label(start)
    return f"{day_name} · {start.strftime('%Y-%m-%d')}"


def _match_holiday(start: datetime, end: datetime) -> str | None:
    """Check if event dates overlap with a known holiday."""
    # Check all dates in the event range
    current = start.date()
    end_date = end.date()

    while current <= end_date:
        # Solar holidays (exact match)
        key = (current.month, current.day)
        if key in HOLIDAYS:
            return HOLIDAYS[key]

        # Lunar holidays (proximity match ±3 days)
        for month, day, name in LUNAR_HOLIDAYS:
            if current.month == month and abs(current.day - day) <= 3:
                return name

        current += timedelta(days=1)

    return None


def _day_label(dt: datetime) -> str:
    """Return a descriptive label for the day of week."""
    weekday = dt.weekday()
    if weekday == 5 or weekday == 6:
        return "周末随拍"
    return "日常记录"
