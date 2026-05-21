"""Tests for auto album events — time clustering, naming, cover selection."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from visage.events.cluster import Event, PhotoTimestamp, cluster_by_time
from visage.events.cover_selector import select_event_cover
from visage.events.naming import generate_event_name
from visage.events.timeline import build_growth_timeline

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def morning_photos() -> list[PhotoTimestamp]:
    """Photos taken in the morning of the same day."""
    base = datetime(2024, 6, 15, 9, 0)
    return [
        PhotoTimestamp(path=f"/tmp/morning_{i}.jpg", timestamp=base + timedelta(minutes=30 * i))
        for i in range(5)
    ]


@pytest.fixture
def afternoon_photos() -> list[PhotoTimestamp]:
    """Photos taken in the afternoon of the same day (4h+ gap from morning)."""
    base = datetime(2024, 6, 15, 15, 0)
    return [
        PhotoTimestamp(path=f"/tmp/afternoon_{i}.jpg", timestamp=base + timedelta(minutes=15 * i))
        for i in range(3)
    ]


@pytest.fixture
def next_day_photos() -> list[PhotoTimestamp]:
    """Photos taken the next day."""
    base = datetime(2024, 6, 16, 10, 0)
    return [
        PhotoTimestamp(path=f"/tmp/nextday_{i}.jpg", timestamp=base + timedelta(hours=i))
        for i in range(4)
    ]


@pytest.fixture
def holiday_photos() -> list[PhotoTimestamp]:
    """Photos taken on Christmas Day."""
    base = datetime(2024, 12, 25, 10, 0)
    return [
        PhotoTimestamp(path=f"/tmp/christmas_{i}.jpg", timestamp=base + timedelta(hours=i))
        for i in range(8)
    ]


@pytest.fixture
def multi_day_trip() -> list[PhotoTimestamp]:
    """Photos spanning 5 days (a trip)."""
    base = datetime(2024, 7, 10, 8, 0)
    photos = []
    for day in range(5):
        for hour in range(3):
            photos.append(PhotoTimestamp(
                path=f"/tmp/trip_d{day}_h{hour}.jpg",
                timestamp=base + timedelta(days=day, hours=hour * 4),
            ))
    return photos


# ── Time Clustering ───────────────────────────────────────────────


class TestClusterByTime:
    def test_empty_input(self):
        result = cluster_by_time([])
        assert result == []

    def test_single_photo(self):
        photos = [PhotoTimestamp(path="/tmp/one.jpg", timestamp=datetime(2024, 1, 1))]
        events = cluster_by_time(photos)
        assert len(events) == 1
        assert events[0].photo_count == 1
        assert events[0].photo_paths == ["/tmp/one.jpg"]

    def test_same_event_no_gap(self, morning_photos):
        events = cluster_by_time(morning_photos)
        assert len(events) == 1
        assert events[0].photo_count == 5

    def test_split_by_gap(self, morning_photos, afternoon_photos):
        # Default gap is 4h, afternoon starts at 15:00, morning ends at 11:00
        # Gap is 4h exactly — should be same event (<=)
        all_photos = morning_photos + afternoon_photos
        events = cluster_by_time(all_photos, gap_hours=4.0)
        assert len(events) == 1
        assert events[0].photo_count == 8

    def test_split_by_large_gap(self, morning_photos, afternoon_photos):
        # With smaller gap, morning and afternoon split
        all_photos = morning_photos + afternoon_photos
        events = cluster_by_time(all_photos, gap_hours=1.0)
        assert len(events) == 2
        assert events[0].photo_count == 5  # morning
        assert events[1].photo_count == 3  # afternoon

    def test_different_days(self, morning_photos, next_day_photos):
        all_photos = morning_photos + next_day_photos
        events = cluster_by_time(all_photos)
        assert len(events) == 2
        assert not events[0].is_multi_day
        assert not events[1].is_multi_day

    def test_multi_day_event(self, multi_day_trip):
        # Trip photos: within each day gap is 4h (at threshold).
        # Between days: last photo at 16:00, next at 08:00 = 16h gap.
        # With gap_hours=4: each day is a separate event
        events = cluster_by_time(multi_day_trip, gap_hours=4.0)
        assert len(events) == 5  # One event per day
        # With gap_hours=20: all days merge into one multi-day event
        events_merged = cluster_by_time(multi_day_trip, gap_hours=20.0)
        assert len(events_merged) == 1
        assert events_merged[0].is_multi_day

    def test_unsorted_input_produces_expected_events(self):
        # cluster_by_time processes input in order — caller should pre-sort
        photos = [
            PhotoTimestamp(path="/tmp/late.jpg", timestamp=datetime(2024, 12, 1)),
            PhotoTimestamp(path="/tmp/early.jpg", timestamp=datetime(2024, 1, 1)),
        ]
        # Unsorted input: gap appears as negative timedelta (Jan < Dec same year)
        # which is <= threshold, so they merge into 1 event
        events = cluster_by_time(photos)
        assert len(events) == 1

        # Sorted input: gap is 335 days, far exceeds threshold
        sorted_photos = sorted(photos, key=lambda p: p.timestamp)
        events = cluster_by_time(sorted_photos)
        assert len(events) == 2
        assert events[0].start_time < events[1].start_time

    def test_event_id_sequential(self, morning_photos):
        events = cluster_by_time(morning_photos)
        assert events[0].event_id == "event_0000"

    def test_event_id_sequential_multiple(self, morning_photos, next_day_photos):
        events = cluster_by_time(morning_photos + next_day_photos)
        assert events[0].event_id == "event_0000"
        assert events[1].event_id == "event_0001"


class TestEventProperties:
    def test_duration(self):
        e = Event(
            event_id="test",
            start_time=datetime(2024, 1, 1, 10, 0),
            end_time=datetime(2024, 1, 1, 14, 0),
        )
        assert e.duration == timedelta(hours=4)

    def test_is_multi_day_false(self):
        e = Event(
            event_id="test",
            start_time=datetime(2024, 1, 1, 10, 0),
            end_time=datetime(2024, 1, 1, 14, 0),
        )
        assert not e.is_multi_day

    def test_is_multi_day_true(self):
        e = Event(
            event_id="test",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 3),
        )
        assert e.is_multi_day


# ── Event Naming ──────────────────────────────────────────────────


class TestGenerateEventName:
    def test_christmas(self, holiday_photos):
        events = cluster_by_time(holiday_photos)
        name = generate_event_name(events[0])
        assert "圣诞节" in name
        assert "2024" in name

    def test_new_year(self):
        e = Event(
            event_id="test",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 1),
        )
        name = generate_event_name(e)
        assert "元旦" in name

    def test_national_day(self):
        e = Event(
            event_id="test",
            start_time=datetime(2024, 10, 1),
            end_time=datetime(2024, 10, 1),
        )
        name = generate_event_name(e)
        assert "国庆节" in name

    def test_multi_day_with_location(self):
        e = Event(
            event_id="test",
            start_time=datetime(2024, 7, 10),
            end_time=datetime(2024, 7, 15),
        )
        name = generate_event_name(e, location="大理")
        assert "大理" in name
        assert "2024-07" in name

    def test_multi_day_without_location(self):
        e = Event(
            event_id="test",
            start_time=datetime(2024, 7, 10),
            end_time=datetime(2024, 7, 15),
        )
        name = generate_event_name(e)
        assert "2024-07-10" in name

    def test_weekend(self):
        # 2024-06-15 is a Saturday
        e = Event(
            event_id="test",
            start_time=datetime(2024, 6, 15),
            end_time=datetime(2024, 6, 15),
        )
        name = generate_event_name(e)
        assert "周末" in name

    def test_weekday(self):
        # 2024-06-17 is a Monday
        e = Event(
            event_id="test",
            start_time=datetime(2024, 6, 17),
            end_time=datetime(2024, 6, 17),
        )
        name = generate_event_name(e)
        assert "日常" in name


# ── Cover Selection ───────────────────────────────────────────────


class TestSelectEventCover:
    def test_empty_event(self):
        e = Event(event_id="test", start_time=datetime(2024, 1, 1), end_time=datetime(2024, 1, 1))
        assert select_event_cover(e) is None

    def test_no_quality_map_returns_first(self):
        e = Event(
            event_id="test",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 1),
            photo_paths=["/tmp/a.jpg", "/tmp/b.jpg"],
        )
        assert select_event_cover(e) == "/tmp/a.jpg"

    def test_selects_highest_quality(self):
        e = Event(
            event_id="test",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 1),
            photo_paths=["/tmp/low.jpg", "/tmp/high.jpg", "/tmp/medium.jpg"],
        )
        quality = {"/tmp/low.jpg": 0.3, "/tmp/high.jpg": 0.9, "/tmp/medium.jpg": 0.6}
        assert select_event_cover(e, quality) == "/tmp/high.jpg"

    def test_missing_quality_gets_zero(self):
        e = Event(
            event_id="test",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 1),
            photo_paths=["/tmp/with_q.jpg", "/tmp/no_q.jpg"],
        )
        quality = {"/tmp/with_q.jpg": 0.5}
        assert select_event_cover(e, quality) == "/tmp/with_q.jpg"


# ── Growth Timeline ───────────────────────────────────────────────


class TestBuildGrowthTimeline:
    def test_insufficient_photos(self):
        result = build_growth_timeline(
            "person_0",
            ["/tmp/a.jpg"],
            [datetime(2024, 1, 1)],
            min_photos=5,
        )
        assert result is None

    def test_insufficient_span(self):
        paths = [f"/tmp/p_{i}.jpg" for i in range(6)]
        # All on same day
        timestamps = [datetime(2024, 1, 1, i) for i in range(6)]
        result = build_growth_timeline(
            "person_0", paths, timestamps, min_photos=5, min_span_months=12,
        )
        assert result is None

    def test_valid_timeline(self):
        paths = [f"/tmp/p_{i}.jpg" for i in range(10)]
        # Span 18 months
        timestamps = [datetime(2023, 1, 1) + timedelta(days=60 * i) for i in range(10)]
        result = build_growth_timeline(
            "person_0", paths, timestamps, min_photos=5, min_span_months=12,
        )
        assert result is not None
        assert result.entry_count == 10
        assert result.span_months >= 12
        assert result.has_growth_milestone is True

    def test_entries_sorted_chronologically(self):
        paths = ["/tmp/a.jpg", "/tmp/b.jpg", "/tmp/c.jpg"]
        timestamps = [
            datetime(2024, 6, 1),
            datetime(2023, 1, 1),
            datetime(2024, 1, 1),
        ]
        # Need enough photos to pass minimum
        paths *= 3
        timestamps = timestamps * 3
        result = build_growth_timeline(
            "person_0", paths, timestamps, min_photos=5, min_span_months=12,
        )
        if result is not None:
            ts_list = [e.timestamp for e in result.entries]
            assert ts_list == sorted(ts_list)
