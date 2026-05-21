"""Auto album events — time clustering, naming, cover selection."""

from visage.events.cluster import Event, cluster_by_time
from visage.events.cover_selector import select_event_cover
from visage.events.naming import generate_event_name
from visage.events.timeline import build_growth_timeline

__all__ = [
    "Event",
    "cluster_by_time",
    "generate_event_name",
    "select_event_cover",
    "build_growth_timeline",
]
