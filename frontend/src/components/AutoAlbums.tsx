import { useState, useEffect, useCallback } from "react";
import type { EventsResponse, EventInfo } from "../api-phase3";
import { fetchEvents } from "../api-phase3";
import { getImageUrl } from "../api";

export function AutoAlbums() {
  const [events, setEvents] = useState<EventInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedEvent, setSelectedEvent] = useState<EventInfo | null>(null);

  useEffect(() => {
    fetchEvents()
      .then((res: EventsResponse) => setEvents(res.events))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const handleSelectEvent = useCallback((event: EventInfo) => {
    setSelectedEvent((prev) => (prev?.event_id === event.event_id ? null : event));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-8 h-8 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <p className="text-red-500 mb-2">Failed to load events</p>
        <p className="text-sm text-gray-500">{error}</p>
      </div>
    );
  }

  if (events.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-500 dark:text-slate-400 text-lg">No events detected</p>
        <p className="text-sm text-gray-400 dark:text-slate-500 mt-1">
          Events are auto-detected from photo timestamps
        </p>
      </div>
    );
  }

  return (
    <div>
      <h2 className="text-lg font-semibold mb-4" style={{ color: "var(--color-text-primary)" }}>
        Auto Albums ({events.length})
      </h2>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {events.map((event) => (
          <button
            key={event.event_id}
            onClick={() => handleSelectEvent(event)}
            className={`text-left rounded-lg border overflow-hidden transition-all hover:shadow-md ${
              selectedEvent?.event_id === event.event_id
                ? "border-blue-400 dark:border-blue-500 ring-2 ring-blue-200 dark:ring-blue-800"
                : "border-gray-200 dark:border-slate-700"
            }`}
          >
            {/* Cover image */}
            {event.cover_path ? (
              <div className="aspect-video bg-gray-100 dark:bg-slate-800 overflow-hidden">
                <img
                  src={getImageUrl(event.cover_path, "thumb")}
                  alt={event.name}
                  className="w-full h-full object-cover"
                  loading="lazy"
                />
              </div>
            ) : (
              <div className="aspect-video bg-gray-100 dark:bg-slate-800 flex items-center justify-center text-3xl">
                📷
              </div>
            )}

            <div className="p-3">
              <h3 className="font-medium text-sm truncate" style={{ color: "var(--color-text-primary)" }}>
                {event.name}
              </h3>
              <p className="text-xs mt-1" style={{ color: "var(--color-text-secondary)" }}>
                {new Date(event.start_time).toLocaleDateString()} · {event.photo_count} photos
              </p>
              {event.is_multi_day && (
                <span className="inline-block mt-1 px-1.5 py-0.5 text-[10px] bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 rounded">
                  Multi-day
                </span>
              )}
              {event.location_name && (
                <p className="text-xs mt-1 text-gray-400 dark:text-slate-500">
                  📍 {event.location_name}
                </p>
              )}
            </div>
          </button>
        ))}
      </div>

      {/* Selected event detail */}
      {selectedEvent && (
        <div className="mt-6 p-4 bg-white dark:bg-slate-800 rounded-lg border border-gray-200 dark:border-slate-700">
          <h3 className="font-semibold mb-2" style={{ color: "var(--color-text-primary)" }}>
            {selectedEvent.name}
          </h3>
          <p className="text-sm mb-3" style={{ color: "var(--color-text-secondary)" }}>
            {new Date(selectedEvent.start_time).toLocaleString()} →{" "}
            {new Date(selectedEvent.end_time).toLocaleString()}
          </p>
          <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 gap-2">
            {selectedEvent.photo_paths.slice(0, 12).map((path) => (
              <div key={path} className="aspect-square rounded overflow-hidden bg-gray-100 dark:bg-slate-700">
                <img
                  src={getImageUrl(path, "thumb")}
                  alt=""
                  className="w-full h-full object-cover"
                  loading="lazy"
                />
              </div>
            ))}
            {selectedEvent.photo_paths.length > 12 && (
              <div className="aspect-square rounded bg-gray-100 dark:bg-slate-700 flex items-center justify-center text-sm text-gray-500 dark:text-slate-400">
                +{selectedEvent.photo_paths.length - 12}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
