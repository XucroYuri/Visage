import { useEffect, useRef, useState } from "react";

const DEFAULT_PAGE_SIZE = 80;

/**
 * Hook for infinite-scroll masonry grids.
 * Resets when `totalItems` changes (new data loaded).
 * Returns a sentinel ref to attach to a loader element, and
 * the current count of visible items.
 */
export function useInfiniteScroll(
  totalItems: number,
  pageSize: number = DEFAULT_PAGE_SIZE,
) {
  const [visibleCount, setVisibleCount] = useState(pageSize);
  const sentinelRef = useRef<HTMLDivElement>(null);

  // Reset when source data changes
  useEffect(() => {
    setVisibleCount(pageSize);
  }, [totalItems, pageSize]);

  // IntersectionObserver on sentinel
  useEffect(() => {
    const sentinel = sentinelRef.current;
    if (!sentinel) return;
    if (visibleCount >= totalItems) return;

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting) {
          setVisibleCount((prev) => Math.min(prev + pageSize, totalItems));
        }
      },
      { rootMargin: "200px" },
    );
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [totalItems, visibleCount, pageSize]);

  return {
    visibleCount,
    sentinelRef,
    hasMore: visibleCount < totalItems,
  };
}
