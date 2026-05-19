import { type ReactNode } from "react";
import Masonry from "react-masonry-css";
import { useInfiniteScroll } from "../hooks/useInfiniteScroll";

interface PhotoGridProps {
  /** Children rendered inside the masonry grid (one per photo) */
  children: ReactNode[];
  /** Total number of items (for infinite scroll tracking) */
  totalCount: number;
  /** Message shown when there are no children */
  emptyMessage?: string;
}

const BREAKPOINT_COLUMNS = {
  default: 6,
  1600: 5,
  1200: 4,
  900: 3,
  600: 2,
};

/**
 * Shared masonry photo grid with infinite scroll.
 * Wrap each photo in a `div` (as required by react-masonry-css) and pass as children.
 */
export function PhotoGrid({
  children,
  totalCount,
  emptyMessage = "No photos",
}: PhotoGridProps) {
  const { visibleCount, sentinelRef, hasMore } = useInfiniteScroll(totalCount);
  const visibleChildren = children.slice(0, visibleCount);

  if (totalCount === 0) {
    return (
      <div className="text-center text-gray-400 mt-20">
        <p className="text-lg">{emptyMessage}</p>
      </div>
    );
  }

  return (
    <>
      <Masonry
        breakpointCols={BREAKPOINT_COLUMNS}
        className="masonry-grid"
        columnClassName="masonry-grid-column"
      >
        {visibleChildren}
      </Masonry>

      {/* Infinite scroll sentinel */}
      {hasMore && (
        <div
          ref={sentinelRef}
          className="py-8 text-center text-gray-400 text-sm"
        >
          Loading more... ({visibleCount} / {totalCount})
        </div>
      )}
    </>
  );
}
