import { type ReactNode, useEffect, useRef, useState } from "react";
import { useInfiniteScroll } from "../hooks/useInfiniteScroll";
import { useVirtualScroll } from "../hooks/useVirtualScroll";

interface PhotoGridProps {
  children: ReactNode[];
  totalCount: number;
  emptyMessage?: string;
  /** Enable virtual scrolling for large photo sets */
  virtualized?: boolean;
  /** Minimum column width for CSS Grid auto-fill (default: 180px) */
  minColumnWidth?: number;
}

const GRID_GAP = 12;
const MIN_COLUMN_WIDTH = 180;
const VIRTUAL_THRESHOLD = 200; // use virtual scrolling above this count

/**
 * Responsive CSS Grid photo grid with optional virtual scrolling.
 * Replaces the previous Masonry-based grid.
 *
 * For small sets (< VIRTUAL_THRESHOLD), uses simple CSS Grid with infinite scroll.
 * For large sets, additionally enables row-level virtual scrolling.
 */
export function PhotoGrid({
  children,
  totalCount,
  emptyMessage = "No photos",
  virtualized,
  minColumnWidth = MIN_COLUMN_WIDTH,
}: PhotoGridProps) {
  // ── Column calculation ──────────────────────────────────
  const gridRef = useRef<HTMLDivElement>(null);
  const [columns, setColumns] = useState(4);

  useEffect(() => {
    const updateColumns = () => {
      if (!gridRef.current) return;
      const width = gridRef.current.offsetWidth;
      const colCount = Math.max(1, Math.floor(width / (minColumnWidth + GRID_GAP)));
      setColumns(colCount);
    };

    updateColumns();
    const observer = new ResizeObserver(updateColumns);
    if (gridRef.current) observer.observe(gridRef.current);
    return () => observer.disconnect();
  }, [minColumnWidth]);

  const useVirtual = virtualized ?? totalCount > VIRTUAL_THRESHOLD;

  // ── Infinite scroll (for non-virtualized mode) ──────────
  const { visibleCount, sentinelRef, hasMore } = useInfiniteScroll(
    useVirtual ? totalCount : totalCount,
  );
  const visibleChildren = children.slice(0, useVirtual ? totalCount : visibleCount);

  // ── Virtual scrolling (for large sets) ──────────────────
  const {
    parentRef,
    virtualizer,
  } = useVirtualScroll({
    count: useVirtual ? totalCount : 0,
    columns,
    gap: GRID_GAP,
  });

  // ── Empty state ─────────────────────────────────────────
  if (totalCount === 0) {
    return (
      <div className="text-center mt-20" style={{ color: "var(--color-text-muted)" }}>
        <p className="text-lg">{emptyMessage}</p>
      </div>
    );
  }

  // ── Virtualized grid ────────────────────────────────────
  if (useVirtual && totalCount > 0) {
    return (
      <div
        ref={parentRef}
        className="virtual-scroll-container"
        style={{ height: "100%" }}
      >
        <div
          className="virtual-scroll-content"
          style={{ height: `${virtualizer.getTotalSize()}px` }}
        >
          {virtualizer.getVirtualItems().map((virtualRow) => {
            const rowIndex = virtualRow.index;
            const startIdx = rowIndex * columns;
            const rowItems = visibleChildren.slice(startIdx, startIdx + columns);

            return (
              <div
                key={virtualRow.key}
                className="photo-grid-row"
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: "100%",
                  transform: `translateY(${virtualRow.start}px)`,
                  display: "grid",
                  gridTemplateColumns: `repeat(${columns}, 1fr)`,
                  gap: `${GRID_GAP}px`,
                  paddingRight: `${GRID_GAP}px`,
                }}
              >
                {rowItems}
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  // ── Simple CSS Grid (non-virtualized, with infinite scroll) ──
  return (
    <>
      <div
        ref={gridRef}
        className="photo-grid"
      >
        {visibleChildren.map((child, i) => (
          <div key={i}>{child}</div>
        ))}
      </div>

      {/* Infinite scroll sentinel */}
      {hasMore && (
        <div
          ref={sentinelRef}
          className="py-8 text-center text-sm"
          style={{ color: "var(--color-text-muted)" }}
        >
          Loading more... ({visibleCount} / {totalCount})
        </div>
      )}
    </>
  );
}
