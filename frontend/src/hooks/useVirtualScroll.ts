import { useCallback, useRef } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";

interface VirtualScrollOptions {
  /** Total number of items */
  count: number;
  /** Estimated height of each item in px (used for scroll bar sizing) */
  estimateSize?: number;
  /** Number of columns in the grid */
  columns?: number;
  /** Gap between items in px */
  gap?: number;
  /** Overscan items to render above/below viewport */
  overscan?: number;
}

const DEFAULT_ESTIMATE_SIZE = 240;
const DEFAULT_OVSERCAN = 5;

/**
 * Hook that returns a virtualizer configured for a photo grid.
 * The virtualizer treats each *row* as one virtual item, where
 * each row contains `columns` photos.
 */
export function useVirtualScroll({
  count,
  estimateSize = DEFAULT_ESTIMATE_SIZE,
  columns = 1,
  gap = 12,
  overscan = DEFAULT_OVSERCAN,
}: VirtualScrollOptions) {
  const parentRef = useRef<HTMLDivElement>(null);

  const rowCount = Math.ceil(count / columns);

  const virtualizer = useVirtualizer({
    count: rowCount,
    getScrollElement: () => parentRef.current,
    estimateSize: () => estimateSize + gap,
    overscan,
  });

  const getVirtualItems = useCallback(
    () => virtualizer.getVirtualItems(),
    [virtualizer],
  );

  const getTotalSize = useCallback(
    () => virtualizer.getTotalSize(),
    [virtualizer],
  );

  return {
    parentRef,
    virtualizer,
    rowCount,
    getVirtualItems,
    getTotalSize,
  };
}
