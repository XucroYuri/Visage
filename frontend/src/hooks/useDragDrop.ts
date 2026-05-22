import { useCallback, useRef } from "react";

interface DragDropOptions {
  onDrop?: (path: string) => void;
}

/**
 * Hook for drag-drop interactions on photo elements.
 *
 * In a Tauri desktop context, file drops include full paths.
 * In the browser, only the file name is accessible.
 */
export function useDragDrop(options: DragDropOptions = {}) {
  const dragRef = useRef<HTMLDivElement>(null);

  const handleDragStart = useCallback(
    (e: React.DragEvent, path: string) => {
      e.dataTransfer.setData("text/plain", path);
      e.dataTransfer.effectAllowed = "move";
    },
    [],
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const path = e.dataTransfer.getData("text/plain");
      if (path && options.onDrop) {
        options.onDrop(path);
      }
    },
    [options],
  );

  return {
    dragRef,
    handleDragStart,
    handleDragOver,
    handleDrop,
  };
}
