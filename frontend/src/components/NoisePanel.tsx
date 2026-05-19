import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ClusterInfo, PhotoInfo } from "../api";
import { PhotoCard } from "./PhotoCard";
import { PhotoGrid } from "./PhotoGrid";
import { PhotoViewer } from "./PhotoViewer";

interface NoisePanelProps {
  noisePhotos: PhotoInfo[];
  clusters: ClusterInfo[];
  nextClusterId: number;
  onAssign: (imagePath: string, toId: number) => void;
  onBatchAssign: (imagePaths: string[], toId: number) => void;
  mutating: boolean;
}

export function NoisePanel({
  noisePhotos,
  clusters,
  nextClusterId,
  onAssign,
  onBatchAssign,
  mutating,
}: NoisePanelProps) {
  const [selectedPaths, setSelectedPaths] = useState<Set<string>>(new Set());
  const [showBatchMenu, setShowBatchMenu] = useState(false);
  const [viewerPhoto, setViewerPhoto] = useState<PhotoInfo | null>(null);
  const batchMenuRef = useRef<HTMLDivElement>(null);

  // Reset selection when photos change
  useEffect(() => {
    setSelectedPaths(new Set());
  }, [noisePhotos]);

  // Close batch menu on outside click
  useEffect(() => {
    if (!showBatchMenu) return;
    const handleClick = (e: MouseEvent) => {
      if (
        batchMenuRef.current &&
        !batchMenuRef.current.contains(e.target as Node)
      ) {
        setShowBatchMenu(false);
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [showBatchMenu]);

  // Ctrl+A handler
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "a") {
        // Only handle if noise panel is visible (check document focus or context)
        if (noisePhotos.length > 0) {
          e.preventDefault();
          setSelectedPaths(new Set(noisePhotos.map((p) => p.path)));
        }
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [noisePhotos]);

  const toggleSelect = useCallback((path: string) => {
    setSelectedPaths((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  }, []);

  const selectAll = useCallback(() => {
    setSelectedPaths(new Set(noisePhotos.map((p) => p.path)));
  }, [noisePhotos]);

  const deselectAll = useCallback(() => {
    setSelectedPaths(new Set());
  }, []);

  const allSelected =
    noisePhotos.length > 0 && selectedPaths.size === noisePhotos.length;

  const handleDragStart = useCallback(
    (e: React.DragEvent, imagePath: string) => {
      e.dataTransfer.setData("text/plain", imagePath);
      e.dataTransfer.effectAllowed = "move";
      // If dragging a selected photo, also drag all selected
      if (selectedPaths.has(imagePath) && selectedPaths.size > 1) {
        e.dataTransfer.setData(
          "application/json",
          JSON.stringify(Array.from(selectedPaths)),
        );
      }
    },
    [selectedPaths],
  );

  const handleBatchAssign = useCallback(
    (toId: number) => {
      if (selectedPaths.size === 0) return;
      onBatchAssign(Array.from(selectedPaths), toId);
      setSelectedPaths(new Set());
      setShowBatchMenu(false);
    },
    [selectedPaths, onBatchAssign],
  );

  // Viewer prev/next
  const flatPhotos = useMemo(() => noisePhotos, [noisePhotos]);
  const viewerIndex = viewerPhoto
    ? flatPhotos.findIndex((p) => p.path === viewerPhoto.path)
    : -1;

  const handlePrev = useCallback(() => {
    if (viewerIndex > 0) setViewerPhoto(flatPhotos[viewerIndex - 1]);
  }, [viewerIndex, flatPhotos]);

  const handleNext = useCallback(() => {
    if (viewerIndex < flatPhotos.length - 1)
      setViewerPhoto(flatPhotos[viewerIndex + 1]);
  }, [viewerIndex, flatPhotos.length]);

  const selectionCount = selectedPaths.size;

  return (
    <div>
      {/* Header + Quick actions toolbar */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-700">
          Unclustered Faces ({noisePhotos.length})
        </h2>

        {noisePhotos.length > 0 && (
          <div className="flex items-center gap-2">
            {/* Selection controls */}
            <button
              onClick={allSelected ? deselectAll : selectAll}
              className="text-xs px-2 py-1 border rounded hover:bg-gray-50 transition-colors text-gray-600"
            >
              {allSelected ? "Deselect All" : "Select All"}
            </button>

            {/* Selection counter */}
            {selectionCount > 0 && (
              <span className="text-xs text-blue-600 font-medium">
                {selectionCount} selected
              </span>
            )}

            {/* Batch assign dropdown */}
            <div className="relative" ref={batchMenuRef}>
              <button
                onClick={() => setShowBatchMenu(!showBatchMenu)}
                disabled={selectionCount === 0 || mutating}
                className="text-xs px-3 py-1.5 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-40 transition-colors flex items-center gap-1"
              >
                {mutating && selectionCount > 0 ? (
                  <span className="inline-block w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                ) : null}
                Assign {selectionCount > 0 ? `(${selectionCount})` : "Selected"}
              </button>

              {showBatchMenu && selectionCount > 0 && (
                <div className="absolute right-0 top-full mt-1 bg-white rounded shadow-lg border max-h-56 overflow-y-auto min-w-44 z-20">
                  {clusters.map((c) => (
                    <button
                      key={c.id}
                      onClick={() => handleBatchAssign(c.id)}
                      className="block w-full text-left px-3 py-2 text-sm hover:bg-blue-50 text-gray-700 transition-colors"
                    >
                      {c.name}{" "}
                      <span className="text-gray-400">({c.photo_count})</span>
                    </button>
                  ))}
                  <button
                    onClick={() => handleBatchAssign(nextClusterId)}
                    className="block w-full text-left px-3 py-2 text-sm hover:bg-green-50 text-green-700 border-t transition-colors font-medium"
                  >
                    + New Cluster
                  </button>
                </div>
              )}
            </div>

            {/* Quick: Assign All to New Cluster */}
            <button
              onClick={() => {
                const allPaths = noisePhotos.map((p) => p.path);
                onBatchAssign(allPaths, nextClusterId);
                setSelectedPaths(new Set());
              }}
              disabled={noisePhotos.length === 0 || mutating}
              className="text-xs px-3 py-1.5 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-40 transition-colors"
            >
              All to New
            </button>
          </div>
        )}
      </div>

      {/* Empty state */}
      {noisePhotos.length === 0 ? (
        <div className="text-center text-gray-400 mt-20">
          <p className="text-lg">No unclustered faces</p>
          <p className="text-sm mt-1">
            All faces have been assigned to clusters.
          </p>
        </div>
      ) : (
        <PhotoGrid
          totalCount={noisePhotos.length}
          emptyMessage="No unclustered faces"
        >
          {noisePhotos.map((photo) => (
            <div key={photo.path}>
              <PhotoCard
                photo={photo}
                onMove={(toId) => onAssign(photo.path, toId)}
                otherClusters={clusters}
                nextClusterId={nextClusterId}
                onViewFull={() => setViewerPhoto(photo)}
                draggable
                onDragStart={handleDragStart}
                selectionMode
                selected={selectedPaths.has(photo.path)}
                onSelectToggle={toggleSelect}
              />
            </div>
          ))}
        </PhotoGrid>
      )}

      {/* Full-size viewer with prev/next */}
      {viewerPhoto && (
        <PhotoViewer
          photo={viewerPhoto}
          onClose={() => setViewerPhoto(null)}
          onPrev={viewerIndex > 0 ? handlePrev : undefined}
          onNext={
            viewerIndex < flatPhotos.length - 1 ? handleNext : undefined
          }
        />
      )}
    </div>
  );
}
