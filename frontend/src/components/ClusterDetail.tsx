import { useState, useCallback, useMemo } from "react";
import type { PhotoInfo } from "../api";
import { useAppContext } from "../context/AppContext";
import { PhotoCard } from "./PhotoCard";
import { PhotoGrid } from "./PhotoGrid";
import { PhotoViewer } from "./PhotoViewer";

export function ClusterDetail() {
  const ctx = useAppContext();
  const {
    ws,
    selectedCluster,
    editingName,
    editValue,
    setEditValue,
    handleStartEdit,
    handleCancelEdit,
    handleRename,
    handleRemove,
    handleMove,
    handleViewAll,
  } = ctx;

  const [viewerPhoto, setViewerPhoto] = useState<PhotoInfo | null>(null);

  const cluster = selectedCluster;
  const editing = editingName === cluster?.id;
  const otherClusters = useMemo(
    () => (ws && cluster ? ws.clusters.filter((c) => c.id !== cluster.id) : []),
    [ws, cluster],
  );

  const viewerIndex =
    viewerPhoto && cluster
      ? cluster.photos.findIndex((p) => p.path === viewerPhoto.path)
      : -1;

  const handlePrev = useCallback(() => {
    if (viewerIndex > 0 && cluster) {
      setViewerPhoto(cluster.photos[viewerIndex - 1]);
    }
  }, [viewerIndex, cluster]);

  const handleNext = useCallback(() => {
    if (cluster && viewerIndex < cluster.photos.length - 1) {
      setViewerPhoto(cluster.photos[viewerIndex + 1]);
    }
  }, [viewerIndex, cluster]);

  if (!cluster || !ws) return null;

  return (
    <div>
      {/* Header */}
      <div className="flex items-center gap-3 mb-4">
        <button
          onClick={handleViewAll}
          className="text-sm text-gray-400 hover:text-gray-600 transition-colors"
        >
          &#8592; All
        </button>
        {editing ? (
          <input
            value={editValue}
            onChange={(e) => setEditValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleRename();
              if (e.key === "Escape") handleCancelEdit();
            }}
            className="text-xl font-semibold border rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-400"
          />
        ) : (
          /* eslint-disable jsx-a11y/no-noninteractive-element-interactions, jsx-a11y/no-noninteractive-tabindex */
          <h2
            onClick={() => handleStartEdit(cluster.id, cluster.name)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                handleStartEdit(cluster.id, cluster.name);
              }
            }}
            tabIndex={0}
            className="text-xl font-semibold text-gray-900 cursor-pointer hover:text-blue-600 transition-colors"
            title="Click to rename"
          >
            {cluster.name}
          </h2>
          /* eslint-enable jsx-a11y/no-noninteractive-element-interactions, jsx-a11y/no-noninteractive-tabindex */
        )}
        <span className="text-sm text-gray-400">
          {cluster.photo_count} photos &middot; confidence{" "}
          {cluster.confidence.toFixed(2)}
        </span>
      </div>

      {/* Empty state */}
      {cluster.photos.length === 0 ? (
        <div className="text-center text-gray-400 mt-20">
          <p className="text-lg">This cluster is empty</p>
          <p className="text-sm mt-1">
            Photos may have been moved or removed.
          </p>
        </div>
      ) : (
        <PhotoGrid totalCount={cluster.photos.length} emptyMessage="This cluster is empty">
          {cluster.photos.map((photo) => (
            <div key={photo.path}>
              <PhotoCard
                photo={photo}
                onRemove={() => handleRemove(cluster.id, photo.path)}
                onMove={(toId) => handleMove(photo.path, cluster.id, toId)}
                otherClusters={otherClusters}
                onViewFull={() => setViewerPhoto(photo)}
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
          onNext={viewerIndex < cluster.photos.length - 1 ? handleNext : undefined}
        />
      )}
    </div>
  );
}
