import { useState, useCallback } from "react";
import type { ClusterInfo, PhotoInfo } from "../api";
import { PhotoCard } from "./PhotoCard";
import { PhotoGrid } from "./PhotoGrid";
import { PhotoViewer } from "./PhotoViewer";

interface ClusterDetailProps {
  cluster: ClusterInfo;
  clusters: ClusterInfo[];
  onRemove: (imagePath: string) => void;
  onMove: (imagePath: string, toId: number) => void;
  onBack: () => void;
  onStartRename: () => void;
  editing: boolean;
  editValue: string;
  onEditChange: (value: string) => void;
  onSaveEdit: () => void;
  onCancelEdit: () => void;
}

export function ClusterDetail({
  cluster,
  clusters,
  onRemove,
  onMove,
  onBack,
  onStartRename,
  editing,
  editValue,
  onEditChange,
  onSaveEdit,
  onCancelEdit,
}: ClusterDetailProps) {
  const [viewerPhoto, setViewerPhoto] = useState<PhotoInfo | null>(null);

  const otherClusters = clusters.filter((c) => c.id !== cluster.id);

  const viewerIndex = viewerPhoto
    ? cluster.photos.findIndex((p) => p.path === viewerPhoto.path)
    : -1;

  const handlePrev = useCallback(() => {
    if (viewerIndex > 0) {
      setViewerPhoto(cluster.photos[viewerIndex - 1]);
    }
  }, [viewerIndex, cluster.photos]);

  const handleNext = useCallback(() => {
    if (viewerIndex < cluster.photos.length - 1) {
      setViewerPhoto(cluster.photos[viewerIndex + 1]);
    }
  }, [viewerIndex, cluster.photos.length]);

  return (
    <div>
      {/* Header */}
      <div className="flex items-center gap-3 mb-4">
        <button
          onClick={onBack}
          className="text-sm text-gray-400 hover:text-gray-600 transition-colors"
        >
          &#8592; All
        </button>
        {editing ? (
          <input
            value={editValue}
            onChange={(e) => onEditChange(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") onSaveEdit();
              if (e.key === "Escape") onCancelEdit();
            }}
            className="text-xl font-semibold border rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-400"
            autoFocus
          />
        ) : (
          <h2
            onClick={onStartRename}
            className="text-xl font-semibold text-gray-900 cursor-pointer hover:text-blue-600 transition-colors"
            title="Click to rename"
          >
            {cluster.name}
          </h2>
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
                onRemove={() => onRemove(photo.path)}
                onMove={(toId) => onMove(photo.path, toId)}
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
