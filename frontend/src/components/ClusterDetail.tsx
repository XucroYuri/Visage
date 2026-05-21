import { useCallback, useMemo, useState } from "react";
import { useParams, useNavigate } from "react-router";
import type { PhotoInfo } from "../api";
import { useKeyboard } from "../hooks/useKeyboard";
import { useUIStore } from "../store/ui";
import { useMoveMutation, useRemoveMutation, useRenameMutation, useWorkspaceStore } from "../store/workspace";
import { BatchActionBar } from "./BatchActionBar";
import { PhotoCard } from "./PhotoCard";
import { PhotoGrid } from "./PhotoGrid";
import { PhotoViewer } from "./PhotoViewer";

export function ClusterDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const clusterId = id ? parseInt(id, 10) : null;

  const ws = useWorkspaceStore((s) => s.ws);
  const editingName = useUIStore((s) => s.editingName);
  const editValue = useUIStore((s) => s.editValue);
  const setEditValue = useUIStore((s) => s.setEditValue);
  const setEditingName = useUIStore((s) => s.setEditingName);

  // Batch selection state
  const selectedPhotoPaths = useUIStore((s) => s.selectedPhotoPaths);
  const setSelectedPhotoPaths = useUIStore((s) => s.setSelectedPhotoPaths);
  const batchMode = useUIStore((s) => s.batchMode);
  const setBatchMode = useUIStore((s) => s.setBatchMode);

  const removeMutation = useRemoveMutation();
  const moveMutation = useMoveMutation();
  const renameMutation = useRenameMutation();

  const [viewerPhoto, setViewerPhoto] = useState<PhotoInfo | null>(null);

  const cluster =
    clusterId != null && ws
      ? ws.clusters.find((c) => c.id === clusterId) ?? null
      : null;

  const editing = editingName === cluster?.id;

  const otherClusters = useMemo(
    () => (ws && cluster ? ws.clusters.filter((c) => c.id !== cluster.id) : []),
    [ws, cluster],
  );

  const viewerIndex =
    viewerPhoto && cluster
      ? cluster.photos.findIndex((p) => p.path === viewerPhoto.path)
      : -1;

  // ── Viewer navigation ──────────────────────────────────────
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

  // ── Single photo actions ───────────────────────────────────
  const handleRemove = useCallback(
    async (removeClusterId: number, imagePath: string) => {
      await removeMutation.mutateAsync({ clusterId: removeClusterId, imagePath });
      const updated = useWorkspaceStore.getState().ws;
      if (updated && !updated.clusters.find((c) => c.id === removeClusterId)) {
        navigate("/");
      }
    },
    [removeMutation, navigate],
  );

  const handleMove = useCallback(
    (imagePath: string, fromId: number, toId: number) => {
      moveMutation.mutate({ imagePath, fromId, toId });
    },
    [moveMutation],
  );

  // ── Rename ─────────────────────────────────────────────────
  const handleStartEdit = useCallback(
    (id: number, name: string) => {
      setEditingName(id);
      setEditValue(name);
    },
    [setEditingName, setEditValue],
  );

  const handleRename = useCallback(() => {
    if (editingName === null || !editValue.trim()) {
      setEditingName(null);
      return;
    }
    const wsState = useWorkspaceStore.getState().ws;
    const oldName =
      wsState?.clusters.find((c) => c.id === editingName)?.name ?? `#${editingName}`;
    const clusterCountBefore = wsState?.clusters.length ?? 0;
    renameMutation.mutate({
      clusterId: editingName,
      name: editValue.trim(),
      oldName,
      clusterCountBefore,
    } as Parameters<typeof renameMutation.mutate>[0]);
    setEditingName(null);
  }, [editingName, editValue, renameMutation, setEditingName]);

  const handleCancelEdit = useCallback(() => {
    setEditingName(null);
  }, [setEditingName]);

  const handleViewAll = useCallback(() => navigate("/"), [navigate]);

  // ── Batch selection ────────────────────────────────────────
  const handleToggleBatchMode = useCallback(() => {
    setBatchMode(!batchMode);
    setSelectedPhotoPaths(new Set());
  }, [batchMode, setBatchMode, setSelectedPhotoPaths]);

  const handleSelectToggle = useCallback(
    (imagePath: string) => {
      const next = new Set(selectedPhotoPaths);
      if (next.has(imagePath)) {
        next.delete(imagePath);
      } else {
        next.add(imagePath);
      }
      setSelectedPhotoPaths(next);
    },
    [selectedPhotoPaths, setSelectedPhotoPaths],
  );

  const handleBatchReject = useCallback(async () => {
    if (!cluster) return;
    const paths = Array.from(selectedPhotoPaths);
    for (const path of paths) {
      await removeMutation.mutateAsync({ clusterId: cluster.id, imagePath: path });
    }
    setSelectedPhotoPaths(new Set());
    const updated = useWorkspaceStore.getState().ws;
    if (updated && !updated.clusters.find((c) => c.id === cluster.id)) {
      navigate("/");
    }
  }, [cluster, selectedPhotoPaths, removeMutation, setSelectedPhotoPaths, navigate]);

  const handleBatchClear = useCallback(() => {
    setSelectedPhotoPaths(new Set());
  }, [setSelectedPhotoPaths]);

  // ── Photo click: select in batch mode, view in normal mode ─
  const handlePhotoClick = useCallback(
    (photo: PhotoInfo) => {
      if (batchMode) {
        handleSelectToggle(photo.path);
      } else {
        setViewerPhoto(photo);
      }
    },
    [batchMode, handleSelectToggle],
  );

  // ── Keyboard shortcuts for this view ───────────────────────
  useKeyboard({
    onPrev: viewerIndex > 0 ? handlePrev : undefined,
    onNext: viewerIndex < (cluster?.photos.length ?? 0) - 1 ? handleNext : undefined,
    onSelectAll: batchMode
      ? () => {
          if (!cluster) return;
          setSelectedPhotoPaths(new Set(cluster.photos.map((p) => p.path)));
        }
      : undefined,
    onEscape: () => {
      if (selectedPhotoPaths.size > 0) {
        setSelectedPhotoPaths(new Set());
      } else if (batchMode) {
        setBatchMode(false);
      } else if (viewerPhoto) {
        setViewerPhoto(null);
      }
    },
  });

  if (!cluster || !ws) return null;

  const isPending = removeMutation.isPending || moveMutation.isPending;

  return (
    <div className="relative min-h-full">
      {/* Header */}
      <div className="flex items-center gap-3 mb-4">
        <button
          onClick={handleViewAll}
          className="text-sm transition-colors"
          style={{ color: "var(--color-text-muted)" }}
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
            className="text-xl font-semibold border rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-400 bg-white dark:bg-slate-800 text-gray-900 dark:text-slate-100 border-gray-300 dark:border-slate-600"
            autoFocus
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
            className="text-xl font-semibold cursor-pointer hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
            style={{ color: "var(--color-text-primary)" }}
            title="Click to rename"
          >
            {cluster.name}
          </h2>
          /* eslint-enable jsx-a11y/no-noninteractive-element-interactions, jsx-a11y/no-noninteractive-tabindex */
        )}
        <span className="text-sm" style={{ color: "var(--color-text-muted)" }}>
          {cluster.photo_count} photos &middot; confidence{" "}
          {cluster.confidence.toFixed(2)}
        </span>

        {/* Batch mode toggle */}
        <button
          onClick={handleToggleBatchMode}
          className={`ml-auto text-xs px-2 py-1 rounded transition-colors ${
            batchMode
              ? "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400"
              : "hover:bg-gray-100 dark:hover:bg-slate-800 text-gray-500 dark:text-slate-400"
          }`}
        >
          {batchMode ? "Exit Select" : "Select"}
        </button>
      </div>

      {/* Empty state */}
      {cluster.photos.length === 0 ? (
        <div className="text-center mt-20" style={{ color: "var(--color-text-muted)" }}>
          <p className="text-lg">This cluster is empty</p>
          <p className="text-sm mt-1">
            Photos may have been moved or removed.
          </p>
        </div>
      ) : (
        <>
          <PhotoGrid
            totalCount={cluster.photos.length}
            emptyMessage="This cluster is empty"
            virtualized={cluster.photos.length > 200}
          >
            {cluster.photos.map((photo) => (
              <div key={photo.path}>
                <PhotoCard
                  photo={photo}
                  selected={selectedPhotoPaths.has(photo.path)}
                  selectionMode={batchMode}
                  onSelectToggle={() => handleSelectToggle(photo.path)}
                  onRemove={() => handleRemove(cluster.id, photo.path)}
                  onMove={(toId) => handleMove(photo.path, cluster.id, toId)}
                  otherClusters={otherClusters}
                  onViewFull={(p) => handlePhotoClick(p)}
                  displayClusterId={cluster.id}
                />
              </div>
            ))}
          </PhotoGrid>

          {/* Batch action bar */}
          <BatchActionBar
            count={selectedPhotoPaths.size}
            onReject={handleBatchReject}
            onClear={handleBatchClear}
            disabled={isPending}
          />

          {/* Bottom padding for batch bar */}
          {selectedPhotoPaths.size > 0 && <div className="h-20" />}
        </>
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
