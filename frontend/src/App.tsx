import { useCallback, useRef, useState } from "react";
import type { WorkspaceState } from "./api";
import { ClusterDetail } from "./components/ClusterDetail";
import { Header } from "./components/Header";
import { NoisePanel } from "./components/NoisePanel";
import { PhotoCard } from "./components/PhotoCard";
import { PhotoGrid } from "./components/PhotoGrid";
import { PipelineLoader } from "./components/PipelineLoader";
import { Sidebar } from "./components/Sidebar";
import { ToastContainer } from "./components/Toast";
import { useKeyboard } from "./hooks/useKeyboard";
import type { ToastMessage } from "./hooks/useWorkspace";
import { useWorkspace } from "./hooks/useWorkspace";

type ViewMode = "all" | "noise" | { clusterId: number };

/** Unique ID counter for toast items. */
let toastIdCounter = 0;

function App() {
  // ── Toast state ──────────────────────────────────────────
  const [toasts, setToasts] = useState<
    Array<{ id: number } & ToastMessage>
  >([]);

  const addToast = useCallback((toast: ToastMessage) => {
    const id = ++toastIdCounter;
    setToasts((prev) => [...prev, { id, ...toast }]);
  }, []);

  const dismissToast = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  // ── Workspace hook ───────────────────────────────────────
  const {
    ws,
    loading,
    error,
    mutating,
    load,
    merge,
    remove,
    move,
    batchAssign,
    rename,
    undoLast,
    saveToDisk,
  } = useWorkspace(addToast);

  // ── View / UI state ──────────────────────────────────────
  const [view, setView] = useState<ViewMode>("all");
  const [mergeMode, setMergeMode] = useState(false);
  const [selectedForMerge, setSelectedForMerge] = useState<Set<number>>(
    new Set(),
  );
  const [editingName, setEditingName] = useState<number | null>(null);
  const [editValue, setEditValue] = useState("");
  const [saving, setSaving] = useState(false);

  // Track whether we've detected pipeline mode (first load)
  const hasLoadedOnce = useRef(false);

  // ── Event handlers ───────────────────────────────────────

  const handleWorkspaceReady = useCallback((_data: WorkspaceState) => {
    // PipelineLoader already fetched the workspace, but useWorkspace
    // manages its own state. Trigger a fresh load to sync.
    load();
  }, [load]);

  const handleError = useCallback((_msg: string) => {
    // Error from pipeline — useWorkspace handles errors from load()
  }, []);

  const handleSelectCluster = useCallback(
    (id: number) => {
      if (mergeMode) {
        setSelectedForMerge((prev) => {
          const next = new Set(prev);
          if (next.has(id)) next.delete(id);
          else next.add(id);
          return next;
        });
      } else {
        setView({ clusterId: id });
      }
    },
    [mergeMode],
  );

  const handleExecuteMerge = useCallback(async () => {
    const ids = Array.from(selectedForMerge);
    if (ids.length < 2) return;
    const toId = ids[0];
    const fromIds = ids.slice(1);
    await merge(fromIds, toId);
    setSelectedForMerge(new Set());
    setMergeMode(false);
    setView({ clusterId: toId });
  }, [selectedForMerge, merge]);

  const handleRemove = useCallback(
    async (clusterId: number, imagePath: string) => {
      const updated = await remove(clusterId, imagePath);
      // If cluster disappeared (merged or empty), navigate back to all
      if (updated && !updated.clusters.find((c) => c.id === clusterId)) {
        setView("all");
      }
    },
    [remove],
  );

  const handleMove = useCallback(
    async (imagePath: string, fromId: number, toId: number) => {
      await move(imagePath, fromId, toId);
    },
    [move],
  );

  const handleRename = useCallback(async () => {
    if (editingName === null || !editValue.trim()) {
      setEditingName(null);
      return;
    }
    await rename(editingName, editValue.trim());
    setEditingName(null);
  }, [editingName, editValue, rename]);

  const handleStartEdit = useCallback((id: number, name: string) => {
    setEditingName(id);
    setEditValue(name);
  }, []);

  const handleCancelEdit = useCallback(() => {
    setEditingName(null);
  }, []);

  const handleMergeCancel = useCallback(() => {
    setMergeMode(false);
    setSelectedForMerge(new Set());
  }, []);

  const handleViewAll = useCallback(() => {
    setView("all");
    setMergeMode(false);
    setSelectedForMerge(new Set());
  }, []);

  const handleViewNoise = useCallback(() => {
    setView("noise");
    setMergeMode(false);
    setSelectedForMerge(new Set());
  }, []);

  const handleToggleMergeMode = useCallback(() => {
    setMergeMode(true);
    setSelectedForMerge(new Set());
    setView("all");
  }, []);

  const handleUndo = useCallback(async () => {
    await undoLast();
  }, [undoLast]);

  const handleSave = useCallback(async () => {
    setSaving(true);
    try {
      await saveToDisk();
    } finally {
      setSaving(false);
    }
  }, [saveToDisk]);

  const handleDropOnCluster = useCallback(
    async (imagePath: string, clusterId: number) => {
      await move(imagePath, -1, clusterId);
    },
    [move],
  );

  // ── Keyboard shortcuts ───────────────────────────────────
  useKeyboard({
    onUndo: ws?.can_undo ? handleUndo : undefined,
    onSave: handleSave,
    onEscape: () => {
      if (editingName !== null) {
        handleCancelEdit();
      } else if (mergeMode) {
        handleMergeCancel();
      }
    },
  });

  // ── Loading / Error states ───────────────────────────────

  // Show pipeline loader when the workspace hasn't loaded yet
  // and we're in pipeline mode (SSE is active).
  // useWorkspace does an initial fetchWorkspace() on mount.
  // If that succeeds immediately (no pipeline), we show the app.
  // If it fails, we fall into pipeline mode.
  if (loading && !ws && !hasLoadedOnce.current) {
    // First attempt failed — show pipeline loader
    if (error) {
      hasLoadedOnce.current = true;
    }
    return (
      <PipelineLoader
        onReady={handleWorkspaceReady}
        onError={handleError}
      />
    );
  }

  if (error && !ws) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-50">
        <div className="text-center">
          <p className="text-xl text-red-500 mb-2">
            Failed to load workspace
          </p>
          <p className="text-sm text-gray-500 mb-4">{error}</p>
          <button
            onClick={load}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!ws) return null;

  const selectedCluster =
    typeof view === "object" && "clusterId" in view
      ? ws.clusters.find((c) => c.id === view.clusterId) ?? null
      : null;

  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <Header
        stats={ws.stats}
        canUndo={ws.can_undo}
        saving={saving}
        mutating={mutating}
        saveResult={null}
        onUndo={handleUndo}
        onSave={handleSave}
      />

      {/* Body */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <Sidebar
          ws={ws}
          view={view}
          mergeMode={mergeMode}
          selectedForMerge={selectedForMerge}
          editingName={editingName}
          editValue={editValue}
          mutating={mutating}
          onSelectCluster={handleSelectCluster}
          onViewAll={handleViewAll}
          onViewNoise={handleViewNoise}
          onToggleMergeMode={handleToggleMergeMode}
          onCancelMerge={handleMergeCancel}
          onExecuteMerge={handleExecuteMerge}
          onStartEdit={handleStartEdit}
          onEditChange={setEditValue}
          onSaveEdit={handleRename}
          onCancelEdit={handleCancelEdit}
          onDropOnCluster={handleDropOnCluster}
        />

        {/* Main content */}
        <main className="flex-1 overflow-y-auto p-6 bg-gray-50/50">
          {selectedCluster ? (
            <ClusterDetail
              cluster={selectedCluster}
              clusters={ws.clusters}
              onRemove={(path) => handleRemove(selectedCluster.id, path)}
              onMove={(path, toId) =>
                handleMove(path, selectedCluster.id, toId)
              }
              onBack={handleViewAll}
              onStartRename={() =>
                handleStartEdit(
                  selectedCluster.id,
                  selectedCluster.name,
                )
              }
              editing={editingName === selectedCluster.id}
              editValue={editValue}
              onEditChange={setEditValue}
              onSaveEdit={handleRename}
              onCancelEdit={handleCancelEdit}
            />
          ) : view === "noise" ? (
            <NoisePanel
              noisePhotos={ws.noise_photos}
              clusters={ws.clusters}
              nextClusterId={ws.next_cluster_id}
              onAssign={(path, toId) => handleMove(path, -1, toId)}
              onBatchAssign={batchAssign}
              mutating={mutating}
            />
          ) : (
            <div>
              <h2 className="text-lg font-semibold text-gray-700 mb-4">
                All Photos ({ws.all_photos.length})
              </h2>
              <PhotoGrid
                totalCount={ws.all_photos.length}
                emptyMessage="No photos"
              >
                {ws.all_photos.map((photo) => (
                  <div key={photo.path}>
                    <PhotoCard photo={photo} />
                  </div>
                ))}
              </PhotoGrid>
            </div>
          )}
        </main>
      </div>

      {/* Toast notifications */}
      <ToastContainer toasts={toasts} onDismiss={dismissToast} />
    </div>
  );
}

export default App;
