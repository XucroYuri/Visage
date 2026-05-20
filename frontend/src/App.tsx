import { Suspense, lazy, useCallback } from "react";
import { Header } from "./components/Header";
import { PhotoCard } from "./components/PhotoCard";
import { PhotoGrid } from "./components/PhotoGrid";
import { Sidebar } from "./components/Sidebar";
import { ToastContainer } from "./components/Toast";
import { AppProvider, useAppContext } from "./context/AppContext";
import { useKeyboard } from "./hooks/useKeyboard";

// ── Lazy-loaded view-level components ───────────────────

const ClusterDetail = lazy(() =>
  import("./components/ClusterDetail").then((m) => ({ default: m.ClusterDetail })),
);
const NoisePanel = lazy(() =>
  import("./components/NoisePanel").then((m) => ({ default: m.NoisePanel })),
);
const PipelineLoader = lazy(() =>
  import("./components/PipelineLoader").then((m) => ({ default: m.PipelineLoader })),
);

// ── Suspense fallback ────────────────────────────────────

function ViewFallback() {
  return (
    <div className="flex items-center justify-center h-64">
      <div className="w-8 h-8 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin" />
    </div>
  );
}

// ── Inner app (rendered inside AppProvider to access context) ──

function AppInner() {
  const ctx = useAppContext();
  const {
    ws,
    loading,
    error,
    mutating,
    load,
    selectedCluster,
    saving,
    toasts,
    dismissToast,
    handleUndo,
    handleSave,
    handleCancelEdit,
    handleMergeCancel,
    view,
    editingName,
    mergeMode,
  } = ctx;

  // ── Pipeline mode handlers ──────────────────────────────
  const handleWorkspaceReady = useCallback(() => {
    load();
  }, [load]);

  const handlePipelineError = useCallback(() => {
    // useWorkspace handles errors from load()
  }, []);

  // ── Keyboard shortcuts ──────────────────────────────────
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

  // ── Loading / Error states ──────────────────────────────
  if (loading && !ws && !error) {
    return (
      <Suspense fallback={<ViewFallback />}>
        <PipelineLoader
          onReady={handleWorkspaceReady}
          onError={handlePipelineError}
        />
      </Suspense>
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
        <Sidebar />

        {/* Main content */}
        <main className="flex-1 overflow-y-auto p-6 bg-gray-50/50">
          {selectedCluster ? (
            <Suspense fallback={<ViewFallback />}>
              <ClusterDetail />
            </Suspense>
          ) : view === "noise" ? (
            <Suspense fallback={<ViewFallback />}>
              <NoisePanel />
            </Suspense>
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

// ── Root app ─────────────────────────────────────────────

export default function App() {
  return (
    <AppProvider>
      <AppInner />
    </AppProvider>
  );
}
