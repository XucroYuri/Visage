import { Suspense, lazy, useCallback, useEffect, useRef, useState } from "react";
import { Routes, Route, useLocation, useNavigate } from "react-router";
import type { SaveSettings, WorkspaceState } from "./api";
import { ApiError, fetchWorkspace } from "./api";
import { Header } from "./components/Header";
import { PhotoCard } from "./components/PhotoCard";
import { PhotoGrid } from "./components/PhotoGrid";
import { SaveDialog } from "./components/SaveDialog";
import { SettingsPanel } from "./components/SettingsPanel";
import { Sidebar } from "./components/Sidebar";
import { ToastContainer } from "./components/Toast";
import { useKeyboard } from "./hooks/useKeyboard";
import { useSettingsStore } from "./store/settings";
import { useToastStore } from "./store/toast";
import { useUIStore } from "./store/ui";
import {
  useIsMutating,
  useMergeMutation,
  useMoveMutation,
  useRenameMutation,
  useSaveMutation,
  useUndoMutation,
  useWorkspaceStore,
} from "./store/workspace";

// ── Lazy-loaded view-level components ──────────────────────────

const ClusterDetail = lazy(() =>
  import("./components/ClusterDetail").then((m) => ({ default: m.ClusterDetail })),
);
const NoisePanel = lazy(() =>
  import("./components/NoisePanel").then((m) => ({ default: m.NoisePanel })),
);
const PipelineLoader = lazy(() =>
  import("./components/PipelineLoader").then((m) => ({ default: m.PipelineLoader })),
);

// ── Dark mode manager ──────────────────────────────────────────

function useDarkMode() {
  const darkMode = useUIStore((s) => s.darkMode);

  // Determine effective dark state
  useEffect(() => {
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");

    const update = () => {
      const isDark = darkMode === true || (darkMode === "system" && mediaQuery.matches);
      document.documentElement.classList.toggle("dark", isDark);
    };

    update();

    // Listen for system preference changes when in "system" mode
    if (darkMode === "system") {
      mediaQuery.addEventListener("change", update);
      return () => mediaQuery.removeEventListener("change", update);
    }
  }, [darkMode]);
}

// ── Suspense fallback ──────────────────────────────────────────

function ViewFallback() {
  return (
    <div className="flex items-center justify-center h-64">
      <div className="w-8 h-8 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin" />
    </div>
  );
}

// ── App ────────────────────────────────────────────────────────

export default function App() {
  const navigate = useNavigate();
  const location = useLocation();

  // Dark mode
  useDarkMode();

  // Responsive layout
  const batchMode = useUIStore((s) => s.batchMode);
  const selectedPhotoPaths = useUIStore((s) => s.selectedPhotoPaths);
  const setSelectedPhotoPaths = useUIStore((s) => s.setSelectedPhotoPaths);

  // Workspace store
  const ws = useWorkspaceStore((s) => s.ws);
  const error = useWorkspaceStore((s) => s.error);
  const setWs = useWorkspaceStore((s) => s.setWs);
  const setError = useWorkspaceStore((s) => s.setError);

  // UI store
  const mergeMode = useUIStore((s) => s.mergeMode);
  const setMergeMode = useUIStore((s) => s.setMergeMode);
  const selectedForMerge = useUIStore((s) => s.selectedForMerge);
  const setSelectedForMerge = useUIStore((s) => s.setSelectedForMerge);
  const editingName = useUIStore((s) => s.editingName);
  const setEditingName = useUIStore((s) => s.setEditingName);
  const editValue = useUIStore((s) => s.editValue);
  const setEditValue = useUIStore((s) => s.setEditValue);
  const saving = useUIStore((s) => s.saving);
  const setSaving = useUIStore((s) => s.setSaving);

  // Toast store
  const toasts = useToastStore((s) => s.toasts);
  const dismissToast = useToastStore((s) => s.dismissToast);

  // Mutations
  const mergeMutation = useMergeMutation();
  const moveMutation = useMoveMutation();
  const renameMutation = useRenameMutation();
  const undoMutation = useUndoMutation();
  const saveMutation = useSaveMutation();
  const isMutating = useIsMutating();

  // UI state for overlay components
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [saveResult, setSaveResult] = useState<string | null>(null);

  // Ref for main content area (for scrolling)
  const mainRef = useRef<HTMLDivElement>(null);

  // ── Event handlers ─────────────────────────────────────────

  const handleExecuteMerge = useCallback(() => {
    const ids = Array.from(selectedForMerge);
    if (ids.length < 2) return;
    const toId = ids[0];
    const fromIds = ids.slice(1);
    mergeMutation.mutate({ fromIds, toId });
    setSelectedForMerge(new Set());
    setMergeMode(false);
    navigate(`/cluster/${toId}`);
  }, [selectedForMerge, mergeMutation, setSelectedForMerge, setMergeMode, navigate]);

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
    } as { clusterId: number; name: string; oldName: string; clusterCountBefore: number });
    setEditingName(null);
  }, [editingName, editValue, renameMutation, setEditingName]);

  const handleStartEdit = useCallback(
    (id: number, name: string) => {
      setEditingName(id);
      setEditValue(name);
    },
    [setEditingName, setEditValue],
  );

  const handleCancelEdit = useCallback(() => {
    setEditingName(null);
  }, [setEditingName]);

  const handleMergeCancel = useCallback(() => {
    setMergeMode(false);
    setSelectedForMerge(new Set());
  }, [setMergeMode, setSelectedForMerge]);

  const handleToggleMergeMode = useCallback(() => {
    setMergeMode(true);
    setSelectedForMerge(new Set());
    navigate("/");
  }, [setMergeMode, setSelectedForMerge, navigate]);

  const handleUndo = useCallback(() => {
    undoMutation.mutate();
  }, [undoMutation]);

  const handleOpenSave = useCallback(() => {
    setSaveDialogOpen(true);
  }, []);

  const handleOpenSettings = useCallback(() => {
    setSettingsOpen(true);
  }, []);

  const handleSave = useCallback(() => {
    setSaving(true);
    const store = useSettingsStore.getState();
    const settings: SaveSettings = {
      output_dir: store.outputDir || undefined,
      copy_mode: store.copyMode,
      folder_prefix: store.folderPrefix,
      include_unclustered: store.includeUnclustered,
      include_no_faces: store.includeNoFaces,
      multi_face_strategy: store.multiFaceStrategy,
    };
    if (
      store.clusterSelectionMode === "selected" &&
      store.selectedClusterIds.size > 0
    ) {
      settings.cluster_ids = Array.from(store.selectedClusterIds);
    }
    saveMutation.mutate(settings, {
      onSuccess: (res, vars) => {
        const action = vars.copy_mode !== false ? "copied" : "moved";
        const count =
          res.stats[action] ??
          Object.values(res.stats).reduce((a, b) => a + b, 0);
        setSaveResult(`${count} files ${action}`);
      },
      onSettled: () => {
        setSaving(false);
        setSaveDialogOpen(false);
      },
    });
  }, [saveMutation, setSaving]);

  // Auto-clear save result after 6 seconds
  useEffect(() => {
    if (!saveResult) return;
    const timer = setTimeout(() => setSaveResult(null), 6000);
    return () => clearTimeout(timer);
  }, [saveResult]);

  const handleDropOnCluster = useCallback(
    (imagePath: string, clusterId: number) => {
      moveMutation.mutate({ imagePath, fromId: -1, toId: clusterId });
    },
    [moveMutation],
  );

  // ── Pipeline mode handlers ──────────────────────────────────

  const handleWorkspaceReady = useCallback(
    (loaded: WorkspaceState) => {
      setWs(loaded);
    },
    [setWs],
  );

  const handlePipelineError = useCallback(
    (msg: string) => {
      fetchWorkspace()
        .then(setWs)
        .catch((e) => {
          if (e instanceof ApiError && e.statusCode === 503) return;
          setError(String(e));
        });
      console.error("Pipeline error:", msg);
    },
    [setWs, setError],
  );

  // ── Keyboard shortcuts ──────────────────────────────────────

  const handleSelectAll = useCallback(() => {
    // Select all photos on the current route
    if (ws && location.pathname === "/" && ws.all_photos.length > 0) {
      setSelectedPhotoPaths(new Set(ws.all_photos.map((p) => p.path)));
    }
  }, [ws, setSelectedPhotoPaths]);

  useKeyboard({
    onUndo: ws?.can_undo ? handleUndo : undefined,
    onSave: saveDialogOpen ? undefined : handleOpenSave,
    onSelectAll: handleSelectAll,
    onEscape: () => {
      if (editingName !== null) {
        handleCancelEdit();
      } else if (mergeMode) {
        handleMergeCancel();
      } else if (selectedPhotoPaths.size > 0) {
        setSelectedPhotoPaths(new Set());
      } else if (batchMode) {
        useUIStore.getState().setBatchMode(false);
      }
    },
  });

  // ── Loading / Error states ──────────────────────────────────

  if (!ws && !error) {
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
      <div className="flex items-center justify-center h-screen" style={{ backgroundColor: "var(--color-bg-primary)" }}>
        <div className="text-center">
          <p className="text-xl text-red-500 mb-2">
            Failed to load workspace
          </p>
          <p className="text-sm mb-4" style={{ color: "var(--color-text-secondary)" }}>{error}</p>
          <button
            onClick={() => {
              setError(null);
              fetchWorkspace()
                .then(setWs)
                .catch((e) => setError(String(e)));
            }}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!ws) return null;

  // ── Context bag passed to Sidebar ──

  const sidebarCtx = {
    ws,
    mergeMode,
    selectedForMerge,
    editingName,
    editValue,
    isMutating,
    handleToggleMergeMode,
    handleMergeCancel,
    handleExecuteMerge,
    handleStartEdit,
    handleRename,
    handleCancelEdit,
    handleDropOnCluster,
    setEditValue,
  };

  // ── Main UI ───────────────────────────────────────────────

  return (
    <div className="flex flex-col h-screen" style={{ backgroundColor: "var(--color-bg-primary)" }}>
      <Header
        stats={ws.stats}
        canUndo={ws.can_undo}
        saving={saving}
        mutating={isMutating}
        saveResult={saveResult}
        onUndo={handleUndo}
        onOpenSave={handleOpenSave}
        onOpenSettings={handleOpenSettings}
      />

      <div className="flex flex-1 overflow-hidden">
        <Sidebar ctx={sidebarCtx} />

        <main
          ref={mainRef}
          className="flex-1 overflow-y-auto p-4 sm:p-6"
          style={{
            backgroundColor: "var(--color-bg-tertiary)",
            color: "var(--color-text-primary)",
          }}
        >
          <Suspense fallback={<ViewFallback />}>
            <Routes>
              <Route
                path="/"
                element={
                  <div>
                    <h2
                      className="text-lg font-semibold mb-4"
                      style={{ color: "var(--color-text-primary)" }}
                    >
                      All Photos ({ws.all_photos.length})
                    </h2>
                    <PhotoGrid
                      totalCount={ws.all_photos.length}
                      emptyMessage="No photos"
                      virtualized={ws.all_photos.length > 200}
                    >
                      {ws.all_photos.map((photo) => (
                        <div key={photo.path}>
                          <PhotoCard photo={photo} />
                        </div>
                      ))}
                    </PhotoGrid>
                  </div>
                }
              />
              <Route path="/cluster/:id" element={<ClusterDetail />} />
              <Route path="/noise" element={<NoisePanel />} />
            </Routes>
          </Suspense>
        </main>
      </div>

      <ToastContainer toasts={toasts} onDismiss={dismissToast} />

      <SettingsPanel
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        inputDir={ws.input_dir}
        embeddingBackend={ws.config.embedding_backend}
        totalImages={ws.stats.total_images}
        imagesWithFaces={ws.stats.images_with_faces}
        clusterMethod={ws.config.embedding_backend ? "hdbscan" : "dbscan"}
        mergeThreshold={0.80}
      />

      <SaveDialog
        open={saveDialogOpen}
        onClose={() => setSaveDialogOpen(false)}
        onSave={handleSave}
        saving={saving}
        clusters={ws.clusters.map((c) => ({
          id: c.id,
          name: c.name,
          photoCount: c.photo_count,
        }))}
        defaultOutputDir={`${ws.input_dir}/visage_output`}
      />
    </div>
  );
}
