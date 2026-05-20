import { createContext, useCallback, useContext, useState } from "react";
import type { ReactNode } from "react";
import type { WorkspaceState } from "../api";
import type { ToastMessage } from "../hooks/useWorkspace";
import { useWorkspace } from "../hooks/useWorkspace";

// ── View mode ────────────────────────────────────────────

export type ViewMode = "all" | "noise" | { clusterId: number };

// ── Toast item type ──────────────────────────────────────

export interface ToastItem {
  id: number;
  type: ToastMessage["type"];
  text: string;
}

// ── Context value shape ──────────────────────────────────

interface AppContextValue {
  // Workspace
  ws: WorkspaceState | null;
  loading: boolean;
  error: string | null;
  mutating: boolean;

  // Mutations
  load: () => void;
  merge: (fromIds: number[], toId: number) => Promise<void>;
  remove: (clusterId: number, imagePath: string) => Promise<WorkspaceState | null>;
  move: (imagePath: string, fromId: number, toId: number) => Promise<void>;
  batchAssign: (imagePaths: string[], toId: number) => Promise<void>;
  rename: (clusterId: number, name: string) => Promise<void>;
  undoLast: () => Promise<void>;
  saveToDisk: (outputDir?: string) => Promise<void>;

  // View state
  view: ViewMode;
  setView: (v: ViewMode) => void;
  mergeMode: boolean;
  setMergeMode: (v: boolean) => void;
  selectedForMerge: Set<number>;
  setSelectedForMerge: (s: Set<number>) => void;
  editingName: number | null;
  setEditingName: (id: number | null) => void;
  editValue: string;
  setEditValue: (v: string) => void;
  saving: boolean;

  // Derived
  selectedCluster: import("../api").ClusterInfo | null;

  // Toast
  toasts: ToastItem[];
  dismissToast: (id: number) => void;

  // Convenience handlers
  handleSelectCluster: (id: number) => void;
  handleExecuteMerge: () => Promise<void>;
  handleRemove: (clusterId: number, imagePath: string) => Promise<void>;
  handleMove: (imagePath: string, fromId: number, toId: number) => Promise<void>;
  handleRename: () => Promise<void>;
  handleStartEdit: (id: number, name: string) => void;
  handleCancelEdit: () => void;
  handleMergeCancel: () => void;
  handleViewAll: () => void;
  handleViewNoise: () => void;
  handleToggleMergeMode: () => void;
  handleUndo: () => Promise<void>;
  handleSave: () => Promise<void>;
  handleDropOnCluster: (imagePath: string, clusterId: number) => Promise<void>;
}

const AppContext = createContext<AppContextValue | null>(null);

// ── Toast ID generator ───────────────────────────────────

let toastIdCounter = 0;

// ── Provider ─────────────────────────────────────────────

export function AppProvider({
  children,
}: {
  children: ReactNode;
}) {
  // ── Toast state ─────────────────────────────────────────
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const addToast = useCallback((toast: ToastMessage) => {
    const id = ++toastIdCounter;
    setToasts((prev) => [...prev, { id, ...toast }]);
  }, []);

  const dismissToast = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  // ── Workspace hook ─────────────────────────────────────
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

  // ── View / UI state ────────────────────────────────────
  const [view, setView] = useState<ViewMode>("all");
  const [mergeMode, setMergeMode] = useState(false);
  const [selectedForMerge, setSelectedForMerge] = useState<Set<number>>(
    new Set(),
  );
  const [editingName, setEditingName] = useState<number | null>(null);
  const [editValue, setEditValue] = useState("");
  const [saving, setSaving] = useState(false);

  // ── Event handlers ────────────────────────────────────

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

  // ── Derived state ──────────────────────────────────────
  const selectedCluster =
    typeof view === "object" && "clusterId" in view
      ? ws?.clusters.find((c) => c.id === view.clusterId) ?? null
      : null;

  // ── Context value ──────────────────────────────────────
  const value: AppContextValue = {
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
    view,
    setView,
    mergeMode,
    setMergeMode,
    selectedForMerge,
    setSelectedForMerge,
    editingName,
    setEditingName,
    editValue,
    setEditValue,
    saving,
    selectedCluster,
    toasts,
    dismissToast,
    handleSelectCluster,
    handleExecuteMerge,
    handleRemove,
    handleMove,
    handleRename,
    handleStartEdit,
    handleCancelEdit,
    handleMergeCancel,
    handleViewAll,
    handleViewNoise,
    handleToggleMergeMode,
    handleUndo,
    handleSave,
    handleDropOnCluster,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

// ── Consumer hook ────────────────────────────────────────

// eslint-disable-next-line react-refresh/only-export-components
export function useAppContext(): AppContextValue {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useAppContext must be used within AppProvider");
  return ctx;
}
