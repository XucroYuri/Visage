import { type DragEvent, useCallback } from "react";
import { useNavigate, useLocation } from "react-router";
import type { WorkspaceState } from "../api";
import { useUIStore } from "../store/ui";
import { ClusterRow } from "./ClusterRow";

interface SidebarContext {
  ws: WorkspaceState;
  mergeMode: boolean;
  selectedForMerge: Set<number>;
  editingName: number | null;
  editValue: string;
  isMutating: boolean;
  handleToggleMergeMode: () => void;
  handleMergeCancel: () => void;
  handleExecuteMerge: () => void;
  handleStartEdit: (id: number, name: string) => void;
  handleRename: () => void;
  handleCancelEdit: () => void;
  handleDropOnCluster: (imagePath: string, clusterId: number) => void;
  setEditValue: (v: string) => void;
}

export function Sidebar({ ctx }: { ctx: SidebarContext }) {
  const navigate = useNavigate();
  const location = useLocation();

  const sidebarCollapsed = useUIStore((s) => s.sidebarCollapsed);
  const toggleSidebar = useUIStore((s) => s.toggleSidebar);

  const {
    ws,
    mergeMode,
    selectedForMerge,
    editingName,
    editValue,
    isMutating: mutating,
    handleToggleMergeMode,
    handleMergeCancel,
    handleExecuteMerge,
    handleStartEdit,
    handleRename,
    handleCancelEdit,
    handleDropOnCluster,
    setEditValue,
  } = ctx;

  const handleDragOver = (e: DragEvent) => {
    e.preventDefault();
  };

  const handleDropOnRow = (e: DragEvent, clusterId: number) => {
    e.preventDefault();
    const imagePath = e.dataTransfer.getData("text/plain");
    if (imagePath) {
      handleDropOnCluster(imagePath, clusterId);
    }
  };

  const handleSelectCluster = useCallback(
    (id: number) => {
      if (mergeMode) {
        const prev = useUIStore.getState().selectedForMerge;
        const next = new Set(prev);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        useUIStore.getState().setSelectedForMerge(next);
      } else {
        navigate(`/cluster/${id}`);
      }
    },
    [mergeMode, navigate],
  );

  const handleViewAll = useCallback(() => {
    useUIStore.getState().setMergeMode(false);
    useUIStore.getState().setSelectedForMerge(new Set());
    navigate("/");
  }, [navigate]);

  const handleViewNoise = useCallback(() => {
    useUIStore.getState().setMergeMode(false);
    useUIStore.getState().setSelectedForMerge(new Set());
    navigate("/noise");
  }, [navigate]);

  const activeClusterId = (() => {
    const match = location.pathname.match(/^\/cluster\/(\d+)$/);
    return match ? parseInt(match[1], 10) : null;
  })();

  // ── Collapsed sidebar: icon-only bar ───────────────────────
  if (sidebarCollapsed) {
    return (
      <aside className="flex flex-col items-center w-14 bg-gray-50 dark:bg-slate-800 border-r border-gray-200 dark:border-slate-700 shrink-0 overflow-hidden">
        <button
          onClick={toggleSidebar}
          className="w-full p-3 text-gray-400 hover:text-gray-600 dark:hover:text-slate-300 hover:bg-gray-100 dark:hover:bg-slate-700 transition-colors"
          title="Expand sidebar"
          aria-label="Expand sidebar"
        >
          <svg className="w-5 h-5 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
          </svg>
        </button>

        <nav className="flex flex-col items-center gap-1 p-2 w-full">
          <button
            onClick={handleViewAll}
            className={`w-10 h-10 flex items-center justify-center rounded-lg text-lg transition-colors ${
              location.pathname === "/"
                ? "bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400"
                : "hover:bg-gray-100 dark:hover:bg-slate-700 text-gray-500 dark:text-slate-400"
            }`}
            title="All Photos"
          >
            &#128247;
          </button>
          <button
            onClick={handleViewNoise}
            className={`w-10 h-10 flex items-center justify-center rounded-lg text-lg transition-colors ${
              location.pathname === "/noise"
                ? "bg-amber-50 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400"
                : "hover:bg-gray-100 dark:hover:bg-slate-700 text-gray-500 dark:text-slate-400"
            }`}
            title="Unclustered"
          >
            &#10067;
          </button>
          <button
            onClick={() => navigate("/albums")}
            className={`w-10 h-10 flex items-center justify-center rounded-lg text-lg transition-colors ${
              location.pathname === "/albums"
                ? "bg-green-50 dark:bg-green-900/30 text-green-700 dark:text-green-400"
                : "hover:bg-gray-100 dark:hover:bg-slate-700 text-gray-500 dark:text-slate-400"
            }`}
            title="Auto Albums"
          >
            📂
          </button>
          <button
            onClick={() => navigate("/search")}
            className={`w-10 h-10 flex items-center justify-center rounded-lg text-lg transition-colors ${
              location.pathname === "/search"
                ? "bg-purple-50 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400"
                : "hover:bg-gray-100 dark:hover:bg-slate-700 text-gray-500 dark:text-slate-400"
            }`}
            title="Search"
          >
            🔍
          </button>
        </nav>
      </aside>
    );
  }

  // ── Expanded sidebar ──────────────────────────────────────
  return (
    <aside className="flex flex-col w-72 bg-gray-50 dark:bg-slate-800 border-r border-gray-200 dark:border-slate-700 overflow-y-auto shrink-0">
      {/* Collapse toggle */}
      <div className="flex items-center justify-between p-2 border-b border-gray-200 dark:border-slate-700">
        <span className="text-xs font-semibold uppercase tracking-wider text-gray-400 dark:text-slate-500 ml-2">
          Clusters
        </span>
        <button
          onClick={toggleSidebar}
          className="p-1.5 text-gray-400 hover:text-gray-600 dark:hover:text-slate-300 hover:bg-gray-100 dark:hover:bg-slate-700 rounded transition-colors"
          title="Collapse sidebar"
          aria-label="Collapse sidebar"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
          </svg>
        </button>
      </div>

      {/* Navigation buttons */}
      <nav className="p-3 border-b border-gray-200 dark:border-slate-700 space-y-1">
        <button
          onClick={handleViewAll}
          className={`w-full text-left px-3 py-2 rounded text-sm font-medium transition-colors ${
            location.pathname === "/"
              ? "bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400"
              : "hover:bg-gray-100 dark:hover:bg-slate-700 text-gray-700 dark:text-slate-300"
          }`}
        >
          &#128247; All Photos
        </button>

        <button
          onClick={handleViewNoise}
          className={`w-full text-left px-3 py-2 rounded text-sm font-medium transition-colors ${
            location.pathname === "/noise"
              ? "bg-amber-50 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400"
              : "hover:bg-gray-100 dark:hover:bg-slate-700 text-gray-700 dark:text-slate-300"
          }`}
        >
          &#10067; Unclustered
          {ws.noise_photos.length > 0 && (
            <span className="ml-2 inline-flex items-center px-1.5 py-0.5 text-xs font-medium rounded-full bg-amber-100 dark:bg-amber-900/50 text-amber-700 dark:text-amber-400">
              {ws.noise_photos.length}
            </span>
          )}
        </button>

        {/* Phase 3 navigation */}
        <button
          onClick={() => navigate("/albums")}
          className={`w-full text-left px-3 py-2 rounded text-sm font-medium transition-colors ${
            location.pathname === "/albums"
              ? "bg-green-50 dark:bg-green-900/30 text-green-700 dark:text-green-400"
              : "hover:bg-gray-100 dark:hover:bg-slate-700 text-gray-700 dark:text-slate-300"
          }`}
        >
          📂 Auto Albums
        </button>

        <button
          onClick={() => navigate("/search")}
          className={`w-full text-left px-3 py-2 rounded text-sm font-medium transition-colors ${
            location.pathname === "/search"
              ? "bg-purple-50 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400"
              : "hover:bg-gray-100 dark:hover:bg-slate-700 text-gray-700 dark:text-slate-300"
          }`}
        >
          🔍 Search
        </button>

        {/* Merge mode toggle */}
        <button
          onClick={mergeMode ? handleMergeCancel : handleToggleMergeMode}
          className={`w-full text-left px-3 py-2 rounded text-sm font-medium transition-colors ${
            mergeMode
              ? "bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400"
              : "hover:bg-gray-100 dark:hover:bg-slate-700 text-gray-700 dark:text-slate-300"
          }`}
        >
          {mergeMode ? "&#10005; Cancel Merge" : "&#9878; Merge Mode"}
        </button>

        {/* Merge execute button */}
        {mergeMode && selectedForMerge.size >= 2 && (
          <button
            onClick={handleExecuteMerge}
            disabled={mutating}
            className="w-full mt-1 px-3 py-2 bg-purple-600 text-white rounded text-sm font-medium hover:bg-purple-700 disabled:opacity-40 transition-colors"
          >
            {mutating ? (
              <span className="flex items-center justify-center gap-1.5">
                <span className="inline-block w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Merging...
              </span>
            ) : (
              `Merge ${selectedForMerge.size} Selected`
            )}
          </button>
        )}
      </nav>

      {/* Cluster list with drop zone */}
      <div
        className="flex-1 divide-y divide-gray-100 dark:divide-slate-700/50"
        onDragOver={handleDragOver}
      >
        {ws.clusters.map((c) => (
          <ClusterRow
            key={c.id}
            cluster={c}
            selected={activeClusterId === c.id}
            mergeMode={mergeMode}
            checkedForMerge={selectedForMerge.has(c.id)}
            editing={editingName === c.id}
            editValue={editValue}
            onSelect={() => handleSelectCluster(c.id)}
            onStartEdit={() => handleStartEdit(c.id, c.name)}
            onEditChange={setEditValue}
            onSaveEdit={handleRename}
            onCancelEdit={handleCancelEdit}
            onDrop={(e, clusterId) => handleDropOnRow(e, clusterId)}
          />
        ))}
      </div>
    </aside>
  );
}
