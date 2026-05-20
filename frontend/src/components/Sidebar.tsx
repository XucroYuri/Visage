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

  return (
    <aside className="w-72 bg-gray-50 border-r border-gray-200 overflow-y-auto shrink-0 flex flex-col">
      {/* Navigation buttons */}
      <nav className="p-3 border-b border-gray-200 space-y-1">
        <button
          onClick={handleViewAll}
          className={`w-full text-left px-3 py-2 rounded text-sm font-medium transition-colors ${
            location.pathname === "/"
              ? "bg-blue-50 text-blue-700"
              : "hover:bg-gray-100 text-gray-700"
          }`}
        >
          &#128247; All Photos
        </button>

        <button
          onClick={handleViewNoise}
          className={`w-full text-left px-3 py-2 rounded text-sm font-medium transition-colors ${
            location.pathname === "/noise"
              ? "bg-amber-50 text-amber-700"
              : "hover:bg-gray-100 text-gray-700"
          }`}
        >
          &#10067; Unclustered
          {ws.noise_photos.length > 0 && (
            <span className="ml-2 inline-flex items-center px-1.5 py-0.5 text-xs font-medium rounded-full bg-amber-100 text-amber-700">
              {ws.noise_photos.length}
            </span>
          )}
        </button>

        {/* Merge mode toggle */}
        <button
          onClick={mergeMode ? handleMergeCancel : handleToggleMergeMode}
          className={`w-full text-left px-3 py-2 rounded text-sm font-medium transition-colors ${
            mergeMode
              ? "bg-purple-100 text-purple-700"
              : "hover:bg-gray-100 text-gray-700"
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
        className="flex-1 divide-y divide-gray-100"
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
