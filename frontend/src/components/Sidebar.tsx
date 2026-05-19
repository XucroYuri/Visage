import type { DragEvent } from "react";
import type { WorkspaceState } from "../api";
import { ClusterRow } from "./ClusterRow";

type ViewMode = "all" | "noise" | { clusterId: number };

interface SidebarProps {
  ws: WorkspaceState;
  view: ViewMode;
  mergeMode: boolean;
  selectedForMerge: Set<number>;
  editingName: number | null;
  editValue: string;
  mutating: boolean;
  onSelectCluster: (id: number) => void;
  onViewAll: () => void;
  onViewNoise: () => void;
  onToggleMergeMode: () => void;
  onCancelMerge: () => void;
  onExecuteMerge: () => void;
  onStartEdit: (id: number, name: string) => void;
  onEditChange: (value: string) => void;
  onSaveEdit: () => void;
  onCancelEdit: () => void;
  /** Called when a noise photo is dropped on a cluster row */
  onDropOnCluster: (imagePath: string, clusterId: number) => void;
}

export function Sidebar({
  ws,
  view,
  mergeMode,
  selectedForMerge,
  editingName,
  editValue,
  mutating,
  onSelectCluster,
  onViewAll,
  onViewNoise,
  onToggleMergeMode,
  onCancelMerge,
  onExecuteMerge,
  onStartEdit,
  onEditChange,
  onSaveEdit,
  onCancelEdit,
  onDropOnCluster,
}: SidebarProps) {
  const handleDragOver = (e: DragEvent) => {
    e.preventDefault();
  };

  const handleDropOnRow = (e: DragEvent, clusterId: number) => {
    const imagePath = e.dataTransfer.getData("text/plain");
    if (imagePath) {
      onDropOnCluster(imagePath, clusterId);
    }
  };

  const activeClusterId =
    typeof view === "object" && "clusterId" in view ? view.clusterId : null;

  return (
    <aside className="w-72 bg-gray-50 border-r border-gray-200 overflow-y-auto shrink-0 flex flex-col">
      {/* Navigation buttons */}
      <nav className="p-3 border-b border-gray-200 space-y-1">
        <button
          onClick={onViewAll}
          className={`w-full text-left px-3 py-2 rounded text-sm font-medium transition-colors ${
            view === "all"
              ? "bg-blue-50 text-blue-700"
              : "hover:bg-gray-100 text-gray-700"
          }`}
        >
          &#128247; All Photos
        </button>

        <button
          onClick={onViewNoise}
          className={`w-full text-left px-3 py-2 rounded text-sm font-medium transition-colors ${
            view === "noise"
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
          onClick={mergeMode ? onCancelMerge : onToggleMergeMode}
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
            onClick={onExecuteMerge}
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

      {/* Cluster list */}
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
            onSelect={() => onSelectCluster(c.id)}
            onStartEdit={() => onStartEdit(c.id, c.name)}
            onEditChange={onEditChange}
            onSaveEdit={onSaveEdit}
            onCancelEdit={onCancelEdit}
            onDrop={(e, clusterId) => handleDropOnRow(e, clusterId)}
          />
        ))}
      </div>
    </aside>
  );
}
