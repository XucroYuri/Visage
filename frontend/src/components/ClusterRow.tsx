import { type KeyboardEvent, type DragEvent, useState } from "react";
import type { ClusterInfo } from "../api";
import { getImageUrl } from "../api";

interface ClusterRowProps {
  cluster: ClusterInfo;
  /** Whether this cluster is the active view */
  selected: boolean;
  /** Whether merge mode is active */
  mergeMode: boolean;
  /** Whether this cluster is checked for merging */
  checkedForMerge: boolean;
  /** Whether we are currently editing this cluster's name */
  editing: boolean;
  /** Current edit input value */
  editValue: string;
  /** Called when the row is clicked (select or merge-check) */
  onSelect: () => void;
  /** Called to start editing the name */
  onStartEdit: () => void;
  /** Called when edit input changes */
  onEditChange: (value: string) => void;
  /** Called to save the edited name */
  onSaveEdit: () => void;
  /** Called to cancel editing */
  onCancelEdit: () => void;
  /** Called when a draggable item hovers over this row (drop target) */
  onDragOver?: (e: DragEvent) => void;
  /** Called when a draggable item is dropped on this row */
  onDrop?: (e: DragEvent, clusterId: number) => void;
}

export function ClusterRow({
  cluster,
  selected,
  mergeMode,
  checkedForMerge,
  editing,
  editValue,
  onSelect,
  onStartEdit,
  onEditChange,
  onSaveEdit,
  onCancelEdit,
  onDragOver,
  onDrop,
}: ClusterRowProps) {
  const [dragOver, setDragOver] = useState(false);

  const handleDragOver = (e: DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
    setDragOver(true);
    onDragOver?.(e);
  };

  const handleDragLeave = () => {
    setDragOver(false);
  };

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    onDrop?.(e, cluster.id);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") onSaveEdit();
    if (e.key === "Escape") onCancelEdit();
  };

  const handleRowKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onSelect();
    }
  };

  return (
    <div
      onClick={onSelect}
      onKeyDown={handleRowKeyDown}
      role="option"
      aria-selected={selected}
      tabIndex={0}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`flex items-center gap-3 px-3 py-2.5 cursor-pointer group transition-all duration-150 ${
        selected
          ? "bg-blue-50 border-l-2 border-blue-500"
          : dragOver
            ? "bg-blue-50 border-l-2 border-blue-400 ring-1 ring-blue-200"
            : "hover:bg-white border-l-2 border-transparent"
      }`}
    >
      {/* Merge checkbox */}
      {mergeMode && (
        <input
          type="checkbox"
          checked={checkedForMerge}
          onChange={onSelect}
          className="shrink-0 w-4 h-4 rounded border-gray-300 text-purple-600 focus:ring-purple-500 cursor-pointer"
        />
      )}

      {/* Thumbnail */}
      <div className="w-10 h-10 bg-gray-200 rounded overflow-hidden shrink-0">
        {cluster.thumbnail && (
          <img
            src={getImageUrl(cluster.thumbnail)}
            alt=""
            className="w-full h-full object-cover"
          />
        )}
      </div>

      {/* Name + info */}
      <div className="flex-1 min-w-0">
        {editing ? (
          <input
            value={editValue}
            onChange={(e) => onEditChange(e.target.value)}
            onKeyDown={handleKeyDown}
            onClick={(e) => e.stopPropagation()}
            className="text-sm font-medium w-full border rounded px-1 py-0.5 focus:outline-none focus:ring-1 focus:ring-blue-400"
          />
        ) : (
          <div
            onClick={(e) => {
              e.stopPropagation();
              onStartEdit();
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                e.stopPropagation();
                onStartEdit();
              }
            }}
            role="button"
            tabIndex={0}
            className="text-sm font-medium text-gray-800 truncate cursor-text hover:text-blue-600 transition-colors"
            title="Click to rename"
          >
            {cluster.name}
          </div>
        )}
        <div className="text-xs text-gray-400">
          {cluster.photo_count} photos &middot; conf:{" "}
          {cluster.confidence.toFixed(2)}
        </div>
      </div>

      {/* Drag indicator (visible on hover when not in merge mode) */}
      {!mergeMode && (
        <div className="opacity-0 group-hover:opacity-100 text-gray-300 text-xs transition-opacity shrink-0">
          &#x2B07;
        </div>
      )}
    </div>
  );
}
