import { useEffect, useRef, useState } from "react";
import type { ClusterInfo, PhotoInfo } from "../api";
import { getImageUrl } from "../api";
import { PhotoViewer } from "./PhotoViewer";

interface PhotoCardProps {
  photo: PhotoInfo;
  /** Called when user clicks Remove */
  onRemove?: () => void;
  /** Called when user selects a target cluster to move to */
  onMove?: (toId: number) => void;
  /** Available clusters for the Move menu (excluding current cluster) */
  otherClusters?: ClusterInfo[];
  /** If provided, "New Cluster" option is shown using this ID */
  nextClusterId?: number;
  /** If provided, the parent manages the full-size viewer. PhotoCard will NOT render its own. */
  onViewFull?: (photo: PhotoInfo) => void;
  /** Make the card draggable (for noise panel drag-to-assign) */
  draggable?: boolean;
  /** Called when drag starts */
  onDragStart?: (e: React.DragEvent, imagePath: string) => void;
  /** Whether this card is selected (for batch operations) */
  selected?: boolean;
  /** Called when selection checkbox is toggled */
  onSelectToggle?: (imagePath: string) => void;
  /** Show selection checkboxes */
  selectionMode?: boolean;
}

export function PhotoCard({
  photo,
  onRemove,
  onMove,
  otherClusters,
  nextClusterId,
  onViewFull,
  draggable = false,
  onDragStart,
  selected = false,
  onSelectToggle,
  selectionMode = false,
}: PhotoCardProps) {
  const [showFull, setShowFull] = useState(false);
  const [showMoveMenu, setShowMoveMenu] = useState(false);
  const [imgSize, setImgSize] = useState<{ w: number; h: number } | null>(null);
  const moveMenuRef = useRef<HTMLDivElement>(null);
  const filename = photo.path.split("/").pop() || photo.path;

  // Close move menu on outside click
  useEffect(() => {
    if (!showMoveMenu) return;
    const handleClick = (e: MouseEvent) => {
      if (
        moveMenuRef.current &&
        !moveMenuRef.current.contains(e.target as Node)
      ) {
        setShowMoveMenu(false);
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [showMoveMenu]);

  const handleImageClick = () => {
    if (onViewFull) {
      onViewFull(photo);
    } else {
      setShowFull(true);
    }
  };

  return (
    <>
      <div
        className={`group relative bg-white border rounded-lg overflow-hidden mb-3 transition-shadow duration-200 hover:shadow-md ${
          selected
            ? "border-blue-500 ring-2 ring-blue-200"
            : "border-gray-200"
        }`}
        draggable={draggable}
        onDragStart={(e) => {
          if (onDragStart) onDragStart(e, photo.path);
        }}
      >
        {/* Selection checkbox overlay */}
        {selectionMode && (
          <div
            className="absolute top-2 left-2 z-10"
            role="presentation"
            onClick={(e) => e.stopPropagation()}
          >
            <input
              type="checkbox"
              checked={selected}
              onChange={() => onSelectToggle?.(photo.path)}
              className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500 cursor-pointer"
            />
          </div>
        )}

        {/* Image */}
        <div className="relative">
          {/* eslint-disable jsx-a11y/no-noninteractive-element-interactions, jsx-a11y/no-noninteractive-tabindex */}
          <img
            src={getImageUrl(photo.path)}
            alt={filename}
            className="w-full h-auto cursor-pointer block"
            onClick={handleImageClick}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                handleImageClick();
              }
            }}
            tabIndex={0}
            loading="lazy"
            onLoad={(e) => {
              const img = e.currentTarget;
              setImgSize({ w: img.naturalWidth, h: img.naturalHeight });
            }}
            draggable={false}
          />
          {/* eslint-enable jsx-a11y/no-noninteractive-element-interactions, jsx-a11y/no-noninteractive-tabindex */}

          {/* Face bounding boxes */}
          {imgSize &&
            photo.faces.map((face, i) => (
              <div
                key={i}
                className="absolute border-2 border-green-400 rounded-sm pointer-events-none"
                style={{
                  left: `${(face.left / imgSize.w) * 100}%`,
                  top: `${(face.top / imgSize.h) * 100}%`,
                  width: `${((face.right - face.left) / imgSize.w) * 100}%`,
                  height: `${((face.bottom - face.top) / imgSize.h) * 100}%`,
                }}
              />
            ))}
        </div>

        {/* Hover actions */}
        <div className="absolute bottom-0 left-0 right-0 p-2 bg-gradient-to-t from-black/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex gap-1">
          {onRemove && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onRemove();
              }}
              className="text-xs text-white bg-red-500/80 hover:bg-red-600 px-2 py-1 rounded transition-colors"
            >
              Remove
            </button>
          )}
          {onMove && otherClusters && (
            <div className="relative" ref={moveMenuRef}>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setShowMoveMenu(!showMoveMenu);
                }}
                className="text-xs text-white bg-blue-500/80 hover:bg-blue-600 px-2 py-1 rounded transition-colors"
              >
                Move
              </button>
              {showMoveMenu && (
                <div className="absolute bottom-full left-0 mb-1 bg-white rounded shadow-lg border max-h-48 overflow-y-auto min-w-40 z-20">
                  {otherClusters.map((c) => (
                    <button
                      key={c.id}
                      onClick={() => {
                        onMove(c.id);
                        setShowMoveMenu(false);
                      }}
                      className="block w-full text-left px-3 py-1.5 text-sm hover:bg-blue-50 text-gray-700 transition-colors"
                    >
                      {c.name} ({c.photo_count})
                    </button>
                  ))}
                  {nextClusterId !== undefined && (
                    <button
                      onClick={() => {
                        onMove(nextClusterId);
                        setShowMoveMenu(false);
                      }}
                      className="block w-full text-left px-3 py-1.5 text-sm hover:bg-green-50 text-green-700 border-t transition-colors"
                    >
                      + New cluster
                    </button>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Filename */}
        <div className="p-1.5">
          <div
            className="text-xs text-gray-500 truncate"
            title={filename}
          >
            {filename}
          </div>
        </div>
      </div>

      {/* Standalone full-size viewer (used when onViewFull is NOT provided) */}
      {showFull && !onViewFull && (
        <PhotoViewer
          photo={photo}
          onClose={() => setShowFull(false)}
        />
      )}
    </>
  );
}
