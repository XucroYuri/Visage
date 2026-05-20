import { useEffect, useRef, useState } from "react";
import type { PhotoInfo } from "../api";
import { getImageUrl } from "../api";

interface PhotoViewerProps {
  photo: PhotoInfo;
  onClose: () => void;
  /** Previous photo for arrow-key navigation (optional) */
  onPrev?: () => void;
  /** Next photo for arrow-key navigation (optional) */
  onNext?: () => void;
  /** Full list of photos for navigation */
  navigationPhotos?: PhotoInfo[];
  /** Index of current photo in navigationPhotos */
  currentIndex?: number;
  /** Navigate to a specific photo by index */
  onNavigate?: (index: number) => void;
}

export function PhotoViewer({
  photo,
  onClose,
  onPrev,
  onNext,
}: PhotoViewerProps) {
  const [fullSize, setFullSize] = useState<{ w: number; h: number } | null>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  const filename = photo.path.split("/").pop() || photo.path;

  // Focus trap: focus the overlay so onKeyDown works
  useEffect(() => {
    overlayRef.current?.focus();
  }, []);

  return (
    /* eslint-disable jsx-a11y/no-noninteractive-element-interactions, jsx-a11y/no-noninteractive-tabindex */
    <div
      ref={overlayRef}
      className="fixed inset-0 bg-black/85 z-50 flex items-center justify-center cursor-pointer animate-fade-in"
      onClick={onClose}
      onKeyDown={(e) => {
        if (e.key === "Escape") onClose();
      }}
      tabIndex={0}
      role="dialog"
      aria-label={`Full-size view of ${filename}`}
    >
      {/* eslint-enable jsx-a11y/no-noninteractive-element-interactions, jsx-a11y/no-noninteractive-tabindex */}
      {/* Close button */}
      <button
        onClick={onClose}
        className="absolute top-4 right-4 text-white/60 hover:text-white text-2xl z-10 transition-colors"
        aria-label="Close"
      >
        &times;
      </button>

      {/* Prev arrow */}
      {onPrev && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            onPrev();
          }}
          className="absolute left-4 top-1/2 -translate-y-1/2 text-white/60 hover:text-white text-4xl px-2 transition-colors z-10"
          aria-label="Previous photo"
        >
          &#8249;
        </button>
      )}

      {/* Image */}
      <div
        className="relative max-h-[90vh] max-w-[90vw] animate-scale-in"
        role="presentation"
        onClick={(e) => e.stopPropagation()}
      >
        <img
          src={getImageUrl(photo.path, "full")}
          alt={filename}
          className="max-h-[90vh] max-w-[90vw] object-contain rounded shadow-2xl"
          onLoad={(e) => {
            const img = e.currentTarget;
            setFullSize({ w: img.naturalWidth, h: img.naturalHeight });
          }}
        />
        {/* Face bounding boxes */}
        {fullSize &&
          photo.faces.map((face, i) => (
            <div
              key={i}
              className="absolute border-2 border-green-400 rounded-sm pointer-events-none"
              style={{
                left: `${(face.left / fullSize.w) * 100}%`,
                top: `${(face.top / fullSize.h) * 100}%`,
                width: `${((face.right - face.left) / fullSize.w) * 100}%`,
                height: `${((face.bottom - face.top) / fullSize.h) * 100}%`,
              }}
            />
          ))}
      </div>

      {/* Next arrow */}
      {onNext && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            onNext();
          }}
          className="absolute right-4 top-1/2 -translate-y-1/2 text-white/60 hover:text-white text-4xl px-2 transition-colors z-10"
          aria-label="Next photo"
        >
          &#8250;
        </button>
      )}

      {/* Filename */}
      <div className="absolute bottom-4 left-4 text-white/80 text-sm bg-black/50 px-3 py-1.5 rounded">
        {filename}
      </div>
    </div>
  );
}
