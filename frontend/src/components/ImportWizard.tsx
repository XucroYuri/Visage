import { useState, useCallback } from "react";

interface ImportWizardProps {
  onImport: (inputDir: string) => void;
  onClose: () => void;
}

export function ImportWizard({ onImport, onClose }: ImportWizardProps) {
  const [inputDir, setInputDir] = useState("");
  const [error, setError] = useState<string | null>(null);

  const handleImport = useCallback(() => {
    const dir = inputDir.trim();
    if (!dir) {
      setError("Please enter a directory path");
      return;
    }
    setError(null);
    onImport(dir);
  }, [inputDir, onImport]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter") handleImport();
      if (e.key === "Escape") onClose();
    },
    [handleImport, onClose],
  );

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="bg-white dark:bg-slate-800 rounded-xl shadow-xl w-full max-w-md mx-4 overflow-hidden">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-200 dark:border-slate-700">
          <h2 className="text-lg font-semibold" style={{ color: "var(--color-text-primary)" }}>
            Import Photos
          </h2>
          <p className="text-sm mt-1" style={{ color: "var(--color-text-secondary)" }}>
            Enter the path to your photo directory
          </p>
        </div>

        {/* Content */}
        <div className="px-6 py-4 space-y-4">
          <div>
            <label
              htmlFor="import-dir"
              className="block text-sm font-medium mb-1.5"
              style={{ color: "var(--color-text-primary)" }}
            >
              Photo Directory
            </label>
            <input
              id="import-dir"
              type="text"
              value={inputDir}
              onChange={(e) => {
                setInputDir(e.target.value);
                setError(null);
              }}
              onKeyDown={handleKeyDown}
              placeholder="/path/to/photos"
              className="w-full px-3 py-2 border border-gray-200 dark:border-slate-600 rounded-lg text-sm bg-transparent outline-none focus:ring-2 focus:ring-blue-400"
              style={{ color: "var(--color-text-primary)" }}
              autoFocus
            />
            {error && (
              <p className="text-red-500 text-xs mt-1">{error}</p>
            )}
          </div>

          {/* Drag-drop zone */}
          <div
            className="border-2 border-dashed border-gray-300 dark:border-slate-600 rounded-lg p-6 text-center"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              const file = e.dataTransfer.files[0];
              if (file) {
                // In browser, we can only get the file name, not full path
                // The user needs to type the path manually
                setError("Browser security prevents reading full paths. Please type the directory path above.");
              }
            }}
          >
            <p className="text-sm" style={{ color: "var(--color-text-secondary)" }}>
              Type the directory path above to import photos
            </p>
            <p className="text-xs mt-1 text-gray-400 dark:text-slate-500">
              Supports JPG, PNG, HEIC, and TIFF files
            </p>
          </div>
        </div>

        {/* Actions */}
        <div className="px-6 py-3 border-t border-gray-200 dark:border-slate-700 flex justify-end gap-2">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm border border-gray-200 dark:border-slate-600 rounded-lg hover:bg-gray-50 dark:hover:bg-slate-700 transition-colors"
            style={{ color: "var(--color-text-primary)" }}
          >
            Cancel
          </button>
          <button
            onClick={handleImport}
            disabled={!inputDir.trim()}
            className="px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-40 transition-colors font-medium"
          >
            Import
          </button>
        </div>
      </div>
    </div>
  );
}
