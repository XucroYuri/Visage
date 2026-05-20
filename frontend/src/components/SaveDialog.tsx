import { useCallback, useEffect } from "react";
import { useSettingsStore } from "../store/settings";

interface SaveDialogProps {
  open: boolean;
  onClose: () => void;
  onSave: () => void;
  saving: boolean;
  clusters: Array<{ id: number; name: string; photoCount: number }>;
  defaultOutputDir: string;
}

export function SaveDialog({
  open,
  onClose,
  onSave,
  saving,
  clusters,
  defaultOutputDir,
}: SaveDialogProps) {
  const outputDir = useSettingsStore((s) => s.outputDir);
  const setOutputDir = useSettingsStore((s) => s.setOutputDir);
  const folderPrefix = useSettingsStore((s) => s.folderPrefix);
  const setFolderPrefix = useSettingsStore((s) => s.setFolderPrefix);
  const copyMode = useSettingsStore((s) => s.copyMode);
  const setCopyMode = useSettingsStore((s) => s.setCopyMode);
  const includeUnclustered = useSettingsStore((s) => s.includeUnclustered);
  const setIncludeUnclustered = useSettingsStore((s) => s.setIncludeUnclustered);
  const includeNoFaces = useSettingsStore((s) => s.includeNoFaces);
  const setIncludeNoFaces = useSettingsStore((s) => s.setIncludeNoFaces);
  const clusterSelectionMode = useSettingsStore((s) => s.clusterSelectionMode);
  const setClusterSelectionMode = useSettingsStore((s) => s.setClusterSelectionMode);
  const selectedClusterIds = useSettingsStore((s) => s.selectedClusterIds);
  const setSelectedClusterIds = useSettingsStore((s) => s.setSelectedClusterIds);

  // ── Keyboard: Escape closes ───────────────────────────────
  useEffect(() => {
    if (!open) return;
    const handleKeyDown = (e: globalThis.KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [open, onClose]);

  // ── Reset cluster selection when dialog opens ─────────────
  useEffect(() => {
    if (open) {
      setClusterSelectionMode("all");
      setSelectedClusterIds(new Set());
    }
  }, [open, setClusterSelectionMode, setSelectedClusterIds]);

  // ── Prevent body scroll while dialog is open ──────────────
  useEffect(() => {
    if (!open) return;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = "";
    };
  }, [open]);

  // ── Cluster selection helpers ──────────────────────────────
  const handleSelectAll = useCallback(() => {
    setSelectedClusterIds(new Set(clusters.map((c) => c.id)));
  }, [clusters, setSelectedClusterIds]);

  const handleDeselectAll = useCallback(() => {
    setSelectedClusterIds(new Set());
  }, [setSelectedClusterIds]);

  const handleToggleCluster = useCallback(
    (id: number) => {
      const next = new Set(selectedClusterIds);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      setSelectedClusterIds(next);
    },
    [selectedClusterIds, setSelectedClusterIds],
  );

  // ── Form submission: Enter in any text field triggers save ─
  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      if (!saving) onSave();
    },
    [saving, onSave],
  );

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Save to Disk"
        className="animate-scale-in bg-white rounded-lg shadow-2xl max-w-md w-full max-h-[85vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* ── Header ────────────────────────────────────────── */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 shrink-0">
          <h2 className="text-lg font-semibold text-gray-900">
            Save to Disk
          </h2>
          <button
            onClick={onClose}
            className="ml-4 text-gray-400 hover:text-gray-600 transition-colors leading-none text-xl"
            aria-label="Close"
          >
            &times;
          </button>
        </div>

        {/* ── Form body (scrollable) ────────────────────────── */}
        <form
          id="save-form"
          onSubmit={handleSubmit}
          className="flex-1 overflow-y-auto px-6 py-5 space-y-5"
        >
          {/* ── 1. Output Directory ─────────────────────────── */}
          <fieldset className="space-y-1.5">
            <label
              htmlFor="save-output-dir"
              className="block text-sm font-medium text-gray-700"
            >
              Output Directory
            </label>
            <input
              id="save-output-dir"
              type="text"
              value={outputDir}
              onChange={(e) => setOutputDir(e.target.value)}
              placeholder={defaultOutputDir}
              className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg
                         focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-blue-400
                         placeholder:text-gray-400 transition-shadow"
            />
            <p className="text-xs text-gray-400">
              Leave empty to use default location
            </p>
          </fieldset>

          {/* ── 2. Folder Prefix ────────────────────────────── */}
          <fieldset className="space-y-1.5">
            <label
              htmlFor="save-folder-prefix"
              className="block text-sm font-medium text-gray-700"
            >
              Folder Prefix
            </label>
            <input
              id="save-folder-prefix"
              type="text"
              value={folderPrefix}
              onChange={(e) => setFolderPrefix(e.target.value)}
              className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg
                         focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-blue-400
                         transition-shadow"
            />
            <p className="text-xs text-gray-400">
              Cluster folders named as:{" "}
              <code className="px-1 py-0.5 bg-gray-100 rounded text-xs font-mono">
                {folderPrefix || "person_"}1
              </code>
            </p>
          </fieldset>

          {/* ── 3. Copy Mode ────────────────────────────────── */}
          <fieldset className="space-y-2">
            <legend className="text-sm font-medium text-gray-700 mb-1">
              Copy Mode
            </legend>

            <label
              className={`flex items-start gap-2.5 p-3 rounded-lg border cursor-pointer transition-colors ${
                copyMode
                  ? "border-blue-400 bg-blue-50/50"
                  : "border-gray-200 hover:bg-gray-50"
              }`}
            >
              <input
                type="radio"
                name="copyMode"
                checked={copyMode}
                onChange={() => setCopyMode(true)}
                className="mt-0.5 shrink-0 w-4 h-4 text-blue-600 border-gray-300 focus:ring-blue-500"
              />
              <div className="min-w-0">
                <span className="text-sm font-medium text-gray-800">
                  Copy files (safe)
                </span>
                <p className="text-xs text-gray-400 mt-0.5">
                  Leave originals untouched. Creates copies in the output
                  directory.
                </p>
              </div>
            </label>

            <label
              className={`flex items-start gap-2.5 p-3 rounded-lg border cursor-pointer transition-colors ${
                !copyMode
                  ? "border-blue-400 bg-blue-50/50"
                  : "border-gray-200 hover:bg-gray-50"
              }`}
            >
              <input
                type="radio"
                name="copyMode"
                checked={!copyMode}
                onChange={() => setCopyMode(false)}
                className="mt-0.5 shrink-0 w-4 h-4 text-blue-600 border-gray-300 focus:ring-blue-500"
              />
              <div className="min-w-0">
                <span className="text-sm font-medium text-gray-800">
                  Move files
                </span>
                <p className="text-xs text-gray-400 mt-0.5">
                  Files are moved from their original location. Frees up space
                  but cannot undo.
                </p>
              </div>
            </label>
          </fieldset>

          {/* ── 4. Include Options ──────────────────────────── */}
          <fieldset className="space-y-2.5">
            <legend className="text-sm font-medium text-gray-700 mb-1">
              Include
            </legend>

            <label className="flex items-center gap-2.5 cursor-pointer">
              <input
                type="checkbox"
                checked={includeUnclustered}
                onChange={(e) => setIncludeUnclustered(e.target.checked)}
                className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <span className="text-sm text-gray-700">
                Include unclustered faces
              </span>
            </label>

            <label className="flex items-center gap-2.5 cursor-pointer">
              <input
                type="checkbox"
                checked={includeNoFaces}
                onChange={(e) => setIncludeNoFaces(e.target.checked)}
                className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <span className="text-sm text-gray-700">
                Include images without faces
              </span>
            </label>
          </fieldset>

          {/* ── 5. Save Summary ────────────────────────────── */}
          <fieldset className="space-y-1.5">
            <legend className="text-sm font-medium text-gray-700">
              Save Summary
            </legend>
            <div className="bg-blue-50 border border-blue-100 rounded-lg p-3 text-xs text-blue-800 space-y-1.5">
              <div className="flex justify-between">
                <span>Clusters to save</span>
                <span className="font-semibold tabular-nums">
                  {clusterSelectionMode === "selected"
                    ? selectedClusterIds.size
                    : clusters.length}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Copy mode</span>
                <span className="font-semibold">
                  {copyMode ? "Copy (safe)" : "Move"}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Output folder</span>
                <span className="font-semibold truncate max-w-[200px] text-right" title={outputDir || defaultOutputDir}>
                  {outputDir || defaultOutputDir}
                </span>
              </div>
            </div>
          </fieldset>

          {/* ── 6. Cluster Selection ────────────────────────── */}
          <fieldset className="space-y-3">
            <legend className="text-sm font-medium text-gray-700 mb-1">
              Cluster Selection
            </legend>

            {/* All clusters radio */}
            <label
              className={`flex items-start gap-2.5 p-3 rounded-lg border cursor-pointer transition-colors ${
                clusterSelectionMode === "all"
                  ? "border-blue-400 bg-blue-50/50"
                  : "border-gray-200 hover:bg-gray-50"
              }`}
            >
              <input
                type="radio"
                name="clusterSelection"
                checked={clusterSelectionMode === "all"}
                onChange={() => setClusterSelectionMode("all")}
                className="mt-0.5 shrink-0 w-4 h-4 text-blue-600 border-gray-300 focus:ring-blue-500"
              />
              <span className="text-sm font-medium text-gray-800">
                All clusters ({clusters.length})
              </span>
            </label>

            {/* Selected only radio */}
            <label
              className={`flex items-start gap-2.5 p-3 rounded-lg border cursor-pointer transition-colors ${
                clusterSelectionMode === "selected"
                  ? "border-blue-400 bg-blue-50/50"
                  : "border-gray-200 hover:bg-gray-50"
              }`}
            >
              <input
                type="radio"
                name="clusterSelection"
                checked={clusterSelectionMode === "selected"}
                onChange={() => setClusterSelectionMode("selected")}
                className="mt-0.5 shrink-0 w-4 h-4 text-blue-600 border-gray-300 focus:ring-blue-500"
              />
              <span className="text-sm font-medium text-gray-800">
                Selected clusters only
              </span>
            </label>

            {/* Selected cluster checkbox list */}
            {clusterSelectionMode === "selected" && (
              <div className="border border-gray-200 rounded-lg overflow-hidden">
                {/* Toggle bar */}
                <div className="flex items-center gap-2 px-3 py-2 bg-gray-50 border-b border-gray-100">
                  <button
                    type="button"
                    onClick={handleSelectAll}
                    className="text-xs text-blue-600 hover:text-blue-700 font-medium transition-colors"
                  >
                    Select All
                  </button>
                  <span className="text-gray-300 select-none">|</span>
                  <button
                    type="button"
                    onClick={handleDeselectAll}
                    className="text-xs text-gray-400 hover:text-gray-600 font-medium transition-colors"
                  >
                    Deselect All
                  </button>
                  <span className="ml-auto text-xs text-gray-400">
                    {selectedClusterIds.size} of {clusters.length}
                  </span>
                </div>

                {/* Scrollable cluster list */}
                <div className="max-h-48 overflow-y-auto divide-y divide-gray-100">
                  {clusters.map((c) => (
                    <label
                      key={c.id}
                      className="flex items-center gap-2.5 px-3 py-2 cursor-pointer hover:bg-gray-50 transition-colors"
                    >
                      <input
                        type="checkbox"
                        checked={selectedClusterIds.has(c.id)}
                        onChange={() => handleToggleCluster(c.id)}
                        className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500 shrink-0"
                      />
                      <span className="text-sm text-gray-700 truncate">
                        {c.name}
                      </span>
                      <span className="ml-auto text-xs text-gray-400 shrink-0">
                        {c.photoCount}
                      </span>
                    </label>
                  ))}
                </div>
              </div>
            )}
          </fieldset>
        </form>

        {/* ── Footer ────────────────────────────────────────── */}
        <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-gray-200 shrink-0">
          <span className="mr-auto text-xs text-gray-400">
            <kbd className="px-1 py-0.5 bg-gray-100 rounded text-[10px] font-mono">Enter</kbd> to save
          </span>
          <button
            type="button"
            onClick={onClose}
            disabled={saving}
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-lg
                       hover:bg-gray-200 disabled:opacity-40 transition-colors"
          >
            Cancel
          </button>
          <button
            type="submit"
            form="save-form"
            disabled={
              saving ||
              (clusterSelectionMode === "selected" &&
                selectedClusterIds.size === 0)
            }
            className="px-5 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg
                       hover:bg-blue-700 disabled:opacity-50 transition-colors
                       flex items-center gap-1.5 min-w-[100px] justify-center"
          >
            {saving && (
              <span className="inline-block w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            )}
            {saving ? "Saving..." : "Save"}
          </button>
        </div>
      </div>
    </div>
  );
}
