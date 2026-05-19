import type { WorkspaceState } from "../api";

interface HeaderProps {
  stats: WorkspaceState["stats"] | null;
  canUndo: boolean;
  saving: boolean;
  mutating: boolean;
  saveResult: string | null;
  onUndo: () => void;
  onSave: () => void;
}

export function Header({
  stats,
  canUndo,
  saving,
  mutating,
  saveResult,
  onUndo,
  onSave,
}: HeaderProps) {
  return (
    <header className="flex items-center justify-between px-6 py-3 bg-white border-b border-gray-200 shadow-sm shrink-0">
      <div className="flex items-center gap-4">
        <h1 className="text-lg font-semibold text-gray-900">Visage Review</h1>
        {stats && (
          <span className="text-sm text-gray-400">
            {stats.num_clusters} clusters &middot;{" "}
            {stats.images_with_faces} images &middot;{" "}
            {stats.num_noise_faces} noise
          </span>
        )}
      </div>

      <div className="flex items-center gap-2">
        {saveResult && (
          <span className="text-sm text-green-600 mr-2 animate-pulse">
            {saveResult}
          </span>
        )}
        <button
          onClick={onUndo}
          disabled={!canUndo || mutating}
          className="px-3 py-1.5 text-sm border rounded disabled:opacity-30 hover:bg-gray-50 transition-colors"
          title="Undo (Ctrl+Z)"
        >
          &#8630; Undo
        </button>
        <button
          onClick={onSave}
          disabled={saving || mutating}
          className="px-4 py-1.5 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 transition-colors flex items-center gap-1.5"
        >
          {saving && (
            <span className="inline-block w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
          )}
          {saving ? "Saving..." : "Save to Disk"}
        </button>
      </div>
    </header>
  );
}
