import type { WorkspaceState } from "../api";

interface HeaderProps {
  stats: WorkspaceState["stats"] | null;
  canUndo: boolean;
  saving: boolean;
  mutating: boolean;
  saveResult: string | null;
  onUndo: () => void;
  onOpenSave: () => void;
  onOpenSettings: () => void;
}

export function Header({
  stats,
  canUndo,
  saving,
  mutating,
  saveResult,
  onUndo,
  onOpenSave,
  onOpenSettings,
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
          onClick={onOpenSettings}
          className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
          title="Settings"
          aria-label="Settings"
        >
          <svg
            className="w-5 h-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
            />
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
            />
          </svg>
        </button>
        <button
          onClick={onUndo}
          disabled={!canUndo || mutating}
          className="px-3 py-1.5 text-sm border rounded disabled:opacity-30 hover:bg-gray-50 transition-colors"
          title="Undo (Ctrl+Z)"
        >
          &#8630; Undo
        </button>
        <button
          onClick={onOpenSave}
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
