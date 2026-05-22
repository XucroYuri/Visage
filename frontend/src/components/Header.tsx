import type { WorkspaceState } from "../api";
import { DarkModeToggle } from "./DarkModeToggle";

interface HeaderProps {
  stats: WorkspaceState["stats"] | null;
  canUndo: boolean;
  saving: boolean;
  mutating: boolean;
  saveResult: string | null;
  onUndo: () => void;
  onOpenSave: () => void;
  onOpenSettings: () => void;
  onOpenImport?: () => void;
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
  onOpenImport,
}: HeaderProps) {
  return (
    <header className="flex items-center justify-between px-4 sm:px-6 py-3 bg-white dark:bg-slate-900 border-b border-gray-200 dark:border-slate-700 shadow-sm shrink-0 transition-colors">
      <div className="flex items-center gap-4 min-w-0">
        <h1 className="text-lg font-semibold text-gray-900 dark:text-slate-100 whitespace-nowrap">
          Visage Review
        </h1>
        {stats && (
          <div className="hidden sm:flex items-center gap-1.5 flex-wrap">
            <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-400 rounded-full text-xs font-medium">
              <span className="w-2 h-2 rounded-full bg-indigo-400" />
              {stats.num_clusters} clusters
            </span>
            <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-green-50 dark:bg-green-900/30 text-green-700 dark:text-green-400 rounded-full text-xs font-medium">
              <span className="w-2 h-2 rounded-full bg-green-400" />
              {stats.images_with_faces} images
            </span>
            <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-amber-50 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400 rounded-full text-xs font-medium">
              <span className="w-2 h-2 rounded-full bg-amber-400" />
              {stats.num_noise_faces} noise
            </span>
          </div>
        )}
      </div>

      <div className="flex items-center gap-2 shrink-0">
        {saveResult && (
          <span className="text-sm text-green-600 dark:text-green-400 mr-2 animate-pulse hidden sm:inline">
            {saveResult}
          </span>
        )}

        {onOpenImport && (
          <button
            onClick={onOpenImport}
            className="p-2 text-gray-400 hover:text-gray-600 dark:text-slate-500 dark:hover:text-slate-300 hover:bg-gray-100 dark:hover:bg-slate-800 rounded-lg transition-colors"
            title="Import Photos"
            aria-label="Import Photos"
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
                d="M12 4v16m8-8H4"
              />
            </svg>
          </button>
        )}

        <DarkModeToggle />

        <button
          onClick={onOpenSettings}
          className="p-2 text-gray-400 hover:text-gray-600 dark:text-slate-500 dark:hover:text-slate-300 hover:bg-gray-100 dark:hover:bg-slate-800 rounded-lg transition-colors"
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
          className="px-3 py-1.5 text-sm border border-gray-200 dark:border-slate-600 rounded disabled:opacity-30 hover:bg-gray-50 dark:hover:bg-slate-800 text-gray-700 dark:text-slate-300 transition-colors"
          title="Undo (Ctrl+Z)"
        >
          <kbd className="hidden sm:inline-flex items-center px-1 py-0.5 mr-1 text-[10px] font-mono bg-gray-200/70 dark:bg-slate-700 rounded">
            Ctrl+Z
          </kbd>
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
          <kbd className="hidden sm:inline-flex items-center px-1 py-0.5 mr-1 text-[10px] font-mono bg-blue-500/40 rounded">
            Ctrl+S
          </kbd>
          {saving ? "Saving..." : "Save to Disk"}
        </button>
      </div>
    </header>
  );
}
