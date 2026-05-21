interface BatchActionBarProps {
  /** Number of selected items */
  count: number;
  /** Called to confirm selected faces (mark as correctly clustered) */
  onConfirm?: () => void;
  /** Called to reject/remove selected faces from current cluster */
  onReject?: () => void;
  /** Called to clear the selection */
  onClear: () => void;
  /** Whether mutations are in progress */
  disabled?: boolean;
}

export function BatchActionBar({
  count,
  onConfirm,
  onReject,
  onClear,
  disabled = false,
}: BatchActionBarProps) {
  if (count === 0) return null;

  return (
    <div className="batch-action-bar mb-4">
      <div className="flex items-center gap-3 bg-white dark:bg-slate-800 rounded-xl shadow-lg border border-gray-200 dark:border-slate-700 px-4 py-3">
        <span className="text-sm font-medium text-gray-700 dark:text-slate-300 whitespace-nowrap">
          {count} selected
        </span>

        <div className="w-px h-5 bg-gray-200 dark:bg-slate-600" />

        {onConfirm && (
          <button
            onClick={onConfirm}
            disabled={disabled}
            className="px-3 py-1.5 text-sm font-medium text-white bg-green-600 hover:bg-green-700 disabled:opacity-50 rounded-lg transition-colors"
          >
            Confirm
          </button>
        )}

        {onReject && (
          <button
            onClick={onReject}
            disabled={disabled}
            className="px-3 py-1.5 text-sm font-medium text-white bg-red-600 hover:bg-red-700 disabled:opacity-50 rounded-lg transition-colors"
          >
            Reject
          </button>
        )}

        <div className="w-px h-5 bg-gray-200 dark:bg-slate-600" />

        <button
          onClick={onClear}
          disabled={disabled}
          className="px-3 py-1.5 text-sm font-medium text-gray-600 dark:text-slate-400 hover:bg-gray-100 dark:hover:bg-slate-700 disabled:opacity-50 rounded-lg transition-colors"
        >
          Clear
        </button>
      </div>
    </div>
  );
}
