import { useEffect } from "react";

export interface KeyboardActions {
  /** Ctrl+Z / Cmd+Z */
  onUndo?: () => void;
  /** Ctrl+S / Cmd+S */
  onSave?: () => void;
  /** Escape key */
  onEscape?: () => void;
  /** Arrow left */
  onPrev?: () => void;
  /** Arrow right */
  onNext?: () => void;
  /** Ctrl+A / Cmd+A */
  onSelectAll?: () => void;
}

/**
 * Global keyboard shortcut hook.
 * Only the provided action callbacks are registered.
 * All handlers receive preventDefault() before firing.
 */
export function useKeyboard(actions: KeyboardActions) {
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;

      // Ctrl+Z / Cmd+Z — Undo
      if (mod && e.key === "z" && !e.shiftKey && actions.onUndo) {
        e.preventDefault();
        actions.onUndo();
        return;
      }

      // Ctrl+S / Cmd+S — Save
      if (mod && e.key === "s" && actions.onSave) {
        e.preventDefault();
        actions.onSave();
        return;
      }

      // Ctrl+A / Cmd+A — Select All
      if (mod && e.key === "a" && actions.onSelectAll) {
        e.preventDefault();
        actions.onSelectAll();
        return;
      }

      // Escape
      if (e.key === "Escape" && actions.onEscape) {
        e.preventDefault();
        actions.onEscape();
        return;
      }

      // Arrow Left
      if (e.key === "ArrowLeft" && actions.onPrev) {
        e.preventDefault();
        actions.onPrev();
        return;
      }

      // Arrow Right
      if (e.key === "ArrowRight" && actions.onNext) {
        e.preventDefault();
        actions.onNext();
        return;
      }
    };

    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [actions]);
}
