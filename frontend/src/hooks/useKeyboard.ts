import { useEffect } from "react";

export interface KeyboardActions {
  /** Ctrl+Z / Cmd+Z */
  onUndo?: () => void;
  /** Ctrl+S / Cmd+S */
  onSave?: () => void;
  /** Escape key */
  onEscape?: () => void;
  /** Arrow left / J */
  onPrev?: () => void;
  /** Arrow right / K */
  onNext?: () => void;
  /** Ctrl+A / Cmd+A */
  onSelectAll?: () => void;
  /** Space — view/preview */
  onView?: () => void;
  /** Ctrl+F / Cmd+F or / — focus search */
  onSearch?: () => void;
  /** G then A — go to all photos */
  onGoAll?: () => void;
  /** G then N — go to noise */
  onGoNoise?: () => void;
}

/**
 * Global keyboard shortcut hook.
 * Only the provided action callbacks are registered.
 * All handlers receive preventDefault() before firing.
 *
 * Navigation:
 *   J / ArrowDown  — next item
 *   K / ArrowUp    — previous item
 *   Space          — view/preview
 *   / or Ctrl+F    — focus search
 *
 * Actions:
 *   Ctrl+Z — undo
 *   Ctrl+S — save
 *   Ctrl+A — select all
 *   Escape — dismiss/close
 */
export function useKeyboard(actions: KeyboardActions) {
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      const target = e.target as HTMLElement;
      const inInput = target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable;

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

      // Don't handle navigation keys when typing in inputs
      if (inInput) return;

      // J / ArrowDown — Next
      if ((e.key === "j" || e.key === "ArrowDown") && actions.onNext) {
        e.preventDefault();
        actions.onNext();
        return;
      }

      // K / ArrowUp — Previous
      if ((e.key === "k" || e.key === "ArrowUp") && actions.onPrev) {
        e.preventDefault();
        actions.onPrev();
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

      // Space — View/Preview
      if (e.key === " " && actions.onView) {
        e.preventDefault();
        actions.onView();
        return;
      }

      // Ctrl+F / Cmd+F or / — Search
      if (actions.onSearch && ((mod && e.key === "f") || e.key === "/")) {
        e.preventDefault();
        actions.onSearch();
        return;
      }
    };

    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [actions]);
}
