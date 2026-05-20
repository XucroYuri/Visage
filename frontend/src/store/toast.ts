import { create } from "zustand";

export type ToastType = "success" | "error" | "info";

export interface ToastItem {
  id: number;
  type: ToastType;
  text: string;
}

interface ToastStore {
  toasts: ToastItem[];
  addToast: (toast: { type: ToastType; text: string }) => void;
  dismissToast: (id: number) => void;
}

let toastIdCounter = 0;

export const useToastStore = create<ToastStore>((set) => ({
  toasts: [],
  addToast: (toast) => {
    const id = ++toastIdCounter;
    set((s) => ({ toasts: [...s.toasts, { ...toast, id }] }));
  },
  dismissToast: (id) => {
    set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) }));
  },
}));

/** Convenience hook: add a typed toast */
export function useToast() {
  return useToastStore((s) => s.addToast);
}
