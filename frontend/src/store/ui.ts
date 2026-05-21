import { create } from "zustand";

interface UIStore {
  mergeMode: boolean;
  setMergeMode: (v: boolean) => void;
  selectedForMerge: Set<number>;
  setSelectedForMerge: (s: Set<number>) => void;
  editingName: number | null;
  setEditingName: (id: number | null) => void;
  editValue: string;
  setEditValue: (v: string) => void;
  saving: boolean;
  setSaving: (v: boolean) => void;

  // Batch operations (M4)
  selectedPhotoPaths: Set<string>;
  setSelectedPhotoPaths: (s: Set<string>) => void;
  batchMode: boolean;
  setBatchMode: (v: boolean) => void;

  // Responsive layout (M4)
  sidebarCollapsed: boolean;
  setSidebarCollapsed: (v: boolean) => void;
  toggleSidebar: () => void;

  // Dark mode (M4)
  darkMode: boolean | "system";
  setDarkMode: (v: boolean | "system") => void;
}

export const useUIStore = create<UIStore>((set, get) => ({
  // ── Existing ──
  mergeMode: false,
  setMergeMode: (mergeMode) => set({ mergeMode }),
  selectedForMerge: new Set(),
  setSelectedForMerge: (selectedForMerge) => set({ selectedForMerge }),
  editingName: null,
  setEditingName: (editingName) => set({ editingName }),
  editValue: "",
  setEditValue: (editValue) => set({ editValue }),
  saving: false,
  setSaving: (saving) => set({ saving }),

  // ── Batch operations ──
  selectedPhotoPaths: new Set(),
  setSelectedPhotoPaths: (selectedPhotoPaths) => set({ selectedPhotoPaths }),
  batchMode: false,
  setBatchMode: (batchMode) => set({ batchMode }),

  // ── Responsive layout ──
  sidebarCollapsed: false,
  setSidebarCollapsed: (sidebarCollapsed) => set({ sidebarCollapsed }),
  toggleSidebar: () => set({ sidebarCollapsed: !get().sidebarCollapsed }),

  // ── Dark mode ──
  darkMode: "system",
  setDarkMode: (darkMode) => set({ darkMode }),
}));
