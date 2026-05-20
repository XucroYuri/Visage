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
}

export const useUIStore = create<UIStore>((set) => ({
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
}));
