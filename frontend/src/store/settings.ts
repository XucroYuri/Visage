import { create } from "zustand";
import { persist } from "zustand/middleware";

interface SettingsStore {
  outputDir: string;           // "" = use default (input_dir/visage_output)
  copyMode: boolean;           // true = copy, false = move
  folderPrefix: string;        // default "person_"
  includeUnclustered: boolean; // default false
  includeNoFaces: boolean;     // default false
  clusterSelectionMode: "all" | "selected";
  selectedClusterIds: Set<number>;

  setOutputDir: (dir: string) => void;
  setCopyMode: (v: boolean) => void;
  setFolderPrefix: (v: string) => void;
  setIncludeUnclustered: (v: boolean) => void;
  setIncludeNoFaces: (v: boolean) => void;
  setClusterSelectionMode: (v: "all" | "selected") => void;
  setSelectedClusterIds: (ids: Set<number>) => void;
  resetToDefaults: () => void;
  resetClusterSelection: () => void;
}

export const DEFAULTS = {
  outputDir: "",
  copyMode: true,
  folderPrefix: "person_",
  includeUnclustered: false,
  includeNoFaces: false,
};

const CLUSTER_SELECTION_DEFAULTS = {
  clusterSelectionMode: "all" as const,
  selectedClusterIds: new Set<number>(),
};

export const useSettingsStore = create<SettingsStore>()(
  persist(
    (set) => ({
      ...DEFAULTS,
      ...CLUSTER_SELECTION_DEFAULTS,
      setOutputDir: (outputDir) => set({ outputDir }),
      setCopyMode: (copyMode) => set({ copyMode }),
      setFolderPrefix: (folderPrefix) => set({ folderPrefix }),
      setIncludeUnclustered: (includeUnclustered) => set({ includeUnclustered }),
      setIncludeNoFaces: (includeNoFaces) => set({ includeNoFaces }),
      setClusterSelectionMode: (clusterSelectionMode) => set({ clusterSelectionMode }),
      setSelectedClusterIds: (selectedClusterIds) => set({ selectedClusterIds }),
      resetToDefaults: () => set({ ...DEFAULTS }),
      resetClusterSelection: () => set({ ...CLUSTER_SELECTION_DEFAULTS }),
    }),
    {
      name: "visage-settings",
      // Sets are not JSON-serializable, so exclude from persistence.
      // selectedClusterIds resets on each dialog open.
      partialize: (state) => ({
        outputDir: state.outputDir,
        copyMode: state.copyMode,
        folderPrefix: state.folderPrefix,
        includeUnclustered: state.includeUnclustered,
        includeNoFaces: state.includeNoFaces,
        clusterSelectionMode: state.clusterSelectionMode,
      }),
    },
  ),
);
