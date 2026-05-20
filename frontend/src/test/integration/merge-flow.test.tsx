import { describe, it, expect, beforeEach } from "vitest";
import { useUIStore } from "../../store/ui";
import { useWorkspaceStore } from "../../store/workspace";
import type { WorkspaceState } from "../../api";

const mockWs: WorkspaceState = {
  input_dir: "/photos",
  config: { copy_mode: true, folder_prefix: "Cluster", embedding_backend: "dlib" },
  stats: {
    total_images: 50,
    images_with_faces: 40,
    total_faces: 100,
    num_clusters: 3,
    num_noise_faces: 5,
  },
  clusters: [
    { id: 0, name: "Cluster 0", photos: [], photo_count: 10, thumbnail: null, confidence: 0.95 },
    { id: 1, name: "Beach", photos: [], photo_count: 8, thumbnail: null, confidence: 0.90 },
    { id: 2, name: "Family", photos: [], photo_count: 12, thumbnail: null, confidence: 0.88 },
  ],
  noise_photos: [],
  all_photos: [],
  next_cluster_id: 3,
  can_undo: true,
};

function resetStores() {
  useUIStore.setState({
    mergeMode: false,
    selectedForMerge: new Set(),
    editingName: null,
    editValue: "",
    saving: false,
  });
  useWorkspaceStore.setState({ ws: mockWs, loading: false, error: null });
}

beforeEach(resetStores);

describe("Merge flow", () => {
  it("enters merge mode", () => {
    expect(useUIStore.getState().mergeMode).toBe(false);

    useUIStore.getState().setMergeMode(true);

    expect(useUIStore.getState().mergeMode).toBe(true);
  });

  it("selects clusters for merge", () => {
    useUIStore.getState().setMergeMode(true);

    // Simulate selecting cluster 0
    const prev = useUIStore.getState().selectedForMerge;
    const next = new Set(prev);
    next.add(0);
    useUIStore.getState().setSelectedForMerge(next);

    // Simulate selecting cluster 1
    const prev2 = useUIStore.getState().selectedForMerge;
    const next2 = new Set(prev2);
    next2.add(1);
    useUIStore.getState().setSelectedForMerge(next2);

    expect(useUIStore.getState().selectedForMerge.has(0)).toBe(true);
    expect(useUIStore.getState().selectedForMerge.has(1)).toBe(true);
    expect(useUIStore.getState().selectedForMerge.size).toBe(2);
  });

  it("deselects a cluster when toggled off", () => {
    useUIStore.getState().setMergeMode(true);

    // Select both
    useUIStore.getState().setSelectedForMerge(new Set([0, 1]));
    expect(useUIStore.getState().selectedForMerge.size).toBe(2);

    // Deselect cluster 1
    const prev = useUIStore.getState().selectedForMerge;
    const next = new Set(prev);
    next.delete(1);
    useUIStore.getState().setSelectedForMerge(next);

    expect(useUIStore.getState().selectedForMerge.has(0)).toBe(true);
    expect(useUIStore.getState().selectedForMerge.has(1)).toBe(false);
    expect(useUIStore.getState().selectedForMerge.size).toBe(1);
  });

  it("cancels merge mode and clears selection", () => {
    useUIStore.getState().setMergeMode(true);
    useUIStore.getState().setSelectedForMerge(new Set([0, 1]));

    useUIStore.getState().setMergeMode(false);
    useUIStore.getState().setSelectedForMerge(new Set());

    expect(useUIStore.getState().mergeMode).toBe(false);
    expect(useUIStore.getState().selectedForMerge.size).toBe(0);
  });

  it("clears selection after merge execute", () => {
    useUIStore.getState().setMergeMode(true);
    useUIStore.getState().setSelectedForMerge(new Set([0, 1]));

    // Simulate post-merge cleanup (what handleExecuteMerge does)
    useUIStore.getState().setSelectedForMerge(new Set());
    useUIStore.getState().setMergeMode(false);

    expect(useUIStore.getState().mergeMode).toBe(false);
    expect(useUIStore.getState().selectedForMerge.size).toBe(0);
  });

  it("workspace reflects cluster data after merge", () => {
    const ws = useWorkspaceStore.getState().ws;
    expect(ws?.clusters).toHaveLength(3);
    expect(ws?.clusters[0].name).toBe("Cluster 0");
    expect(ws?.clusters[1].name).toBe("Beach");
  });
});
