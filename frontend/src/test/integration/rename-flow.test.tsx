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
    { id: 1, name: "Beach Trip", photos: [], photo_count: 8, thumbnail: null, confidence: 0.90 },
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

describe("Rename flow", () => {
  it("starts editing a cluster name", () => {
    useUIStore.getState().setEditingName(1);
    useUIStore.getState().setEditValue("Beach Trip");

    expect(useUIStore.getState().editingName).toBe(1);
    expect(useUIStore.getState().editValue).toBe("Beach Trip");
  });

  it("updates edit value as user types", () => {
    useUIStore.getState().setEditingName(1);
    useUIStore.getState().setEditValue("Beach Vacation 2024");

    expect(useUIStore.getState().editValue).toBe("Beach Vacation 2024");
  });

  it("cancels editing without saving", () => {
    useUIStore.getState().setEditingName(1);
    useUIStore.getState().setEditValue("New Name");

    // Cancel
    useUIStore.getState().setEditingName(null);

    expect(useUIStore.getState().editingName).toBeNull();
    // editValue persists but editing is done
    expect(useUIStore.getState().editValue).toBe("New Name");
  });

  it("clears editing when rename completes", () => {
    useUIStore.getState().setEditingName(1);
    useUIStore.getState().setEditValue("Final Name");

    // Simulate rename submission (handleRename sets editingName to null)
    useUIStore.getState().setEditingName(null);

    expect(useUIStore.getState().editingName).toBeNull();
  });

  it("resolves old cluster name from workspace", () => {
    useUIStore.getState().setEditingName(1);
    useUIStore.getState().setEditValue("Beach Trip");

    const ws = useWorkspaceStore.getState().ws;
    const oldName = ws?.clusters.find((c) => c.id === 1)?.name;

    expect(oldName).toBe("Beach Trip");
  });

  it("handles blank edit value as cancel", () => {
    // When editValue is empty/whitespace and user submits, it's treated as cancel
    useUIStore.getState().setEditingName(1);
    useUIStore.getState().setEditValue("  ");

    // handleRename would check: if (!editValue.trim()) { setEditingName(null); return; }
    const editValue = useUIStore.getState().editValue;
    const isEmpty = !editValue.trim();
    expect(isEmpty).toBe(true);

    // Simulate cancel
    useUIStore.getState().setEditingName(null);
    expect(useUIStore.getState().editingName).toBeNull();
  });
});
