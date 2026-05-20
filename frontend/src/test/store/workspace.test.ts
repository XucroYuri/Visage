import { describe, it, expect, beforeEach } from "vitest";
import { useWorkspaceStore } from "../../store/workspace";
import type { WorkspaceState } from "../../api";

function resetStore() {
  useWorkspaceStore.setState({ ws: null, loading: true, error: null });
}

beforeEach(resetStore);

const mockWs: WorkspaceState = {
  input_dir: "/photos",
  config: {
    copy_mode: true,
    folder_prefix: "Cluster",
    embedding_backend: "dlib",
  },
  stats: {
    total_images: 100,
    images_with_faces: 80,
    total_faces: 200,
    num_clusters: 5,
    num_noise_faces: 10,
  },
  clusters: [
    {
      id: 0,
      name: "Cluster 0",
      photos: [],
      photo_count: 0,
      thumbnail: null,
      confidence: 0.95,
    },
  ],
  noise_photos: [],
  all_photos: [],
  next_cluster_id: 1,
  can_undo: false,
};

describe("useWorkspaceStore", () => {
  it("starts with loading true and no data", () => {
    const s = useWorkspaceStore.getState();
    expect(s.ws).toBeNull();
    expect(s.loading).toBe(true);
    expect(s.error).toBeNull();
  });

  it("sets workspace and clears loading/error", () => {
    useWorkspaceStore.getState().setWs(mockWs);
    const s = useWorkspaceStore.getState();
    expect(s.ws).toEqual(mockWs);
    expect(s.loading).toBe(false);
    expect(s.error).toBeNull();
  });

  it("sets loading state", () => {
    useWorkspaceStore.getState().setLoading(true);
    expect(useWorkspaceStore.getState().loading).toBe(true);

    useWorkspaceStore.getState().setLoading(false);
    expect(useWorkspaceStore.getState().loading).toBe(false);
  });

  it("sets error state and clears loading", () => {
    useWorkspaceStore.getState().setError("Network error");
    const s = useWorkspaceStore.getState();
    expect(s.error).toBe("Network error");
    expect(s.loading).toBe(false);
  });

  it("clears error when setting workspace", () => {
    useWorkspaceStore.getState().setError("Previous error");
    useWorkspaceStore.getState().setWs(mockWs);
    expect(useWorkspaceStore.getState().error).toBeNull();
  });
});
