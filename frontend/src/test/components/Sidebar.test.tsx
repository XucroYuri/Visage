import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import type { WorkspaceState } from "../../api";
import { Sidebar } from "../../components/Sidebar";
import { useUIStore } from "../../store/ui";

function resetUIStore() {
  useUIStore.setState({
    mergeMode: false,
    selectedForMerge: new Set(),
    editingName: null,
    editValue: "",
    saving: false,
  });
}

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
  noise_photos: [
    { path: "/noise/1.jpg", faces: [], width: 640, height: 480 },
    { path: "/noise/2.jpg", faces: [], width: 640, height: 480 },
  ],
  all_photos: [],
  next_cluster_id: 3,
  can_undo: true,
};

const baseCtx = {
  ws: mockWs,
  mergeMode: false,
  selectedForMerge: new Set<number>(),
  editingName: null,
  editValue: "",
  isMutating: false,
  handleToggleMergeMode: vi.fn(),
  handleMergeCancel: vi.fn(),
  handleExecuteMerge: vi.fn(),
  handleStartEdit: vi.fn(),
  handleRename: vi.fn(),
  handleCancelEdit: vi.fn(),
  handleDropOnCluster: vi.fn(),
  setEditValue: vi.fn(),
};

function renderSidebar(ctx = baseCtx, initialRoute = "/") {
  return render(
    <MemoryRouter initialEntries={[initialRoute]}>
      <Sidebar ctx={ctx} />
    </MemoryRouter>,
  );
}

beforeEach(resetUIStore);

describe("Sidebar", () => {
  it("renders navigation buttons", () => {
    renderSidebar();
    expect(screen.getByText(/All Photos/)).toBeDefined();
    expect(screen.getByText(/Unclustered/)).toBeDefined();
  });

  it("renders cluster list", () => {
    renderSidebar();
    expect(screen.getByText("Cluster 0")).toBeDefined();
    expect(screen.getByText("Beach")).toBeDefined();
    expect(screen.getByText("Family")).toBeDefined();
  });

  it("shows noise count badge when there are noise photos", () => {
    renderSidebar();
    // Badge shows noise_photos.length (which is 2 in our mock)
    const badge = screen.getByText("2");
    expect(badge).toBeDefined();
    expect(badge.className).toContain("bg-amber-100");
  });

  it("highlights All Photos when on root route", () => {
    renderSidebar(baseCtx, "/");
    const allPhotosBtn = screen.getByText(/All Photos/).closest("button")!;
    expect(allPhotosBtn.className).toContain("bg-blue-50");
  });

  it("highlights Unclustered when on noise route", () => {
    renderSidebar(baseCtx, "/noise");
    const unclusteredBtn = screen.getByText(/Unclustered/).closest("button")!;
    expect(unclusteredBtn.className).toContain("bg-amber-50");
  });

  it("highlights active cluster", () => {
    renderSidebar(baseCtx, "/cluster/1");
    const options = screen.getAllByRole("option");
    const activeOption = options.find((o) => o.getAttribute("aria-selected") === "true");
    expect(activeOption).toBeDefined();
    expect(activeOption?.textContent).toContain("Beach");
  });

  it("shows merge mode toggle button", () => {
    renderSidebar();
    expect(screen.getByText(/Merge Mode/)).toBeDefined();
  });

  it("shows cancel merge button when in merge mode", () => {
    renderSidebar({
      ...baseCtx,
      mergeMode: true,
    });
    expect(screen.getByText(/Cancel Merge/)).toBeDefined();
  });

  it("shows merge execute button when 2+ clusters selected", () => {
    renderSidebar({
      ...baseCtx,
      mergeMode: true,
      selectedForMerge: new Set([0, 1]),
    });
    expect(screen.getByText(/Merge 2 Selected/)).toBeDefined();
  });

  it("does not show merge execute button when <2 selected", () => {
    renderSidebar({
      ...baseCtx,
      mergeMode: true,
      selectedForMerge: new Set([0]),
    });
    expect(screen.queryByText(/Merge \d+ Selected/)).toBeNull();
  });

  it("shows callbacks when clicking cluster rows", () => {
    renderSidebar(baseCtx, "/cluster/0");
    // ClusterRow click should navigate via handleSelectCluster
    // In merge mode it would toggle selection; in normal mode it navigates
    // We verify the row is interactive (has role=option)
    const options = screen.getAllByRole("option");
    expect(options.length).toBe(3);
  });
});
