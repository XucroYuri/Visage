import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  ApiError,
  assignNoise,
  fetchWorkspace,
  getImageUrl,
  mergeClusters,
  moveFace,
  pipelineStatusUrl,
  removeFace,
  renameCluster,
  save,
  undo,
} from "../api";

// ── fetch mock ────────────────────────────────────────────

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

beforeEach(() => {
  mockFetch.mockReset();
});

function mockOk(data: unknown) {
  mockFetch.mockResolvedValueOnce({
    ok: true,
    json: () => Promise.resolve(data),
  });
}

// ── Tests ─────────────────────────────────────────────────

describe("ApiError", () => {
  it("stores statusCode", () => {
    const err = new ApiError("Not Found", 404);
    expect(err.message).toBe("Not Found");
    expect(err.statusCode).toBe(404);
    expect(err.name).toBe("ApiError");
  });
});

describe("getImageUrl", () => {
  it("encodes path for thumb", () => {
    const url = getImageUrl("/photos/img.jpg");
    expect(url).toContain("size=thumb");
    expect(url).toContain("path=");
  });

  it("supports full size", () => {
    const url = getImageUrl("/photos/img.jpg", "full");
    expect(url).toContain("size=full");
  });
});

describe("pipelineStatusUrl", () => {
  it("returns the SSE endpoint", () => {
    expect(pipelineStatusUrl()).toBe("/api/pipeline-status");
  });
});

describe("fetchWorkspace", () => {
  it("returns workspace data on success", async () => {
    const mockWs = { clusters: [], noise_photos: [], all_photos: [] };
    mockOk(mockWs);
    const result = await fetchWorkspace();
    expect(result).toEqual(mockWs);
  });

  it("throws ApiError on failure", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      text: () => Promise.resolve("Server Error"),
    });
    await expect(fetchWorkspace()).rejects.toThrow(ApiError);
  });
});

describe("mergeClusters", () => {
  it("sends POST with correct body", async () => {
    mockOk({ ok: true, workspace: {} });
    await mergeClusters(1, 5);
    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toBe("/api/clusters/merge");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ from_id: 1, to_id: 5 });
  });
});

describe("removeFace", () => {
  it("sends POST to cluster remove endpoint", async () => {
    mockOk({ ok: true, workspace: {} });
    await removeFace(3, "/photos/img.jpg");
    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toBe("/api/clusters/3/remove");
    expect(JSON.parse(init.body)).toEqual({ image_path: "/photos/img.jpg" });
  });
});

describe("moveFace", () => {
  it("sends POST with from_id and to_id", async () => {
    mockOk({ ok: true, workspace: {} });
    await moveFace("/photos/img.jpg", 2, 4);
    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toBe("/api/clusters/move");
    expect(JSON.parse(init.body)).toEqual({
      image_path: "/photos/img.jpg",
      from_id: 2,
      to_id: 4,
    });
  });
});

describe("assignNoise", () => {
  it("sends POST to assign endpoint", async () => {
    mockOk({ ok: true, workspace: {} });
    await assignNoise("/photos/img.jpg", 7);
    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toBe("/api/clusters/assign");
    expect(JSON.parse(init.body)).toEqual({
      image_path: "/photos/img.jpg",
      to_id: 7,
    });
  });
});

describe("renameCluster", () => {
  it("sends PUT with name", async () => {
    mockOk({ ok: true, workspace: {} });
    await renameCluster(3, "Vacation");
    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toBe("/api/clusters/3");
    expect(init.method).toBe("PUT");
    expect(JSON.parse(init.body)).toEqual({ name: "Vacation" });
  });
});

describe("undo", () => {
  it("sends POST to undo endpoint", async () => {
    mockOk({ ok: true, undo: {}, workspace: {} });
    await undo();
    expect(mockFetch.mock.calls[0][0]).toBe("/api/clusters/undo");
  });
});

describe("save", () => {
  it("sends POST with output_dir", async () => {
    mockOk({ ok: true, stats: {} });
    await save({ output_dir: "/output" });
    const [, init] = mockFetch.mock.calls[0];
    expect(JSON.parse(init.body)).toEqual({ output_dir: "/output" });
  });

  it("sends empty body by default", async () => {
    mockOk({ ok: true, stats: {} });
    await save();
    expect(JSON.parse(mockFetch.mock.calls[0][1].body)).toEqual({});
  });
});
