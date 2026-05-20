import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  ApiError,
  fetchWorkspace,
  getImageUrl,
  pipelineStatusUrl,
} from "../api";

// ── fetch mock ────────────────────────────────────────────

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

beforeEach(() => {
  mockFetch.mockReset();
});

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
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockWs),
    });
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
