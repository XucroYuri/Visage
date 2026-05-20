import { describe, it, expect, beforeEach } from "vitest";
import { useUIStore } from "../../store/ui";

function resetStore() {
  useUIStore.setState({
    mergeMode: false,
    selectedForMerge: new Set(),
    editingName: null,
    editValue: "",
    saving: false,
  });
}

beforeEach(resetStore);

describe("useUIStore", () => {
  describe("initial state", () => {
    it("has mergeMode disabled", () => {
      expect(useUIStore.getState().mergeMode).toBe(false);
    });

    it("has empty merge selection", () => {
      expect(useUIStore.getState().selectedForMerge.size).toBe(0);
    });

    it("has no editing state", () => {
      expect(useUIStore.getState().editingName).toBeNull();
      expect(useUIStore.getState().editValue).toBe("");
    });

    it("has saving disabled", () => {
      expect(useUIStore.getState().saving).toBe(false);
    });
  });

  describe("merge mode", () => {
    it("enables merge mode", () => {
      useUIStore.getState().setMergeMode(true);
      expect(useUIStore.getState().mergeMode).toBe(true);
    });

    it("disables merge mode", () => {
      useUIStore.getState().setMergeMode(true);
      useUIStore.getState().setMergeMode(false);
      expect(useUIStore.getState().mergeMode).toBe(false);
    });

    it("sets selected clusters for merge", () => {
      const sel = new Set([1, 2, 3]);
      useUIStore.getState().setSelectedForMerge(sel);
      expect(useUIStore.getState().selectedForMerge).toEqual(sel);
    });
  });

  describe("inline editing", () => {
    it("sets editing name and value", () => {
      useUIStore.getState().setEditingName(5);
      useUIStore.getState().setEditValue("NewName");
      expect(useUIStore.getState().editingName).toBe(5);
      expect(useUIStore.getState().editValue).toBe("NewName");
    });

    it("clears editing name", () => {
      useUIStore.getState().setEditingName(5);
      useUIStore.getState().setEditingName(null);
      expect(useUIStore.getState().editingName).toBeNull();
    });
  });

  describe("saving", () => {
    it("toggles saving state", () => {
      useUIStore.getState().setSaving(true);
      expect(useUIStore.getState().saving).toBe(true);
      useUIStore.getState().setSaving(false);
      expect(useUIStore.getState().saving).toBe(false);
    });
  });
});
