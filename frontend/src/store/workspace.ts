import { create } from "zustand";
import { useMutation, useMutationState } from "@tanstack/react-query";
import type { ClusterInfo, SaveSettings, WorkspaceState } from "../api";
import {
  assignNoise,
  mergeClusters,
  moveFace,
  removeFace,
  renameCluster,
  save,
  undo,
} from "../api";
import { useToastStore } from "./toast";

// ── Zustand workspace store ────────────────────────────────────

interface WorkspaceStore {
  ws: WorkspaceState | null;
  loading: boolean;
  error: string | null;
  setWs: (ws: WorkspaceState) => void;
  setLoading: (v: boolean) => void;
  setError: (err: string | null) => void;
}

export const useWorkspaceStore = create<WorkspaceStore>((set) => ({
  ws: null,
  loading: true,
  error: null,
  setWs: (ws) => set({ ws, loading: false, error: null }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error, loading: false }),
}));

// ── Helper: update store after mutation ────────────────────────

function updateStore(ws: WorkspaceState) {
  useWorkspaceStore.getState().setWs(ws);
}

// ── TanStack Query mutation hooks ──────────────────────────────

export function useMergeMutation() {
  const addToast = useToastStore((s) => s.addToast);

  return useMutation({
    mutationFn: async ({
      fromIds,
      toId,
    }: {
      fromIds: number[];
      toId: number;
    }) => {
      for (const fromId of fromIds) {
        const res = await mergeClusters(fromId, toId);
        updateStore(res.workspace);
      }
    },
    onSuccess: (_data, vars) => {
      addToast({
        type: "success",
        text: `Merged ${vars.fromIds.length} cluster(s) into cluster #${vars.toId}`,
      });
    },
    onError: (error: Error) => {
      addToast({ type: "error", text: `Merge failed: ${error.message}` });
    },
  });
}

export function useRemoveMutation() {
  const addToast = useToastStore((s) => s.addToast);

  return useMutation({
    mutationFn: async ({
      clusterId,
      imagePath,
    }: {
      clusterId: number;
      imagePath: string;
    }) => {
      const res = await removeFace(clusterId, imagePath);
      updateStore(res.workspace);
      return res.workspace;
    },
    onSuccess: (_data, vars) => {
      addToast({
        type: "success",
        text: `Removed photo from cluster #${vars.clusterId}`,
      });
    },
    onError: (error: Error) => {
      addToast({ type: "error", text: `Remove failed: ${error.message}` });
    },
  });
}

export function useMoveMutation() {
  const addToast = useToastStore((s) => s.addToast);

  return useMutation({
    mutationFn: async ({
      imagePath,
      fromId,
      toId,
    }: {
      imagePath: string;
      fromId: number;
      toId: number;
    }) => {
      let res;
      if (fromId === -1) {
        res = await assignNoise(imagePath, toId);
      } else {
        res = await moveFace(imagePath, fromId, toId);
      }
      updateStore(res.workspace);
      return res.workspace;
    },
    onSuccess: (_data, vars) => {
      addToast({
        type: "success",
        text: `Moved photo to cluster #${vars.toId}`,
      });
    },
    onError: (error: Error) => {
      addToast({ type: "error", text: `Move failed: ${error.message}` });
    },
  });
}

export function useBatchAssignMutation() {
  const addToast = useToastStore((s) => s.addToast);

  return useMutation({
    mutationFn: async ({
      imagePaths,
      toId,
    }: {
      imagePaths: string[];
      toId: number;
    }) => {
      for (const path of imagePaths) {
        const res = await assignNoise(path, toId);
        updateStore(res.workspace);
      }
    },
    onSuccess: (_data, vars) => {
      addToast({
        type: "success",
        text: `Assigned ${vars.imagePaths.length} photo(s) to cluster #${vars.toId}`,
      });
    },
    onError: (error: Error) => {
      addToast({
        type: "error",
        text: `Batch assign failed: ${error.message}`,
      });
    },
  });
}

export function useRenameMutation() {
  const addToast = useToastStore((s) => s.addToast);

  return useMutation<
    WorkspaceState,
    Error,
    { clusterId: number; name: string; oldName: string; clusterCountBefore: number }
  >({
    mutationFn: async ({ clusterId, name }) => {
      const res = await renameCluster(clusterId, name);
      updateStore(res.workspace);
      return res.workspace;
    },
    onSuccess: (ws, vars) => {
      const stillExists = ws.clusters.some(
        (c: ClusterInfo) => c.id === vars.clusterId,
      );
      if (!stillExists || ws.clusters.length < vars.clusterCountBefore) {
        addToast({
          type: "info",
          text: `Renamed "${vars.oldName}" merged into existing "${vars.name}" cluster`,
        });
      } else {
        addToast({
          type: "success",
          text: `Renamed "${vars.oldName}" to "${vars.name}"`,
        });
      }
    },
    onError: (error: Error) => {
      addToast({ type: "error", text: `Rename failed: ${error.message}` });
    },
  });
}

export function useUndoMutation() {
  const addToast = useToastStore((s) => s.addToast);

  return useMutation({
    mutationFn: async () => {
      const res = await undo();
      updateStore(res.workspace);
      return res;
    },
    onSuccess: (res) => {
      const undoInfo = res.undo as Record<string, unknown> | undefined;
      const action = undoInfo?.["action"] as string | undefined;
      const msg =
        action === "merge"
          ? "Undone: merge"
          : action === "remove"
            ? "Undone: remove"
            : action === "move"
              ? "Undone: move"
              : action === "rename"
                ? "Undone: rename"
                : "Action undone";
      addToast({ type: "success", text: msg });
    },
    onError: (error: Error) => {
      addToast({ type: "error", text: `Undo failed: ${error.message}` });
    },
  });
}

export function useSaveMutation() {
  const addToast = useToastStore((s) => s.addToast);

  return useMutation({
    mutationFn: async (settings: SaveSettings) => {
      return save(settings);
    },
    onSuccess: (res, vars) => {
      const action = vars.copy_mode !== false ? "copied" : "moved";
      const count = res.stats[action] ?? Object.values(res.stats).reduce((a, b) => a + b, 0);
      addToast({ type: "success", text: `${count} files ${action}` });
    },
    onError: (error: Error) => {
      addToast({ type: "error", text: `Save failed: ${error.message}` });
    },
  });
}

// ── Aggregated mutation state ──────────────────────────────────

/** Returns true if any workspace mutation is pending. */
export function useIsMutating(): boolean {
  return (
    useMutationState({
      filters: { status: "pending" },
    }).length > 0
  );
}
