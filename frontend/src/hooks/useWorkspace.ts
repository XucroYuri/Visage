import { useCallback, useEffect, useRef, useState } from "react";
import type { ClusterInfo, WorkspaceState } from "../api";
import {
  assignNoise,
  fetchWorkspace,
  mergeClusters,
  moveFace,
  removeFace,
  renameCluster,
  save,
  undo,
} from "../api";

export interface ToastMessage {
  type: "success" | "error" | "info";
  text: string;
}

export function useWorkspace(onToast: (toast: ToastMessage) => void) {
  const [ws, setWs] = useState<WorkspaceState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [mutating, setMutating] = useState(false);

  // Track cluster count before rename to detect auto-merge
  const clusterCountBeforeRename = useRef<number>(0);

  /** Load workspace (initial or refresh). */
  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchWorkspace();
      setWs(data);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  // Initial load
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    load();
  }, [load]);

  /** Wrap a mutation call — handles error toasts and mutating state. */
  const mutate = useCallback(
    async <T>(
      fn: () => Promise<T>,
      onSuccess: (result: T) => void,
      successToast?: string,
      errorPrefix: string = "Operation failed",
    ) => {
      setMutating(true);
      try {
        const result = await fn();
        onSuccess(result);
        if (successToast) {
          onToast({ type: "success", text: successToast });
        }
        return result;
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        onToast({ type: "error", text: `${errorPrefix}: ${msg}` });
        throw e;
      } finally {
        setMutating(false);
      }
    },
    [onToast],
  );

  /** Merge clusters: merge each fromId into toId. */
  const merge = useCallback(
    async (fromIds: number[], toId: number) => {
      const targetName = ws?.clusters.find((c) => c.id === toId)?.name ?? `#${toId}`;
      await mutate(
        async () => {
          for (const fromId of fromIds) {
            const res = await mergeClusters(fromId, toId);
            setWs(res.workspace);
          }
        },
        () => {},
        `Merged ${fromIds.length} cluster(s) into "${targetName}"`,
        "Merge failed",
      );
    },
    [ws, mutate],
  );

  /** Remove a face from a cluster. Returns the updated workspace for post-mutation checks. */
  const remove = useCallback(
    async (clusterId: number, imagePath: string): Promise<WorkspaceState | null> => {
      const clusterName = ws?.clusters.find((c) => c.id === clusterId)?.name ?? `#${clusterId}`;
      let updated: WorkspaceState | null = null;
      await mutate(
        async () => {
          const res = await removeFace(clusterId, imagePath);
          setWs(res.workspace);
          updated = res.workspace;
        },
        () => {},
        `Removed photo from "${clusterName}"`,
        "Remove failed",
      );
      return updated;
    },
    [ws, mutate],
  );

  /** Move a face between clusters or from noise to a cluster. */
  const move = useCallback(
    async (imagePath: string, fromId: number, toId: number) => {
      const targetName = ws?.clusters.find((c) => c.id === toId)?.name ?? `#${toId}`;
      await mutate(
        async () => {
          if (fromId === -1) {
            const res = await assignNoise(imagePath, toId);
            setWs(res.workspace);
          } else {
            const res = await moveFace(imagePath, fromId, toId);
            setWs(res.workspace);
          }
        },
        () => {},
        `Moved photo to "${targetName}"`,
        "Move failed",
      );
    },
    [ws, mutate],
  );

  /** Assign multiple noise photos to a cluster at once. */
  const batchAssign = useCallback(
    async (imagePaths: string[], toId: number) => {
      const targetName = ws?.clusters.find((c) => c.id === toId)?.name ?? `#${toId}`;
      await mutate(
        async () => {
          for (const path of imagePaths) {
            const res = await assignNoise(path, toId);
            setWs(res.workspace);
          }
        },
        () => {},
        `Assigned ${imagePaths.length} photo(s) to "${targetName}"`,
        "Batch assign failed",
      );
    },
    [ws, mutate],
  );

  /** Rename a cluster. Detects auto-merge if the cluster disappears. */
  const rename = useCallback(
    async (clusterId: number, name: string) => {
      clusterCountBeforeRename.current = ws?.clusters.length ?? 0;
      const oldName = ws?.clusters.find((c) => c.id === clusterId)?.name ?? `#${clusterId}`;

      await mutate(
        async () => {
          const res = await renameCluster(clusterId, name);
          setWs(res.workspace);
          return res;
        },
        (res) => {
          // Detect auto-merge: cluster disappeared or count decreased
          const stillExists = res.workspace.clusters.some((c: ClusterInfo) => c.id === clusterId);
          if (!stillExists || res.workspace.clusters.length < clusterCountBeforeRename.current) {
            onToast({
              type: "info",
              text: `Renamed "${oldName}" merged into existing "${name}" cluster`,
            });
          } else {
            onToast({ type: "success", text: `Renamed "${oldName}" to "${name}"` });
          }
        },
        undefined,
        "Rename failed",
      );
    },
    [ws, mutate, onToast],
  );

  /** Undo the last workspace mutation. */
  const undoLast = useCallback(async () => {
    await mutate(
      async () => {
        const res = await undo();
        setWs(res.workspace);
        return res;
      },
      (res) => {
        const undoInfo = res.undo as Record<string, unknown> | undefined;
        const action = undoInfo?.["action"] as string | undefined;
        if (action === "merge") {
          onToast({ type: "success", text: `Undone: merge` });
        } else if (action === "remove") {
          onToast({ type: "success", text: `Undone: remove` });
        } else if (action === "move") {
          onToast({ type: "success", text: `Undone: move` });
        } else if (action === "rename") {
          onToast({ type: "success", text: `Undone: rename` });
        } else {
          onToast({ type: "success", text: `Action undone` });
        }
      },
      undefined,
      "Undo failed",
    );
  }, [mutate, onToast]);

  /** Save changes to disk. */
  const saveToDisk = useCallback(
    async (outputDir?: string) => {
      await mutate(
        async () => {
          const res = await save(outputDir);
          return res;
        },
        (res) => {
          const action = ws?.config.copy_mode ? "copied" : "moved";
          const count = res.stats[action] ?? 0;
          onToast({ type: "success", text: `${count} files ${action}` });
        },
        undefined,
        "Save failed",
      );
    },
    [ws, mutate, onToast],
  );

  return {
    ws,
    loading,
    error,
    mutating,
    load,
    merge,
    remove,
    move,
    batchAssign,
    rename,
    undoLast,
    saveToDisk,
  };
}
