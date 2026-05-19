import { useCallback, useEffect, useRef, useState } from "react";
import type { ClusterInfo, PhotoInfo, PipelineEvent, WorkspaceState } from "./api";
import {
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
} from "./api";

type ViewMode = "all" | "noise" | { clusterId: number };

/* ── Pipeline loading screen ────────────────────────────── */

function PipelineLoader({ onReady, onError }: {
  onReady: (ws: WorkspaceState) => void;
  onError: (msg: string) => void;
}) {
  const [phase, setPhase] = useState(0);
  const [message, setMessage] = useState("Starting pipeline...");
  const phases = [
    "Scanning images",
    "Detecting faces",
    "Generating embeddings",
    "Clustering faces",
    "Merging clusters",
  ];

  useEffect(() => {
    const es = new EventSource(pipelineStatusUrl());
    es.onmessage = (e) => {
      const data: PipelineEvent = JSON.parse(e.data);
      setPhase(data.phase);
      setMessage(data.message);
      if (data.done) {
        es.close();
        // Small delay to let workspace state settle
        setTimeout(() => {
          fetchWorkspace().then(onReady).catch((err) => onError(String(err)));
        }, 200);
      }
      if (data.error) {
        es.close();
        onError(data.message);
      }
    };
    es.onerror = () => {
      es.close();
      // Fallback: try to fetch workspace directly
      fetchWorkspace()
        .then(onReady)
        .catch(() => onError("Connection lost"));
    };
    return () => es.close();
  }, [onReady, onError]);

  const progress = Math.min(100, (phase / 5) * 100 + 10);

  return (
    <div className="flex items-center justify-center h-screen bg-gray-50">
      <div className="w-96 text-center">
        <div className="mb-6">
          <div className="inline-block w-16 h-16 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin" />
        </div>
        <h2 className="text-xl font-semibold text-gray-800 mb-2">
          Processing Photos
        </h2>
        <p className="text-sm text-gray-500 mb-4">{message}</p>
        <div className="w-full bg-gray-200 rounded-full h-2 mb-3">
          <div
            className="bg-blue-600 h-2 rounded-full transition-all duration-500"
            style={{ width: `${progress}%` }}
          />
        </div>
        <div className="flex justify-between text-xs text-gray-400">
          {phases.map((label, i) => (
            <span key={i} className={i < phase ? "text-blue-600" : ""}>
              {label}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ── Main App ───────────────────────────────────────────── */

function App() {
  const [ws, setWs] = useState<WorkspaceState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [view, setView] = useState<ViewMode>("all");
  const [mergeMode, setMergeMode] = useState(false);
  const [selectedForMerge, setSelectedForMerge] = useState<Set<number>>(new Set());
  const [editingName, setEditingName] = useState<number | null>(null);
  const [editValue, setEditValue] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveResult, setSaveResult] = useState<string | null>(null);

  const handleWorkspaceReady = useCallback((data: WorkspaceState) => {
    setWs(data);
    setLoading(false);
    setError(null);
  }, []);

  const handleError = useCallback((msg: string) => {
    setError(msg);
    setLoading(false);
  }, []);

  useEffect(() => {
    // Try fetching workspace first — if pipeline is already done, skip SSE
    fetchWorkspace()
      .then(handleWorkspaceReady)
      .catch(() => {
        // Workspace not ready yet — pipeline still loading, show loader
        setLoading(true);
      });
  }, [handleWorkspaceReady]);

  const refresh = async () => {
    setLoading(true);
    try {
      const data = await fetchWorkspace();
      setWs(data);
      setError(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const handleSelectCluster = (id: number) => {
    if (mergeMode) {
      setSelectedForMerge((prev) => {
        const next = new Set(prev);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        return next;
      });
    } else {
      setView({ clusterId: id });
    }
  };

  const handleMerge = async () => {
    const ids = Array.from(selectedForMerge);
    if (ids.length < 2) return;
    const toId = ids[0];
    for (let i = 1; i < ids.length; i++) {
      const res = await mergeClusters(ids[i], toId);
      setWs(res.workspace);
    }
    setSelectedForMerge(new Set());
    setMergeMode(false);
    setView({ clusterId: toId });
  };

  const handleRemove = async (clusterId: number, imagePath: string) => {
    const res = await removeFace(clusterId, imagePath);
    setWs(res.workspace);
    if (!res.workspace.clusters.find((c) => c.id === clusterId)) {
      setView("all");
    }
  };

  const handleMove = async (imagePath: string, fromId: number, toId: number) => {
    if (fromId === -1) {
      const res = await assignNoise(imagePath, toId);
      setWs(res.workspace);
    } else {
      const res = await moveFace(imagePath, fromId, toId);
      setWs(res.workspace);
    }
  };

  const handleRename = async (clusterId: number) => {
    if (editValue.trim()) {
      const res = await renameCluster(clusterId, editValue.trim());
      setWs(res.workspace);
    }
    setEditingName(null);
  };

  const handleUndo = async () => {
    const res = await undo();
    setWs(res.workspace);
  };

  const handleSave = async () => {
    setSaving(true);
    setSaveResult(null);
    try {
      const res = await save();
      const action = ws?.config.copy_mode ? "copied" : "moved";
      const count = res.stats[action] || 0;
      setSaveResult(`Saved: ${count} files ${action}`);
      setTimeout(() => setSaveResult(null), 4000);
    } catch (e) {
      setSaveResult(`Error: ${e}`);
    } finally {
      setSaving(false);
    }
  };

  // Keyboard shortcuts: Ctrl+Z = undo, Ctrl+S = save
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "z" && ws?.can_undo) {
        e.preventDefault();
        undo().then((res) => setWs(res.workspace));
      }
      if ((e.metaKey || e.ctrlKey) && e.key === "s" && !saving) {
        e.preventDefault();
        handleSave();
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [ws?.can_undo, saving]);

  // Pipeline still loading
  if (loading && !ws) {
    return <PipelineLoader onReady={handleWorkspaceReady} onError={handleError} />;
  }

  if (error && !ws) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center text-red-500">
          <p className="text-xl mb-2">Failed to load workspace</p>
          <p className="text-sm text-gray-500">{error}</p>
          <button onClick={refresh} className="mt-4 px-4 py-2 bg-blue-500 text-white rounded">
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!ws) return null;

  const selectedCluster =
    typeof view === "object" && "clusterId" in view
      ? ws.clusters.find((c) => c.id === view.clusterId)
      : null;

  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-3 bg-white border-b border-gray-200 shadow-sm shrink-0">
        <div className="flex items-center gap-4">
          <h1 className="text-lg font-semibold text-gray-900">Visage Review</h1>
          <span className="text-sm text-gray-400">
            {ws.stats.num_clusters} clusters &middot;{" "}
            {ws.stats.images_with_faces} images &middot;{" "}
            {ws.stats.num_noise_faces} noise
          </span>
        </div>
        <div className="flex items-center gap-2">
          {saveResult && (
            <span className="text-sm text-green-600 mr-2">{saveResult}</span>
          )}
          <button
            onClick={handleUndo}
            disabled={!ws.can_undo}
            className="px-3 py-1.5 text-sm border rounded disabled:opacity-30 hover:bg-gray-50"
            title="Undo (Ctrl+Z)"
          >
            &#8630; Undo
          </button>
          <button
            onClick={handleSave}
            disabled={saving}
            className="px-4 py-1.5 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
          >
            {saving ? "Saving..." : "Save to Disk"}
          </button>
        </div>
      </header>

      {/* Body */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside className="w-72 bg-gray-50 border-r border-gray-200 overflow-y-auto shrink-0">
          <div className="p-3 border-b border-gray-200">
            <button
              onClick={() => { setView("all"); setMergeMode(false); setSelectedForMerge(new Set()); }}
              className={`w-full text-left px-3 py-2 rounded text-sm font-medium ${
                view === "all" ? "bg-blue-50 text-blue-700" : "hover:bg-gray-100"
              }`}
            >
              &#128247; All Photos
            </button>
            <button
              onClick={() => { setView("noise"); setMergeMode(false); setSelectedForMerge(new Set()); }}
              className={`w-full text-left px-3 py-2 rounded text-sm font-medium mt-1 ${
                view === "noise" ? "bg-amber-50 text-amber-700" : "hover:bg-gray-100"
              }`}
            >
              &#10067; Unclustered ({ws.noise_photos.length})
            </button>
            <button
              onClick={() => { setMergeMode(!mergeMode); setSelectedForMerge(new Set()); if (!mergeMode) setView("all"); }}
              className={`w-full text-left px-3 py-2 rounded text-sm font-medium mt-1 ${
                mergeMode ? "bg-purple-100 text-purple-700" : "hover:bg-gray-100"
              }`}
            >
              {mergeMode ? "&#10005; Cancel Merge" : "&#9878; Merge Mode"}
            </button>
            {mergeMode && selectedForMerge.size >= 2 && (
              <button
                onClick={handleMerge}
                className="w-full mt-1 px-3 py-2 bg-purple-600 text-white rounded text-sm font-medium"
              >
                Merge {selectedForMerge.size} Selected
              </button>
            )}
          </div>

          <div className="divide-y divide-gray-100">
            {ws.clusters.map((c) => (
              <ClusterRow
                key={c.id}
                cluster={c}
                selected={
                  typeof view === "object" && "clusterId" in view && view.clusterId === c.id
                }
                mergeMode={mergeMode}
                checkedForMerge={selectedForMerge.has(c.id)}
                editing={editingName === c.id}
                editValue={editValue}
                onSelect={() => handleSelectCluster(c.id)}
                onStartEdit={() => { setEditingName(c.id); setEditValue(c.name); }}
                onEditChange={setEditValue}
                onSaveEdit={() => handleRename(c.id)}
                onCancelEdit={() => setEditingName(null)}
              />
            ))}
          </div>
        </aside>

        {/* Main content */}
        <main className="flex-1 overflow-y-auto p-6">
          {selectedCluster ? (
            <ClusterDetail
              cluster={selectedCluster}
              clusters={ws.clusters}
              onRemove={(path) => handleRemove(selectedCluster.id, path)}
              onMove={(path, toId) => handleMove(path, selectedCluster.id, toId)}
              onBack={() => setView("all")}
              onStartRename={() => { setEditingName(selectedCluster.id); setEditValue(selectedCluster.name); }}
              editing={editingName === selectedCluster.id}
              editValue={editValue}
              onEditChange={setEditValue}
              onSaveEdit={() => handleRename(selectedCluster.id)}
              onCancelEdit={() => setEditingName(null)}
            />
          ) : view === "noise" ? (
            <NoisePanel
              noisePhotos={ws.noise_photos}
              clusters={ws.clusters}
              nextClusterId={ws.next_cluster_id}
              onAssign={(path, toId) => handleMove(path, -1, toId)}
            />
          ) : (
            <div>
              <h2 className="text-lg font-semibold text-gray-700 mb-4">
                All Photos ({ws.all_photos.length})
              </h2>
              <div className="grid grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
                {ws.all_photos.map((photo) => (
                  <PhotoCard key={photo.path} photo={photo} />
                ))}
              </div>
              {ws.all_photos.length === 0 && (
                <div className="text-center text-gray-400 mt-20">
                  <p className="text-lg">No clustered photos</p>
                </div>
              )}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

/* ── Cluster row in sidebar ──────────────────────────────── */

function ClusterRow({ cluster, selected, mergeMode, checkedForMerge, editing, editValue,
  onSelect, onStartEdit, onEditChange, onSaveEdit, onCancelEdit }: {
  cluster: ClusterInfo; selected: boolean; mergeMode: boolean; checkedForMerge: boolean;
  editing: boolean; editValue: string; onSelect: () => void; onStartEdit: () => void;
  onEditChange: (v: string) => void; onSaveEdit: () => void; onCancelEdit: () => void;
}) {
  return (
    <div
      onClick={onSelect}
      className={`flex items-center gap-3 px-3 py-2.5 cursor-pointer group ${
        selected ? "bg-blue-50 border-l-2 border-blue-500" : "hover:bg-white"
      }`}
    >
      {mergeMode && (
        <input type="checkbox" checked={checkedForMerge} onChange={onSelect} className="shrink-0" />
      )}
      <div className="w-10 h-10 bg-gray-200 rounded overflow-hidden shrink-0">
        {cluster.thumbnail && (
          <img src={getImageUrl(cluster.thumbnail)} alt="" className="w-full h-full object-cover" />
        )}
      </div>
      <div className="flex-1 min-w-0">
        {editing ? (
          <input
            value={editValue} onChange={(e) => onEditChange(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") onSaveEdit(); if (e.key === "Escape") onCancelEdit(); }}
            onClick={(e) => e.stopPropagation()}
            className="text-sm font-medium w-full border rounded px-1 py-0.5" autoFocus
          />
        ) : (
          <div
            onClick={(e) => { e.stopPropagation(); onStartEdit(); }}
            className="text-sm font-medium text-gray-800 truncate cursor-text hover:text-blue-600"
            title="Click to rename"
          >
            {cluster.name}
          </div>
        )}
        <div className="text-xs text-gray-400">
          {cluster.photo_count} photos &middot; conf: {cluster.confidence.toFixed(2)}
        </div>
      </div>
    </div>
  );
}

/* ── Cluster detail view ──────────────────────────────────── */

function ClusterDetail({ cluster, clusters, onRemove, onMove, onBack, onStartRename,
  editing, editValue, onEditChange, onSaveEdit, onCancelEdit }: {
  cluster: ClusterInfo; clusters: ClusterInfo[];
  onRemove: (path: string) => void; onMove: (path: string, toId: number) => void;
  onBack: () => void; onStartRename: () => void; editing: boolean; editValue: string;
  onEditChange: (v: string) => void; onSaveEdit: () => void; onCancelEdit: () => void;
}) {
  return (
    <div>
      <div className="flex items-center gap-3 mb-4">
        <button onClick={onBack} className="text-sm text-gray-400 hover:text-gray-600">
          &#8592; All
        </button>
        {editing ? (
          <input
            value={editValue} onChange={(e) => onEditChange(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") onSaveEdit(); if (e.key === "Escape") onCancelEdit(); }}
            className="text-xl font-semibold border rounded px-2 py-1" autoFocus
          />
        ) : (
          <h2
            onClick={onStartRename}
            className="text-xl font-semibold text-gray-900 cursor-pointer hover:text-blue-600"
            title="Click to rename"
          >
            {cluster.name}
          </h2>
        )}
        <span className="text-sm text-gray-400">
          {cluster.photo_count} photos &middot; confidence {cluster.confidence.toFixed(2)}
        </span>
      </div>

      <div className="grid grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
        {cluster.photos.map((photo) => (
          <PhotoCard
            key={photo.path}
            photo={photo}
            onRemove={() => onRemove(photo.path)}
            onMove={(toId) => onMove(photo.path, toId)}
            otherClusters={clusters.filter((c) => c.id !== cluster.id)}
          />
        ))}
      </div>
    </div>
  );
}

/* ── Noise / Unclustered panel ───────────────────────────── */

function NoisePanel({ noisePhotos, clusters, nextClusterId, onAssign }: {
  noisePhotos: PhotoInfo[]; clusters: ClusterInfo[]; nextClusterId: number;
  onAssign: (path: string, toId: number) => void;
}) {
  return (
    <div>
      <h2 className="text-lg font-semibold text-gray-700 mb-4">
        Unclustered Faces ({noisePhotos.length})
      </h2>
      {noisePhotos.length === 0 ? (
        <p className="text-gray-400 text-center mt-10">No unclustered faces</p>
      ) : (
        <div className="grid grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
          {noisePhotos.map((photo) => (
            <div key={photo.path} className="relative group">
              <PhotoCard
                photo={photo}
                onMove={(toId) => onAssign(photo.path, toId)}
                otherClusters={clusters}
                nextClusterId={nextClusterId}
              />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Photo card with face overlay ───────────────────────────── */

function PhotoCard({ photo, onRemove, onMove, otherClusters, nextClusterId }: {
  photo: PhotoInfo; onRemove?: () => void;
  onMove?: (toId: number) => void;
  otherClusters?: ClusterInfo[];
  nextClusterId?: number;
}) {
  const [showFull, setShowFull] = useState(false);
  const [showMoveMenu, setShowMoveMenu] = useState(false);
  const [imgSize, setImgSize] = useState<{ w: number; h: number } | null>(null);
  const [fullSize, setFullSize] = useState<{ w: number; h: number } | null>(null);
  const filename = photo.path.split("/").pop() || photo.path;

  return (
    <>
      <div className="group relative bg-white border border-gray-200 rounded-lg overflow-hidden">
        <div className="relative">
          <img
            src={getImageUrl(photo.path)} alt={filename}
            className="w-full aspect-square object-cover cursor-pointer"
            onClick={() => setShowFull(true)}
            loading="lazy"
            onLoad={(e) => {
              const img = e.currentTarget;
              setImgSize({ w: img.naturalWidth, h: img.naturalHeight });
            }}
          />
          {imgSize && photo.faces.map((face, i) => (
            <div
              key={i}
              className="absolute border-2 border-green-400 rounded-sm pointer-events-none"
              style={{
                left: `${(face.left / imgSize.w) * 100}%`,
                top: `${(face.top / imgSize.h) * 100}%`,
                width: `${((face.right - face.left) / imgSize.w) * 100}%`,
                height: `${((face.bottom - face.top) / imgSize.h) * 100}%`,
              }}
            />
          ))}
        </div>
        {/* Hover actions */}
        <div className="absolute bottom-0 left-0 right-0 p-2 bg-gradient-to-t from-black/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex gap-1">
          {onRemove && (
            <button
              onClick={(e) => { e.stopPropagation(); onRemove(); }}
              className="text-xs text-white bg-red-500/80 hover:bg-red-600 px-2 py-1 rounded"
            >
              Remove
            </button>
          )}
          {onMove && otherClusters && (
            <div className="relative">
              <button
                onClick={(e) => { e.stopPropagation(); setShowMoveMenu(!showMoveMenu); }}
                className="text-xs text-white bg-blue-500/80 hover:bg-blue-600 px-2 py-1 rounded"
              >
                Move
              </button>
              {showMoveMenu && (
                <div
                  className="absolute bottom-full left-0 mb-1 bg-white rounded shadow-lg border max-h-48 overflow-y-auto min-w-40 z-10"
                  onClick={(e) => e.stopPropagation()}
                >
                  {otherClusters.map((c) => (
                    <button
                      key={c.id}
                      onClick={() => { onMove(c.id); setShowMoveMenu(false); }}
                      className="block w-full text-left px-3 py-1.5 text-sm hover:bg-blue-50 text-gray-700"
                    >
                      {c.name} ({c.photo_count})
                    </button>
                  ))}
                  {nextClusterId !== undefined && (
                    <button
                      onClick={() => { onMove(nextClusterId); setShowMoveMenu(false); }}
                      className="block w-full text-left px-3 py-1.5 text-sm hover:bg-green-50 text-green-700 border-t"
                    >
                      + New cluster
                    </button>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
        <div className="p-1.5">
          <div className="text-xs text-gray-500 truncate" title={filename}>{filename}</div>
        </div>
      </div>

      {/* Full-size modal with navigation */}
      {showFull && (
        <PhotoViewer
          photo={photo}
          onClose={() => { setShowFull(false); setFullSize(null); }}
          fullSize={fullSize}
          setFullSize={setFullSize}
        />
      )}
    </>
  );
}

/* ── Full-size photo viewer with prev/next ───────────────── */

function PhotoViewer({ photo, onClose, fullSize, setFullSize }: {
  photo: PhotoInfo;
  onClose: () => void;
  fullSize: { w: number; h: number } | null;
  setFullSize: (s: { w: number; h: number } | null) => void;
}) {
  const filename = photo.path.split("/").pop() || photo.path;
  const overlayRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [onClose]);

  return (
    <div
      ref={overlayRef}
      className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center cursor-pointer"
      onClick={onClose}
    >
      <div className="relative max-h-[90vh] max-w-[90vw]">
        <img
          src={getImageUrl(photo.path, "full")} alt={filename}
          className="max-h-[90vh] max-w-[90vw] object-contain"
          onLoad={(e) => {
            const img = e.currentTarget;
            setFullSize({ w: img.naturalWidth, h: img.naturalHeight });
          }}
        />
        {fullSize && photo.faces.map((face, i) => (
          <div
            key={i}
            className="absolute border-2 border-green-400 rounded-sm pointer-events-none"
            style={{
              left: `${(face.left / fullSize.w) * 100}%`,
              top: `${(face.top / fullSize.h) * 100}%`,
              width: `${((face.right - face.left) / fullSize.w) * 100}%`,
              height: `${((face.bottom - face.top) / fullSize.h) * 100}%`,
            }}
          />
        ))}
      </div>
      <div className="absolute bottom-4 left-4 text-white text-sm bg-black/50 px-3 py-1.5 rounded">
        {filename}
      </div>
    </div>
  );
}

export default App;
