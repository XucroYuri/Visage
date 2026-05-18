export interface FaceBox {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

export interface PhotoInfo {
  path: string;
  faces: FaceBox[];
}

export interface ClusterInfo {
  id: number;
  name: string;
  photos: PhotoInfo[];
  photo_count: number;
  thumbnail: string | null;
  confidence: number;
}

export interface WorkspaceState {
  input_dir: string;
  config: {
    copy_mode: boolean;
    folder_prefix: string;
    embedding_backend: string;
  };
  stats: {
    total_images: number;
    images_with_faces: number;
    total_faces: number;
    num_clusters: number;
    num_noise_faces: number;
  };
  clusters: ClusterInfo[];
  noise_photos: PhotoInfo[];
  all_photos: PhotoInfo[];
  next_cluster_id: number;
  can_undo: boolean;
}

export interface PipelineEvent {
  phase: number;
  message: string;
  done?: boolean;
  error?: boolean;
  count?: number;
  elapsed?: number;
}

const BASE = "/api";

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(body || res.statusText);
  }
  return res.json();
}

export function fetchWorkspace(): Promise<WorkspaceState> {
  return request<WorkspaceState>("/workspace");
}

export function mergeClusters(fromId: number, toId: number): Promise<{ ok: boolean; workspace: WorkspaceState }> {
  return request("/clusters/merge", {
    method: "POST",
    body: JSON.stringify({ from_id: fromId, to_id: toId }),
  });
}

export function removeFace(clusterId: number, imagePath: string): Promise<{ ok: boolean; workspace: WorkspaceState }> {
  return request(`/clusters/${clusterId}/remove`, {
    method: "POST",
    body: JSON.stringify({ image_path: imagePath }),
  });
}

export function moveFace(
  imagePath: string,
  fromId: number,
  toId: number,
): Promise<{ ok: boolean; workspace: WorkspaceState }> {
  return request("/clusters/move", {
    method: "POST",
    body: JSON.stringify({ image_path: imagePath, from_id: fromId, to_id: toId }),
  });
}

export function assignNoise(
  imagePath: string,
  toId: number,
): Promise<{ ok: boolean; workspace: WorkspaceState }> {
  return request("/clusters/assign", {
    method: "POST",
    body: JSON.stringify({ image_path: imagePath, to_id: toId }),
  });
}

export function renameCluster(clusterId: number, name: string): Promise<{ ok: boolean; workspace: WorkspaceState }> {
  return request(`/clusters/${clusterId}`, {
    method: "PUT",
    body: JSON.stringify({ name }),
  });
}

export function undo(): Promise<{ ok: boolean; undo: object; workspace: WorkspaceState }> {
  return request("/clusters/undo", { method: "POST" });
}

export function save(outputDir?: string): Promise<{ ok: boolean; stats: Record<string, number> }> {
  return request("/save", {
    method: "POST",
    body: JSON.stringify({ output_dir: outputDir }),
  });
}

export function getImageUrl(path: string, size: "thumb" | "full" = "thumb"): string {
  const encoded = encodeURIComponent(path);
  return `${BASE}/image?path=${encoded}&size=${size}`;
}

export function pipelineStatusUrl(): string {
  return `${BASE}/pipeline-status`;
}
