export interface FaceBox {
  top: number;
  right: number;
  bottom: number;
  left: number;
  cluster_id: number;  // which cluster this face belongs to (-1 = noise/unclustered)
}

export interface PhotoInfo {
  path: string;
  faces: FaceBox[];
  width: number;
  height: number;
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

export interface MutationResult {
  ok: boolean;
  workspace: WorkspaceState;
}

export interface UndoResult {
  ok: boolean;
  undo: Record<string, unknown>;
  workspace: WorkspaceState;
}

export interface ReclusterSettings {
  cluster_method?: string;
  min_samples?: number;
  min_cluster_size?: number;
  cluster_selection_epsilon?: number;
  cluster_selection_method?: string;
  merge_threshold?: number;
  small_merge_threshold?: number;
  min_reliable_size?: number;
  head_feature_weight?: number;
}

export interface ReclusterResult {
  ok: boolean;
  workspace: WorkspaceState;
}

export interface SaveSettings {
  output_dir?: string;
  copy_mode?: boolean;
  folder_prefix?: string;
  include_unclustered?: boolean;
  include_no_faces?: boolean;
  cluster_ids?: number[];
  multi_face_strategy?: "primary" | "all";
}

export interface SaveResult {
  ok: boolean;
  stats: Record<string, number>;
}

/** API error with optional HTTP status code. */
export class ApiError extends Error {
  statusCode?: number;

  constructor(message: string, statusCode?: number) {
    super(message);
    this.name = "ApiError";
    this.statusCode = statusCode;
  }
}

const BASE = "/api";

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new ApiError(body || res.statusText, res.status);
  }
  return res.json() as Promise<T>;
}

export function fetchWorkspace(): Promise<WorkspaceState> {
  return request<WorkspaceState>("/workspace");
}

export function mergeClusters(
  fromId: number,
  toId: number,
): Promise<MutationResult> {
  return request("/clusters/merge", {
    method: "POST",
    body: JSON.stringify({ from_id: fromId, to_id: toId }),
  });
}

export function removeFace(
  clusterId: number,
  imagePath: string,
): Promise<MutationResult> {
  return request(`/clusters/${clusterId}/remove`, {
    method: "POST",
    body: JSON.stringify({ image_path: imagePath }),
  });
}

export function moveFace(
  imagePath: string,
  fromId: number,
  toId: number,
): Promise<MutationResult> {
  return request("/clusters/move", {
    method: "POST",
    body: JSON.stringify({ image_path: imagePath, from_id: fromId, to_id: toId }),
  });
}

export function assignNoise(
  imagePath: string,
  toId: number,
): Promise<MutationResult> {
  return request("/clusters/assign", {
    method: "POST",
    body: JSON.stringify({ image_path: imagePath, to_id: toId }),
  });
}

export function renameCluster(
  clusterId: number,
  name: string,
): Promise<MutationResult> {
  return request(`/clusters/${clusterId}`, {
    method: "PUT",
    body: JSON.stringify({ name }),
  });
}

export function undo(): Promise<UndoResult> {
  return request("/clusters/undo", { method: "POST" });
}

export function save(settings?: SaveSettings): Promise<SaveResult> {
  return request("/save", {
    method: "POST",
    body: JSON.stringify(settings || {}),
  });
}

export function getImageUrl(
  path: string,
  size: "thumb" | "full" = "thumb",
): string {
  const encoded = encodeURIComponent(path);
  return `${BASE}/image?path=${encoded}&size=${size}`;
}

export function pipelineStatusUrl(): string {
  return `${BASE}/pipeline-status`;
}

export function recluster(
  settings: ReclusterSettings,
): Promise<ReclusterResult> {
  return request("/recluster", {
    method: "POST",
    body: JSON.stringify(settings),
  });
}

export interface ConfigResponse {
  copy_mode: boolean;
  folder_prefix: string;
  embedding_backend: string;
  cluster_method: string;
  min_samples: number;
  min_cluster_size: number;
  cluster_selection_epsilon: number;
  cluster_selection_method: string;
  merge_threshold: number;
  small_merge_threshold: number;
  min_reliable_size: number;
  head_feature_weight: number;
}

export function fetchConfig(): Promise<ConfigResponse> {
  return request("/config");
}
