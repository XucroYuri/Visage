/** Face search API types and client. */

export interface SearchFaceRequest {
  face_id: string;
  top_k?: number;
  min_score?: number;
  cluster_id?: number | null;
  page?: number;
  page_size?: number;
}

export interface FaceSearchResult {
  face_id: string;
  image_path: string;
  similarity: number;
  cluster_id?: number | null;
  bbox?: number[] | null;
}

export interface FaceSearchResponse {
  query_face_id: string;
  results: FaceSearchResult[];
  total: number;
  page: number;
  page_size: number;
}

const BASE = "/api";

export async function searchFace(
  req: SearchFaceRequest,
): Promise<FaceSearchResponse> {
  const res = await fetch(`${BASE}/search/face`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    throw new Error(`Search failed: ${res.statusText}`);
  }
  return res.json();
}
