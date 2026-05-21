/** API client for Phase 3 endpoints — events, search, active learning. */

import { request } from "./api";

// ── Events ────────────────────────────────────────────────────────

export interface EventInfo {
  event_id: string;
  name: string;
  start_time: string;
  end_time: string;
  photo_count: number;
  photo_paths: string[];
  cover_path: string | null;
  location_name: string | null;
  is_multi_day: boolean;
}

export interface EventsResponse {
  events: EventInfo[];
  total: number;
}

export function fetchEvents(): Promise<EventsResponse> {
  return request("/events");
}

export function fetchEvent(eventId: string): Promise<EventInfo> {
  return request(`/events/${encodeURIComponent(eventId)}`);
}

export function fetchPeopleIntersection(personIds: string): Promise<{
  photo_paths: string[];
  total: number;
}> {
  return request(`/events/people-intersection?person_ids=${encodeURIComponent(personIds)}`);
}

// ── Semantic Search ───────────────────────────────────────────────

export interface SearchResult {
  image_path: string;
  score: number;
}

export interface SemanticSearchResponse {
  query: string;
  results: SearchResult[];
  total: number;
}

export function semanticSearch(
  query: string,
  topK = 20,
  minScore = 0.3,
): Promise<SemanticSearchResponse> {
  return request(
    `/search/semantic?q=${encodeURIComponent(query)}&top_k=${topK}&min_score=${minScore}`,
  );
}

export interface TagSearchResponse {
  tags: string[];
  results: SearchResult[];
  total: number;
}

export function searchByTags(
  tags: string,
  minScore = 0,
  limit = 100,
): Promise<TagSearchResponse> {
  return request(
    `/search/tags?tags=${encodeURIComponent(tags)}&min_score=${minScore}&limit=${limit}`,
  );
}

export interface TagCountsResponse {
  counts: Record<string, number>;
  unique_tags: number;
}

export function fetchTagCounts(): Promise<TagCountsResponse> {
  return request("/search/tags/counts");
}

export interface ImageTagsResponse {
  image_path: string;
  tags: Record<string, string[]>;
}

export function fetchImageTags(imagePath: string): Promise<ImageTagsResponse> {
  return request(`/search/image/${encodeURIComponent(imagePath)}`);
}

// ── Active Learning ───────────────────────────────────────────────

export interface CorrectionInfo {
  id: number;
  action: string;
  face_ids: string[];
  source_cluster: number | null;
  target_cluster: number | null;
  details: Record<string, unknown> | null;
  created_at: number;
}

export interface CorrectionsResponse {
  corrections: CorrectionInfo[];
  total: number;
}

export function fetchCorrections(action?: string, limit = 100): Promise<CorrectionsResponse> {
  const params = new URLSearchParams();
  if (action) params.set("action", action);
  params.set("limit", String(limit));
  return request(`/active/corrections?${params}`);
}

export function recordCorrection(
  action: string,
  faceIds: string[],
  sourceCluster?: number,
  targetCluster?: number,
): Promise<{ correction_id: number; action: string; recorded: boolean }> {
  return request("/active/correction", {
    method: "POST",
    body: JSON.stringify({
      action,
      face_ids: faceIds,
      source_cluster: sourceCluster,
      target_cluster: targetCluster,
    }),
  });
}

export interface CorrectionStatsResponse {
  counts?: Record<string, number>;
  total?: number;
  threshold?: {
    threshold: number;
    merge_count: number;
    split_count: number;
    total_corrections: number;
    merge_ratio: number;
  };
}

export function fetchCorrectionStats(): Promise<CorrectionStatsResponse> {
  return request("/active/corrections/stats");
}

export interface PrototypeInfo {
  cluster_id: number;
  member_count: number;
  total_weight: number;
}

export function fetchPrototypes(): Promise<{ prototypes: PrototypeInfo[]; total: number }> {
  return request("/active/prototypes");
}

export interface ThresholdInfo {
  active: boolean;
  threshold?: number;
  merge_count?: number;
  split_count?: number;
  total_corrections?: number;
}

export function fetchThreshold(): Promise<ThresholdInfo> {
  return request("/active/threshold");
}

export function resetThreshold(): Promise<ThresholdInfo> {
  return request("/active/threshold/reset", { method: "POST" });
}
