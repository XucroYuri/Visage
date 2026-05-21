/** Hook for face search state management. */

import { useState, useCallback } from "react";
import {
  searchFace,
  type FaceSearchResult,
  type SearchFaceRequest,
} from "../api/search";

export interface UseFaceSearchReturn {
  results: FaceSearchResult[];
  total: number;
  loading: boolean;
  error: string | null;
  page: number;
  search: (req: SearchFaceRequest) => Promise<void>;
  loadMore: () => Promise<void>;
  clear: () => void;
}

export function useFaceSearch(): UseFaceSearchReturn {
  const [results, setResults] = useState<FaceSearchResult[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const [lastReq, setLastReq] = useState<SearchFaceRequest | null>(null);

  const search = useCallback(async (req: SearchFaceRequest) => {
    setLoading(true);
    setError(null);
    setLastReq(req);
    setPage(0);
    try {
      const resp = await searchFace({ ...req, page: 0 });
      setResults(resp.results);
      setTotal(resp.total);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Search failed");
    } finally {
      setLoading(false);
    }
  }, []);

  const loadMore = useCallback(async () => {
    if (!lastReq || loading) return;
    const nextPage = page + 1;
    setLoading(true);
    try {
      const resp = await searchFace({ ...lastReq, page: nextPage });
      setResults((prev) => [...prev, ...resp.results]);
      setPage(nextPage);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Load more failed");
    } finally {
      setLoading(false);
    }
  }, [lastReq, loading, page]);

  const clear = useCallback(() => {
    setResults([]);
    setTotal(0);
    setError(null);
    setPage(0);
    setLastReq(null);
  }, []);

  return { results, total, loading, error, page, search, loadMore, clear };
}
