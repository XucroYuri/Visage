/** Search results grid with similarity scores. */

import type { FaceSearchResult } from "../api/search";

interface SearchResultsProps {
  results: FaceSearchResult[];
  total: number;
  loading: boolean;
  error: string | null;
  hasMore: boolean;
  onLoadMore: () => void;
  onResultClick: (result: FaceSearchResult) => void;
}

function similarityColor(score: number): string {
  if (score > 0.85) return "text-green-600 dark:text-green-400";
  if (score > 0.70) return "text-yellow-600 dark:text-yellow-400";
  return "text-red-600 dark:text-red-400";
}

function similarityBg(score: number): string {
  if (score > 0.85) return "bg-green-100 dark:bg-green-900/30";
  if (score > 0.70) return "bg-yellow-100 dark:bg-yellow-900/30";
  return "bg-red-100 dark:bg-red-900/30";
}

export function SearchResults({
  results,
  total,
  loading,
  error,
  hasMore,
  onLoadMore,
  onResultClick,
}: SearchResultsProps) {
  if (error) {
    return (
      <div className="p-4 text-center text-red-600 dark:text-red-400">
        {error}
      </div>
    );
  }

  if (results.length === 0 && !loading) {
    return null;
  }

  return (
    <div>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-3">
        {total} similar face{total !== 1 ? "s" : ""} found
      </p>
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
        {results.map((r) => (
          <button
            key={r.face_id}
            onClick={() => onResultClick(r)}
            className="group relative rounded-lg overflow-hidden border border-gray-200
                       dark:border-gray-700 hover:border-blue-400 dark:hover:border-blue-500
                       transition-colors text-left"
          >
            <div className="aspect-square bg-gray-100 dark:bg-gray-800 flex items-center justify-center">
              <span className="text-gray-400 text-xs truncate px-2">
                {r.image_path.split("/").pop()}
              </span>
            </div>
            <div
              className={`px-2 py-1 text-xs font-medium ${similarityColor(r.similarity)} ${similarityBg(r.similarity)}`}
            >
              {(r.similarity * 100).toFixed(0)}% match
            </div>
          </button>
        ))}
      </div>
      {hasMore && (
        <button
          onClick={onLoadMore}
          disabled={loading}
          className="mt-4 w-full py-2 text-sm text-blue-600 dark:text-blue-400
                     hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-lg transition-colors"
        >
          {loading ? "Loading..." : "Load more results"}
        </button>
      )}
    </div>
  );
}
