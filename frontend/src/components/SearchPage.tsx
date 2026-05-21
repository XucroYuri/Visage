import { useState, useCallback } from "react";
import { semanticSearch, searchByTags } from "../api-phase3";
import type { SearchResult } from "../api-phase3";
import { getImageUrl } from "../api";

export function SearchPage() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<"semantic" | "tags">("semantic");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searched, setSearched] = useState(false);

  const handleSearch = useCallback(async () => {
    const q = query.trim();
    if (!q) return;

    setLoading(true);
    setError(null);
    setSearched(true);

    try {
      if (mode === "semantic") {
        const res = await semanticSearch(q);
        setResults(res.results);
      } else {
        const res = await searchByTags(q);
        setResults(res.results);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Search failed");
    } finally {
      setLoading(false);
    }
  }, [query, mode]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter") handleSearch();
    },
    [handleSearch],
  );

  return (
    <div>
      <h2 className="text-lg font-semibold mb-4" style={{ color: "var(--color-text-primary)" }}>
        Search Photos
      </h2>

      {/* Search bar */}
      <div className="flex gap-2 mb-4">
        <div className="flex rounded-lg border border-gray-200 dark:border-slate-600 overflow-hidden flex-1">
          {/* Mode toggle */}
          <button
            onClick={() => setMode(mode === "semantic" ? "tags" : "semantic")}
            className="px-3 py-2 text-xs font-medium border-r border-gray-200 dark:border-slate-600 whitespace-nowrap transition-colors"
            style={{
              color: "var(--color-text-secondary)",
              backgroundColor: "var(--color-bg-secondary)",
            }}
          >
            {mode === "semantic" ? "Natural Language" : "Tags"}
          </button>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              mode === "semantic"
                ? "sunset at the beach..."
                : "sunset, beach, outdoor"
            }
            className="flex-1 px-3 py-2 text-sm bg-transparent outline-none"
            style={{ color: "var(--color-text-primary)" }}
          />
        </div>
        <button
          onClick={handleSearch}
          disabled={loading || !query.trim()}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-40 transition-colors"
        >
          {loading ? "Searching..." : "Search"}
        </button>
      </div>

      {/* Error */}
      {error && (
        <p className="text-red-500 text-sm mb-4">{error}</p>
      )}

      {/* Results */}
      {searched && !loading && results.length === 0 && (
        <p className="text-center py-8 text-gray-500 dark:text-slate-400">
          No results found
        </p>
      )}

      {results.length > 0 && (
        <div>
          <p className="text-sm mb-3" style={{ color: "var(--color-text-secondary)" }}>
            {results.length} result{results.length !== 1 ? "s" : ""}
          </p>
          <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-2">
            {results.map((r) => (
              <div key={r.image_path} className="group relative">
                <div className="aspect-square rounded-lg overflow-hidden bg-gray-100 dark:bg-slate-700">
                  <img
                    src={getImageUrl(r.image_path, "thumb")}
                    alt=""
                    className="w-full h-full object-cover"
                    loading="lazy"
                  />
                </div>
                <div className="absolute bottom-0 left-0 right-0 px-1.5 py-1 bg-black/60 text-white text-[10px] rounded-b-lg opacity-0 group-hover:opacity-100 transition-opacity">
                  {Math.round(r.score * 100)}% match
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
