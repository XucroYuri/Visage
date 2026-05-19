import { useEffect, useState } from "react";
import type { PipelineEvent, WorkspaceState } from "../api";
import { fetchWorkspace, pipelineStatusUrl } from "../api";

interface PipelineLoaderProps {
  onReady: (ws: WorkspaceState) => void;
  onError: (msg: string) => void;
}

const PHASES = [
  "Scanning images",
  "Detecting faces",
  "Generating embeddings",
  "Clustering faces",
  "Merging clusters",
];

export function PipelineLoader({ onReady, onError }: PipelineLoaderProps) {
  const [phase, setPhase] = useState(0);
  const [message, setMessage] = useState("Starting pipeline...");

  useEffect(() => {
    const es = new EventSource(pipelineStatusUrl());
    es.onmessage = (e) => {
      const data: PipelineEvent = JSON.parse(e.data);
      setPhase(data.phase);
      setMessage(data.message);
      if (data.done) {
        es.close();
        setTimeout(() => {
          fetchWorkspace()
            .then(onReady)
            .catch((err) => onError(String(err)));
        }, 200);
      }
      if (data.error) {
        es.close();
        onError(data.message);
      }
    };
    es.onerror = () => {
      es.close();
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
        {/* Spinner */}
        <div className="mb-6">
          <div className="inline-block w-16 h-16 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin" />
        </div>

        <h2 className="text-xl font-semibold text-gray-800 mb-2">
          Processing Photos
        </h2>
        <p className="text-sm text-gray-500 mb-4">{message}</p>

        {/* Progress bar */}
        <div className="w-full bg-gray-200 rounded-full h-2 mb-3 overflow-hidden">
          <div
            className="bg-blue-600 h-2 rounded-full transition-all duration-500 ease-out"
            style={{ width: `${progress}%` }}
          />
        </div>

        {/* Phase labels */}
        <div className="flex justify-between text-xs text-gray-400">
          {PHASES.map((label, i) => (
            <span
              key={i}
              className={`transition-colors duration-300 ${
                i < phase ? "text-blue-600 font-medium" : ""
              } ${i === phase ? "text-blue-500" : ""}`}
            >
              {label}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
