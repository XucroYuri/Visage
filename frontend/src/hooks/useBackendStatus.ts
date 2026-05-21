import { useEffect, useRef, useState } from "react";

export type BackendStatus = "starting" | "ready" | "error";

const HEALTH_URL = "/api/health";
const POLL_INTERVAL = 2000; // 2 seconds
const MAX_RETRIES = 30; // 60 seconds total before declaring error

interface UseBackendStatusResult {
  status: BackendStatus;
  errorMessage: string | null;
  retryCount: number;
}

/**
 * Polls the backend health endpoint to determine when the Python engine is ready.
 *
 * States:
 *  - "starting": polling health endpoint, engine not yet ready
 *  - "ready": health endpoint responds with 200
 *  - "error": max retries exceeded or network error
 */
export function useBackendStatus(): UseBackendStatusResult {
  const [status, setStatus] = useState<BackendStatus>("starting");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const retryRef = useRef(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    retryRef.current = 0;
    setStatus("starting");
    setErrorMessage(null);

    function poll() {
      fetch(HEALTH_URL, { method: "GET", cache: "no-store" })
        .then((res) => {
          if (res.ok) {
            setStatus("ready");
            setErrorMessage(null);
            return;
          }
          throw new Error(`Health check returned ${res.status}`);
        })
        .catch((err: Error) => {
          retryRef.current += 1;
          setRetryCount(retryRef.current);

          if (retryRef.current >= MAX_RETRIES) {
            setStatus("error");
            setErrorMessage(
              `Backend did not start after ${(MAX_RETRIES * POLL_INTERVAL) / 1000} seconds. ${err.message || ""}`,
            );
            return;
          }

          // Schedule next poll
          timerRef.current = setTimeout(poll, POLL_INTERVAL);
        });
    }

    // Start polling
    poll();

    return () => {
      if (timerRef.current !== null) {
        clearTimeout(timerRef.current);
      }
    };
  }, []);

  return { status, errorMessage, retryCount };
}
