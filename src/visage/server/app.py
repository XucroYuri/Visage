"""FastAPI application for the Visage review UI."""

from __future__ import annotations

import json
import logging
import queue
import time
import webbrowser
from collections.abc import Generator
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from visage.backends import get_backend
from visage.cache import EmbeddingCache
from visage.cluster import (
    cluster_faces,
    extract_embeddings,
    merge_clusters,
)
from visage.config import VisageConfig
from visage.detector import detect_faces_batch
from visage.detectors import get_detector
from visage.embedder import generate_embeddings_batch
from visage.scanner import scan_images

from .routes import router
from .routes_active import router as active_router
from .routes_events import router as events_router
from .routes_search import router as search_router
from .workspace import Workspace

logger = logging.getLogger(__name__)


def _run_pipeline(
    input_dir: str,
    config: VisageConfig,
    progress_queue: queue.Queue | None = None,
) -> Workspace:
    """Run the full face clustering pipeline and return a Workspace.

    If progress_queue is provided, emits phase updates as {"phase": int, "message": str}.
    """
    cache = EmbeddingCache(input_dir)

    def _emit(phase: int, message: str, **extra: object) -> None:
        if progress_queue is not None:
            progress_queue.put({"phase": phase, "message": message, **extra})

    # Phase 1: Scan
    _emit(1, "Scanning images...")
    t0 = time.time()
    image_paths = scan_images(input_dir)
    total_images = len(image_paths)
    _emit(1, f"Found {total_images} images", count=total_images, elapsed=time.time() - t0)

    if total_images == 0:
        raise ValueError(f"No images found in {input_dir}")

    # Phase 2: Detect faces
    _emit(2, "Detecting faces...")
    t0 = time.time()
    detector = get_detector(
        config.detection_backend,
        min_confidence=config.detection_confidence,
        min_face_size=config.min_face_size,
    )
    image_results, _detection_stats = detect_faces_batch(
        image_paths,
        min_confidence=config.detection_confidence,
        min_face_size=config.min_face_size,
        max_workers=config.max_workers,
        detector=detector,
    )
    images_with_faces = sum(1 for r in image_results if r.faces and not r.error)
    _emit(
        2,
        f"Detected faces in {images_with_faces} images",
        count=images_with_faces,
        elapsed=time.time() - t0,
    )

    if images_with_faces == 0:
        raise ValueError("No faces detected in any images")

    # Phase 3: Generate embeddings
    _emit(3, "Generating embeddings...")
    t0 = time.time()
    backend = get_backend(
        config.embedding_backend,
        model=config.embedding_model,
        num_jitters=config.num_jitters,
    )
    image_results, cache_hits = generate_embeddings_batch(
        image_results,
        model=config.embedding_model,
        num_jitters=config.num_jitters,
        cache=cache,
        backend=backend,
        min_face_quality=config.min_face_quality,
    )
    _emit(3, "Embeddings generated", cache_hits=cache_hits, elapsed=time.time() - t0)

    # Phase 4: Cluster
    _emit(4, "Clustering faces...")
    t0 = time.time()
    embeddings, face_to_image = extract_embeddings(
        image_results, embedding_dim=backend.embedding_dim,
    )

    if len(embeddings) == 0:
        raise ValueError("No embeddings generated")

    cluster_result = cluster_faces(
        embeddings,
        eps=config.dbscan_eps,
        min_samples=config.dbscan_min_samples,
        auto_eps=config.auto_eps,
        cluster_method=config.cluster_method,
        min_cluster_size=config.hdbscan_min_cluster_size,
        cluster_selection_epsilon=config.cluster_selection_epsilon,
        cluster_selection_method=config.cluster_selection_method,
    )
    _emit(
        4,
        f"Found {cluster_result.num_clusters} clusters",
        count=cluster_result.num_clusters,
        elapsed=time.time() - t0,
    )

    # Phase 5: Post-clustering merge
    if config.merge_threshold > 0.0:
        _emit(5, "Merging similar clusters...")
        t0 = time.time()
        cluster_result = merge_clusters(
            cluster_result,
            merge_threshold=config.merge_threshold,
            min_reliable_size=config.min_reliable_size,
            small_merge_threshold=config.small_merge_threshold,
        )
        _emit(5, "Merge complete", elapsed=time.time() - t0)

    cache.close()

    ws = Workspace(
        input_dir=input_dir,
        config=config,
        image_results=image_results,
        cluster_result=cluster_result,
        face_to_image=face_to_image,
    )
    _emit(5, "Pipeline complete", done=True, clusters=len(ws.cluster_ids), noise=ws.num_noise_faces)
    return ws


def create_app(input_dir: str, config: VisageConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    The pipeline runs in a background thread so the server starts immediately.
    Frontend can poll /api/pipeline-status for progress.

    Args:
        input_dir: Path to the photo directory.
        config: VisageConfig (uses defaults if None).

    Returns:
        Configured FastAPI app.
    """
    cfg = config or VisageConfig()

    app = FastAPI(
        title="Visage Review",
        description="Manual face clustering review UI",
        version="0.1.0",
    )

    # Allow CORS for dev mode (Vite on port 5173)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Pipeline state
    app.state.workspace = None
    app.state.input_dir = input_dir
    app.state.pipeline_error = None
    progress_queue: queue.Queue = queue.Queue()
    app.state.progress_queue = progress_queue

    # Run pipeline in background thread
    def _background_pipeline() -> None:
        try:
            ws = _run_pipeline(input_dir, cfg, progress_queue)
            app.state.workspace = ws
            logger.info(
                "Pipeline complete: %d clusters, %d noise faces",
                len(ws.cluster_ids), ws.num_noise_faces,
            )
        except Exception as exc:
            app.state.pipeline_error = str(exc)
            progress_queue.put({"phase": -1, "message": str(exc), "error": True})
            logger.error("Pipeline failed: %s", exc)

    import threading
    thread = threading.Thread(target=_background_pipeline, daemon=True)
    thread.start()

    # Health check endpoint (used by Tauri sidecar and useBackendStatus hook)
    @app.get("/api/health")
    def health_check() -> dict[str, str]:
        """Return health status for sidecar process monitoring."""
        status = "ok" if app.state.workspace is not None else "starting"
        return {"status": status}

    # SSE endpoint for pipeline progress
    @app.get("/api/pipeline-status")
    def pipeline_status(request: Request) -> StreamingResponse:
        """Stream pipeline progress via Server-Sent Events."""
        def event_stream() -> Generator[str, None, None]:
            q: queue.Queue = request.app.state.progress_queue
            # Flush any already-queued events (pipeline may have finished
            # before this SSE client connected)
            while True:
                try:
                    data = q.get_nowait()
                    yield f"data: {json.dumps(data)}\n\n"
                    if data.get("done") or data.get("error"):
                        return
                except queue.Empty:
                    break

            # If workspace already loaded, send done immediately
            if request.app.state.workspace is not None:
                done_msg = json.dumps(
                    {"phase": 5, "message": "Pipeline complete", "done": True}
                )
                yield f"data: {done_msg}\n\n"
                return

            # Wait for live events
            while True:
                try:
                    data = q.get(timeout=30)
                    yield f"data: {json.dumps(data)}\n\n"
                    if data.get("done") or data.get("error"):
                        break
                except queue.Empty:
                    # Keep-alive
                    yield ": keepalive\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Register API routes
    app.include_router(router)
    app.include_router(events_router)
    app.include_router(search_router)
    app.include_router(active_router)

    # Serve React static files (production build)
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists() and (static_dir / "index.html").exists():
        # Mount static assets directory (JS, CSS bundles)
        assets_dir = static_dir / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        # Root path serves index.html directly
        @app.get("/", include_in_schema=False)
        async def serve_root():
            return FileResponse(str(static_dir / "index.html"))

        # SPA fallback: serve index.html for all non-API, non-asset routes.
        # API routes (/api/*) take priority because they are registered earlier
        # and are more specific than the {full_path:path} catch-all.
        @app.get("/{full_path:path}", include_in_schema=False)
        async def serve_spa(full_path: str):
            # Prevent path traversal: resolve and validate within static dir
            resolved = (static_dir / full_path).resolve()
            try:
                resolved.relative_to(static_dir.resolve())
            except ValueError:
                raise HTTPException(status_code=403) from None
            if resolved.exists() and resolved.is_file():
                return FileResponse(str(resolved))
            # Everything else → SPA index.html
            return FileResponse(str(static_dir / "index.html"))

    return app


def serve(
    input_dir: str,
    config: VisageConfig | None = None,
    port: int = 8787,
    open_browser: bool = True,
) -> None:
    """Run the Visage review server and optionally open the browser.

    Args:
        input_dir: Path to the photo directory.
        config: VisageConfig (uses defaults if None).
        port: Port to listen on (default 8787).
        open_browser: Whether to auto-open the browser.
    """
    import uvicorn

    app = create_app(input_dir, config)
    url = f"http://localhost:{port}"

    if open_browser:
        def _open_browser() -> None:
            time.sleep(1.5)
            webbrowser.open(url)

        import threading
        threading.Thread(target=_open_browser, daemon=True).start()

    print(f"\n  Visage Review UI: {url}")
    print("  Press Ctrl+C to stop\n")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
