"""Embedding service — persistent HTTP server for embedding generation.

Runs as a standalone process, accepts HTTP requests for embedding generation,
supports GPU acceleration, request queuing, batch optimization, and backend hot-swap.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from visage.models import FaceBox

from .backend import EmbeddingBackend, create_backend
from .batcher import RequestBatcher
from .gpu import DeviceInfo, detect_device

logger = logging.getLogger(__name__)

# ── Global service state ──────────────────────────────────────────

_backend: EmbeddingBackend | None = None
_batcher: RequestBatcher | None = None
_device: DeviceInfo | None = None
_processing_task: asyncio.Task[Any] | None = None
_shutdown_event = asyncio.Event()


class EmbedRequest(BaseModel):
    """HTTP request body for embedding generation."""

    face_id: str = ""
    image_b64: str = ""  # Base64-encoded image
    bbox: list[float]  # [top, right, bottom, left]
    priority: str = "low"  # "high" or "low"


class EmbedResponse(BaseModel):
    """HTTP response for embedding generation."""

    face_id: str
    embedding: list[float] | None = None
    dim: int = 0
    error: str | None = None
    elapsed_ms: float = 0.0


class StatusResponse(BaseModel):
    """Service status."""

    status: str  # "ready", "starting", "error"
    backend: str = ""
    device: str = ""
    embedding_dim: int = 0
    uptime_seconds: float = 0.0
    pending_requests: int = 0


class HotSwapRequest(BaseModel):
    """Request to switch embedding backend at runtime."""

    backend: str  # "dlib" or "insightface"


_start_time = time.time()


def _get_backend() -> EmbeddingBackend:
    if _backend is None:
        raise RuntimeError("Embedding service not initialized")
    return _backend


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage service lifecycle: start processor, handle shutdown."""
    global _processing_task
    _processing_task = asyncio.create_task(_batch_processor())
    logger.info("Embedding service started on device=%s backend=%s",
                _device.device if _device else "unknown",
                _backend.name if _backend else "none")
    yield
    _shutdown_event.set()
    if _processing_task:
        _processing_task.cancel()
        try:
            await _processing_task
        except asyncio.CancelledError:
            pass
    logger.info("Embedding service shut down")


app = FastAPI(title="Visage Embedding Service", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    be = _backend
    return StatusResponse(
        status="ready" if be else "starting",
        backend=be.name if be else "",
        device=_device.device if _device else "",
        embedding_dim=be.embedding_dim if be else 0,
        uptime_seconds=time.time() - _start_time,
        pending_requests=_batcher.pending_count if _batcher else 0,
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest) -> EmbedResponse:
    """Generate embedding for a single face."""
    if not _backend or not _batcher:
        raise HTTPException(503, "Service not ready")

    t0 = time.time()
    try:
        image = _decode_image(req.image_b64) if req.image_b64 else None
        if image is None:
            return EmbedResponse(face_id=req.face_id, error="No image provided")

        face_box = FaceBox(
            top=int(req.bbox[0]),
            right=int(req.bbox[1]),
            bottom=int(req.bbox[2]),
            left=int(req.bbox[3]),
        )

        # For now, generate synchronously (batching in future iteration)
        emb = _backend.generate(image, face_box)
        elapsed = (time.time() - t0) * 1000

        return EmbedResponse(
            face_id=req.face_id,
            embedding=emb.tolist() if emb is not None else None,
            dim=len(emb) if emb is not None else 0,
            elapsed_ms=elapsed,
        )
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        logger.error("Embedding generation failed: %s", e, exc_info=True)
        return EmbedResponse(face_id=req.face_id, error=str(e), elapsed_ms=elapsed)


@app.post("/embed/batch", response_model=list[EmbedResponse])
async def embed_batch(requests: list[EmbedRequest]) -> list[EmbedResponse]:
    """Generate embeddings for multiple faces."""
    if not _backend:
        raise HTTPException(503, "Service not ready")

    results: list[EmbedResponse] = []
    for req in requests:
        results.append(await embed(req))
    return results


@app.post("/hotswap")
async def hotswap(req: HotSwapRequest) -> dict[str, str]:
    """Switch embedding backend at runtime without dropping requests."""
    global _backend
    if not _backend:
        raise HTTPException(503, "Service not ready")

    try:
        new_backend = create_backend(req.backend)
        if not new_backend.is_available():
            raise HTTPException(400, f"Backend {req.backend!r} not available")

        old_name = _backend.name
        _backend = new_backend
        logger.info("Hot-swapped backend: %s -> %s", old_name, new_backend.name)
        return {"status": "ok", "previous": old_name, "current": new_backend.name}
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


async def _batch_processor() -> None:
    """Background task that processes batched requests."""
    while not _shutdown_event.is_set():
        if _batcher and _batcher.pending_count > 0:
            batch = _batcher.drain()
            if batch and _backend:
                for req in batch:
                    try:
                        if req.image is not None and req.face_box is not None:
                            emb = _backend.generate(req.image, req.face_box)
                            if not req._future.done():  # noqa: SLF001
                                req._future.set_result(emb)  # noqa: SLF001
                        else:
                            if not req._future.done():  # noqa: SLF001
                                req._future.set_result(None)
                    except Exception as e:
                        if not req._future.done():  # noqa: SLF001
                            req._future.set_exception(e)
        await asyncio.sleep(0.05)  # 50ms poll interval


def _decode_image(b64: str) -> np.ndarray | None:
    """Decode base64 image to RGB numpy array."""
    import base64
    import io

    from PIL import Image

    try:
        data = base64.b64decode(b64)
        img = Image.open(io.BytesIO(data))
        return np.array(img.convert("RGB"))
    except Exception:
        logger.warning("Failed to decode base64 image", exc_info=True)
        return None


def init_service(backend_name: str = "insightface", device_preference: str | None = None) -> None:
    """Initialize global service state (called before uvicorn.run)."""
    global _backend, _batcher, _device

    _device = detect_device(prefer=device_preference)
    logger.info("Detected device: %s (%s)", _device.device, _device.name)

    _backend = create_backend(backend_name)
    if not _backend.is_available():
        logger.warning("Backend %r not available, falling back to dlib", backend_name)
        _backend = create_backend("dlib")

    _batcher = RequestBatcher()


def run_server(host: str = "127.0.0.1", port: int = 0, **kwargs: Any) -> None:
    """Entry point for the embedding service process."""
    import uvicorn

    init_service(
        backend_name=kwargs.get("backend", "insightface"),
        device_preference=kwargs.get("device"),
    )

    uvicorn_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(uvicorn_config)

    # Signal handling
    loop = asyncio.new_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, server.should_exit.set, True)

    loop.run_until_complete(server.serve())


def main() -> None:
    """CLI entry point for visage-engine."""
    parser = argparse.ArgumentParser(description="Visage embedding service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--backend", default="insightface", choices=["insightface", "dlib"])
    parser.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )

    run_server(host=args.host, port=args.port, backend=args.backend, device=args.device)


if __name__ == "__main__":
    main()
