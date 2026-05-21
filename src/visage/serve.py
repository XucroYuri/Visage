"""Visage — sidecar entry point for Tauri (production).

Wraps the FastAPI server with signal handling for graceful shutdown
when launched as a Tauri sidecar process.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys

from visage.config import VisageConfig, build_config
from visage.server.app import create_app

logger = logging.getLogger(__name__)

_shutdown_requested = False


def _handle_sigterm(signum: int, _frame: object) -> None:
    """Handle SIGTERM for graceful shutdown."""
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("Received signal %d, shutting down gracefully...", signum)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visage engine (sidecar)")
    parser.add_argument("input_dir", nargs="?", default=None, help="Photo directory")
    parser.add_argument("--port", type=int, default=8787, help="Server port")
    parser.add_argument("--no-open", action="store_true", default=True,
                        help="Don't open browser")
    parser.add_argument("--config", default=None, help="Config file path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, _handle_sigterm)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, _handle_sigterm)

    config: VisageConfig | None = None
    if args.input_dir:
        config = build_config(
            config_file=args.config,
            input_dir=args.input_dir,
        )

    import uvicorn

    app = create_app(args.input_dir or ".", config=config)

    # Patch the app state to include shutdown flag
    app.state._shutdown_requested = lambda: _shutdown_requested  # noqa: SLF001

    log_level = "info"
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=args.port,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
