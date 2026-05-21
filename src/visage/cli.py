from __future__ import annotations

import argparse
import json
import logging
import sys

from . import __version__
from .cache import EmbeddingCache
from .config import build_config
from .pipeline import run_pipeline
from .progress import ProgressDisplay


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visage",
        description="macOS-native face clustering and photo sorting tool.",
    )
    parser.add_argument("input", help="Input folder containing photos")
    parser.add_argument("--version", action="version", version=f"visage {__version__}")

    # Output options
    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "-o", "--output-dir", default=None,
        help="Output directory (default: <input>/visage_output)",
    )
    output_group.add_argument(
        "--move", action="store_true", default=False,
        help="Move files instead of copying (default: copy)",
    )
    output_group.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Show organization plan without modifying files",
    )

    # Detection options
    detect_group = parser.add_argument_group("detection")
    detect_group.add_argument(
        "--detection-backend", choices=["auto", "vision", "scrfd", "yunet"], default=None,
        help="Face detection backend (default: auto — macOS: Vision, other: SCRFD→YuNet)",
    )
    detect_group.add_argument(
        "--min-confidence", type=float, default=None,
        help="Minimum face detection confidence (default: 0.5)",
    )
    detect_group.add_argument(
        "--max-workers", type=int, default=None,
        help="Max parallel detection workers (default: 4)",
    )

    # Embedding options
    embed_group = parser.add_argument_group("embedding")
    embed_group.add_argument(
        "--backend", choices=["dlib", "insightface"], default=None,
        help="Embedding backend (default: dlib)",
    )
    embed_group.add_argument(
        "--model", choices=["small", "large"], default=None,
        help="Face embedding model size (default: small, dlib only)",
    )
    embed_group.add_argument(
        "--num-jitters", type=int, default=None,
        help="Re-sample count for embeddings (default: 1, dlib only)",
    )

    # Quality options
    quality_group = parser.add_argument_group("quality")
    quality_group.add_argument(
        "--min-quality", type=float, default=None,
        help="Minimum face quality score 0-1 (default: 0, no filtering)",
    )

    # Clustering options
    cluster_group = parser.add_argument_group("clustering")
    cluster_group.add_argument(
        "--cluster-method", choices=["dbscan", "hdbscan"], default=None,
        help="Clustering algorithm (default: hdbscan)",
    )
    cluster_group.add_argument(
        "--eps", type=float, default=None,
        help="DBSCAN epsilon threshold (default: 0.5, DBSCAN only)",
    )
    cluster_group.add_argument(
        "--min-samples", type=int, default=None,
        help="Minimum samples per cluster (default: 2, also used as HDBSCAN min_samples)",
    )
    cluster_group.add_argument(
        "--auto-eps", action="store_true", default=False,
        help="Automatically estimate eps using k-distance elbow method (DBSCAN only)",
    )
    cluster_group.add_argument(
        "--min-cluster-size", type=int, default=None,
        help="Minimum cluster size for HDBSCAN (default: 2)",
    )
    cluster_group.add_argument(
        "--cluster-selection-epsilon", type=float, default=None,
        help="HDBSCAN cluster selection epsilon (default: 0, disabled; >0 may hit sklearn bug)",
    )
    cluster_group.add_argument(
        "--head-feature-weight", type=float, default=None,
        help="Weight for head features in clustering 0-1 (default: 0.2)",
    )
    cluster_group.add_argument(
        "--cluster-selection-method", choices=["eom", "leaf"], default=None,
        help="HDBSCAN cluster selection method (default: eom)",
    )
    cluster_group.add_argument(
        "--merge-threshold", type=float, default=None,
        help="Cosine similarity threshold for merging clusters 0-1 (default: 0.85)",
    )
    cluster_group.add_argument(
        "--small-merge-threshold", type=float, default=None,
        help="Relaxed merge threshold for small clusters 0-1 (default: 0.75)",
    )
    cluster_group.add_argument(
        "--min-reliable-size", type=int, default=None,
        help="Clusters below this size use relaxed threshold (default: 10)",
    )

    # Include options
    include_group = parser.add_argument_group("include")
    include_group.add_argument(
        "--include-unclustered", action="store_true", default=False,
        help="Include _unclustered folder for unmatched faces",
    )
    include_group.add_argument(
        "--include-no-faces", action="store_true", default=False,
        help="Include _no_faces folder for images without faces",
    )

    # Display options
    display_group = parser.add_argument_group("display")
    display_group.add_argument(
        "--json", action="store_true", default=False,
        help="Output results as JSON",
    )
    display_group.add_argument(
        "-q", "--quiet", action="store_true", default=False,
        help="Suppress progress output",
    )
    display_group.add_argument(
        "-v", "--verbose", action="store_true", default=False,
        help="Show detailed log output (warnings and debug info)",
    )

    # Config file
    parser.add_argument(
        "--config", default=None,
        help="Path to TOML config file",
    )

    # Serve mode (web UI)
    serve_group = parser.add_argument_group("serve")
    serve_group.add_argument(
        "--serve", action="store_true", default=False,
        help="Start web review UI instead of batch processing",
    )
    serve_group.add_argument(
        "--port", type=int, default=8787,
        help="Port for the review web server (default: 8787)",
    )
    serve_group.add_argument(
        "--no-open", action="store_true", default=False,
        help="Don't auto-open browser on serve",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for visage."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Configure logging based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )

    # Build config from file + CLI overrides
    overrides = {
        "copy_mode": not args.move,
        "detection_backend": args.detection_backend,
        "detection_confidence": args.min_confidence,
        "embedding_backend": args.backend,
        "embedding_model": args.model,
        "num_jitters": args.num_jitters,
        "min_face_quality": args.min_quality,
        "cluster_method": args.cluster_method,
        "dbscan_eps": args.eps,
        "dbscan_min_samples": args.min_samples,
        "auto_eps": args.auto_eps,
        "hdbscan_min_cluster_size": args.min_cluster_size,
        "cluster_selection_epsilon": args.cluster_selection_epsilon,
        "cluster_selection_method": args.cluster_selection_method,
        "head_feature_weight": args.head_feature_weight,
        "merge_threshold": args.merge_threshold,
        "small_merge_threshold": args.small_merge_threshold,
        "min_reliable_size": args.min_reliable_size,
        "max_workers": args.max_workers,
        "include_unclustered": args.include_unclustered,
        "include_no_faces": args.include_no_faces,
    }

    try:
        config = build_config(
            config_file=args.config,
            input_dir=args.input,
            overrides=overrides,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # ── Serve mode: start web review UI ──────────────────────────
    if args.serve:
        try:
            from .server.app import serve
        except ImportError as exc:
            print(
                f"Error: Web UI dependencies not installed. "
                f"Run: pip install fastapi uvicorn\n"
                f"Details: {exc}",
                file=sys.stderr,
            )
            return 1
        serve(args.input, config=config, port=args.port,
              open_browser=not args.no_open)
        return 0

    progress = ProgressDisplay(quiet=args.quiet)

    # Create a single cache instance for both checkpoint display and pipeline
    cache = EmbeddingCache(args.input)
    checkpoint = cache.load_checkpoint()
    if checkpoint is not None:
        phase = checkpoint.get("phase", 0)
        msg = checkpoint.get("message", "")
        cached_images = checkpoint.get("cached_images", 0)
        cached_faces = checkpoint.get("cached_faces", 0)
        progress._print(
            f"  Resuming from phase {phase}/5 — {msg}"
            f" ({cached_images} images, {cached_faces} faces cached)"
        )

    try:
        result = run_pipeline(
            input_path=args.input,
            config=config,
            dry_run=args.dry_run,
            output_dir=args.output_dir,
            progress=progress,
            cache=cache,
        )
    except KeyboardInterrupt:
        progress.error("Interrupted by user")
        return 130
    except Exception as exc:
        progress.error(f"Fatal error: {exc}")
        cache.close()
        return 1
    finally:
        cache.close()

    # Output results
    if args.json:
        output = {
            "total_images": result.total_images,
            "images_with_faces": result.images_with_faces,
            "total_faces": result.total_faces,
            "num_clusters": result.num_clusters,
            "num_noise_faces": result.num_noise_faces,
            "duration_seconds": round(result.duration_seconds, 2),
            "phase_durations": {
                k: round(v, 3) for k, v in result.phase_durations.items()
            },
            "errors": result.errors,
        }
        if result.organize_plan:
            output["persons"] = {
                f"person_{k:02d}": {
                    "photos": len(v),
                    "confidence": round(
                        result.cluster_confidences.get(k, 0.0), 3
                    ),
                }
                for k, v in sorted(
                    result.organize_plan.person_folders.items()
                )
            }
        print(json.dumps(output, indent=2))
    else:
        # Print summary
        action = "copied" if config.copy_mode else "moved"
        summary = (
            f"\nDone in {result.duration_seconds:.1f}s\n"
            f"  Images scanned:   {result.total_images}\n"
            f"  With faces:       {result.images_with_faces}\n"
            f"  Faces detected:   {result.total_faces}\n"
            f"  People found:     {result.num_clusters}\n"
            f"  Unclustered:      {result.num_noise_faces} faces\n"
        )
        if not args.dry_run and result.organize_plan:
            total_files = sum(
                len(v) for v in result.organize_plan.person_folders.values()
            )
            summary += f"  Files {action}:  {total_files}\n"
        if result.phase_durations:
            summary += "\n  Phase timings:\n"
            for phase_name, dur in result.phase_durations.items():
                summary += f"    {phase_name}: {dur:.2f}s\n"
        if result.organize_plan and result.cluster_confidences:
            summary += "\n  Per-person confidence:\n"
            for cid, paths in sorted(
                result.organize_plan.person_folders.items()
            ):
                conf = result.cluster_confidences.get(cid, 0.0)
                summary += f"    person_{cid:02d}: {conf:.2f} ({len(paths)} photos)\n"
        if result.errors:
            summary += f"  Errors:           {len(result.errors)}\n"

        progress.finish(summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
