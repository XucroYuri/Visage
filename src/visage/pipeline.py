from __future__ import annotations

import time
from typing import Optional

from .cache import EmbeddingCache
from .cluster import build_cluster_mapping, cluster_faces, compute_cluster_confidences, extract_embeddings
from .config import DEFAULT_OUTPUT_DIRNAME, VisageConfig
from .detector import detect_faces_batch
from .embedder import generate_embeddings_batch
from .models import OrganizePlan, PipelineResult
from .organizer import build_organize_plan, execute_organize_plan
from .progress import ProgressDisplay
from .scanner import scan_images


def run_pipeline(
    input_path: str,
    config: Optional[VisageConfig] = None,
    dry_run: bool = False,
    output_dir: Optional[str] = None,
    progress: Optional[ProgressDisplay] = None,
) -> PipelineResult:
    """Execute the full face clustering pipeline.

    Phase 1: Scan for images
    Phase 2: Detect faces (Vision framework)
    Phase 3: Generate embeddings (face_recognition)
    Phase 4: Cluster (DBSCAN)
    Phase 5: Organize into folders

    Args:
        input_path: Root folder containing photos.
        config: Configuration (uses defaults if None).
        dry_run: If True, show plan without copying files.
        output_dir: Override output directory.
        progress: Progress display instance.

    Returns:
        PipelineResult with full statistics and plan.
    """
    cfg = config or VisageConfig()
    prog = progress or ProgressDisplay()
    errors: list[str] = []
    start_time = time.time()
    cache = EmbeddingCache(input_path)

    # ── Phase 1: Scan ──────────────────────────────────────────────
    prog.update("1/5 Scan", 0, 1)
    try:
        image_paths = scan_images(input_path)
    except ValueError as exc:
        prog.error(str(exc))
        return PipelineResult(
            total_images=0, images_with_faces=0, total_faces=0,
            num_clusters=0, num_noise_faces=0,
            errors=[str(exc)], duration_seconds=time.time() - start_time,
        )

    total_images = len(image_paths)
    prog.finish_phase("1/5 Scan", f"Found {total_images} images")
    cache.save_checkpoint(1, message=f"Scanned {total_images} images")

    if total_images == 0:
        return PipelineResult(
            total_images=0, images_with_faces=0, total_faces=0,
            num_clusters=0, num_noise_faces=0,
            errors=["No images found"], duration_seconds=time.time() - start_time,
        )

    # ── Phase 2: Detect faces ──────────────────────────────────────
    def detection_progress(completed: int, total: int) -> None:
        prog.update("2/5 Detection", completed, total)

    image_results = detect_faces_batch(
        image_paths,
        min_confidence=cfg.detection_confidence,
        min_face_size=cfg.min_face_size,
        max_workers=cfg.max_workers,
        progress_callback=detection_progress,
    )

    images_with_faces = sum(1 for r in image_results if r.faces and not r.error)
    total_faces = sum(len(r.faces) for r in image_results)
    detection_errors = sum(1 for r in image_results if r.error)
    errors.extend(r.error for r in image_results if r.error)

    prog.finish_phase(
        "2/5 Detection",
        f"{images_with_faces} images with faces, {total_faces} faces detected"
        + (f", {detection_errors} errors" if detection_errors else ""),
    )
    cache.save_checkpoint(2, message=f"{total_faces} faces in {images_with_faces} images")

    if images_with_faces == 0:
        return PipelineResult(
            total_images=total_images, images_with_faces=0, total_faces=0,
            num_clusters=0, num_noise_faces=0, errors=errors,
            duration_seconds=time.time() - start_time,
        )

    # ── Phase 3: Generate embeddings ───────────────────────────────
    def embedding_progress(completed: int, total: int) -> None:
        prog.update("3/5 Embedding", completed, total)

    image_results, cache_hits = generate_embeddings_batch(
        image_results,
        model=cfg.embedding_model,
        num_jitters=cfg.num_jitters,
        progress_callback=embedding_progress,
        cache=cache,
    )

    faces_with_embeddings = sum(
        len(r.faces) for r in image_results if r.faces
    )
    cache_msg = f", {cache_hits} from cache" if cache_hits > 0 else ""
    prog.finish_phase("3/5 Embedding", f"{faces_with_embeddings} faces encoded{cache_msg}")
    cache.save_checkpoint(3, message=f"{faces_with_embeddings} embeddings ({cache_hits} cached)")

    # ── Phase 4: Cluster ───────────────────────────────────────────
    prog.update("4/5 Clustering", 0, 1)

    embeddings, face_to_image = extract_embeddings(image_results)

    if len(embeddings) == 0:
        prog.finish_phase("4/5 Clustering", "No embeddings to cluster")
        return PipelineResult(
            total_images=total_images, images_with_faces=images_with_faces,
            total_faces=total_faces, num_clusters=0, num_noise_faces=0,
            errors=errors, duration_seconds=time.time() - start_time,
        )

    cluster_result = cluster_faces(
        embeddings,
        eps=cfg.dbscan_eps,
        min_samples=cfg.dbscan_min_samples,
        auto_eps=cfg.auto_eps,
    )

    cluster_mapping = build_cluster_mapping(cluster_result, face_to_image)
    cluster_confidences = compute_cluster_confidences(cluster_result)

    prog.finish_phase(
        "4/5 Clustering",
        f"{cluster_result.num_clusters} people identified, "
        f"{cluster_result.num_noise} unclustered faces",
    )
    cache.save_checkpoint(4, message=f"{cluster_result.num_clusters} clusters found")

    # ── Phase 5: Organize ──────────────────────────────────────────
    # Determine output directory
    if output_dir:
        out = output_dir
    elif cfg.output_dir:
        out = cfg.output_dir
    else:
        out = f"{input_path.rstrip('/')}/{DEFAULT_OUTPUT_DIRNAME}"

    plan = build_organize_plan(
        image_results,
        cluster_mapping,
        folder_prefix=cfg.folder_prefix,
        include_unclustered=cfg.include_unclustered,
        include_no_faces=cfg.include_no_faces,
    )

    if dry_run:
        prog.finish_phase("5/5 Organize", "dry-run: no files modified")
        _print_dry_run_plan(plan, out, prog, cluster_confidences)
    else:
        def organize_progress(completed: int, total: int, current: str) -> None:
            prog.update("5/5 Organize", completed, total)

        stats = execute_organize_plan(
            plan,
            output_dir=out,
            folder_prefix=cfg.folder_prefix,
            copy_mode=cfg.copy_mode,
            dry_run=dry_run,
            progress_callback=organize_progress,
        )

        action = "copied" if cfg.copy_mode else "moved"
        prog.finish_phase(
            "5/5 Organize",
            f"{stats.get(action, 0)} files {action} to {out}",
        )

    duration = time.time() - start_time

    cache.clear_checkpoint()
    return PipelineResult(
        total_images=total_images,
        images_with_faces=images_with_faces,
        total_faces=total_faces,
        num_clusters=cluster_result.num_clusters,
        num_noise_faces=cluster_result.num_noise,
        organize_plan=plan,
        cluster_confidences=cluster_confidences,
        duration_seconds=duration,
        errors=errors,
    )


def _print_dry_run_plan(
    plan: OrganizePlan,
    output_dir: str,
    prog: ProgressDisplay,
    cluster_confidences: dict[int, float] | None = None,
) -> None:
    """Print the organize plan for dry-run mode."""
    lines = ["\nDRY RUN — would organize as follows:", f"Output: {output_dir}", ""]

    for cluster_id, paths in sorted(plan.person_folders.items()):
        conf = cluster_confidences.get(cluster_id) if cluster_confidences else None
        conf_str = f", confidence: {conf:.2f}" if conf is not None else ""
        lines.append(f"  person_{cluster_id:02d}/ ({len(paths)} photos{conf_str})")
        for p in paths[:5]:
            lines.append(f"    {p}")
        if len(paths) > 5:
            lines.append(f"    ... and {len(paths) - 5} more")

    if plan.unclustered:
        lines.append(f"\n  _unclustered/ ({len(plan.unclustered)} photos)")
    if plan.no_faces:
        lines.append(f"  _no_faces/ ({len(plan.no_faces)} photos)")

    prog.print_plan("\n".join(lines))
