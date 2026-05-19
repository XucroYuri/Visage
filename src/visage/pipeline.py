from __future__ import annotations

import time

import numpy as np

from .backends import get_backend
from .cache import EmbeddingCache
from .cluster import (
    _normalize_embeddings,
    build_cluster_mapping,
    cluster_faces,
    compute_cluster_confidences,
    compute_composite_distance,
    extract_embeddings,
    merge_clusters,
)
from .config import DEFAULT_OUTPUT_DIRNAME, VisageConfig
from .detector import detect_faces_batch
from .embedder import generate_embeddings_batch
from .head_features import FEATURE_DIM
from .models import OrganizePlan, PipelineResult
from .organizer import build_organize_plan, execute_organize_plan
from .progress import ProgressDisplay
from .scanner import scan_images


def _extract_head_features(
    image_results: list,
    face_to_image: list[tuple[str, int]],
) -> tuple[np.ndarray | None, np.ndarray]:
    """Extract head feature vectors aligned with face_to_image ordering.

    Returns:
        Tuple of ((N, FEATURE_DIM) array or None, (N,) boolean mask of valid features).
        Missing head features use zero vectors and are marked False in the mask.
    """
    # Build lookup: (path, face_index) -> DetectedFace
    face_lookup: dict[tuple[str, int], object] = {}
    for result in image_results:
        if result.error:
            continue
        for face in result.faces:
            if face.embedding is not None:
                face_lookup[(result.path, face.face_index)] = face

    feats: list[np.ndarray] = []
    valid_mask: list[bool] = []
    for path, face_idx in face_to_image:
        face = face_lookup.get((path, face_idx))
        if face is not None and face.head_features is not None:
            feats.append(face.head_features)
            valid_mask.append(True)
        else:
            feats.append(np.zeros(FEATURE_DIM, dtype=np.float64))
            valid_mask.append(False)

    if not feats:
        return None, np.array([], dtype=bool)
    return np.stack(feats), np.array(valid_mask)


def run_pipeline(
    input_path: str,
    config: VisageConfig | None = None,
    dry_run: bool = False,
    output_dir: str | None = None,
    progress: ProgressDisplay | None = None,
    cache: EmbeddingCache | None = None,
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
        cache: Optional EmbeddingCache (created if not provided).

    Returns:
        PipelineResult with full statistics and plan.
    """
    cfg = config or VisageConfig()
    prog = progress or ProgressDisplay()
    errors: list[str] = []
    phase_durations: dict[str, float] = {}
    start_time = time.time()
    own_cache = cache is None
    if cache is None:
        cache = EmbeddingCache(input_path)

    # ── Phase 1: Scan ──────────────────────────────────────────────
    phase_start = time.time()
    prog.update("1/5 Scan", 0, 1)
    try:
        image_paths = scan_images(input_path)
    except ValueError as exc:
        prog.error(str(exc))
        if own_cache:
            cache.close()
        return PipelineResult(
            total_images=0, images_with_faces=0, total_faces=0,
            num_clusters=0, num_noise_faces=0,
            errors=[str(exc)], duration_seconds=time.time() - start_time,
        )

    total_images = len(image_paths)
    phase_durations["scan"] = time.time() - phase_start
    prog.finish_phase("1/5 Scan", f"Found {total_images} images")
    cache.save_checkpoint(1, message=f"Scanned {total_images} images")

    if total_images == 0:
        if own_cache:
            cache.close()
        return PipelineResult(
            total_images=0, images_with_faces=0, total_faces=0,
            num_clusters=0, num_noise_faces=0,
            errors=["No images found"], duration_seconds=time.time() - start_time,
            phase_durations=phase_durations,
        )

    # ── Phase 2: Detect faces ──────────────────────────────────────
    phase_start = time.time()

    def detection_progress(completed: int, total: int) -> None:
        prog.update("2/5 Detection", completed, total)

    image_results, detection_stats = detect_faces_batch(
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

    # Build detection quality summary
    det_total = detection_stats.get("total", 0)
    det_detail = ""
    if det_total > 0:
        contour_pct = detection_stats.get("contour", 0) / det_total * 100
        shrunk = detection_stats.get("shrunk", 0)
        aspect_sum = detection_stats.get("aspect_ratio_sum", 0.0)
        mean_aspect = aspect_sum / det_total if det_total > 0 else 0.0
        det_detail = (
            f" [{contour_pct:.0f}% contour, shrunk={shrunk}, "
            f"aspect={mean_aspect:.2f}]"
        )

    phase_durations["detection"] = time.time() - phase_start
    prog.finish_phase(
        "2/5 Detection",
        f"{images_with_faces} images with faces, {total_faces} faces detected"
        + det_detail
        + (f", {detection_errors} errors" if detection_errors else ""),
    )
    cache.save_checkpoint(2, message=f"{total_faces} faces in {images_with_faces} images")

    if images_with_faces == 0:
        if own_cache:
            cache.close()
        return PipelineResult(
            total_images=total_images, images_with_faces=0, total_faces=0,
            num_clusters=0, num_noise_faces=0, errors=errors,
            duration_seconds=time.time() - start_time,
            phase_durations=phase_durations,
        )

    # ── Phase 3: Generate embeddings ───────────────────────────────
    phase_start = time.time()

    def embedding_progress(completed: int, total: int) -> None:
        prog.update("3/5 Embedding", completed, total)

    # Create embedding backend
    backend = get_backend(
        cfg.embedding_backend,
        model=cfg.embedding_model,
        num_jitters=cfg.num_jitters,
    )

    image_results, cache_hits = generate_embeddings_batch(
        image_results,
        model=cfg.embedding_model,
        num_jitters=cfg.num_jitters,
        progress_callback=embedding_progress,
        cache=cache,
        backend=backend,
        min_face_quality=cfg.min_face_quality,
    )

    faces_with_embeddings = sum(
        len(r.faces) for r in image_results if r.faces
    )
    cache_msg = f", {cache_hits} from cache" if cache_hits > 0 else ""
    phase_durations["embedding"] = time.time() - phase_start
    prog.finish_phase("3/5 Embedding", f"{faces_with_embeddings} faces encoded{cache_msg}")
    cache.save_checkpoint(3, message=f"{faces_with_embeddings} embeddings ({cache_hits} cached)")

    # ── Phase 4: Cluster ───────────────────────────────────────────
    phase_start = time.time()
    prog.update("4/5 Clustering", 0, 1)

    embeddings, face_to_image = extract_embeddings(
        image_results, embedding_dim=backend.embedding_dim,
    )

    if len(embeddings) == 0:
        phase_durations["clustering"] = time.time() - phase_start
        prog.finish_phase("4/5 Clustering", "No embeddings to cluster")
        if own_cache:
            cache.close()
        return PipelineResult(
            total_images=total_images, images_with_faces=images_with_faces,
            total_faces=total_faces, num_clusters=0, num_noise_faces=0,
            errors=errors, duration_seconds=time.time() - start_time,
            phase_durations=phase_durations,
        )

    # Build composite distance matrix if using head features
    distance_matrix = None
    if cfg.head_feature_weight > 0.0 and len(embeddings) > 1:
        head_result = _extract_head_features(image_results, face_to_image)
        head_feats, head_valid = head_result
        if head_feats is not None and head_feats.shape[1] > 0:
            # L2-normalize embeddings for composite distance
            normed = _normalize_embeddings(embeddings)
            distance_matrix = compute_composite_distance(
                normed, head_feats, head_weight=cfg.head_feature_weight,
            )
            # Fix zero-vector entries: use face-only distance for pairs
            # where either face has missing head features
            if not head_valid.all():
                face_sim = normed @ normed.T
                face_only_dist = np.clip(1.0 - face_sim, 0.0, 2.0)
                invalid = ~head_valid
                # For any pair where either face is invalid, use face-only distance
                use_face_only = invalid[:, None] | invalid[None, :]
                distance_matrix = np.where(use_face_only, face_only_dist, distance_matrix)

    cluster_result = cluster_faces(
        embeddings,
        eps=cfg.dbscan_eps,
        min_samples=cfg.dbscan_min_samples,
        auto_eps=cfg.auto_eps,
        cluster_method=cfg.cluster_method,
        min_cluster_size=cfg.hdbscan_min_cluster_size,
        cluster_selection_epsilon=cfg.cluster_selection_epsilon,
        cluster_selection_method=cfg.cluster_selection_method,
        distance_matrix=distance_matrix,
    )

    # Post-clustering merge: combine over-segmented clusters
    if cfg.merge_threshold > 0.0:
        cluster_result = merge_clusters(
            cluster_result, merge_threshold=cfg.merge_threshold,
            min_reliable_size=cfg.min_reliable_size,
            small_merge_threshold=cfg.small_merge_threshold,
        )

    cluster_mapping = build_cluster_mapping(cluster_result, face_to_image)
    cluster_confidences = compute_cluster_confidences(cluster_result)

    phase_durations["clustering"] = time.time() - phase_start
    prog.finish_phase(
        "4/5 Clustering",
        f"{cluster_result.num_clusters} people identified, "
        f"{cluster_result.num_noise} unclustered faces",
    )
    cache.save_checkpoint(4, message=f"{cluster_result.num_clusters} clusters found")

    # ── Phase 5: Organize ──────────────────────────────────────────
    phase_start = time.time()
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

    phase_durations["organize"] = time.time() - phase_start
    duration = time.time() - start_time

    cache.clear_checkpoint()
    if own_cache:
        cache.close()

    return PipelineResult(
        total_images=total_images,
        images_with_faces=images_with_faces,
        total_faces=total_faces,
        num_clusters=cluster_result.num_clusters,
        num_noise_faces=cluster_result.num_noise,
        organize_plan=plan,
        cluster_confidences=cluster_confidences,
        duration_seconds=duration,
        phase_durations=phase_durations,
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
