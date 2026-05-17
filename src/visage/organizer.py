from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Callable
from pathlib import Path

from .config import DEFAULT_FOLDER_PREFIX
from .models import ImageResult, OrganizePlan

logger = logging.getLogger(__name__)


def build_organize_plan(
    image_results: list[ImageResult],
    cluster_mapping: dict[int, list[str]],
    folder_prefix: str = DEFAULT_FOLDER_PREFIX,
    include_unclustered: bool = False,
    include_no_faces: bool = False,
) -> OrganizePlan:
    """Build a plan for organizing files based on clustering results.

    Does not touch the filesystem. Used for dry-run display.

    Args:
        image_results: List of ImageResult from the pipeline.
        cluster_mapping: cluster_id -> list of image paths.
        folder_prefix: Prefix for person folder names.
        include_unclustered: Include _unclustered folder.
        include_no_faces: Include _no_faces folder.

    Returns:
        OrganizePlan with all mappings ready for execution.
    """
    person_folders: dict[int, list[str]] = {}
    for cluster_id, paths in cluster_mapping.items():
        person_folders[cluster_id] = paths

    # Collect unclustered images (faces detected but didn't cluster)
    clustered_paths: set[str] = set()
    for paths in cluster_mapping.values():
        clustered_paths.update(paths)

    unclustered: list[str] = []
    if include_unclustered:
        for result in image_results:
            if (
                result.faces
                and not result.error
                and not result.skipped
                and result.path not in clustered_paths
            ):
                unclustered.append(result.path)

    # Collect images with no detected faces
    no_faces: list[str] = []
    if include_no_faces:
        for result in image_results:
            if result.skipped and not result.error:
                no_faces.append(result.path)

    return OrganizePlan(
        person_folders=person_folders,
        unclustered=sorted(unclustered),
        no_faces=sorted(no_faces),
    )


def _unique_dest_path(dest_dir: str, filename: str) -> str:
    """Resolve filename collisions by appending _1, _2, etc.

    Args:
        dest_dir: Destination directory.
        filename: Original filename.

    Returns:
        Unique file path in the destination directory.
    """
    dest = Path(dest_dir) / filename
    if not dest.exists():
        return str(dest)

    stem = dest.stem
    suffix = dest.suffix
    counter = 1
    while True:
        new_name = f"{stem}_{counter}{suffix}"
        new_dest = Path(dest_dir) / new_name
        if not new_dest.exists():
            return str(new_dest)
        counter += 1


def execute_organize_plan(
    plan: OrganizePlan,
    output_dir: str,
    folder_prefix: str = DEFAULT_FOLDER_PREFIX,
    copy_mode: bool = True,
    dry_run: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, int]:
    """Execute the organize plan: copy/move files into subfolders.

    Args:
        plan: The organization plan.
        output_dir: Root output directory.
        folder_prefix: Prefix for person folder names.
        copy_mode: True = copy, False = move.
        dry_run: If True, only print what would be done.
        progress_callback: Called with (completed, total, current_file).

    Returns:
        Dict with statistics: {copied/moved: N, skipped: N, errors: N}.
    """
    action = "copy" if copy_mode else "move"
    stats = {action: 0, "skipped": 0, "errors": 0}

    # Count total operations
    total_ops = sum(len(paths) for paths in plan.person_folders.values())
    total_ops += len(plan.unclustered) + len(plan.no_faces)
    completed = 0

    if dry_run:
        if progress_callback:
            progress_callback(1, 1, "dry-run: no files modified")
        stats["skipped"] = total_ops
        return stats

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Organize person folders
    for cluster_id, image_paths in plan.person_folders.items():
        folder_name = f"{folder_prefix}{cluster_id:02d}"
        person_dir = os.path.join(output_dir, folder_name)
        Path(person_dir).mkdir(parents=True, exist_ok=True)

        for image_path in image_paths:
            try:
                filename = os.path.basename(image_path)
                dest = _unique_dest_path(person_dir, filename)

                if copy_mode:
                    shutil.copy2(image_path, dest)
                else:
                    shutil.move(image_path, dest)

                stats[action] += 1
            except Exception as exc:
                logger.warning("Failed to %s %s: %s", action, image_path, exc)
                stats["errors"] += 1

            completed += 1
            if progress_callback:
                progress_callback(completed, total_ops, image_path)

    # Organize unclustered
    if plan.unclustered:
        unclustered_dir = os.path.join(output_dir, "_unclustered")
        Path(unclustered_dir).mkdir(parents=True, exist_ok=True)

        for image_path in plan.unclustered:
            try:
                filename = os.path.basename(image_path)
                dest = _unique_dest_path(unclustered_dir, filename)

                if copy_mode:
                    shutil.copy2(image_path, dest)
                else:
                    shutil.move(image_path, dest)

                stats[action] += 1
            except Exception as exc:
                logger.warning("Failed to %s %s (unclustered): %s", action, image_path, exc)
                stats["errors"] += 1

            completed += 1
            if progress_callback:
                progress_callback(completed, total_ops, image_path)

    # Organize no-faces
    if plan.no_faces:
        no_faces_dir = os.path.join(output_dir, "_no_faces")
        Path(no_faces_dir).mkdir(parents=True, exist_ok=True)

        for image_path in plan.no_faces:
            try:
                filename = os.path.basename(image_path)
                dest = _unique_dest_path(no_faces_dir, filename)

                if copy_mode:
                    shutil.copy2(image_path, dest)
                else:
                    shutil.move(image_path, dest)

                stats[action] += 1
            except Exception as exc:
                logger.warning("Failed to %s %s (no-faces): %s", action, image_path, exc)
                stats["errors"] += 1

            completed += 1
            if progress_callback:
                progress_callback(completed, total_ops, image_path)

    return stats
