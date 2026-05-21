"""In-memory mutable workspace wrapping pipeline results.

Provides merge, split, rename, and undo operations for the review UI.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from visage.cluster import (
    build_cluster_mapping,
    compute_cluster_confidences,
)
from visage.config import DEFAULT_OUTPUT_DIRNAME, VisageConfig
from visage.models import ClusterResult, ImageResult
from visage.organizer import build_organize_plan, execute_organize_plan


@dataclass
class _Operation:
    """A reversible mutation on the workspace."""
    kind: str  # "merge", "remove", "rename", "move", "batch_assign", "batch_remove"
    data: dict  # parameters to reverse the operation


class Workspace:
    """Mutable in-memory workspace wrapping a pipeline result.

    Holds all state the frontend needs and provides mutation methods
    with undo support.
    """

    def __init__(
        self,
        input_dir: str,
        config: VisageConfig,
        image_results: list[ImageResult],
        cluster_result: ClusterResult,
        face_to_image: list[tuple[str, int]],
    ) -> None:
        self.input_dir = input_dir
        self.config = config

        self._image_results = image_results
        self._cluster_result = cluster_result
        self._face_to_image = face_to_image

        # Build initial cluster mapping
        self._cluster_mapping = build_cluster_mapping(cluster_result, face_to_image)

        # User-assigned names for clusters
        self._cluster_names: dict[int, str] = {}

        # Cluster confidences
        self._cluster_confidences = compute_cluster_confidences(cluster_result)

        # Undo stack
        self._history: list[_Operation] = []

        # Image path → face bounding boxes (for frontend overlay)
        self._face_boxes: dict[str, list[dict]] = {}
        # Image path → (width, height) of original image
        self._image_sizes: dict[str, tuple[int, int]] = {}
        for r in image_results:
            self._image_sizes[r.path] = (r.image_width, r.image_height)
            if r.faces:
                self._face_boxes[r.path] = [
                    {
                        "top": int(f.face_box.top),
                        "right": int(f.face_box.right),
                        "bottom": int(f.face_box.bottom),
                        "left": int(f.face_box.left),
                        "face_index": int(f.face_index),
                    }
                    for f in r.faces
                ]

        # ── Face-level cluster tracking ───────────────────────
        # {image_path: {face_index: cluster_id}}
        # Enables per-face cluster display in multi-face images
        self._face_clusters: dict[str, dict[int, int]] = {}
        for i, (path, face_idx) in enumerate(face_to_image):
            if i < len(cluster_result.labels):
                label = int(cluster_result.labels[i])
                if path not in self._face_clusters:
                    self._face_clusters[path] = {}
                self._face_clusters[path][face_idx] = label

        # Total images/faces stats (precomputed)
        self._total_images = len(image_results)
        self._images_with_faces = sum(1 for r in image_results if r.faces and not r.error)
        self._total_faces = sum(len(r.faces) for r in image_results)

    # ── Properties ──────────────────────────────────────────────

    @property
    def cluster_ids(self) -> list[int]:
        return sorted(self._cluster_mapping.keys())

    @property
    def cluster_names(self) -> dict[int, str]:
        return dict(self._cluster_names)

    @property
    def num_noise_faces(self) -> int:
        return len(self.noise_photos)

    def cluster_photos(self, cluster_id: int) -> list[str]:
        return list(self._cluster_mapping.get(cluster_id, []))

    def cluster_count(self, cluster_id: int) -> int:
        return len(self._cluster_mapping.get(cluster_id, []))

    def cluster_confidence(self, cluster_id: int) -> float:
        return self._cluster_confidences.get(cluster_id, 0.0)

    def cluster_name(self, cluster_id: int) -> str:
        return self._cluster_names.get(cluster_id, "")

    # ── Mutations ───────────────────────────────────────────────

    def merge_clusters(self, from_id: int, to_id: int) -> None:
        """Merge cluster `from_id` into `to_id`.

        Saves undo state before applying the merge.
        """
        if from_id == to_id:
            return
        if from_id not in self._cluster_mapping or to_id not in self._cluster_mapping:
            raise ValueError(f"Cluster not found: {from_id} or {to_id}")

        # Save undo state (including face-level snapshot)
        from_photos = list(self._cluster_mapping[from_id])
        to_photos_before = list(self._cluster_mapping[to_id])
        from_name = self._cluster_names.pop(from_id, None)
        to_name_before = self._cluster_names.get(to_id)

        self._history.append(_Operation(
            kind="merge",
            data={
                "from_id": from_id,
                "to_id": to_id,
                "from_photos": from_photos,
                "to_photos_before": to_photos_before,
                "from_name": from_name,
                "to_name_before": to_name_before,
                "from_confidence": self._cluster_confidences.pop(from_id, 0.0),
                "to_confidence_before": self._cluster_confidences.get(to_id, 0.0),
                "face_snapshot": self._snapshot_face_clusters(from_photos),
            },
        ))

        # Apply merge
        self._cluster_mapping[to_id].extend(from_photos)
        del self._cluster_mapping[from_id]

        # Update face-level tracking: all faces from from_id → to_id
        self._update_face_clusters_for_cluster(from_id, to_id)

    def remove_face(self, image_path: str, from_cluster_id: int) -> None:
        """Remove a face/image from a cluster.

        The removed file goes into a virtual "unclustered" set
        (tracked as noise — re-assigns to cluster -1 at save time).
        """
        if from_cluster_id not in self._cluster_mapping:
            raise ValueError(f"Cluster not found: {from_cluster_id}")

        photos = self._cluster_mapping[from_cluster_id]
        if image_path not in photos:
            raise ValueError(f"Image {image_path} not in cluster {from_cluster_id}")

        cluster_deleted = len(photos) == 1
        saved_name = self._cluster_names.pop(from_cluster_id, None) if cluster_deleted else None
        saved_confidence = (
            self._cluster_confidences.pop(from_cluster_id, None) if cluster_deleted else None
        )

        self._history.append(_Operation(
            kind="remove",
            data={
                "image_path": image_path,
                "from_cluster_id": from_cluster_id,
                "cluster_deleted": cluster_deleted,
                "saved_name": saved_name,
                "saved_confidence": saved_confidence,
                "face_snapshot": self._snapshot_face_clusters([image_path]),
            },
        ))

        photos.remove(image_path)

        # Update face-level tracking: faces in this cluster → noise (-1)
        if image_path in self._face_clusters:
            for face_idx, cur_id in self._face_clusters[image_path].items():
                if cur_id == from_cluster_id:
                    self._face_clusters[image_path][face_idx] = -1

        # Remove cluster if empty
        if not photos:
            del self._cluster_mapping[from_cluster_id]

    def move_face(self, image_path: str, from_cluster_id: int, to_cluster_id: int) -> None:
        """Move a face/image from one cluster to another.

        Use from_cluster_id=-1 to move a noise/unclustered photo into a cluster.
        If to_cluster_id does not exist, a new cluster is created.
        """
        if from_cluster_id == to_cluster_id:
            return

        from_noise = from_cluster_id == -1

        if not from_noise:
            if from_cluster_id not in self._cluster_mapping:
                raise ValueError(f"Source cluster not found: {from_cluster_id}")

            from_photos = self._cluster_mapping[from_cluster_id]
            if image_path not in from_photos:
                raise ValueError(f"Image {image_path} not in cluster {from_cluster_id}")
        else:
            # Verify the image is actually a noise photo
            if image_path not in self.noise_photos:
                raise ValueError(f"Image {image_path} is not in noise set")

        # Create destination cluster if it doesn't exist
        is_new_cluster = to_cluster_id not in self._cluster_mapping
        if is_new_cluster:
            self._cluster_mapping[to_cluster_id] = []
            self._cluster_confidences[to_cluster_id] = 0.0

        self._history.append(_Operation(
            kind="move",
            data={
                "image_path": image_path,
                "from_cluster_id": from_cluster_id,
                "to_cluster_id": to_cluster_id,
                "was_new_cluster": is_new_cluster,
                "from_noise": from_noise,
                "from_deleted": False,
                "from_name": None,
                "from_confidence": None,
                "face_snapshot": self._snapshot_face_clusters([image_path]),
            },
        ))

        if not from_noise:
            from_photos.remove(image_path)

            # Remove source cluster if empty
            if not from_photos:
                self._history[-1].data["from_deleted"] = True
                self._history[-1].data["from_name"] = self._cluster_names.pop(
                    from_cluster_id, None,
                )
                self._history[-1].data["from_confidence"] = (
                    self._cluster_confidences.pop(from_cluster_id, None)
                )
                del self._cluster_mapping[from_cluster_id]

        self._cluster_mapping[to_cluster_id].append(image_path)

        # Update face-level tracking
        if from_noise:
            self._update_face_clusters_for_image(image_path, to_cluster_id)
        else:
            if image_path in self._face_clusters:
                for face_idx, cur_id in self._face_clusters[image_path].items():
                    if cur_id == from_cluster_id:
                        self._face_clusters[image_path][face_idx] = to_cluster_id

    def batch_assign_noise(self, image_paths: list[str], to_id: int) -> None:
        """Assign multiple noise photos to a cluster at once.

        If to_id does not exist, a new cluster is created.
        Pushes a single undo operation that can revert all assignments.
        """
        if not image_paths:
            return

        # Validate all paths are noise photos
        for path in image_paths:
            if path not in self.noise_photos:
                raise ValueError(f"Image {path} is not in noise set")

        # Create destination cluster if it doesn't exist
        is_new_cluster = to_id not in self._cluster_mapping
        if is_new_cluster:
            self._cluster_mapping[to_id] = []
            self._cluster_confidences[to_id] = 0.0

        self._history.append(_Operation(
            kind="batch_assign",
            data={
                "image_paths": list(image_paths),
                "to_id": to_id,
                "was_new_cluster": is_new_cluster,
                "face_snapshot": self._snapshot_face_clusters(image_paths),
            },
        ))

        self._cluster_mapping[to_id].extend(image_paths)

        # Update face-level: all faces in these images → to_id
        for path in image_paths:
            self._update_face_clusters_for_image(path, to_id)

    def batch_remove_faces(self, cluster_id: int, image_paths: list[str]) -> None:
        """Remove multiple faces from a cluster at once.

        All removed photos go to noise. Pushes a single undo operation
        that can restore all removed photos in one step.
        """
        if not image_paths:
            return

        if cluster_id not in self._cluster_mapping:
            raise ValueError(f"Cluster not found: {cluster_id}")

        photos = self._cluster_mapping[cluster_id]
        for path in image_paths:
            if path not in photos:
                raise ValueError(f"Image {path} not in cluster {cluster_id}")

        cluster_deleted = len(photos) == len(image_paths)
        saved_name = self._cluster_names.pop(cluster_id, None) if cluster_deleted else None
        saved_confidence = (
            self._cluster_confidences.pop(cluster_id, None) if cluster_deleted else None
        )

        self._history.append(_Operation(
            kind="batch_remove",
            data={
                "image_paths": list(image_paths),
                "from_cluster_id": cluster_id,
                "cluster_deleted": cluster_deleted,
                "saved_name": saved_name,
                "saved_confidence": saved_confidence,
                "face_snapshot": self._snapshot_face_clusters(image_paths),
            },
        ))

        for path in image_paths:
            photos.remove(path)

            # Update face-level: faces in this cluster → noise
            if path in self._face_clusters:
                for face_idx, cur_id in self._face_clusters[path].items():
                    if cur_id == cluster_id:
                        self._face_clusters[path][face_idx] = -1

        if not photos:
            del self._cluster_mapping[cluster_id]

    @property
    def noise_photos(self) -> list[str]:
        """Return image paths for faces that weren't assigned to any cluster."""
        clustered = set()
        for photos in self._cluster_mapping.values():
            clustered.update(photos)
        return sorted(
            p for r in self._image_results
            if r.faces and not r.error and r.path not in clustered
            for p in [r.path]
        )

    def next_cluster_id(self) -> int:
        """Return the next available cluster ID (max existing + 1)."""
        if not self._cluster_mapping:
            return 0
        return max(self._cluster_mapping.keys()) + 1

    def rename_cluster(self, cluster_id: int, name: str) -> None:
        """Assign a user-friendly name to a cluster.

        If another cluster already has this name, the current cluster is
        automatically merged into the existing one (same-name auto-merge).
        """
        if cluster_id not in self._cluster_mapping:
            raise ValueError(f"Cluster not found: {cluster_id}")

        old_name = self._cluster_names.get(cluster_id)

        # Push rename undo operation
        self._history.append(_Operation(
            kind="rename",
            data={
                "cluster_id": cluster_id,
                "old_name": old_name,
            },
        ))

        self._cluster_names[cluster_id] = name

        # Auto-merge: if another cluster already has this name, merge into it
        if name:
            for other_id, other_name in self._cluster_names.items():
                if other_id != cluster_id and other_name == name:
                    self.merge_clusters(cluster_id, other_id)
                    break

    def undo(self) -> dict | None:
        """Undo the last operation. Returns info about what was undone, or None."""
        if not self._history:
            return None

        op = self._history.pop()
        result: dict = {"kind": op.kind}

        if op.kind == "merge":
            from_id = op.data["from_id"]
            to_id = op.data["to_id"]

            # Restore from_id cluster
            self._cluster_mapping[from_id] = op.data["from_photos"]

            # Restore to_id to its pre-merge state
            to_photos_before = op.data["to_photos_before"]
            if to_photos_before:
                self._cluster_mapping[to_id] = to_photos_before
            elif to_id in self._cluster_mapping:
                del self._cluster_mapping[to_id]

            # Restore names
            if op.data["from_name"] is not None:
                self._cluster_names[from_id] = op.data["from_name"]
            if op.data["to_name_before"] is not None:
                self._cluster_names[to_id] = op.data["to_name_before"]
            elif to_id in self._cluster_names:
                del self._cluster_names[to_id]

            # Restore confidences
            self._cluster_confidences[from_id] = op.data["from_confidence"]
            self._cluster_confidences[to_id] = op.data["to_confidence_before"]

            # Restore face-level tracking
            if "face_snapshot" in op.data:
                self._restore_face_clusters(op.data["face_snapshot"])

            result["from_id"] = from_id
            result["to_id"] = to_id

        elif op.kind == "remove":
            image_path = op.data["image_path"]
            from_cluster_id = op.data["from_cluster_id"]

            # Restore the image to its cluster
            if from_cluster_id not in self._cluster_mapping:
                self._cluster_mapping[from_cluster_id] = []
            self._cluster_mapping[from_cluster_id].append(image_path)

            # Restore cluster metadata if it was deleted
            if op.data.get("cluster_deleted"):
                if op.data["saved_name"] is not None:
                    self._cluster_names[from_cluster_id] = op.data["saved_name"]
                if op.data["saved_confidence"] is not None:
                    self._cluster_confidences[from_cluster_id] = op.data["saved_confidence"]

            # Restore face-level tracking
            if "face_snapshot" in op.data:
                self._restore_face_clusters(op.data["face_snapshot"])

            result["image_path"] = image_path
            result["cluster_id"] = from_cluster_id

        elif op.kind == "rename":
            cluster_id = op.data["cluster_id"]
            old_name = op.data["old_name"]
            if old_name is not None:
                self._cluster_names[cluster_id] = old_name
            else:
                self._cluster_names.pop(cluster_id, None)
            result["cluster_id"] = cluster_id
            result["old_name"] = old_name

        elif op.kind == "move":
            image_path = op.data["image_path"]
            from_id = op.data["from_cluster_id"]
            to_id = op.data["to_cluster_id"]
            from_noise = op.data.get("from_noise", False)

            # Remove from destination
            self._cluster_mapping[to_id].remove(image_path)

            # Remove destination cluster if it was newly created and now empty
            if op.data["was_new_cluster"] and not self._cluster_mapping[to_id]:
                del self._cluster_mapping[to_id]
                self._cluster_names.pop(to_id, None)
                self._cluster_confidences.pop(to_id, None)

            # Restore to source (unless it came from noise)
            if not from_noise:
                if op.data["from_deleted"]:
                    self._cluster_mapping[from_id] = []
                    if op.data["from_name"] is not None:
                        self._cluster_names[from_id] = op.data["from_name"]
                    if op.data["from_confidence"] is not None:
                        self._cluster_confidences[from_id] = op.data["from_confidence"]

                if from_id in self._cluster_mapping:
                    self._cluster_mapping[from_id].append(image_path)
            # If from_noise, the photo just goes back to being unclustered — no action needed

            # Restore face-level tracking
            if "face_snapshot" in op.data:
                self._restore_face_clusters(op.data["face_snapshot"])

            result["image_path"] = image_path
            result["from_cluster_id"] = from_id
            result["to_cluster_id"] = to_id

        elif op.kind == "batch_assign":
            image_paths = op.data["image_paths"]
            to_id = op.data["to_id"]

            # Remove all assigned paths from destination cluster
            dest_photos = self._cluster_mapping[to_id]
            for path in image_paths:
                dest_photos.remove(path)

            # Remove destination cluster if it was newly created and now empty
            if op.data["was_new_cluster"] and not dest_photos:
                del self._cluster_mapping[to_id]
                self._cluster_names.pop(to_id, None)
                self._cluster_confidences.pop(to_id, None)
            # Photos go back to noise — no action needed (noise is computed dynamically)

            # Restore face-level tracking
            if "face_snapshot" in op.data:
                self._restore_face_clusters(op.data["face_snapshot"])

            result["image_paths"] = image_paths
            result["to_cluster_id"] = to_id

        elif op.kind == "batch_remove":
            image_paths = op.data["image_paths"]
            from_cluster_id = op.data["from_cluster_id"]

            # Restore the cluster if it was deleted
            if from_cluster_id not in self._cluster_mapping:
                self._cluster_mapping[from_cluster_id] = []

            # Restore all removed paths
            self._cluster_mapping[from_cluster_id].extend(image_paths)

            # Restore cluster metadata if it was deleted
            if op.data.get("cluster_deleted"):
                if op.data["saved_name"] is not None:
                    self._cluster_names[from_cluster_id] = op.data["saved_name"]
                if op.data["saved_confidence"] is not None:
                    self._cluster_confidences[from_cluster_id] = op.data["saved_confidence"]

            # Restore face-level tracking
            if "face_snapshot" in op.data:
                self._restore_face_clusters(op.data["face_snapshot"])

            result["image_paths"] = image_paths
            result["cluster_id"] = from_cluster_id

        return result

    def can_undo(self) -> bool:
        return len(self._history) > 0

    # ── Save ────────────────────────────────────────────────────

    # ── Multi-face strategy ──────────────────────────────────

    def _build_primary_face_cluster_map(self) -> dict[str, int]:
        """Map each image path to the cluster ID of its primary face.

        Only images whose primary face belongs to a valid cluster (not noise)
        are included in the result.

        Returns:
            Dict of ``{image_path: cluster_id}``.
        """
        result: dict[str, int] = {}
        for r in self._image_results:
            if not r.faces:
                continue
            path = r.path
            # Find the primary face
            primary_face = next((f for f in r.faces if f.is_primary), None)
            if primary_face is None:
                continue
            # Look up its current cluster from face-level tracking
            fc = self._face_clusters.get(path, {})
            cid = fc.get(primary_face.face_index, -1)
            if cid >= 0:
                result[path] = cid
        return result

    def _apply_multi_face_strategy(
        self,
        mapping: dict[int, list[str]],
        strategy: str,
    ) -> dict[int, list[str]]:
        """Filter a cluster mapping according to the multi-face strategy.

        ``"all"`` — preserve current behavior: images may appear in multiple
        cluster folders when they contain faces from different people.

        ``"primary"`` — each image only appears in the folder of its primary
        (main) face. Secondary faces' cluster folders do not receive a copy.

        Args:
            mapping: cluster_id → list of image paths.
            strategy: ``"all"`` or ``"primary"``.

        Returns:
            Filtered cluster mapping.
        """
        if strategy == "all":
            return mapping

        primary_map = self._build_primary_face_cluster_map()
        filtered: dict[int, list[str]] = {}
        for cid, paths in mapping.items():
            cid_paths = [p for p in paths if primary_map.get(p) == cid]
            if cid_paths:
                filtered[cid] = cid_paths
        return filtered

    # ── Save ────────────────────────────────────────────────────

    def save_to_disk(
        self,
        output_dir: str | None = None,
        copy_mode: bool | None = None,
        folder_prefix: str | None = None,
        include_unclustered: bool | None = None,
        include_no_faces: bool | None = None,
        cluster_ids: list[int] | None = None,
        multi_face_strategy: str | None = None,
    ) -> dict:
        """Write organized files to disk using the current cluster mapping.

        All optional parameters default to ``None``, which means the
        corresponding :class:`VisageConfig` value is used.

        Args:
            output_dir: Target directory. Defaults to ``<input_dir>/Visage``.
            copy_mode: ``True`` to copy files, ``False`` to move them.
            folder_prefix: Prefix for cluster folder names.
            include_unclustered: Export unclustered faces.
            include_no_faces: Export images with no detected faces.
            cluster_ids: If provided, export only the specified clusters
                (the in-memory mapping is not modified).
            multi_face_strategy: ``"primary"`` or ``"all"``. Controls how
                images with multiple faces from different clusters are handled.
                ``"primary"`` only places each image in its primary face's
                cluster folder. ``"all"`` places images in all matching cluster
                folders (current behavior).

        Returns:
            Stats dict for display.
        """
        out = output_dir or self.input_dir.rstrip("/") + "/" + DEFAULT_OUTPUT_DIRNAME

        # Base mapping (with optional cluster filter)
        mapping = self._cluster_mapping
        if cluster_ids is not None:
            mapping = {
                cid: paths
                for cid, paths in self._cluster_mapping.items()
                if cid in cluster_ids
            }

        # Apply multi-face strategy
        mfs = (
            multi_face_strategy
            if multi_face_strategy is not None
            else self.config.multi_face_strategy
        )
        mapping = self._apply_multi_face_strategy(mapping, mfs)

        fp = folder_prefix if folder_prefix is not None else self.config.folder_prefix
        iu = (
            include_unclustered
            if include_unclustered is not None
            else self.config.include_unclustered
        )
        inf = include_no_faces if include_no_faces is not None else self.config.include_no_faces
        cm = copy_mode if copy_mode is not None else self.config.copy_mode

        plan = build_organize_plan(
            self._image_results,
            mapping,
            folder_prefix=fp,
            include_unclustered=iu,
            include_no_faces=inf,
            cluster_names=self._cluster_names or None,
        )

        stats = execute_organize_plan(
            plan,
            output_dir=out,
            folder_prefix=fp,
            copy_mode=cm,
            dry_run=False,
            cluster_names=self._cluster_names or None,
        )
        return stats

    # ── Face-level helpers ───────────────────────────────────

    def _update_face_clusters_for_image(self, path: str, new_cluster_id: int) -> None:
        """Set all tracked face_indices for an image to a cluster (or -1 for noise)."""
        if path not in self._face_clusters:
            return
        for face_idx in self._face_clusters[path]:
            self._face_clusters[path][face_idx] = new_cluster_id

    def _update_face_clusters_for_cluster(
        self, from_cluster_id: int, to_cluster_id: int,
    ) -> None:
        """Update all faces in a cluster to a new cluster ID.

        Used when merging clusters or removing faces.
        """
        for path in self._cluster_mapping.get(from_cluster_id, []):
            if path in self._face_clusters:
                for face_idx, cur_id in list(self._face_clusters[path].items()):
                    if cur_id == from_cluster_id:
                        self._face_clusters[path][face_idx] = to_cluster_id

    def _snapshot_face_clusters(self, paths: list[str]) -> dict:
        """Snapshot face_clusters for a set of image paths (for undo)."""
        return {p: deepcopy(self._face_clusters.get(p, {})) for p in paths}

    def _restore_face_clusters(self, snapshot: dict) -> None:
        """Restore face_clusters from a snapshot (undo)."""
        for path, fc in snapshot.items():
            if fc:
                self._face_clusters[path] = fc
            elif path in self._face_clusters:
                del self._face_clusters[path]

    # ── API serialization ────────────────────────────────────────

    def _photo_dict(self, path: str, filter_cluster: int | None = None) -> dict:
        """Build a photo entry with face bounding boxes and original image dimensions.

        Args:
            path: Image path.
            filter_cluster: If set, only include faces belonging to this cluster.
        """
        w, h = self._image_sizes.get(path, (0, 0))
        face_clusters = self._face_clusters.get(path, {})

        faces = []
        for face in self._face_boxes.get(path, []):
            fi = face.get("face_index", 0)
            cid = face_clusters.get(fi, -1)
            face_entry = {
                "top": face["top"],
                "right": face["right"],
                "bottom": face["bottom"],
                "left": face["left"],
                "cluster_id": cid,
            }
            if filter_cluster is not None and cid != filter_cluster:
                continue
            faces.append(face_entry)

        return {
            "path": path,
            "faces": faces,
            "width": w,
            "height": h,
        }

    def to_api_dict(self) -> dict:
        """Serialize workspace state for the frontend API.

        All numeric values are explicitly cast to Python native types
        to avoid FastAPI jsonable_encoder issues with numpy integers.
        """
        clusters = []
        all_photo_paths: set[str] = set()
        for cid in sorted(self._cluster_mapping.keys()):
            cid_int = int(cid)
            photos = self._cluster_mapping[cid]
            thumbnail = photos[0] if photos else None
            # Filter face boxes to only show faces belonging to this cluster
            clusters.append({
                "id": cid_int,
                "name": self._cluster_names.get(cid, f"person_{cid_int:02d}"),
                "photos": [self._photo_dict(p, filter_cluster=cid_int) for p in sorted(photos)],
                "photo_count": len(photos),
                "thumbnail": thumbnail,
                "confidence": round(float(self._cluster_confidences.get(cid, 0.0)), 3),
            })
            all_photo_paths.update(photos)

        # All photos across all clusters — show all face boxes with cluster labels
        all_photos = [self._photo_dict(p) for p in sorted(all_photo_paths)]

        # Noise/unclustered photos — only show unclustered faces
        noise = [self._photo_dict(p, filter_cluster=-1) for p in self.noise_photos]

        return {
            "input_dir": self.input_dir,
            "config": {
                "copy_mode": self.config.copy_mode,
                "folder_prefix": self.config.folder_prefix,
                "embedding_backend": self.config.embedding_backend,
            },
            "stats": {
                "total_images": int(self._total_images),
                "images_with_faces": int(self._images_with_faces),
                "total_faces": int(self._total_faces),
                "num_clusters": int(len(self._cluster_mapping)),
                "num_noise_faces": len(noise),
            },
            "clusters": clusters,
            "noise_photos": noise,
            "all_photos": all_photos,
            "next_cluster_id": int(self.next_cluster_id()),
            "can_undo": bool(self.can_undo()),
        }

    def get_recluster_data(self) -> dict:
        """Return raw data needed for re-clustering.

        Returns embeddings array, face_to_image mapping, and image results
        so the server can re-run clustering without redoing detection/embedding.
        """
        return {
            "embeddings": self._cluster_result.embeddings,
            "face_to_image": list(self._face_to_image),
            "image_results": self._image_results,
        }

    # ── Factory ─────────────────────────────────────────────────

    @classmethod
    def from_pipeline(
        cls,
        input_dir: str,
        config: VisageConfig,
        image_results: list[ImageResult],
        cluster_result: ClusterResult,
        face_to_image: list[tuple[str, int]],
    ) -> Workspace:
        """Create a Workspace from pipeline outputs."""
        return cls(input_dir, config, image_results, cluster_result, face_to_image)
