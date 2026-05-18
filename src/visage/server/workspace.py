"""In-memory mutable workspace wrapping pipeline results.

Provides merge, split, rename, and undo operations for the review UI.
"""

from __future__ import annotations

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
    kind: str  # "merge", "remove", "rename", "move"
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
        for r in image_results:
            if r.faces:
                self._face_boxes[r.path] = [
                    {
                        "top": int(f.face_box.top),
                        "right": int(f.face_box.right),
                        "bottom": int(f.face_box.bottom),
                        "left": int(f.face_box.left),
                    }
                    for f in r.faces
                ]

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

        # Save undo state
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
            },
        ))

        # Apply merge
        self._cluster_mapping[to_id].extend(from_photos)
        del self._cluster_mapping[from_id]

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
            },
        ))

        photos.remove(image_path)

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
        """Assign a user-friendly name to a cluster."""
        if cluster_id not in self._cluster_mapping:
            raise ValueError(f"Cluster not found: {cluster_id}")

        old_name = self._cluster_names.get(cluster_id)

        self._history.append(_Operation(
            kind="rename",
            data={
                "cluster_id": cluster_id,
                "old_name": old_name,
            },
        ))

        self._cluster_names[cluster_id] = name

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

            result["image_path"] = image_path
            result["from_cluster_id"] = from_id
            result["to_cluster_id"] = to_id

        return result

    def can_undo(self) -> bool:
        return len(self._history) > 0

    # ── Save ────────────────────────────────────────────────────

    def save_to_disk(self, output_dir: str | None = None) -> dict:
        """Write organized files to disk using the current cluster mapping.

        Returns stats dict for display.
        """
        out = output_dir or self.input_dir.rstrip("/") + "/" + DEFAULT_OUTPUT_DIRNAME

        plan = build_organize_plan(
            self._image_results,
            self._cluster_mapping,
            folder_prefix=self.config.folder_prefix,
            include_unclustered=self.config.include_unclustered,
            include_no_faces=self.config.include_no_faces,
        )

        stats = execute_organize_plan(
            plan,
            output_dir=out,
            folder_prefix=self.config.folder_prefix,
            copy_mode=self.config.copy_mode,
            dry_run=False,
        )
        return stats

    # ── API serialization ────────────────────────────────────────

    def _photo_dict(self, path: str) -> dict:
        """Build a photo entry with face bounding boxes."""
        return {"path": path, "faces": self._face_boxes.get(path, [])}

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
            clusters.append({
                "id": cid_int,
                "name": self._cluster_names.get(cid, f"person_{cid_int:02d}"),
                "photos": [self._photo_dict(p) for p in sorted(photos)],
                "photo_count": len(photos),
                "thumbnail": thumbnail,
                "confidence": round(float(self._cluster_confidences.get(cid, 0.0)), 3),
            })
            all_photo_paths.update(photos)

        # All photos across all clusters (for "All Photos" view)
        all_photos = [self._photo_dict(p) for p in sorted(all_photo_paths)]

        # Noise/unclustered photos
        noise = [self._photo_dict(p) for p in self.noise_photos]

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
