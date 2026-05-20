from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import logging
import tomllib

from .hwdetect import detect_hardware, recommend_config

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff"})

DEFAULT_FOLDER_PREFIX = "person_"
DEFAULT_OUTPUT_DIRNAME = "visage_output"


@dataclass
class VisageConfig:
    """All tunable parameters with sensible defaults."""

    # Face detection (Vision framework)
    detection_confidence: float = 0.5
    min_face_size: int = 40  # minimum face bounding box dimension in pixels

    # Face embedding
    embedding_backend: str = "insightface"  # "dlib" or "insightface"
    embedding_model: str = "small"  # "small" (fast) or "large" (accurate) — dlib only
    num_jitters: int = 1  # times to re-sample for embedding — dlib only

    # Face quality filtering
    min_face_quality: float = 0.0  # minimum quality score [0, 1]; 0 = no filtering

    # Clustering
    cluster_method: str = "hdbscan"  # "dbscan" or "hdbscan"
    dbscan_eps: float = 0.5  # max distance between embeddings in same cluster — DBSCAN only
    dbscan_min_samples: int = 2  # min faces to form a cluster (also used as HDBSCAN min_samples)
    auto_eps: bool = False  # automatically estimate eps using k-distance elbow — DBSCAN only
    hdbscan_min_cluster_size: int = 2  # minimum cluster size for HDBSCAN
    # >0 can trigger sklearn Cython bug with certain datasets
    cluster_selection_epsilon: float = 0.0
    cluster_selection_method: str = "eom"  # "eom" (stable) or "leaf" (fine-grained)
    # cosine similarity threshold for post-clustering merge (0.85 ≈ 32° angle)
    merge_threshold: float = 0.85
    small_merge_threshold: float = 0.80  # relaxed threshold when one cluster is small
    min_reliable_size: int = 10  # clusters below this size use relaxed threshold

    # Head features (supplementary signal for clustering)
    head_feature_weight: float = 0.2  # weight for head features in composite distance (0–1)

    # Processing
    batch_size: int = 100  # images per batch for progress reporting
    max_workers: int = 4  # parallel detection workers
    max_image_dimension: int = 0  # 0 = no downscaling; e.g. 2048 to resize large images
    use_float32_cluster: bool = False  # use float32 for distance matrix (halves memory)
    cluster_chunk_size: int = 0  # 0 = full NxN matrix; >0 = row-block size for chunking
    sample_limit: int | None = None  # max faces before sampling (None = no limit)

    # Output
    copy_mode: bool = True  # True = copy, False = move
    output_dir: str | None = None  # None = create subdirs inside input folder
    folder_prefix: str = DEFAULT_FOLDER_PREFIX
    include_unclustered: bool = False
    include_no_faces: bool = False

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.detection_confidence <= 1.0:
            raise ValueError(
                f"detection_confidence must be 0-1, got {self.detection_confidence}"
            )
        if self.min_face_size < 1:
            raise ValueError(f"min_face_size must be >= 1, got {self.min_face_size}")
        if self.embedding_backend not in ("dlib", "insightface"):
            raise ValueError(
                f"embedding_backend must be 'dlib' or 'insightface', "
                f"got {self.embedding_backend!r}"
            )
        if self.embedding_model not in ("small", "large"):
            raise ValueError(
                f"embedding_model must be 'small' or 'large', "
                f"got {self.embedding_model!r}"
            )
        if self.num_jitters < 1:
            raise ValueError(f"num_jitters must be >= 1, got {self.num_jitters}")
        if not 0.0 <= self.min_face_quality <= 1.0:
            raise ValueError(
                f"min_face_quality must be 0-1, got {self.min_face_quality}"
            )
        if self.cluster_method not in ("dbscan", "hdbscan"):
            raise ValueError(
                f"cluster_method must be 'dbscan' or 'hdbscan', "
                f"got {self.cluster_method!r}"
            )
        if self.dbscan_eps <= 0:
            raise ValueError(f"dbscan_eps must be > 0, got {self.dbscan_eps}")
        if self.dbscan_min_samples < 1:
            raise ValueError(
                f"dbscan_min_samples must be >= 1, got {self.dbscan_min_samples}"
            )
        if self.hdbscan_min_cluster_size < 2:
            raise ValueError(
                f"hdbscan_min_cluster_size must be >= 2, "
                f"got {self.hdbscan_min_cluster_size}"
            )
        if self.cluster_selection_epsilon < 0:
            raise ValueError(
                f"cluster_selection_epsilon must be >= 0, "
                f"got {self.cluster_selection_epsilon}"
            )
        if self.cluster_selection_method not in ("eom", "leaf"):
            raise ValueError(
                f"cluster_selection_method must be 'eom' or 'leaf', "
                f"got {self.cluster_selection_method!r}"
            )
        if not 0.0 <= self.merge_threshold <= 1.0:
            raise ValueError(
                f"merge_threshold must be 0-1, "
                f"got {self.merge_threshold}"
            )
        if not 0.0 <= self.small_merge_threshold <= 1.0:
            raise ValueError(
                f"small_merge_threshold must be 0-1, "
                f"got {self.small_merge_threshold}"
            )
        if self.min_reliable_size < 2:
            raise ValueError(
                f"min_reliable_size must be >= 2, "
                f"got {self.min_reliable_size}"
            )
        if not 0.0 <= self.head_feature_weight <= 1.0:
            raise ValueError(
                f"head_feature_weight must be 0-1, "
                f"got {self.head_feature_weight}"
            )
        if self.max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {self.max_workers}")



def load_config_from_file(path: Path) -> dict[str, Any]:
    """Load configuration from a TOML file."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return data


def _apply_toml_section(
    config_dict: dict[str, Any], section: str, mapping: dict[str, str],
) -> dict[str, Any]:
    """Apply a TOML section to config dict, renaming keys via mapping."""
    if section not in config_dict:
        return {}
    section_data = config_dict[section]
    return {mapping.get(k, k): v for k, v in section_data.items()}


_TOML_KEY_MAP = {
    "detection": {
        "confidence": "detection_confidence",
        "min_face_size": "min_face_size",
    },
    "embedding": {
        "backend": "embedding_backend",
        "model": "embedding_model",
        "num_jitters": "num_jitters",
    },
    "quality": {
        "min_face_quality": "min_face_quality",
    },
    "clustering": {
        "method": "cluster_method",
        "eps": "dbscan_eps",
        "min_samples": "dbscan_min_samples",
        "min_cluster_size": "hdbscan_min_cluster_size",
        "cluster_selection_epsilon": "cluster_selection_epsilon",
        "cluster_selection_method": "cluster_selection_method",
        "head_feature_weight": "head_feature_weight",
        "merge_threshold": "merge_threshold",
        "small_merge_threshold": "small_merge_threshold",
        "min_reliable_size": "min_reliable_size",
    },
    "output": {
        "copy_mode": "copy_mode",
        "folder_prefix": "folder_prefix",
        "include_unclustered": "include_unclustered",
        "include_no_faces": "include_no_faces",
    },
}


def build_config(
    config_file: str | None = None,
    input_dir: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> VisageConfig:
    """Build a VisageConfig from file, input directory, CLI overrides, and hardware.

    Priority: CLI overrides > --config file > visage.toml in input dir >
              hardware-aware recommendations > code defaults.
    """
    kwargs: dict[str, Any] = {}
    user_set: set[str] = set()  # fields explicitly set by user (file or CLI)

    # Try loading from config file
    config_paths: list[Path] = []
    if config_file:
        config_paths.append(Path(config_file))
    if input_dir:
        config_paths.append(Path(input_dir) / "visage.toml")

    for cp in config_paths:
        if cp.exists():
            toml_data = load_config_from_file(cp)
            for section, key_map in _TOML_KEY_MAP.items():
                applied = _apply_toml_section(toml_data, section, key_map)
                kwargs.update(applied)
                user_set.update(applied.keys())
            break  # first found wins

    # Apply CLI overrides (highest priority)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                kwargs[k] = v
                user_set.add(k)

    # ── Hardware-aware defaults (applied only if not set by user) ──
    try:
        hw = detect_hardware()
        rec = recommend_config(hw)
        logger.info(
            "Hardware: %.1f GB RAM, %d cores → backend=%s, workers=%d",
            hw.total_ram_gb, hw.physical_cores, rec.backend, rec.max_workers,
        )

        hw_defaults = {
            "max_workers": rec.max_workers,
            "max_image_dimension": rec.max_image_dimension,
            "use_float32_cluster": rec.use_float32_cluster,
            "cluster_chunk_size": rec.cluster_chunk_size,
            "head_feature_weight": rec.head_feature_weight,
            "sample_limit": rec.sample_limit,
        }
        # Only apply backend recommendation if user didn't specify one
        if "embedding_backend" not in user_set:
            hw_defaults["embedding_backend"] = rec.backend

        for key, val in hw_defaults.items():
            if key not in user_set:
                kwargs.setdefault(key, val)

        if rec.use_float32_cluster:
            logger.info("Clustering: float32 mode (memory-optimized)")
        if rec.max_image_dimension > 0:
            logger.info(
                "Image downscaling: max %dpx (memory-optimized)",
                rec.max_image_dimension,
            )
    except Exception:
        logger.warning("Hardware detection failed, using code defaults", exc_info=True)

    return VisageConfig(**kwargs)
