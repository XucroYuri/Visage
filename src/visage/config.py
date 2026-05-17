from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib

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
    embedding_backend: str = "dlib"  # "dlib" or "insightface"
    embedding_model: str = "small"  # "small" (fast) or "large" (accurate) — dlib only
    num_jitters: int = 1  # times to re-sample for embedding — dlib only

    # Face quality filtering
    min_face_quality: float = 0.0  # minimum quality score [0, 1]; 0 = no filtering

    # Clustering
    cluster_method: str = "dbscan"  # "dbscan" or "hdbscan"
    dbscan_eps: float = 0.5  # max distance between embeddings in same cluster — DBSCAN only
    dbscan_min_samples: int = 2  # min faces to form a cluster
    auto_eps: bool = False  # automatically estimate eps using k-distance elbow — DBSCAN only

    # Processing
    batch_size: int = 100  # images per batch for progress reporting
    max_workers: int = 4  # parallel detection workers

    # Output
    copy_mode: bool = True  # True = copy, False = move
    output_dir: str | None = None  # None = create subdirs inside input folder
    folder_prefix: str = DEFAULT_FOLDER_PREFIX
    include_unclustered: bool = False
    include_no_faces: bool = False



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
    """Build a VisageConfig from file, input directory, and CLI overrides.

    Priority: CLI overrides > --config file > visage.toml in input dir > defaults.
    """
    kwargs: dict[str, Any] = {}

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
                kwargs.update(_apply_toml_section(toml_data, section, key_map))
            break  # first found wins

    # Apply CLI overrides (highest priority)
    if overrides:
        kwargs.update({k: v for k, v in overrides.items() if v is not None})

    return VisageConfig(**kwargs)
