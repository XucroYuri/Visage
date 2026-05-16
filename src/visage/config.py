from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff"})

DEFAULT_FOLDER_PREFIX = "person_"
DEFAULT_OUTPUT_DIRNAME = "visage_output"


@dataclass
class VisageConfig:
    """All tunable parameters with sensible defaults."""

    # Face detection (Vision framework)
    detection_confidence: float = 0.5
    min_face_size: int = 40  # minimum face bounding box dimension in pixels

    # Face embedding (face_recognition library)
    embedding_model: str = "small"  # "small" (fast) or "large" (accurate)
    num_jitters: int = 1  # times to re-sample for embedding

    # Clustering (DBSCAN)
    dbscan_eps: float = 0.5  # max distance between embeddings in same cluster
    dbscan_min_samples: int = 2  # min faces to form a cluster
    auto_eps: bool = False  # automatically estimate eps using k-distance elbow

    # Processing
    batch_size: int = 100  # images per batch for progress reporting
    max_workers: int = 4  # parallel detection workers

    # Output
    copy_mode: bool = True  # True = copy, False = move
    output_dir: Optional[str] = None  # None = create subdirs inside input folder
    folder_prefix: str = DEFAULT_FOLDER_PREFIX
    include_unclustered: bool = False
    include_no_faces: bool = False

    # HEIC handling
    heic_converter: str = "pillow_heif"  # "pillow_heif" or "sips"


def load_config_from_file(path: Path) -> dict[str, Any]:
    """Load configuration from a TOML file."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return data


def _apply_toml_section(config_dict: dict[str, Any], section: str, mapping: dict[str, str]) -> dict[str, Any]:
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
        "model": "embedding_model",
        "num_jitters": "num_jitters",
    },
    "clustering": {
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
    config_file: Optional[str] = None,
    input_dir: Optional[str] = None,
    overrides: Optional[dict[str, Any]] = None,
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
