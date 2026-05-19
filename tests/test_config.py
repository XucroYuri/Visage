"""Tests for visage.config — TOML loading, config merging, overrides."""

from __future__ import annotations

import pytest

from visage.config import (
    _TOML_KEY_MAP,
    SUPPORTED_EXTENSIONS,
    VisageConfig,
    _apply_toml_section,
    build_config,
    load_config_from_file,
)

# ── VisageConfig defaults ─────────────────────────────────────────


class TestVisageConfigDefaults:
    def test_default_config(self):
        config = VisageConfig()
        assert isinstance(config, VisageConfig)

    def test_default_values(self):
        config = VisageConfig()
        assert config.detection_confidence == 0.5
        assert config.min_face_size == 40
        assert config.embedding_backend == "dlib"
        assert config.embedding_model == "small"
        assert config.num_jitters == 1
        assert config.min_face_quality == 0.0
        assert config.cluster_method == "hdbscan"
        assert config.dbscan_eps == 0.5
        assert config.dbscan_min_samples == 3
        assert config.hdbscan_min_cluster_size == 5
        assert config.cluster_selection_epsilon == 0.0
        assert config.cluster_selection_method == "eom"
        assert config.head_feature_weight == 0.2
        assert config.merge_threshold == 0.85
        assert config.small_merge_threshold == 0.80
        assert config.min_reliable_size == 10
        assert config.auto_eps is False
        assert config.copy_mode is True
        assert config.folder_prefix == "person_"
        assert config.include_unclustered is False
        assert config.include_no_faces is False
        assert config.max_workers == 4
        assert config.output_dir is None

    def test_custom_values(self):
        config = VisageConfig(dbscan_eps=0.7, min_face_size=80, embedding_model="large")
        assert config.dbscan_eps == 0.7
        assert config.min_face_size == 80
        assert config.embedding_model == "large"


# ── build_config ──────────────────────────────────────────────────


class TestBuildConfig:
    def test_no_args_returns_defaults(self):
        config = build_config()
        assert config.detection_confidence == 0.5
        assert config.dbscan_eps == 0.5

    def test_overrides(self):
        config = build_config(overrides={"dbscan_eps": 0.7, "min_face_size": 80})
        assert config.dbscan_eps == 0.7
        assert config.min_face_size == 80
        # Unrelated fields remain default
        assert config.detection_confidence == 0.5

    def test_override_none_ignored(self):
        config = build_config(overrides={"dbscan_eps": None, "min_face_size": 80})
        assert config.dbscan_eps == 0.5  # None filtered out, stays default
        assert config.min_face_size == 80

    def test_override_multiple(self):
        config = build_config(overrides={
            "detection_confidence": 0.8,
            "embedding_model": "large",
            "num_jitters": 5,
            "auto_eps": True,
        })
        assert config.detection_confidence == 0.8
        assert config.embedding_model == "large"
        assert config.num_jitters == 5
        assert config.auto_eps is True


# ── TOML file loading ─────────────────────────────────────────────


class TestTomlLoading:
    def test_load_config_from_file(self, tmp_path):
        toml_file = tmp_path / "test.toml"
        toml_file.write_text("""
[detection]
confidence = 0.8
min_face_size = 60

[embedding]
model = "large"
num_jitters = 3

[clustering]
eps = 0.4
min_samples = 3

[output]
copy_mode = false
folder_prefix = "face_"
""")
        data = load_config_from_file(toml_file)
        assert data["detection"]["confidence"] == 0.8
        assert data["embedding"]["model"] == "large"
        assert data["clustering"]["eps"] == 0.4
        assert data["output"]["copy_mode"] is False

    def test_build_config_from_file(self, tmp_path):
        toml_file = tmp_path / "config.toml"
        toml_file.write_text("""
[detection]
confidence = 0.8
""")
        config = build_config(config_file=str(toml_file))
        assert config.detection_confidence == 0.8

    def test_build_config_from_input_dir(self, tmp_path):
        toml_file = tmp_path / "visage.toml"
        toml_file.write_text("""
[clustering]
eps = 0.3
""")
        config = build_config(input_dir=str(tmp_path))
        assert config.dbscan_eps == 0.3

    def test_config_file_priority_over_input_dir(self, tmp_path):
        # input_dir has its own visage.toml
        (tmp_path / "visage.toml").write_text("""
[clustering]
eps = 0.2
""")
        # config_file is explicitly given
        config_file = tmp_path / "custom.toml"
        config_file.write_text("""
[clustering]
eps = 0.9
""")
        config = build_config(config_file=str(config_file), input_dir=str(tmp_path))
        # config_file wins (first found, break logic)
        assert config.dbscan_eps == 0.9

    def test_cli_overrides_config_file(self, tmp_path):
        toml_file = tmp_path / "config.toml"
        toml_file.write_text("""
[detection]
confidence = 0.8
""")
        config = build_config(
            config_file=str(toml_file),
            overrides={"detection_confidence": 0.9},
        )
        assert config.detection_confidence == 0.9

    def test_missing_config_file_uses_defaults(self, tmp_path):
        config = build_config(config_file=str(tmp_path / "nonexistent.toml"))
        assert config.dbscan_eps == 0.5

    def test_new_fields_from_toml(self, tmp_path):
        toml_file = tmp_path / "config.toml"
        toml_file.write_text("""
[embedding]
backend = "insightface"

[quality]
min_face_quality = 0.3

[clustering]
method = "hdbscan"
""")
        config = build_config(config_file=str(toml_file))
        assert config.embedding_backend == "insightface"
        assert config.min_face_quality == 0.3
        assert config.cluster_method == "hdbscan"

    def test_cli_overrides_new_fields(self):
        config = build_config(overrides={
            "embedding_backend": "insightface",
            "min_face_quality": 0.5,
            "cluster_method": "hdbscan",
        })
        assert config.embedding_backend == "insightface"
        assert config.min_face_quality == 0.5
        assert config.cluster_method == "hdbscan"


# ── _apply_toml_section ───────────────────────────────────────────


class TestApplyTomlSection:
    def test_applies_mapping(self):
        toml_data = {"detection": {"confidence": 0.9, "min_face_size": 60}}
        result = _apply_toml_section(toml_data, "detection", _TOML_KEY_MAP["detection"])
        assert result == {"detection_confidence": 0.9, "min_face_size": 60}

    def test_missing_section(self):
        result = _apply_toml_section({}, "detection", _TOML_KEY_MAP["detection"])
        assert result == {}

    def test_partial_keys(self):
        toml_data = {"detection": {"confidence": 0.7}}
        result = _apply_toml_section(toml_data, "detection", _TOML_KEY_MAP["detection"])
        assert result == {"detection_confidence": 0.7}

    def test_all_sections_mapped(self):
        toml_data = {
            "detection": {"confidence": 0.6, "min_face_size": 50},
            "embedding": {"model": "large", "num_jitters": 2},
            "clustering": {"eps": 0.45, "min_samples": 3},
            "output": {"copy_mode": False, "folder_prefix": "face_"},
        }
        kwargs = {}
        for section, key_map in _TOML_KEY_MAP.items():
            kwargs.update(_apply_toml_section(toml_data, section, key_map))

        config = VisageConfig(**kwargs)
        assert config.detection_confidence == 0.6
        assert config.embedding_model == "large"
        assert config.dbscan_eps == 0.45
        assert config.copy_mode is False
        assert config.folder_prefix == "face_"


# ── SUPPORTED_EXTENSIONS ──────────────────────────────────────────


class TestSupportedExtensions:
    def test_contains_jpg(self):
        assert ".jpg" in SUPPORTED_EXTENSIONS

    def test_contains_jpeg(self):
        assert ".jpeg" in SUPPORTED_EXTENSIONS

    def test_contains_png(self):
        assert ".png" in SUPPORTED_EXTENSIONS

    def test_contains_heic(self):
        assert ".heic" in SUPPORTED_EXTENSIONS

    def test_contains_heif(self):
        assert ".heif" in SUPPORTED_EXTENSIONS

    def test_contains_tif(self):
        assert ".tif" in SUPPORTED_EXTENSIONS

    def test_contains_tiff(self):
        assert ".tiff" in SUPPORTED_EXTENSIONS

    def test_no_uppercase(self):
        assert ".JPG" not in SUPPORTED_EXTENSIONS

    def test_no_raw(self):
        assert ".raw" not in SUPPORTED_EXTENSIONS

    def test_count(self):
        assert len(SUPPORTED_EXTENSIONS) == 7


# ── VisageConfig validation ────────────────────────────────────────


class TestVisageConfigValidation:
    def test_defaults_pass_validation(self):
        VisageConfig()  # should not raise

    def test_invalid_confidence_low(self):
        with pytest.raises(ValueError, match="detection_confidence"):
            VisageConfig(detection_confidence=-0.1)

    def test_invalid_confidence_high(self):
        with pytest.raises(ValueError, match="detection_confidence"):
            VisageConfig(detection_confidence=1.5)

    def test_invalid_min_face_size(self):
        with pytest.raises(ValueError, match="min_face_size"):
            VisageConfig(min_face_size=0)

    def test_invalid_backend(self):
        with pytest.raises(ValueError, match="embedding_backend"):
            VisageConfig(embedding_backend="invalid")

    def test_invalid_model(self):
        with pytest.raises(ValueError, match="embedding_model"):
            VisageConfig(embedding_model="huge")

    def test_invalid_num_jitters(self):
        with pytest.raises(ValueError, match="num_jitters"):
            VisageConfig(num_jitters=0)

    def test_invalid_min_quality_low(self):
        with pytest.raises(ValueError, match="min_face_quality"):
            VisageConfig(min_face_quality=-0.1)

    def test_invalid_min_quality_high(self):
        with pytest.raises(ValueError, match="min_face_quality"):
            VisageConfig(min_face_quality=1.5)

    def test_invalid_cluster_method(self):
        with pytest.raises(ValueError, match="cluster_method"):
            VisageConfig(cluster_method="kmeans")

    def test_invalid_eps(self):
        with pytest.raises(ValueError, match="dbscan_eps"):
            VisageConfig(dbscan_eps=0.0)

    def test_invalid_eps_negative(self):
        with pytest.raises(ValueError, match="dbscan_eps"):
            VisageConfig(dbscan_eps=-0.5)

    def test_invalid_min_samples(self):
        with pytest.raises(ValueError, match="dbscan_min_samples"):
            VisageConfig(dbscan_min_samples=0)

    def test_invalid_max_workers(self):
        with pytest.raises(ValueError, match="max_workers"):
            VisageConfig(max_workers=0)

    def test_valid_edge_values(self):
        VisageConfig(detection_confidence=0.0)
        VisageConfig(detection_confidence=1.0)
        VisageConfig(min_face_quality=0.0)
        VisageConfig(min_face_quality=1.0)
        VisageConfig(dbscan_eps=0.001)
