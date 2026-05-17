"""Tests for visage.cli — argument parser and main() entry point with mocked pipeline."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from visage.cli import _build_parser, main
from visage.models import OrganizePlan, PipelineResult

# ── _build_parser ─────────────────────────────────────────────────


class TestBuildParser:
    def test_creates_parser(self):
        parser = _build_parser()
        assert parser is not None

    def test_input_required(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_move_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["/path", "--move"])
        assert args.move is True

    def test_dry_run_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["/path", "--dry-run"])
        assert args.dry_run is True

    def test_json_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["/path", "--json"])
        assert args.json is True

    def test_eps_option(self):
        parser = _build_parser()
        args = parser.parse_args(["/path", "--eps", "0.6"])
        assert args.eps == 0.6

    def test_auto_eps_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["/path", "--auto-eps"])
        assert args.auto_eps is True

    def test_backend_option(self):
        parser = _build_parser()
        args = parser.parse_args(["/path", "--backend", "insightface"])
        assert args.backend == "insightface"

    def test_backend_default_is_none(self):
        parser = _build_parser()
        args = parser.parse_args(["/path"])
        assert args.backend is None

    def test_cluster_method_option(self):
        parser = _build_parser()
        args = parser.parse_args(["/path", "--cluster-method", "hdbscan"])
        assert args.cluster_method == "hdbscan"

    def test_cluster_method_invalid_exits(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["/path", "--cluster-method", "kmeans"])

    def test_min_quality_option(self):
        parser = _build_parser()
        args = parser.parse_args(["/path", "--min-quality", "0.3"])
        assert args.min_quality == 0.3

    def test_model_choices(self):
        parser = _build_parser()
        args = parser.parse_args(["/path", "--model", "large"])
        assert args.model == "large"

    def test_model_invalid_exits(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["/path", "--model", "tiny"])

    def test_quiet_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["/path", "-q"])
        assert args.quiet is True

    def test_verbose_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["/path", "-v"])
        assert args.verbose is True

    def test_min_confidence(self):
        parser = _build_parser()
        args = parser.parse_args(["/path", "--min-confidence", "0.8"])
        assert args.min_confidence == 0.8

    def test_max_workers(self):
        parser = _build_parser()
        args = parser.parse_args(["/path", "--max-workers", "8"])
        assert args.max_workers == 8

    def test_num_jitters(self):
        parser = _build_parser()
        args = parser.parse_args(["/path", "--num-jitters", "5"])
        assert args.num_jitters == 5

    def test_min_samples(self):
        parser = _build_parser()
        args = parser.parse_args(["/path", "--min-samples", "3"])
        assert args.min_samples == 3

    def test_config_option(self):
        parser = _build_parser()
        args = parser.parse_args(["/path", "--config", "/etc/visage.toml"])
        assert args.config == "/etc/visage.toml"


# ── main() ────────────────────────────────────────────────────────


def _mock_pipeline_result(**overrides) -> PipelineResult:
    """Create a standard success PipelineResult for mocking."""
    defaults = dict(
        total_images=100,
        images_with_faces=80,
        total_faces=120,
        num_clusters=5,
        num_noise_faces=10,
        organize_plan=OrganizePlan(
            person_folders={0: ["/a.jpg", "/b.jpg"], 1: ["/c.jpg"]},
            unclustered=[], no_faces=[],
        ),
        cluster_confidences={0: 0.95, 1: 0.88},
        duration_seconds=5.0,
        errors=[],
    )
    defaults.update(overrides)
    return PipelineResult(**defaults)


class TestMain:
    def test_returns_zero_on_success(self, tmp_path):
        (tmp_path / "photo.jpg").write_text("img")
        with patch("visage.cli.run_pipeline") as mock_pipeline, \
             patch("visage.cli.EmbeddingCache") as mock_cache_cls:

            mock_pipeline.return_value = _mock_pipeline_result()
            mock_cache = MagicMock()
            mock_cache.load_checkpoint.return_value = None
            mock_cache_cls.return_value = mock_cache

            result = main([str(tmp_path)])
            assert result == 0

    def test_returns_one_on_exception(self, tmp_path):
        (tmp_path / "photo.jpg").write_text("img")
        with patch("visage.cli.run_pipeline") as mock_pipeline, \
             patch("visage.cli.EmbeddingCache") as mock_cache_cls:

            mock_pipeline.side_effect = Exception("fatal")
            mock_cache = MagicMock()
            mock_cache.load_checkpoint.return_value = None
            mock_cache_cls.return_value = mock_cache

            result = main([str(tmp_path)])
            assert result == 1

    def test_returns_130_on_keyboard_interrupt(self, tmp_path):
        (tmp_path / "photo.jpg").write_text("img")
        with patch("visage.cli.run_pipeline") as mock_pipeline, \
             patch("visage.cli.EmbeddingCache") as mock_cache_cls:

            mock_pipeline.side_effect = KeyboardInterrupt()
            mock_cache = MagicMock()
            mock_cache.load_checkpoint.return_value = None
            mock_cache_cls.return_value = mock_cache

            result = main([str(tmp_path)])
            assert result == 130

    def test_json_output(self, tmp_path, capsys):
        (tmp_path / "photo.jpg").write_text("img")
        with patch("visage.cli.run_pipeline") as mock_pipeline, \
             patch("visage.cli.EmbeddingCache") as mock_cache_cls:

            mock_pipeline.return_value = _mock_pipeline_result()
            mock_cache = MagicMock()
            mock_cache.load_checkpoint.return_value = None
            mock_cache_cls.return_value = mock_cache

            result = main([str(tmp_path), "--json"])
            assert result == 0

            output = capsys.readouterr().out
            data = json.loads(output)
            assert "total_images" in data
            assert data["total_images"] == 100
            assert "persons" in data
            assert "person_00" in data["persons"]

    def test_copy_mode_default(self):
        parser = _build_parser()
        args = parser.parse_args(["/path"])
        overrides = {"copy_mode": not args.move}
        assert overrides["copy_mode"] is True

    def test_copy_mode_from_move(self):
        parser = _build_parser()
        args = parser.parse_args(["/path", "--move"])
        overrides = {"copy_mode": not args.move}
        assert overrides["copy_mode"] is False
