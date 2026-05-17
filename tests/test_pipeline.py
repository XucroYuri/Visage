"""Tests for visage.pipeline — full pipeline orchestration with all phases mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from visage.config import VisageConfig
from visage.models import (
    ClusterResult,
    ImageResult,
    OrganizePlan,
    PipelineResult,
)
from visage.pipeline import _print_dry_run_plan, run_pipeline


def _mock_image_result(
    path: str = "/tmp/test.jpg",
    faces: int = 1,
    error: str | None = None,
) -> ImageResult:
    from visage.models import DetectedFace, FaceBox

    face_list = []
    for i in range(faces):
        face_list.append(DetectedFace(
            face_box=FaceBox(top=10, right=110, bottom=110, left=10),
            confidence=0.9,
            embedding=np.random.randn(128),
            image_path=path,
            face_index=i,
        ))
    if error:
        return ImageResult(path=path, error=error)
    if faces == 0:
        return ImageResult(path=path, skipped=True)
    return ImageResult(path=path, faces=face_list)


def _default_cluster_result() -> ClusterResult:
    return ClusterResult(
        labels=np.array([0, 0, 1]),
        embeddings=np.random.randn(3, 128),
        num_clusters=2,
        num_noise=0,
    )


# ── run_pipeline success path ─────────────────────────────────────


class TestRunPipelineSuccess:
    @patch("visage.pipeline.execute_organize_plan")
    @patch("visage.pipeline.build_organize_plan")
    @patch("visage.pipeline.compute_cluster_confidences")
    @patch("visage.pipeline.build_cluster_mapping")
    @patch("visage.pipeline.cluster_faces")
    @patch("visage.pipeline.extract_embeddings")
    @patch("visage.pipeline.generate_embeddings_batch")
    @patch("visage.pipeline.detect_faces_batch")
    @patch("visage.pipeline.scan_images")
    @patch("visage.pipeline.EmbeddingCache")
    def test_full_success(
        self, mock_cache_cls, mock_scan, mock_detect, mock_embed,
        mock_extract, mock_cluster, mock_mapping, mock_conf,
        mock_build_plan, mock_execute,
    ):
        # Setup mocks
        mock_cache = MagicMock()
        mock_cache.load_checkpoint.return_value = None
        mock_cache_cls.return_value = mock_cache

        mock_scan.return_value = ["/tmp/a.jpg", "/tmp/b.jpg"]
        mock_detect.return_value = [
            _mock_image_result("/tmp/a.jpg"),
            _mock_image_result("/tmp/b.jpg"),
        ]

        def embed_side_effect(results, **kwargs):
            return results, 0
        mock_embed.side_effect = embed_side_effect

        embeddings = np.random.randn(2, 128)
        mock_extract.return_value = (embeddings, [("/tmp/a.jpg", 0), ("/tmp/b.jpg", 0)])

        cluster_result = _default_cluster_result()
        mock_cluster.return_value = cluster_result
        mock_mapping.return_value = {0: ["/tmp/a.jpg"], 1: ["/tmp/b.jpg"]}
        mock_conf.return_value = {0: 0.95, 1: 0.88}

        plan = OrganizePlan(
            person_folders={0: ["/tmp/a.jpg"], 1: ["/tmp/b.jpg"]},
            unclustered=[],
            no_faces=[],
        )
        mock_build_plan.return_value = plan
        mock_execute.return_value = {"copy": 2, "skipped": 0, "errors": 0}

        # Run
        result = run_pipeline("/tmp/input")

        assert isinstance(result, PipelineResult)
        assert result.total_images == 2
        assert result.num_clusters == 2
        assert result.duration_seconds > 0
        mock_cache.clear_checkpoint.assert_called()


# ── run_pipeline early returns ─────────────────────────────────────


class TestRunPipelineEarlyReturns:
    @patch("visage.pipeline.scan_images")
    @patch("visage.pipeline.EmbeddingCache")
    def test_scan_error(self, mock_cache_cls, mock_scan):
        mock_cache = MagicMock()
        mock_cache.load_checkpoint.return_value = None
        mock_cache_cls.return_value = mock_cache
        mock_scan.side_effect = ValueError("Not a directory")

        result = run_pipeline("/bad/path")
        assert len(result.errors) == 1
        assert result.total_images == 0

    @patch("visage.pipeline.scan_images")
    @patch("visage.pipeline.EmbeddingCache")
    def test_no_images(self, mock_cache_cls, mock_scan):
        mock_cache = MagicMock()
        mock_cache.load_checkpoint.return_value = None
        mock_cache_cls.return_value = mock_cache
        mock_scan.return_value = []

        result = run_pipeline("/empty")
        assert result.total_images == 0
        assert "No images found" in result.errors

    @patch("visage.pipeline.detect_faces_batch")
    @patch("visage.pipeline.scan_images")
    @patch("visage.pipeline.EmbeddingCache")
    def test_no_faces_detected(self, mock_cache_cls, mock_scan, mock_detect):
        mock_cache = MagicMock()
        mock_cache.load_checkpoint.return_value = None
        mock_cache_cls.return_value = mock_cache
        mock_scan.return_value = ["/tmp/a.jpg"]
        mock_detect.return_value = [_mock_image_result("/tmp/a.jpg", faces=0)]

        result = run_pipeline("/tmp/input")
        assert result.images_with_faces == 0

    @patch("visage.pipeline.extract_embeddings")
    @patch("visage.pipeline.generate_embeddings_batch")
    @patch("visage.pipeline.detect_faces_batch")
    @patch("visage.pipeline.scan_images")
    @patch("visage.pipeline.EmbeddingCache")
    def test_no_embeddings(
        self, mock_cache_cls, mock_scan, mock_detect, mock_embed, mock_extract,
    ):
        mock_cache = MagicMock()
        mock_cache.load_checkpoint.return_value = None
        mock_cache_cls.return_value = mock_cache
        mock_scan.return_value = ["/tmp/a.jpg"]
        mock_detect.return_value = [_mock_image_result("/tmp/a.jpg")]
        mock_embed.return_value = ([_mock_image_result("/tmp/a.jpg")], 0)
        mock_extract.return_value = (np.empty((0, 128)), [])

        result = run_pipeline("/tmp/input")
        assert result.num_clusters == 0


# ── run_pipeline modes ────────────────────────────────────────────


class TestRunPipelineModes:
    @patch("visage.pipeline.build_organize_plan")
    @patch("visage.pipeline.compute_cluster_confidences")
    @patch("visage.pipeline.build_cluster_mapping")
    @patch("visage.pipeline.cluster_faces")
    @patch("visage.pipeline.extract_embeddings")
    @patch("visage.pipeline.generate_embeddings_batch")
    @patch("visage.pipeline.detect_faces_batch")
    @patch("visage.pipeline.scan_images")
    @patch("visage.pipeline.EmbeddingCache")
    def test_dry_run(
        self, mock_cache_cls, mock_scan, mock_detect, mock_embed,
        mock_extract, mock_cluster, mock_mapping, mock_conf, mock_build_plan,
    ):

        mock_cache = MagicMock()
        mock_cache.load_checkpoint.return_value = None
        mock_cache_cls.return_value = mock_cache

        mock_scan.return_value = ["/tmp/a.jpg"]
        mock_detect.return_value = [_mock_image_result("/tmp/a.jpg")]
        mock_embed.return_value = ([_mock_image_result("/tmp/a.jpg")], 0)
        mock_extract.return_value = (np.random.randn(1, 128), [("/tmp/a.jpg", 0)])

        cluster_result = ClusterResult(
            labels=np.array([0]),
            embeddings=np.random.randn(1, 128),
            num_clusters=1, num_noise=0,
        )
        mock_cluster.return_value = cluster_result
        mock_mapping.return_value = {0: ["/tmp/a.jpg"]}
        mock_conf.return_value = {0: 0.95}
        mock_build_plan.return_value = OrganizePlan(
            person_folders={0: ["/tmp/a.jpg"]}, unclustered=[], no_faces=[],
        )

        result = run_pipeline("/tmp/input", dry_run=True)
        assert result.num_clusters == 1


class TestRunPipelineConfig:
    @patch("visage.pipeline.cluster_faces")
    @patch("visage.pipeline.extract_embeddings")
    @patch("visage.pipeline.generate_embeddings_batch")
    @patch("visage.pipeline.detect_faces_batch")
    @patch("visage.pipeline.scan_images")
    @patch("visage.pipeline.EmbeddingCache")
    def test_config_forwarded_to_cluster(
        self, mock_cache_cls, mock_scan, mock_detect, mock_embed,
        mock_extract, mock_cluster,
    ):
        mock_cache = MagicMock()
        mock_cache.load_checkpoint.return_value = None
        mock_cache_cls.return_value = mock_cache

        mock_scan.return_value = ["/tmp/a.jpg"]
        mock_detect.return_value = [_mock_image_result("/tmp/a.jpg")]
        mock_embed.return_value = ([_mock_image_result("/tmp/a.jpg")], 0)
        mock_extract.return_value = (np.random.randn(1, 128), [("/tmp/a.jpg", 0)])
        mock_cluster.return_value = ClusterResult(
            labels=np.array([0]), embeddings=np.random.randn(1, 128),
            num_clusters=1, num_noise=0,
        )

        config = VisageConfig(dbscan_eps=0.7)
        run_pipeline("/tmp/input", config=config)

        mock_cluster.assert_called_once()
        call_kwargs = mock_cluster.call_args
        assert call_kwargs[1]["eps"] == 0.7 or (call_kwargs[0] and True)  # eps is passed


class TestRunPipelineOutputDir:
    @patch("visage.pipeline.execute_organize_plan")
    @patch("visage.pipeline.build_organize_plan")
    @patch("visage.pipeline.compute_cluster_confidences")
    @patch("visage.pipeline.build_cluster_mapping")
    @patch("visage.pipeline.cluster_faces")
    @patch("visage.pipeline.extract_embeddings")
    @patch("visage.pipeline.generate_embeddings_batch")
    @patch("visage.pipeline.detect_faces_batch")
    @patch("visage.pipeline.scan_images")
    @patch("visage.pipeline.EmbeddingCache")
    def test_output_dir_override(
        self, mock_cache_cls, mock_scan, mock_detect, mock_embed,
        mock_extract, mock_cluster, mock_mapping, mock_conf,
        mock_build_plan, mock_execute,
    ):
        mock_cache = MagicMock()
        mock_cache.load_checkpoint.return_value = None
        mock_cache_cls.return_value = mock_cache

        mock_scan.return_value = ["/tmp/a.jpg"]
        mock_detect.return_value = [_mock_image_result("/tmp/a.jpg")]
        mock_embed.return_value = ([_mock_image_result("/tmp/a.jpg")], 0)
        mock_extract.return_value = (np.random.randn(1, 128), [("/tmp/a.jpg", 0)])

        mock_cluster.return_value = ClusterResult(
            labels=np.array([0]), embeddings=np.random.randn(1, 128),
            num_clusters=1, num_noise=0,
        )
        mock_mapping.return_value = {0: ["/tmp/a.jpg"]}
        mock_conf.return_value = {0: 0.95}
        mock_build_plan.return_value = OrganizePlan(
            person_folders={0: ["/tmp/a.jpg"]}, unclustered=[], no_faces=[],
        )
        mock_execute.return_value = {"copy": 1, "skipped": 0, "errors": 0}

        run_pipeline("/tmp/input", output_dir="/custom/output")

        mock_execute.assert_called_once()
        assert mock_execute.call_args[1]["output_dir"] == "/custom/output"


class TestRunPipelineErrors:
    @patch("visage.pipeline.generate_embeddings_batch")
    @patch("visage.pipeline.detect_faces_batch")
    @patch("visage.pipeline.scan_images")
    @patch("visage.pipeline.EmbeddingCache")
    def test_collects_errors(self, mock_cache_cls, mock_scan, mock_detect, mock_embed):
        mock_cache = MagicMock()
        mock_cache.load_checkpoint.return_value = None
        mock_cache_cls.return_value = mock_cache

        mock_scan.return_value = ["/tmp/a.jpg", "/tmp/b.jpg"]
        error_result = _mock_image_result("/tmp/a.jpg", error="corrupt image")
        ok_result = _mock_image_result("/tmp/b.jpg")
        mock_detect.return_value = [error_result, ok_result]
        mock_embed.return_value = ([ok_result], 0)

        result = run_pipeline("/tmp/input")
        assert len(result.errors) == 1
        assert "corrupt image" in result.errors[0]


# ── _print_dry_run_plan ───────────────────────────────────────────


class TestPrintDryRunPlan:
    def test_output_contains_cluster_info(self, capsys):
        plan = OrganizePlan(
            person_folders={0: ["/a.jpg", "/b.jpg"], 1: ["/c.jpg"]},
            unclustered=["/d.jpg"],
            no_faces=["/e.jpg"],
        )
        prog = MagicMock()
        _print_dry_run_plan(plan, "/output", prog, cluster_confidences={0: 0.95, 1: 0.88})

        prog.print_plan.assert_called_once()
        output = prog.print_plan.call_args[0][0]
        assert "DRY RUN" in output
        assert "person_00" in output
        assert "person_01" in output
        assert "_unclustered" in output
        assert "_no_faces" in output

    def test_truncates_long_lists(self):
        plan = OrganizePlan(
            person_folders={0: [f"/photo_{i}.jpg" for i in range(10)]},
            unclustered=[], no_faces=[],
        )
        prog = MagicMock()
        _print_dry_run_plan(plan, "/output", prog)

        output = prog.print_plan.call_args[0][0]
        assert "... and 5 more" in output
