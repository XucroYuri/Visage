"""Tests for visage.organizer — plan building and file operations with tmp_path."""

from __future__ import annotations

import os

import pytest

from visage.models import DetectedFace, FaceBox, ImageResult, OrganizePlan
from visage.organizer import (
    _unique_dest_path,
    build_organize_plan,
    execute_organize_plan,
)


def _make_result(path: str, faces: int = 0, error: str | None = None, skipped: bool = False) -> ImageResult:
    """Helper to build ImageResult for organizer tests."""
    face_list = []
    for i in range(faces):
        face_list.append(DetectedFace(
            face_box=FaceBox(top=10, right=110, bottom=110, left=10),
            confidence=0.9,
            image_path=path,
            face_index=i,
        ))
    return ImageResult(path=path, faces=face_list, error=error, skipped=skipped)


# ── build_organize_plan ───────────────────────────────────────────


class TestBuildOrganizePlan:
    def test_basic(self):
        results = [_make_result("/a.jpg", faces=1), _make_result("/b.jpg", faces=1)]
        mapping = {0: ["/a.jpg"], 1: ["/b.jpg"]}
        plan = build_organize_plan(results, mapping)
        assert plan.person_folders[0] == ["/a.jpg"]
        assert plan.person_folders[1] == ["/b.jpg"]

    def test_empty_inputs(self):
        plan = build_organize_plan([], {})
        assert plan.person_folders == {}
        assert plan.unclustered == []
        assert plan.no_faces == []

    def test_unclustered(self):
        results = [
            _make_result("/clustered.jpg", faces=1),
            _make_result("/unclustered.jpg", faces=1),
        ]
        mapping = {0: ["/clustered.jpg"]}
        plan = build_organize_plan(results, mapping, include_unclustered=True)
        assert "/clustered.jpg" not in plan.unclustered
        assert "/unclustered.jpg" in plan.unclustered

    def test_no_unclustered_when_disabled(self):
        results = [_make_result("/a.jpg", faces=1)]
        mapping = {}
        plan = build_organize_plan(results, mapping, include_unclustered=False)
        assert plan.unclustered == []

    def test_no_faces_folder(self):
        results = [_make_result("/no_face.jpg", skipped=True)]
        plan = build_organize_plan(results, {}, include_no_faces=True)
        assert "/no_face.jpg" in plan.no_faces

    def test_no_faces_excludes_errors(self):
        results = [_make_result("/bad.jpg", error="corrupt")]
        plan = build_organize_plan(results, {}, include_no_faces=True)
        assert "/bad.jpg" not in plan.no_faces

    def test_sorted_unclustered(self):
        results = [
            _make_result("/c.jpg", faces=1),
            _make_result("/a.jpg", faces=1),
            _make_result("/b.jpg", faces=1),
        ]
        mapping = {0: ["/c.jpg"]}
        plan = build_organize_plan(results, mapping, include_unclustered=True)
        assert plan.unclustered == sorted(plan.unclustered)

    def test_sorted_no_faces(self):
        results = [
            _make_result("/c.jpg", skipped=True),
            _make_result("/a.jpg", skipped=True),
        ]
        plan = build_organize_plan(results, {}, include_no_faces=True)
        assert plan.no_faces == sorted(plan.no_faces)


# ── _unique_dest_path ─────────────────────────────────────────────


class TestUniqueDestPath:
    def test_no_collision(self, tmp_path):
        result = _unique_dest_path(str(tmp_path), "photo.jpg")
        assert result == str(tmp_path / "photo.jpg")

    def test_with_collision(self, tmp_path):
        (tmp_path / "photo.jpg").write_text("existing")
        result = _unique_dest_path(str(tmp_path), "photo.jpg")
        assert result.endswith("photo_1.jpg")

    def test_multiple_collisions(self, tmp_path):
        (tmp_path / "photo.jpg").write_text("existing")
        (tmp_path / "photo_1.jpg").write_text("existing")
        result = _unique_dest_path(str(tmp_path), "photo.jpg")
        assert result.endswith("photo_2.jpg")

    def test_preserves_extension(self, tmp_path):
        result = _unique_dest_path(str(tmp_path), "scan.heic")
        assert result.endswith(".heic")


# ── execute_organize_plan ─────────────────────────────────────────


class TestExecuteOrganizePlan:
    def _create_source_files(self, tmp_path, filenames: list[str]) -> list[str]:
        """Create source files and return their paths."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        paths = []
        for name in filenames:
            p = src_dir / name
            p.write_text(f"content-{name}")
            paths.append(str(p))
        return paths

    def test_copy(self, tmp_path):
        paths = self._create_source_files(tmp_path, ["a.jpg", "b.jpg"])
        plan = OrganizePlan(person_folders={0: paths}, unclustered=[], no_faces=[])
        out_dir = str(tmp_path / "output")
        stats = execute_organize_plan(plan, out_dir, copy_mode=True)
        assert stats["copy"] == 2
        # Source files still exist
        assert all(os.path.exists(p) for p in paths)
        # Output files exist
        assert os.path.exists(os.path.join(out_dir, "person_00", "a.jpg"))
        assert os.path.exists(os.path.join(out_dir, "person_00", "b.jpg"))

    def test_move(self, tmp_path):
        paths = self._create_source_files(tmp_path, ["a.jpg"])
        plan = OrganizePlan(person_folders={0: paths}, unclustered=[], no_faces=[])
        out_dir = str(tmp_path / "output")
        stats = execute_organize_plan(plan, out_dir, copy_mode=False)
        assert stats["move"] == 1
        # Source file gone
        assert not os.path.exists(paths[0])
        # Output file exists
        assert os.path.exists(os.path.join(out_dir, "person_00", "a.jpg"))

    def test_dry_run(self, tmp_path):
        paths = self._create_source_files(tmp_path, ["a.jpg"])
        plan = OrganizePlan(person_folders={0: paths}, unclustered=[], no_faces=[])
        out_dir = str(tmp_path / "output")
        stats = execute_organize_plan(plan, out_dir, dry_run=True)
        assert stats["skipped"] == 1
        # Nothing created
        assert not os.path.exists(out_dir)

    def test_creates_subdirs(self, tmp_path):
        paths = self._create_source_files(tmp_path, ["a.jpg"])
        plan = OrganizePlan(person_folders={0: paths}, unclustered=[], no_faces=[])
        out_dir = str(tmp_path / "output")
        execute_organize_plan(plan, out_dir)
        assert os.path.isdir(os.path.join(out_dir, "person_00"))

    def test_progress_callback(self, tmp_path):
        paths = self._create_source_files(tmp_path, ["a.jpg", "b.jpg"])
        plan = OrganizePlan(person_folders={0: paths}, unclustered=[], no_faces=[])
        out_dir = str(tmp_path / "output")
        callbacks = []
        execute_organize_plan(
            plan, out_dir,
            progress_callback=lambda c, t, f: callbacks.append((c, t, f)),
        )
        assert len(callbacks) == 2
        assert callbacks[0][0] == 1
        assert callbacks[1][0] == 2
        assert callbacks[0][1] == 2  # total

    def test_error_handling(self, tmp_path):
        # Include a non-existent file path
        paths = ["/nonexistent/deleted_file.jpg"]
        plan = OrganizePlan(person_folders={0: paths}, unclustered=[], no_faces=[])
        out_dir = str(tmp_path / "output")
        stats = execute_organize_plan(plan, out_dir)
        assert stats["errors"] == 1

    def test_unclustered_folder(self, tmp_path):
        paths = self._create_source_files(tmp_path, ["a.jpg"])
        plan = OrganizePlan(person_folders={}, unclustered=paths, no_faces=[])
        out_dir = str(tmp_path / "output")
        execute_organize_plan(plan, out_dir)
        assert os.path.exists(os.path.join(out_dir, "_unclustered", "a.jpg"))

    def test_no_faces_folder(self, tmp_path):
        paths = self._create_source_files(tmp_path, ["empty.jpg"])
        plan = OrganizePlan(person_folders={}, unclustered=[], no_faces=paths)
        out_dir = str(tmp_path / "output")
        execute_organize_plan(plan, out_dir)
        assert os.path.exists(os.path.join(out_dir, "_no_faces", "empty.jpg"))

    def test_collision_resolution(self, tmp_path):
        # Two clusters both reference a file with the same basename
        paths_a = self._create_source_files(tmp_path, ["shared.jpg"])
        # Create a second source file with same name in different dir
        src2 = tmp_path / "src2"
        src2.mkdir()
        p2 = src2 / "shared.jpg"
        p2.write_text("different-content")
        paths_b = [str(p2)]

        plan = OrganizePlan(
            person_folders={0: paths_a, 1: paths_b},
            unclustered=[], no_faces=[],
        )
        out_dir = str(tmp_path / "output")
        stats = execute_organize_plan(plan, out_dir)
        assert stats["copy"] == 2
        # Both should exist (one may have _1 suffix)
        files_in_00 = os.listdir(os.path.join(out_dir, "person_00"))
        files_in_01 = os.listdir(os.path.join(out_dir, "person_01"))
        assert "shared.jpg" in files_in_00
        assert "shared.jpg" in files_in_01

    def test_multiple_person_folders(self, tmp_path):
        src_a = tmp_path / "src_a"
        src_a.mkdir()
        (src_a / "face_a.jpg").write_text("a")
        src_b = tmp_path / "src_b"
        src_b.mkdir()
        (src_b / "face_b.jpg").write_text("b")

        plan = OrganizePlan(
            person_folders={0: [str(src_a / "face_a.jpg")], 1: [str(src_b / "face_b.jpg")]},
            unclustered=[], no_faces=[],
        )
        out_dir = str(tmp_path / "output")
        execute_organize_plan(plan, out_dir)
        assert os.path.exists(os.path.join(out_dir, "person_00", "face_a.jpg"))
        assert os.path.exists(os.path.join(out_dir, "person_01", "face_b.jpg"))
