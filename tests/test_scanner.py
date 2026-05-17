"""Tests for visage.scanner — file discovery with tmp_path, no mocking."""

from __future__ import annotations

import os

import pytest

from visage.scanner import is_supported_image, scan_images


# ── is_supported_image ────────────────────────────────────────────


class TestIsSupportedImage:
    def test_jpg(self):
        assert is_supported_image("photo.jpg") is True

    def test_jpeg(self):
        assert is_supported_image("photo.jpeg") is True

    def test_png(self):
        assert is_supported_image("photo.png") is True

    def test_heic(self):
        assert is_supported_image("photo.heic") is True

    def test_heif(self):
        assert is_supported_image("photo.heif") is True

    def test_tif(self):
        assert is_supported_image("scan.tif") is True

    def test_tiff(self):
        assert is_supported_image("scan.tiff") is True

    def test_uppercase_extension(self):
        assert is_supported_image("photo.JPG") is True

    def test_mixed_case(self):
        assert is_supported_image("photo.JpG") is True

    def test_unsupported_pdf(self):
        assert is_supported_image("doc.pdf") is False

    def test_unsupported_gif(self):
        assert is_supported_image("anim.gif") is False

    def test_unsupported_raw(self):
        assert is_supported_image("photo.cr2") is False

    def test_dotfile(self):
        assert is_supported_image(".DS_Store") is False

    def test_no_extension(self):
        assert is_supported_image("README") is False

    def test_empty_string(self):
        assert is_supported_image("") is False

    def test_only_extension(self):
        assert is_supported_image(".jpg") is False


# ── scan_images ───────────────────────────────────────────────────


class TestScanImages:
    def test_empty_directory(self, tmp_path):
        result = scan_images(str(tmp_path))
        assert result == []

    def test_finds_jpg(self, tmp_path):
        (tmp_path / "photo.jpg").write_text("img")
        result = scan_images(str(tmp_path))
        assert len(result) == 1
        assert result[0].endswith("photo.jpg")

    def test_finds_multiple_extensions(self, tmp_image_dir):
        result = scan_images(str(tmp_image_dir))
        basenames = [os.path.basename(p) for p in result]
        assert "photo1.jpg" in basenames
        assert "photo2.png" in basenames
        # photo3.JPG should match (case-insensitive)
        assert any("photo3" in b for b in basenames)
        # subdir photo
        assert any("photo4.heic" in b for b in basenames)

    def test_recursive(self, tmp_image_dir):
        result = scan_images(str(tmp_image_dir))
        basenames = [os.path.basename(p) for p in result]
        assert any("photo4.heic" in b for b in basenames)

    def test_skips_hidden_dirs(self, tmp_image_dir):
        result = scan_images(str(tmp_image_dir))
        basenames = [os.path.basename(p) for p in result]
        assert "hidden_photo.jpg" not in basenames

    def test_skips_visage_output(self, tmp_image_dir):
        result = scan_images(str(tmp_image_dir))
        basenames = [os.path.basename(p) for p in result]
        assert "output_photo.jpg" not in basenames

    def test_skips_dot_visage_cache(self, tmp_image_dir):
        result = scan_images(str(tmp_image_dir))
        basenames = [os.path.basename(p) for p in result]
        assert "embeddings.db" not in basenames

    def test_skips_unsupported_files(self, tmp_image_dir):
        result = scan_images(str(tmp_image_dir))
        basenames = [os.path.basename(p) for p in result]
        assert "doc.pdf" not in basenames

    def test_returns_sorted(self, tmp_image_dir):
        result = scan_images(str(tmp_image_dir))
        assert result == sorted(result)

    def test_returns_absolute_paths(self, tmp_image_dir):
        result = scan_images(str(tmp_image_dir))
        for path in result:
            assert os.path.isabs(path)

    def test_nonexistent_dir(self):
        with pytest.raises(ValueError, match="Not a directory"):
            scan_images("/nonexistent/path/xyz")

    def test_file_not_dir(self, tmp_path):
        file_path = tmp_path / "file.txt"
        file_path.write_text("not a dir")
        with pytest.raises(ValueError, match="Not a directory"):
            scan_images(str(file_path))

    def test_count_expected_files(self, tmp_image_dir):
        result = scan_images(str(tmp_image_dir))
        # photo1.jpg, photo2.png, photo3.JPG, subdir/photo4.heic = 4
        assert len(result) == 4
