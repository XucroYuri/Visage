"""Tests for visage.heic — image loading with real PIL for standard formats, mocked HEIC."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from visage.heic import load_image_as_numpy, load_image_as_pil

# ── Standard format loading (real PIL, no mocking) ────────────────


class TestLoadStandardFormats:
    def test_load_jpeg_as_numpy(self, real_image_path):
        arr = load_image_as_numpy(real_image_path)
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 3
        assert arr.shape[2] == 3  # RGB
        assert arr.dtype == np.uint8

    def test_load_jpeg_as_pil(self, real_image_path):
        img = load_image_as_pil(real_image_path)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_load_png_converts_to_rgb(self, real_png_path):
        img = load_image_as_pil(real_png_path)
        assert img.mode == "RGB"

    def test_load_png_as_numpy(self, real_png_path):
        arr = load_image_as_numpy(real_png_path)
        assert arr.ndim == 3
        assert arr.shape[2] == 3

    def test_nonexistent_file(self):
        with pytest.raises(ValueError, match="Cannot load image"):
            load_image_as_pil("/nonexistent/file.jpg")

    def test_nonexistent_numpy(self):
        with pytest.raises(ValueError, match="Cannot load image"):
            load_image_as_numpy("/nonexistent/file.jpg")


# ── HEIC loading with pillow-heif ─────────────────────────────────


class TestLoadHeic:
    def test_heic_via_pillow_heif(self, tmp_path):
        # Mock pillow_heif available and PIL Image.open succeeds
        with patch("visage.heic._HEIF_AVAILABLE", True):
            mock_img = MagicMock(spec=Image.Image)
            mock_img.mode = "RGB"
            with patch("visage.heic.Image.open", return_value=mock_img):
                result = load_image_as_pil(str(tmp_path / "test.heic"))
                assert result is mock_img

    def test_heic_fallback_to_sips(self, tmp_path):
        heic_file = tmp_path / "test.heic"
        heic_file.write_text("fake-heic")

        with patch("visage.heic._HEIF_AVAILABLE", False):
            # Mock subprocess.run to succeed
            mock_result = MagicMock()
            mock_result.returncode = 0
            # Mock Image.open to return a valid image
            mock_img = MagicMock(spec=Image.Image)
            mock_img.mode = "RGB"
            mock_img.copy.return_value = mock_img

            with patch("visage.heic.subprocess.run", return_value=mock_result), \
                 patch("visage.heic.Image.open", return_value=mock_img):
                result = load_image_as_pil(str(heic_file))
                assert result is mock_img

    def test_heic_sips_failure(self, tmp_path):
        heic_file = tmp_path / "test.heic"
        heic_file.write_text("fake-heic")

        with patch("visage.heic._HEIF_AVAILABLE", False):
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "sips error"

            with patch("visage.heic.subprocess.run", return_value=mock_result):
                with pytest.raises(ValueError, match="sips conversion failed"):
                    load_image_as_pil(str(heic_file))

    def test_heic_sips_not_found(self, tmp_path):
        heic_file = tmp_path / "test.heic"
        heic_file.write_text("fake-heic")

        with patch("visage.heic._HEIF_AVAILABLE", False):
            with patch("visage.heic.subprocess.run", side_effect=FileNotFoundError):
                with pytest.raises(ValueError, match="sips command not found"):
                    load_image_as_pil(str(heic_file))

    def test_heic_sips_timeout(self, tmp_path):
        heic_file = tmp_path / "test.heic"
        heic_file.write_text("fake-heic")

        with patch("visage.heic._HEIF_AVAILABLE", False):
            with patch(
                "visage.heic.subprocess.run",
                side_effect=subprocess.TimeoutExpired("sips", 30),
            ):
                with pytest.raises(ValueError, match="timed out"):
                    load_image_as_pil(str(heic_file))

    def test_heic_nonexistent_file(self):
        with patch("visage.heic._HEIF_AVAILABLE", False):
            with pytest.raises(ValueError, match="File not found"):
                load_image_as_pil("/nonexistent/test.heic")
