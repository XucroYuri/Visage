from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_HEIC_EXTENSIONS = frozenset({".heic", ".heif"})

# Register pillow-heif as a Pillow opener on import
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    _HEIF_AVAILABLE = True
except ImportError:
    _HEIF_AVAILABLE = False


def load_image_as_numpy(path: str, max_dimension: int = 0) -> np.ndarray:
    """Load any supported image as an RGB numpy array.

    Args:
        path: Path to the image file.
        max_dimension: If > 0, downscale the image so its longest side
                       does not exceed this value (preserves aspect ratio).

    Returns:
        numpy array of shape (H, W, 3) with dtype uint8.

    Raises:
        ValueError: If the image cannot be loaded.
    """
    img = load_image_as_pil(path)

    if max_dimension > 0:
        w, h = img.size
        longest = max(w, h)
        if longest > max_dimension:
            ratio = max_dimension / longest
            new_w, new_h = int(w * ratio), int(h * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)

    return np.array(img)


def load_image_as_pil(path: str) -> Image.Image:
    """Load any supported image as a PIL Image in RGB mode.

    For HEIC files, tries pillow-heif first, falls back to macOS sips command.

    Args:
        path: Path to the image file.

    Returns:
        PIL Image in RGB mode.

    Raises:
        ValueError: If the image cannot be loaded.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in _HEIC_EXTENSIONS:
        return _load_heic(path)
    else:
        try:
            img = Image.open(path)
            return img.convert("RGB") if img.mode != "RGB" else img
        except Exception as exc:
            raise ValueError(f"Cannot load image {path}: {exc}") from exc


def _load_heic(path: str) -> Image.Image:
    """Load a HEIC/HEIF image, with sips fallback.

    Args:
        path: Path to the HEIC file.

    Returns:
        PIL Image in RGB mode.

    Raises:
        ValueError: If loading fails.
    """
    # Try pillow-heif first
    if _HEIF_AVAILABLE:
        try:
            img = Image.open(path)
            return img.convert("RGB") if img.mode != "RGB" else img
        except Exception:
            logger.debug("pillow-heif failed for %s, falling back to sips", path, exc_info=True)

    # Fallback: convert with macOS sips command
    return _load_heic_via_sips(path)


def _load_heic_via_sips(path: str) -> Image.Image:
    """Convert HEIC to JPEG using macOS sips and load with Pillow.

    Args:
        path: Path to the HEIC file.

    Returns:
        PIL Image in RGB mode.

    Raises:
        ValueError: If sips conversion fails.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise ValueError(f"File not found: {path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "converted.jpg")
        try:
            result = subprocess.run(
                ["sips", "-s", "format", "jpeg", path, "--out", output_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                raise ValueError(f"sips conversion failed: {result.stderr}")

            img = Image.open(output_path)
            return img.convert("RGB") if img.mode != "RGB" else img.copy()
        except FileNotFoundError as exc:
            raise ValueError("sips command not found (macOS only)") from exc
        except subprocess.TimeoutExpired as exc:
            raise ValueError(f"sips conversion timed out for {path}") from exc
