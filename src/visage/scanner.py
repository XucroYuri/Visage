from __future__ import annotations

import os
from pathlib import Path

from .config import SUPPORTED_EXTENSIONS


def scan_images(
    root_path: str,
    extensions: frozenset[str] = SUPPORTED_EXTENSIONS,
    skip_dirs: frozenset[str] = frozenset({"visage_output", ".visage_cache"}),
) -> list[str]:
    """Walk directory tree and return sorted list of image file paths.

    Args:
        root_path: Root directory to scan.
        extensions: File extensions to include (case-insensitive).
        skip_dirs: Directory names to skip during traversal.

    Returns:
        Sorted list of absolute image file paths.
    """
    root = Path(root_path).resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    images: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden and output directories (modifies in-place to prune os.walk)
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".") and d not in skip_dirs
        ]
        for filename in filenames:
            if is_supported_image(filename, extensions):
                images.append(str(Path(dirpath) / filename))

    images.sort()
    return images


def is_supported_image(
    filename: str,
    extensions: frozenset[str] = SUPPORTED_EXTENSIONS,
) -> bool:
    """Check if a filename has a supported image extension.

    Args:
        filename: The filename (not full path) to check.
        extensions: Allowed extensions, lowercase with dot prefix.

    Returns:
        True if the file extension is supported.
    """
    if filename.startswith("."):
        return False
    ext = os.path.splitext(filename)[1].lower()
    return ext in extensions
