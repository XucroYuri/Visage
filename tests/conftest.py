from __future__ import annotations

import numpy as np
import pytest

from visage.models import DetectedFace, FaceBox, ImageResult

# ── Custom pytest markers ───────────────────────────────────────────


def pytest_configure(config):
    config.addinivalue_line("markers", "vision: requires macOS Vision framework")
    config.addinivalue_line("markers", "scrfd: requires insightface package")
    config.addinivalue_line("markers", "yunet: requires opencv with FaceDetectorYN")
    config.addinivalue_line("markers", "desktop: requires Tauri desktop runtime")


# ── Seed for reproducibility ──────────────────────────────────────
np.random.seed(42)


@pytest.fixture
def sample_face_box() -> FaceBox:
    """A 100x100 face box."""
    return FaceBox(top=10, right=110, bottom=110, left=10)


@pytest.fixture
def wide_face_box() -> FaceBox:
    """A 200x50 face box (width > height)."""
    return FaceBox(top=0, right=200, bottom=50, left=0)


@pytest.fixture
def narrow_face_box() -> FaceBox:
    """A 50x200 face box (height > width)."""
    return FaceBox(top=0, right=50, bottom=200, left=0)


@pytest.fixture
def sample_embedding() -> np.ndarray:
    """A deterministic 128-dim embedding vector."""
    return np.random.randn(128).astype(np.float64)


@pytest.fixture
def sample_detected_face(sample_face_box: FaceBox, sample_embedding: np.ndarray) -> DetectedFace:
    """A detected face with embedding populated."""
    return DetectedFace(
        face_box=sample_face_box,
        confidence=0.95,
        embedding=sample_embedding.copy(),
        image_path="/tmp/test.jpg",
        face_index=0,
    )


@pytest.fixture
def sample_image_result(sample_detected_face: DetectedFace) -> ImageResult:
    """An ImageResult with one detected face."""
    return ImageResult(path="/tmp/test.jpg", faces=[sample_detected_face])


@pytest.fixture
def empty_image_result() -> ImageResult:
    """An ImageResult with no faces (skipped)."""
    return ImageResult(path="/tmp/empty.jpg", skipped=True)


@pytest.fixture
def error_image_result() -> ImageResult:
    """An ImageResult with an error."""
    return ImageResult(path="/tmp/error.jpg", error="load failed")


@pytest.fixture
def multi_face_image_result() -> ImageResult:
    """An ImageResult with two detected faces."""
    face1 = DetectedFace(
        face_box=FaceBox(top=10, right=110, bottom=110, left=10),
        confidence=0.9,
        embedding=np.random.randn(128).astype(np.float64),
        image_path="/tmp/multi.jpg",
        face_index=0,
    )
    face2 = DetectedFace(
        face_box=FaceBox(top=200, right=300, bottom=300, left=200),
        confidence=0.85,
        embedding=np.random.randn(128).astype(np.float64),
        image_path="/tmp/multi.jpg",
        face_index=1,
    )
    return ImageResult(path="/tmp/multi.jpg", faces=[face1, face2])


# ── Clustering test data ──────────────────────────────────────────


@pytest.fixture
def clusterable_embeddings() -> tuple[np.ndarray, list[tuple[str, int]]]:
    """20 embeddings in two well-separated clusters + face_to_image mapping.

    Returns:
        (embeddings (20, 128), face_to_image list of 20 entries).
    """
    np.random.seed(42)
    centroid_a = np.random.randn(128).astype(np.float64)
    centroid_b = np.random.randn(128).astype(np.float64)
    # Push centroids apart
    centroid_b += 3.0

    cluster_a: list[np.ndarray] = []
    cluster_b: list[np.ndarray] = []
    for _ in range(10):
        cluster_a.append(centroid_a + 0.1 * np.random.randn(128).astype(np.float64))
        cluster_b.append(centroid_b + 0.1 * np.random.randn(128).astype(np.float64))

    embeddings = np.array(cluster_a + cluster_b)

    face_to_image = [(f"/tmp/photo_{i}.jpg", 0) for i in range(20)]
    # Make the last 5 photos of each cluster share image paths to test multi-face
    for i in range(5, 10):
        face_to_image[i] = (f"/tmp/shared_{i}.jpg", 0)
        face_to_image[i + 10] = (f"/tmp/shared_{i}.jpg", 1)

    return embeddings, face_to_image


# ── Filesystem fixtures ───────────────────────────────────────────


@pytest.fixture
def tmp_image_dir(tmp_path):
    """Create a temp directory with various image files and subdirectories.

    Structure:
        tmp_path/
        ├── photo1.jpg
        ├── photo2.png
        ├── photo3.JPG  (uppercase)
        ├── doc.pdf      (unsupported)
        ├── .DS_Store     (hidden)
        ├── subdir/
        │   └── photo4.heic
        ├── .hidden/
        │   └── hidden_photo.jpg
        └── visage_output/
            └── output_photo.jpg
    """
    # Root-level files
    (tmp_path / "photo1.jpg").write_text("fake-jpeg")
    (tmp_path / "photo2.png").write_text("fake-png")
    (tmp_path / "photo3.JPG").write_text("fake-jpeg-upper")
    (tmp_path / "doc.pdf").write_text("not an image")
    (tmp_path / ".DS_Store").write_text("hidden file")

    # Normal subdirectory
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "photo4.heic").write_text("fake-heic")

    # Hidden subdirectory (should be skipped)
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "hidden_photo.jpg").write_text("hidden")

    # visage_output directory (should be skipped)
    vo = tmp_path / "visage_output"
    vo.mkdir()
    (vo / "output_photo.jpg").write_text("output")

    # .visage_cache (should be skipped)
    vc = tmp_path / ".visage_cache"
    vc.mkdir()
    (vc / "embeddings.db").write_text("cache")

    return tmp_path


@pytest.fixture
def real_image_path(tmp_path):
    """Create a small real JPEG image for PIL-based tests."""
    from PIL import Image

    path = tmp_path / "real.jpg"
    img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    img.save(path, "JPEG")
    return str(path)


@pytest.fixture
def real_png_path(tmp_path):
    """Create a small real PNG image for PIL-based tests."""
    from PIL import Image

    path = tmp_path / "real.png"
    img = Image.new("RGBA", (320, 240), color=(255, 0, 0, 128))
    img.save(path, "PNG")
    return str(path)
