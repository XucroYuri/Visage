from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from .models import ClusterResult, ImageResult


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize each embedding vector to unit length.

    After normalization, euclidean distance is monotonically related to
    cosine distance, making it suitable for DBSCAN with metric="euclidean".
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)  # avoid division by zero
    return embeddings / norms


def extract_embeddings(image_results: list[ImageResult]) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Extract all valid embeddings from image results.

    Args:
        image_results: List of ImageResult with embeddings populated.

    Returns:
        Tuple of:
        - (N, 128) numpy array of embeddings
        - List of (image_path, face_index) tuples, one per row
    """
    embeddings: list[np.ndarray] = []
    face_to_image: list[tuple[str, int]] = []

    for result in image_results:
        if result.error:
            continue
        for face in result.faces:
            if face.embedding is not None:
                embeddings.append(face.embedding)
                face_to_image.append((result.path, face.face_index))

    if not embeddings:
        return np.empty((0, 128)), []

    return np.stack(embeddings), face_to_image


def estimate_eps(embeddings: np.ndarray, k: int = 5) -> float:
    """Estimate optimal DBSCAN eps using k-distance graph elbow method.

    Computes the k-th nearest neighbor distance for each point,
    sorts them, and finds the elbow point using the maximum curvature.

    Args:
        embeddings: (N, D) array of face embeddings.
        k: Number of nearest neighbors to consider.

    Returns:
        Estimated eps value.
    """
    if len(embeddings) < k + 1:
        return 0.5  # default for very small datasets

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)

    # Use the distance to the k-th neighbor
    k_distances = np.sort(distances[:, -1])

    # Find elbow: point of maximum curvature (second derivative peak)
    # Simple approach: use the point with max perpendicular distance
    # from the line connecting first and last points
    n = len(k_distances)
    coords = np.column_stack([np.arange(n), k_distances])

    # Normalize to [0, 1] for distance calculation
    x_range = coords[-1, 0] - coords[0, 0] or 1
    y_range = coords[-1, 1] - coords[0, 1] or 1
    norm_coords = coords.copy()
    norm_coords[:, 0] /= x_range
    norm_coords[:, 1] /= y_range

    # Line from first to last point
    p1 = norm_coords[0]
    p2 = norm_coords[-1]
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec) or 1

    # Perpendicular distances
    distances_to_line = np.abs(
        np.cross(line_vec, p1 - norm_coords)
    ) / line_len

    elbow_idx = np.argmax(distances_to_line)
    return float(k_distances[elbow_idx])


def cluster_faces(
    embeddings: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 2,
    auto_eps: bool = False,
    eps_k: int = 5,
) -> ClusterResult:
    """Cluster face embeddings using DBSCAN.

    Args:
        embeddings: (N, 128) array of face embeddings.
        eps: Maximum distance between embeddings in the same cluster.
        min_samples: Minimum number of faces to form a cluster.
        auto_eps: If True, estimate eps automatically.
        eps_k: k value for eps estimation when auto_eps is True.

    Returns:
        ClusterResult with labels and statistics.
    """
    if len(embeddings) == 0:
        return ClusterResult(
            labels=np.array([], dtype=int),
            embeddings=embeddings,
            num_clusters=0,
            num_noise=0,
        )

    # L2-normalize embeddings
    normalized = _normalize_embeddings(embeddings)

    # Auto-estimate eps if requested
    if auto_eps and len(normalized) > eps_k + 1:
        eps = estimate_eps(normalized, k=eps_k)

    # Run DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = dbscan.fit_predict(normalized)

    unique_labels = set(labels)
    unique_labels.discard(-1)  # -1 is noise
    num_clusters = len(unique_labels)
    num_noise = int(np.sum(labels == -1))

    return ClusterResult(
        labels=labels,
        embeddings=normalized,
        num_clusters=num_clusters,
        num_noise=num_noise,
    )


def build_cluster_mapping(
    cluster_result: ClusterResult,
    face_to_image: list[tuple[str, int]],
) -> dict[int, list[str]]:
    """Map cluster IDs to lists of image paths.

    A single image may appear in multiple clusters if it contains
    faces from different people.

    Args:
        cluster_result: ClusterResult from cluster_faces().
        face_to_image: List of (image_path, face_index) from extract_embeddings().

    Returns:
        Dict mapping cluster_id -> sorted list of unique image paths.
    """
    mapping: dict[int, set[str]] = {}

    for i, label in enumerate(cluster_result.labels):
        if label == -1:
            continue  # skip noise
        if i >= len(face_to_image):
            continue

        image_path = face_to_image[i][0]
        if label not in mapping:
            mapping[label] = set()
        mapping[label].add(image_path)

    # Convert sets to sorted lists
    return {k: sorted(v) for k, v in sorted(mapping.items())}


def compute_cluster_confidences(
    cluster_result: ClusterResult,
) -> dict[int, float]:
    """Compute a confidence score for each cluster.

    The confidence is the average cosine similarity of each face embedding
    to the cluster centroid. Higher values indicate a more cohesive cluster.

    Args:
        cluster_result: ClusterResult with normalized embeddings and labels.

    Returns:
        Dict mapping cluster_id -> confidence score (0 to 1).
    """
    unique_labels = set(cluster_result.labels)
    unique_labels.discard(-1)

    confidences: dict[int, float] = {}
    for label in unique_labels:
        mask = cluster_result.labels == label
        cluster_embs = cluster_result.embeddings[mask]

        # Compute centroid (already L2-normalized, but re-normalize centroid)
        centroid = cluster_embs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)

        # Cosine similarity of each embedding to centroid
        similarities = cluster_embs @ centroid
        confidences[label] = float(np.clip(similarities.mean(), 0.0, 1.0))

    return confidences
