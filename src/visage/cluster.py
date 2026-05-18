from __future__ import annotations

import logging

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from .models import ClusterResult, ImageResult

logger = logging.getLogger(__name__)

# HDBSCAN is available in scikit-learn >= 1.3
try:
    from sklearn.cluster import HDBSCAN
    _HDBSCAN_AVAILABLE = True
except ImportError:
    _HDBSCAN_AVAILABLE = False


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize each embedding vector to unit length.

    After normalization, euclidean distance is monotonically related to
    cosine distance, making it suitable for DBSCAN with metric="euclidean".
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)  # avoid division by zero
    return embeddings / norms


def extract_embeddings(
    image_results: list[ImageResult],
    embedding_dim: int = 128,
) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Extract all valid embeddings from image results.

    Args:
        image_results: List of ImageResult with embeddings populated.
        embedding_dim: Dimensionality of the embedding vectors (default 128 for dlib).

    Returns:
        Tuple of:
        - (N, D) numpy array of embeddings
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
        return np.empty((0, embedding_dim)), []

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

    # Perpendicular distances (explicit 2D formula to avoid np.cross deprecation)
    # |line_vec x (p1 - point)| / |line_vec| = |dx*(y1-yy) - dy*(x1-xx)| / line_len
    dx, dy = line_vec
    distances_to_line = np.abs(
        dx * (p1[1] - norm_coords[:, 1]) - dy * (p1[0] - norm_coords[:, 0])
    ) / line_len

    elbow_idx = np.argmax(distances_to_line)
    return float(k_distances[elbow_idx])


def cluster_faces(
    embeddings: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 3,
    auto_eps: bool = False,
    eps_k: int = 5,
    cluster_method: str = "hdbscan",
    min_cluster_size: int = 2,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: str = "eom",
    distance_matrix: np.ndarray | None = None,
) -> ClusterResult:
    """Cluster face embeddings using DBSCAN or HDBSCAN.

    Args:
        embeddings: (N, D) array of face embeddings.
        eps: Maximum distance between embeddings in the same cluster (DBSCAN only).
        min_samples: Minimum number of faces to form a cluster.
        auto_eps: If True, estimate eps automatically (DBSCAN only).
        eps_k: k value for eps estimation when auto_eps is True.
        cluster_method: "dbscan" or "hdbscan".
        min_cluster_size: Minimum cluster size for HDBSCAN.
        cluster_selection_epsilon: Distance threshold for HDBSCAN cluster selection
            (0 = disabled; >0 may trigger sklearn Cython bug with certain data).
        cluster_selection_method: "eom" (stable, fewer clusters) or "leaf" (fine-grained).
        distance_matrix: Optional precomputed (N, N) distance matrix for HDBSCAN.

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

    if cluster_method == "hdbscan":
        # HDBSCAN requires at least 2 samples
        if len(normalized) < 2:
            return ClusterResult(
                labels=np.full(len(normalized), -1, dtype=int),
                embeddings=normalized,
                num_clusters=0,
                num_noise=len(normalized),
            )
        return _cluster_hdbscan(
            normalized,
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method=cluster_selection_method,
            distance_matrix=distance_matrix,
        )

    # Default: DBSCAN
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


def _cluster_hdbscan(
    normalized: np.ndarray,
    min_samples: int = 3,
    min_cluster_size: int = 2,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: str = "eom",
    distance_matrix: np.ndarray | None = None,
) -> ClusterResult:
    """Cluster using HDBSCAN with tunable parameters.

    HDBSCAN automatically adapts to varying cluster densities and
    does not require an eps parameter.

    Args:
        normalized: (N, D) array of L2-normalized embeddings.
        min_samples: Minimum samples parameter for HDBSCAN.
        min_cluster_size: Minimum size of a cluster.
        cluster_selection_epsilon: Distance threshold for cluster merging.
        cluster_selection_method: "eom" (stable) or "leaf" (fine-grained).
        distance_matrix: Optional precomputed (N, N) distance matrix.
    """
    if not _HDBSCAN_AVAILABLE:
        raise RuntimeError(
            "HDBSCAN not available. Requires scikit-learn >= 1.3. "
            "Install it: pip install scikit-learn>=1.3"
        )

    if distance_matrix is not None:
        clusterer = HDBSCAN(
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method=cluster_selection_method,
            metric="precomputed",
            copy=False,
        )
        labels = clusterer.fit_predict(distance_matrix)
    else:
        clusterer = HDBSCAN(
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method=cluster_selection_method,
            metric="euclidean",
            copy=False,
        )
        labels = clusterer.fit_predict(normalized)

    unique_labels = set(labels)
    unique_labels.discard(-1)
    num_clusters = len(unique_labels)
    num_noise = int(np.sum(labels == -1))

    probabilities = None
    if hasattr(clusterer, "probabilities_"):
        probabilities = clusterer.probabilities_

    return ClusterResult(
        labels=labels,
        embeddings=normalized,
        num_clusters=num_clusters,
        num_noise=num_noise,
        probabilities=probabilities,
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


def compute_composite_distance(
    face_embeddings: np.ndarray,
    head_features: np.ndarray,
    head_weight: float = 0.2,
) -> np.ndarray:
    """Compute a composite distance matrix from face embeddings and head features.

    Combines L2-normalized face embedding distances with head feature distances
    using a weighted sum. The face embeddings should already be L2-normalized.

    Args:
        face_embeddings: (N, D) array of L2-normalized face embeddings.
        head_features: (N, H) array of head feature vectors.
        head_weight: Weight for head feature distance (0 = face only, 1 = head only).

    Returns:
        (N, N) symmetric distance matrix.
    """
    face_weight = 1.0 - head_weight

    # Face embedding distance: euclidean on L2-normalized = sqrt(2 - 2*cos_sim)
    # Use cosine distance directly for better scaling
    face_sim = face_embeddings @ face_embeddings.T
    face_dist = np.clip(1.0 - face_sim, 0.0, 2.0)

    if head_weight <= 0.0 or head_features.shape[1] == 0:
        return face_dist

    # Normalize head features to unit length
    norms = np.linalg.norm(head_features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    head_norm = head_features / norms

    # Head feature distance: cosine distance
    head_sim = head_norm @ head_norm.T
    head_dist = np.clip(1.0 - head_sim, 0.0, 2.0)

    return face_weight * face_dist + head_weight * head_dist


def merge_clusters(
    cluster_result: ClusterResult,
    merge_threshold: float = 0.85,
    min_reliable_size: int = 10,
    small_merge_threshold: float = 0.75,
) -> ClusterResult:
    """Merge clusters whose centroids are above a cosine similarity threshold.

    Uses iterative greedy merging: repeatedly merges the most similar pair,
    recomputes the merged centroid, and continues until no pair exceeds the
    threshold. This avoids the transitive chain-merging problem (where A~B
    and B~C causes all three to merge even if A and C are dissimilar).

    Uses size-aware thresholding: when either cluster is below
    min_reliable_size, uses small_merge_threshold (more permissive) instead
    of merge_threshold. Small clusters have unreliable centroids (2-4 faces
    produce unstable means), so they need a lower bar to merge into larger
    clusters without collapsing large clusters together.

    Args:
        cluster_result: ClusterResult from cluster_faces().
        merge_threshold: Cosine similarity threshold for reliable clusters.
        min_reliable_size: Clusters below this size use the relaxed threshold.
        small_merge_threshold: Relaxed threshold when one cluster is small.

    Returns:
        New ClusterResult with merged labels and updated statistics.
    """
    if merge_threshold <= 0.0 or cluster_result.num_clusters <= 1:
        return cluster_result

    unique_labels = set(cluster_result.labels)
    unique_labels.discard(-1)
    labels = cluster_result.labels.copy()
    label_list = sorted(unique_labels)

    if len(label_list) <= 1:
        return cluster_result

    # Store embeddings per cluster for fast centroid recomputation
    cluster_embeddings: dict[int, np.ndarray] = {}
    for label in label_list:
        mask = labels == label
        cluster_embeddings[label] = cluster_result.embeddings[mask]

    # Compute initial centroids
    centroids: dict[int, np.ndarray] = {}
    for label in label_list:
        centroid = cluster_embeddings[label].mean(axis=0)
        centroids[label] = centroid / (np.linalg.norm(centroid) + 1e-10)

    active = set(label_list)
    merges_performed = 0

    def _threshold_for(size_a: int, size_b: int) -> float:
        """Use relaxed threshold when either cluster is too small for a
        reliable centroid."""
        if size_a < min_reliable_size or size_b < min_reliable_size:
            return small_merge_threshold
        return merge_threshold

    # Iterative greedy merging: find best pair, merge, recompute centroid
    while len(active) > 1:
        best_sim = -1.0
        best_pair: tuple[int, int] | None = None
        sorted_active = sorted(active)
        for i, label_a in enumerate(sorted_active):
            size_a = cluster_embeddings[label_a].shape[0]
            for label_b in sorted_active[i + 1:]:
                sim = float(centroids[label_a] @ centroids[label_b])
                if sim <= best_sim:
                    continue
                size_b = cluster_embeddings[label_b].shape[0]
                needed = _threshold_for(size_a, size_b)
                if sim < needed:
                    continue
                best_sim = sim
                best_pair = (label_a, label_b)

        if best_pair is None:
            break

        a, b = best_pair
        # Merge b into a
        merged_embs = np.vstack([cluster_embeddings[a], cluster_embeddings[b]])
        cluster_embeddings[a] = merged_embs
        centroid = merged_embs.mean(axis=0)
        centroids[a] = centroid / (np.linalg.norm(centroid) + 1e-10)

        active.discard(b)
        del centroids[b]
        del cluster_embeddings[b]
        labels[labels == b] = a
        merges_performed += 1

        logger.debug(
            "Merging cluster %d into %d (similarity: %.3f, sizes: %d+%d)",
            b, a, best_sim, merged_embs.shape[0] - cluster_embeddings[a].shape[0],
            cluster_embeddings[a].shape[0],
        )

    # Renumber remaining labels to be sequential starting from 0
    remaining = sorted(active)
    old_to_new: dict[int, int] = {-1: -1}
    for new_id, old_label in enumerate(remaining):
        old_to_new[old_label] = new_id

    new_labels = np.array([old_to_new.get(label, -1) for label in labels], dtype=int)

    unique_new = set(new_labels)
    unique_new.discard(-1)
    num_clusters = len(unique_new)
    num_noise = int(np.sum(new_labels == -1))

    if merges_performed > 0:
        logger.info(
            "Post-clustering merge: %d → %d clusters (%d merges)",
            len(label_list), num_clusters, merges_performed,
        )

    return ClusterResult(
        labels=new_labels,
        embeddings=cluster_result.embeddings,
        num_clusters=num_clusters,
        num_noise=num_noise,
        probabilities=cluster_result.probabilities,
    )
