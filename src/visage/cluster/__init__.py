"""Clustering engine — HDBSCAN/DBSCAN clustering and incremental assignment."""

from visage.cluster.core import (
    _normalize_embeddings,
    build_cluster_mapping,
    cluster_faces,
    compute_cluster_confidences,
    compute_composite_distance,
    compute_composite_distance_chunked,
    estimate_eps,
    extract_embeddings,
    merge_clusters,
)

__all__ = [
    "_normalize_embeddings",
    "build_cluster_mapping",
    "cluster_faces",
    "compute_cluster_confidences",
    "compute_composite_distance",
    "compute_composite_distance_chunked",
    "estimate_eps",
    "extract_embeddings",
    "merge_clusters",
]
