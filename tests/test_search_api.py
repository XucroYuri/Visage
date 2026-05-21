"""Tests for face search API."""

from __future__ import annotations

import numpy as np

from visage.server.search import SearchResponse, SearchResult, search_faces


class TestSearchResult:
    def test_fields(self):
        r = SearchResult(
            face_id="f1", image_path="/a.jpg",
            similarity=0.95, quality_score=0.8,
            cluster_id="c1", bbox=(10, 110, 110, 10),
        )
        assert r.face_id == "f1"
        assert r.similarity == 0.95
        assert r.bbox == (10, 110, 110, 10)


class TestSearchResponse:
    def test_defaults(self):
        resp = SearchResponse(query_face_id="q1")
        assert resp.results == []
        assert resp.total == 0
        assert resp.page == 0


class TestSearchFaces:
    def test_no_search_fn(self):
        resp = search_faces("q1")
        assert resp.total == 0

    def test_basic_search(self):
        rng = np.random.RandomState(42)
        center = rng.randn(128).astype(np.float32)
        center = center / np.linalg.norm(center)

        # Create similar vectors
        vecs = [center + 0.01 * rng.randn(128).astype(np.float32) for _ in range(5)]
        vecs = [v / np.linalg.norm(v) for v in vecs]

        def mock_search(query, top_k):
            results = []
            q = query / np.linalg.norm(query)
            for i, v in enumerate(vecs):
                score = float(np.dot(q, v))
                results.append((f"f{i}", score))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

        def mock_lookup(fid):
            return {"image_path": f"/photo/{fid}.jpg", "cluster_id": "c1"}

        resp = search_faces(
            query_face_id="query",
            query_vector=center,
            search_fn=mock_search,
            metadata_lookup=mock_lookup,
            min_score=0.5,
        )

        assert resp.query_face_id == "query"
        assert resp.total > 0
        assert all(r.similarity >= 0.5 for r in resp.results)

    def test_pagination(self):
        def mock_search(query, top_k):
            return [(f"f{i}", 0.9 - i * 0.01) for i in range(10)]

        def mock_lookup(fid):
            return {"image_path": f"/{fid}.jpg"}

        resp = search_faces(
            query_face_id="q1",
            query_vector=np.random.randn(128).astype(np.float32),
            search_fn=mock_search,
            metadata_lookup=mock_lookup,
            min_score=0.0,
            page=0,
            page_size=3,
        )
        assert resp.total == 10
        assert len(resp.results) == 3

    def test_cluster_filter(self):
        def mock_search(query, top_k):
            return [(f"f{i}", 0.9) for i in range(5)]

        def mock_lookup(fid):
            cluster = "c1" if int(fid[1:]) < 3 else "c2"
            return {"image_path": f"/{fid}.jpg", "cluster_id": cluster}

        resp = search_faces(
            query_face_id="q1",
            query_vector=np.random.randn(128).astype(np.float32),
            search_fn=mock_search,
            metadata_lookup=mock_lookup,
            cluster_id="c1",
            min_score=0.0,
        )
        assert resp.total == 3
        assert all(r.cluster_id == "c1" for r in resp.results)

    def test_min_score_filter(self):
        def mock_search(query, top_k):
            return [("f1", 0.95), ("f2", 0.5), ("f3", 0.3)]

        def mock_lookup(fid):
            return {"image_path": f"/{fid}.jpg"}

        resp = search_faces(
            query_face_id="q1",
            query_vector=np.random.randn(128).astype(np.float32),
            search_fn=mock_search,
            metadata_lookup=mock_lookup,
            min_score=0.6,
        )
        assert resp.total == 1
        assert resp.results[0].face_id == "f1"
