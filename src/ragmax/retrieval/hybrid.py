"""Hybrid retriever combining dense and sparse (BM25) search."""

from __future__ import annotations

from typing import Any

from ragmax.core.models import Chunk, SearchResult
from ragmax.core.protocols import Embedder, VectorStore


class BM25Index:
    """Lightweight in-memory BM25 index for sparse retrieval."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._bm25 = None

    def add(self, chunks: list[Chunk]) -> None:
        self._chunks.extend(chunks)
        self._rebuild()

    def _rebuild(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return
        corpus = [c.content.lower().split() for c in self._chunks]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        if self._bm25 is None or not self._chunks:
            return []
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            SearchResult(chunk=self._chunks[idx], score=float(score), source="sparse")
            for idx, score in ranked
            if score > 0
        ]


class HybridRetriever:
    """Combine dense vector search with BM25 sparse retrieval.

    Uses a weighted reciprocal rank fusion (alpha blending) as described
    in the book's Chapter 4 on hybrid search strategies.

    Parameters
    ----------
    alpha : float
        Weight for dense scores (1-alpha for sparse).  Default 0.7.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        bm25_index: BM25Index | None = None,
        alpha: float = 0.7,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.bm25 = bm25_index or BM25Index()
        self.alpha = alpha

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        # Dense retrieval
        embeddings = await self.embedder.embed([query])
        dense_results = await self.store.search(
            embeddings[0], top_k=top_k * 2, filters=filters
        )

        # Sparse retrieval
        sparse_results = self.bm25.search(query, top_k=top_k * 2)

        # Reciprocal Rank Fusion
        return self._fuse(dense_results, sparse_results, top_k)

    def _fuse(
        self,
        dense: list[SearchResult],
        sparse: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Weighted reciprocal rank fusion."""
        k = 60  # RRF constant
        scores: dict[str, float] = {}
        chunk_map: dict[str, SearchResult] = {}

        for rank, r in enumerate(dense):
            rrf = self.alpha / (k + rank + 1)
            scores[r.chunk.id] = scores.get(r.chunk.id, 0) + rrf
            chunk_map[r.chunk.id] = r

        for rank, r in enumerate(sparse):
            rrf = (1 - self.alpha) / (k + rank + 1)
            scores[r.chunk.id] = scores.get(r.chunk.id, 0) + rrf
            if r.chunk.id not in chunk_map:
                chunk_map[r.chunk.id] = r

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            SearchResult(
                chunk=chunk_map[cid].chunk,
                score=score,
                source="hybrid",
            )
            for cid, score in ranked
        ]
