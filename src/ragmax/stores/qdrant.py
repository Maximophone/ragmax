"""Qdrant vector store backend."""

from __future__ import annotations

from typing import Any

from ragmax.core.exceptions import StoreError
from ragmax.core.models import Chunk, SearchResult
from ragmax.core.utils import require_dependency


class QdrantStore:
    """Vector store backed by Qdrant.

    Qdrant supports HNSW indexing, scalar/binary quantization,
    pre-filtering, and hybrid search via sparse vectors.
    Ideal for production workloads.
    """

    def __init__(
        self,
        collection: str = "ragmax",
        url: str | None = None,
        path: str | None = None,
        api_key: str | None = None,
        dimension: int = 1536,
        on_disk: bool = False,
    ) -> None:
        qdrant_mod = require_dependency("qdrant_client", "qdrant")
        from qdrant_client.models import Distance, VectorParams

        if url:
            self._client = qdrant_mod.QdrantClient(url=url, api_key=api_key)
        elif path:
            self._client = qdrant_mod.QdrantClient(path=path)
        else:
            self._client = qdrant_mod.QdrantClient(":memory:")

        self._collection = collection
        self._dimension = dimension

        # Create collection if it doesn't exist
        collections = [c.name for c in self._client.get_collections().collections]
        if collection not in collections:
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE,
                    on_disk=on_disk,
                ),
            )

    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        from qdrant_client.models import PointStruct

        try:
            points = [
                PointStruct(
                    id=c.id,
                    vector=emb,
                    payload={
                        "content": c.content,
                        "document_id": c.document_id,
                        "index": c.index,
                        **c.metadata,
                    },
                )
                for c, emb in zip(chunks, embeddings)
            ]
            self._client.upsert(collection_name=self._collection, points=points)
        except Exception as exc:
            raise StoreError(f"Qdrant upsert failed: {exc}") from exc

    async def search(
        self,
        embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        try:
            query_filter = None
            if filters:
                conditions = [
                    FieldCondition(key=k, match=MatchValue(value=v))
                    for k, v in filters.items()
                ]
                query_filter = Filter(must=conditions)

            hits = self._client.query_points(
                collection_name=self._collection,
                query=embedding,
                limit=top_k,
                query_filter=query_filter,
            ).points

            results: list[SearchResult] = []
            for hit in hits:
                payload = hit.payload or {}
                chunk = Chunk(
                    id=str(hit.id),
                    content=payload.get("content", ""),
                    document_id=payload.get("document_id", ""),
                    index=payload.get("index", 0),
                    metadata={
                        k: v
                        for k, v in payload.items()
                        if k not in ("content", "document_id", "index")
                    },
                )
                results.append(SearchResult(chunk=chunk, score=hit.score, source="dense"))

            return results
        except Exception as exc:
            raise StoreError(f"Qdrant search failed: {exc}") from exc

    async def delete(self, ids: list[str]) -> None:
        from qdrant_client.models import PointIdsList

        try:
            self._client.delete(
                collection_name=self._collection,
                points_selector=PointIdsList(points=ids),
            )
        except Exception as exc:
            raise StoreError(f"Qdrant delete failed: {exc}") from exc

    async def count(self) -> int:
        info = self._client.get_collection(self._collection)
        return info.points_count or 0
