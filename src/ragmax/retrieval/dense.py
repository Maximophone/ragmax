"""Standard dense retriever — embed query, search store."""

from __future__ import annotations

from typing import Any

from ragmax.core.models import SearchResult
from ragmax.core.protocols import Embedder, VectorStore


class DenseRetriever:
    """Classic dense retrieval: embed the query, search the vector store.

    This is the default retriever used when no advanced strategy is configured.
    """

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        embeddings = await self.embedder.embed([query])
        query_embedding = embeddings[0]
        return await self.store.search(query_embedding, top_k=top_k, filters=filters)
