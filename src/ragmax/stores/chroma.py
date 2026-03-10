"""ChromaDB vector store backend."""

from __future__ import annotations

from typing import Any

from ragmax.core.exceptions import StoreError
from ragmax.core.models import Chunk, SearchResult
from ragmax.core.utils import require_dependency


class ChromaStore:
    """Vector store backed by ChromaDB.

    ChromaDB is ideal for development, prototyping, and moderate-scale
    deployments.  Supports persistent local storage and client/server mode.
    """

    def __init__(
        self,
        collection: str = "ragmax",
        path: str | None = None,
        host: str | None = None,
        port: int = 8000,
    ) -> None:
        chromadb = require_dependency("chromadb", "chroma")

        if host:
            self._client = chromadb.HttpClient(host=host, port=port)
        elif path:
            self._client = chromadb.PersistentClient(path=path)
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        try:
            self._collection.upsert(
                ids=[c.id for c in chunks],
                embeddings=embeddings,
                documents=[c.content for c in chunks],
                metadatas=[
                    {
                        "document_id": c.document_id,
                        "index": c.index,
                        **{k: str(v) for k, v in c.metadata.items()},
                    }
                    for c in chunks
                ],
            )
        except Exception as exc:
            raise StoreError(f"ChromaDB upsert failed: {exc}") from exc

    async def search(
        self,
        embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        try:
            kwargs: dict[str, Any] = {
                "query_embeddings": [embedding],
                "n_results": top_k,
            }
            if filters:
                kwargs["where"] = filters

            results = self._collection.query(**kwargs)

            search_results: list[SearchResult] = []
            ids = results.get("ids", [[]])[0]
            docs = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]

            for i, (cid, doc, dist, meta) in enumerate(
                zip(ids, docs, distances, metadatas)
            ):
                chunk = Chunk(
                    id=cid,
                    content=doc or "",
                    document_id=meta.get("document_id", ""),
                    index=int(meta.get("index", 0)),
                    metadata={k: v for k, v in meta.items() if k not in ("document_id", "index")},
                )
                # ChromaDB returns distances; convert to similarity score
                score = 1.0 - dist if dist is not None else 0.0
                search_results.append(SearchResult(chunk=chunk, score=score, source="dense"))

            return search_results
        except Exception as exc:
            raise StoreError(f"ChromaDB search failed: {exc}") from exc

    async def delete(self, ids: list[str]) -> None:
        try:
            self._collection.delete(ids=ids)
        except Exception as exc:
            raise StoreError(f"ChromaDB delete failed: {exc}") from exc

    async def count(self) -> int:
        return self._collection.count()
