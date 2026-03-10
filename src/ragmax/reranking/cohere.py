"""Cohere reranker — cloud-based neural reranking."""

from __future__ import annotations

from ragmax.core.exceptions import RerankingError
from ragmax.core.models import RerankedResult, SearchResult
from ragmax.core.utils import require_dependency


class CohereReranker:
    """Rerank using Cohere's Rerank API.

    High quality, cloud-hosted reranking.  Default model:
    ``rerank-v3.5``.
    """

    def __init__(
        self,
        model: str = "rerank-v3.5",
        api_key: str | None = None,
    ) -> None:
        cohere = require_dependency("cohere", "cohere")
        self._client = cohere.AsyncClientV2(api_key=api_key)
        self.model = model

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[RerankedResult]:
        if not results:
            return []
        try:
            documents = [r.chunk.content for r in results]
            response = await self._client.rerank(
                query=query,
                documents=documents,
                model=self.model,
                top_n=top_k,
            )
            reranked: list[RerankedResult] = []
            for item in response.results:
                idx = item.index
                reranked.append(
                    RerankedResult(
                        chunk=results[idx].chunk,
                        score=item.relevance_score,
                        original_rank=idx,
                        reranker_score=item.relevance_score,
                    )
                )
            return reranked
        except Exception as exc:
            raise RerankingError(f"Cohere reranking failed: {exc}") from exc
