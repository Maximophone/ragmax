"""Multi-query retriever — generate query variants for broader recall."""

from __future__ import annotations

from typing import Any

from ragmax.core.models import SearchResult
from ragmax.core.protocols import Embedder, LLMProvider, VectorStore

MULTI_QUERY_PROMPT = """You are a helpful assistant that generates search queries.
Given the user's question, generate {n} different versions of the question
that capture different aspects or phrasings.  Return ONLY the queries,
one per line, with no numbering or extra text.

User question: {query}"""


class MultiQueryRetriever:
    """Generate multiple query reformulations and merge results.

    This addresses the "Missed Top-Ranked Documents" failure mode
    (Book Ch. 2) by casting a wider retrieval net.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm: LLMProvider,
        num_variants: int = 3,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.num_variants = num_variants

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        # Generate query variants
        prompt = MULTI_QUERY_PROMPT.format(n=self.num_variants, query=query)
        response = await self.llm.generate([{"role": "user", "content": prompt}])
        variants = [q.strip() for q in response.strip().split("\n") if q.strip()]
        all_queries = [query] + variants[: self.num_variants]

        # Retrieve for each variant
        seen_ids: set[str] = set()
        results: list[SearchResult] = []

        for q in all_queries:
            embeddings = await self.embedder.embed([q])
            hits = await self.store.search(embeddings[0], top_k=top_k, filters=filters)
            for hit in hits:
                if hit.chunk.id not in seen_ids:
                    seen_ids.add(hit.chunk.id)
                    results.append(hit)

        # Sort by score and return top_k
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
