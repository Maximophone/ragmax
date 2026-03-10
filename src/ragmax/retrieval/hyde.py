"""HyDE retriever — Hypothetical Document Embeddings.

Generate a hypothetical answer, embed it, and use that embedding
for retrieval.  Particularly useful when the query phrasing is
very different from the document language (e.g. questions vs. statements).
"""

from __future__ import annotations

from typing import Any

from ragmax.core.models import SearchResult
from ragmax.core.protocols import Embedder, LLMProvider, VectorStore

HYDE_PROMPT = """Write a short, factual passage that would answer the following question.
Write ONLY the passage, no preamble or citations.

Question: {query}"""


class HyDERetriever:
    """Hypothetical Document Embeddings retriever.

    Instead of embedding the raw query, first ask the LLM to generate
    a hypothetical answer, then embed that answer.  This often produces
    embeddings closer to relevant documents in the vector space.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm: LLMProvider,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.llm = llm

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        # Generate hypothetical document
        prompt = HYDE_PROMPT.format(query=query)
        hypothetical = await self.llm.generate([{"role": "user", "content": prompt}])

        # Embed the hypothetical document (not the query)
        embeddings = await self.embedder.embed([hypothetical])
        return await self.store.search(embeddings[0], top_k=top_k, filters=filters)
