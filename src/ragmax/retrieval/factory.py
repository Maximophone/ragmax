"""Factory for creating retriever instances."""

from __future__ import annotations

from ragmax.core.config import RetrievalConfig
from ragmax.core.protocols import Embedder, LLMProvider, Retriever, VectorStore


def create_retriever(
    config: RetrievalConfig,
    embedder: Embedder,
    store: VectorStore,
    llm: LLMProvider | None = None,
) -> Retriever:
    """Build the appropriate retriever based on config."""
    if config.hyde and llm is not None:
        from ragmax.retrieval.hyde import HyDERetriever

        return HyDERetriever(embedder=embedder, store=store, llm=llm)

    if config.multi_query and llm is not None:
        from ragmax.retrieval.multi_query import MultiQueryRetriever

        return MultiQueryRetriever(embedder=embedder, store=store, llm=llm)

    if config.hybrid:
        from ragmax.retrieval.hybrid import HybridRetriever

        return HybridRetriever(
            embedder=embedder,
            store=store,
            alpha=config.hybrid_alpha,
        )

    from ragmax.retrieval.dense import DenseRetriever

    return DenseRetriever(embedder=embedder, store=store)
