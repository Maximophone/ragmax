"""Factory for creating reranker instances."""

from __future__ import annotations

from typing import Any

from ragmax.core.config import RerankerConfig
from ragmax.core.exceptions import ConfigError
from ragmax.core.protocols import LLMProvider, Reranker


class NoOpReranker:
    """Pass-through reranker that returns results unchanged."""

    async def rerank(self, query: str, results: list, top_k: int = 5) -> list:
        from ragmax.core.models import RerankedResult

        return [
            RerankedResult(
                chunk=r.chunk,
                score=r.score,
                original_rank=i,
                reranker_score=r.score,
            )
            for i, r in enumerate(results[:top_k])
        ]


def create_reranker(
    config: RerankerConfig | None = None,
    llm: LLMProvider | None = None,
    **kwargs: Any,
) -> Reranker:
    """Instantiate a Reranker from config or keyword arguments."""
    if config is None:
        config = RerankerConfig(**kwargs)

    provider = config.provider

    if provider == "none":
        return NoOpReranker()
    elif provider == "cross_encoder":
        from ragmax.reranking.cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker(model=config.model or "cross-encoder/ms-marco-MiniLM-L-6-v2")
    elif provider == "cohere":
        from ragmax.reranking.cohere import CohereReranker

        return CohereReranker(model=config.model or "rerank-v3.5", **config.extra)
    elif provider == "llm":
        if llm is None:
            raise ConfigError("LLM reranker requires an LLM provider")
        from ragmax.reranking.llm_reranker import LLMReranker

        return LLMReranker(llm=llm)
    elif provider == "colbert":
        from ragmax.reranking.cross_encoder import CrossEncoderReranker

        # ColBERT-style models can be loaded via CrossEncoder interface
        return CrossEncoderReranker(
            model=config.model or "colbert-ir/colbertv2.0",
            **config.extra,
        )
    else:
        raise ConfigError(f"Unknown reranker provider: {provider!r}")
