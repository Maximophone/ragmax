"""Factory function to create a chunker from config."""

from __future__ import annotations

from typing import Any

from ragmax.core.config import ChunkerConfig
from ragmax.core.exceptions import ConfigError
from ragmax.core.protocols import Chunker


def create_chunker(config: ChunkerConfig | None = None, **kwargs: Any) -> Chunker:
    """Instantiate a Chunker from a :class:`ChunkerConfig` or keyword arguments."""
    if config is None:
        config = ChunkerConfig(**kwargs)

    strategy = config.strategy

    if strategy == "character":
        from ragmax.chunking.character import CharacterChunker

        return CharacterChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separator=config.separator or "",
        )
    elif strategy == "recursive":
        from ragmax.chunking.recursive import RecursiveChunker

        return RecursiveChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
    elif strategy == "sentence":
        from ragmax.chunking.sentence import SentenceChunker

        return SentenceChunker(**config.extra)
    elif strategy == "semantic":
        from ragmax.chunking.semantic import SemanticChunker

        return SemanticChunker(**config.extra)
    elif strategy == "context_enriched":
        from ragmax.chunking.context_enriched import ContextEnrichedChunker

        return ContextEnrichedChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            **config.extra,
        )
    elif strategy == "late":
        from ragmax.chunking.late import LateChunker

        return LateChunker(**config.extra)
    elif strategy == "agentic":
        from ragmax.chunking.agentic import AgenticChunker

        return AgenticChunker(**config.extra)
    else:
        raise ConfigError(f"Unknown chunking strategy: {strategy!r}")
