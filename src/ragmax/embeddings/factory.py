"""Factory for creating embedder instances from config."""

from __future__ import annotations

from typing import Any

from ragmax.core.config import EmbedderConfig
from ragmax.core.exceptions import ConfigError
from ragmax.core.protocols import Embedder


def create_embedder(config: EmbedderConfig | None = None, **kwargs: Any) -> Embedder:
    """Instantiate an Embedder from config or keyword arguments."""
    if config is None:
        config = EmbedderConfig(**kwargs)

    provider = config.provider

    if provider == "openai":
        from ragmax.embeddings.openai import OpenAIEmbedder

        return OpenAIEmbedder(
            model=config.model,
            dimensions=config.dimensions,
            batch_size=config.batch_size,
            **config.extra,
        )
    elif provider == "google":
        from ragmax.embeddings.google import GoogleEmbedder

        return GoogleEmbedder(
            model=config.model,
            dimensions=config.dimensions,
            batch_size=config.batch_size,
            **config.extra,
        )
    elif provider == "anthropic":
        from ragmax.embeddings.anthropic import AnthropicEmbedder

        return AnthropicEmbedder(
            model=config.model,
            batch_size=config.batch_size,
            **config.extra,
        )
    elif provider == "sentence_transformers":
        from ragmax.embeddings.sentence_transformers import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(
            model=config.model,
            batch_size=config.batch_size,
            **config.extra,
        )
    else:
        raise ConfigError(f"Unknown embedding provider: {provider!r}")
