"""Factory for creating vector store instances."""

from __future__ import annotations

from typing import Any

from ragmax.core.config import StoreConfig
from ragmax.core.exceptions import ConfigError
from ragmax.core.protocols import VectorStore


def create_store(config: StoreConfig | None = None, **kwargs: Any) -> VectorStore:
    """Instantiate a VectorStore from config or keyword arguments."""
    if config is None:
        config = StoreConfig(**kwargs)

    provider = config.provider

    if provider == "chroma":
        from ragmax.stores.chroma import ChromaStore

        return ChromaStore(
            collection=config.collection,
            path=config.path,
            **config.extra,
        )
    elif provider == "qdrant":
        from ragmax.stores.qdrant import QdrantStore

        return QdrantStore(
            collection=config.collection,
            url=config.url,
            path=config.path,
            **config.extra,
        )
    else:
        raise ConfigError(f"Unknown store provider: {provider!r}")
