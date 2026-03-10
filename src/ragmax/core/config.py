"""Configuration models for the RAG pipeline."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class EmbedderConfig(BaseModel):
    """Configuration for an embedding provider."""

    provider: Literal["openai", "anthropic", "google", "sentence_transformers"] = "openai"
    model: str = "text-embedding-3-small"
    dimensions: int | None = None
    batch_size: int = 64
    extra: dict[str, Any] = Field(default_factory=dict)


class StoreConfig(BaseModel):
    """Configuration for a vector store backend."""

    provider: Literal["qdrant", "chroma"] = "chroma"
    collection: str = "ragmax"
    url: str | None = None
    path: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ChunkerConfig(BaseModel):
    """Configuration for a chunking strategy."""

    strategy: Literal[
        "character",
        "recursive",
        "sentence",
        "semantic",
        "context_enriched",
        "late",
        "agentic",
    ] = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64
    separator: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class RerankerConfig(BaseModel):
    """Configuration for a reranker."""

    provider: Literal[
        "cross_encoder",
        "cohere",
        "colbert",
        "llm",
        "none",
    ] = "none"
    model: str | None = None
    top_k: int = 5
    extra: dict[str, Any] = Field(default_factory=dict)


class LLMConfig(BaseModel):
    """Configuration for the generation LLM."""

    provider: Literal["openai", "anthropic", "google"] = "openai"
    model: str = "gpt-5.4"
    temperature: float = 0.1
    max_tokens: int = 2048
    extra: dict[str, Any] = Field(default_factory=dict)


class RetrievalConfig(BaseModel):
    """Configuration for retrieval behaviour."""

    top_k: int = 10
    hybrid: bool = False
    hybrid_alpha: float = 0.7
    filters: dict[str, Any] | None = None
    multi_query: bool = False
    hyde: bool = False


class GuardrailsConfig(BaseModel):
    """Configuration for input/output guardrails."""

    input_guardrails: list[str] = Field(default_factory=list)
    output_guardrails: list[str] = Field(default_factory=list)


class RAGConfig(BaseModel):
    """Top-level configuration for the entire RAG pipeline."""

    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RAGConfig:
        return cls.model_validate(data)
