"""ragmax — State-of-the-art RAG library.

Modular, async-first, production-grade Retrieval-Augmented Generation.

Quick start::

    from ragmax import RAG

    rag = RAG.from_config(llm="openai", store="chroma")
    await rag.ingest("./docs/")
    result = await rag.query("What is our refund policy?")
    print(result.answer)

Builder API::

    from ragmax import RAGBuilder

    rag = (RAGBuilder()
        .with_llm("anthropic", model="claude-sonnet-4-20250514")
        .with_embedder("google", model="gemini-embedding-2-preview", dimensions=768)
        .with_store("qdrant", url="localhost:6333")
        .with_chunker("semantic", threshold=0.5)
        .with_reranker("cross_encoder")
        .with_input_guardrails(["pii", "injection"])
        .with_output_guardrails(["hallucination"])
        .with_hybrid_search(alpha=0.7)
        .build())
"""

from ragmax._version import __version__
from ragmax.pipeline import RAG, RAGBuilder, RAGPipeline
from ragmax.core.models import (
    Chunk,
    Document,
    GenerationResult,
    GuardrailResult,
    QueryContext,
    RerankedResult,
    SearchResult,
)
from ragmax.core.config import RAGConfig
from ragmax.core.exceptions import (
    RagmaxError,
    ConfigError,
    DependencyMissing,
    EmbeddingError,
    GenerationError,
    GuardrailTriggered,
    ParsingError,
    RetrievalError,
    StoreError,
)

__all__ = [
    # High-level APIs
    "RAG",
    "RAGBuilder",
    "RAGPipeline",
    "RAGConfig",
    # Data models
    "Chunk",
    "Document",
    "GenerationResult",
    "GuardrailResult",
    "QueryContext",
    "RerankedResult",
    "SearchResult",
    # Exceptions
    "ConfigError",
    "DependencyMissing",
    "EmbeddingError",
    "GenerationError",
    "GuardrailTriggered",
    "ParsingError",
    "RagmaxError",
    "RetrievalError",
    "StoreError",
    # Version
    "__version__",
]
