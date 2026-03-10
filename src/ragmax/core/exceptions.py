"""Custom exception hierarchy for ragmax."""

from __future__ import annotations


class RagmaxError(Exception):
    """Base exception for all ragmax errors."""


class DependencyMissing(RagmaxError):
    """A required optional dependency is not installed."""

    def __init__(self, package: str, extra: str) -> None:
        self.package = package
        self.extra = extra
        super().__init__(
            f"'{package}' is required but not installed. "
            f"Install it with: pip install ragmax[{extra}]"
        )


class ConfigError(RagmaxError):
    """Invalid or missing configuration."""


class ParsingError(RagmaxError):
    """Failed to parse a document."""


class EmbeddingError(RagmaxError):
    """Failed to generate embeddings."""


class StoreError(RagmaxError):
    """Vector store operation failed."""


class RetrievalError(RagmaxError):
    """Retrieval pipeline failed."""


class GenerationError(RagmaxError):
    """LLM generation failed."""


class GuardrailTriggered(RagmaxError):
    """A guardrail check did not pass."""

    def __init__(self, guardrail_name: str, message: str) -> None:
        self.guardrail_name = guardrail_name
        super().__init__(f"Guardrail '{guardrail_name}' triggered: {message}")


class RerankingError(RagmaxError):
    """Reranking operation failed."""
