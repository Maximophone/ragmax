"""Protocol definitions for all pluggable components."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from ragmax.core.models import (
    Chunk,
    Document,
    GuardrailResult,
    QueryContext,
    RerankedResult,
    SearchResult,
)


@runtime_checkable
class Embedder(Protocol):
    """Produces vector embeddings from text."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts and return their vectors."""
        ...

    @property
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Stores and retrieves vector embeddings."""

    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Insert or update chunks with their embeddings."""
        ...

    async def search(
        self,
        embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Find the nearest chunks to the given embedding."""
        ...

    async def delete(self, ids: list[str]) -> None:
        """Delete chunks by their IDs."""
        ...


@runtime_checkable
class Retriever(Protocol):
    """Retrieves relevant chunks for a query."""

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Return the most relevant chunks for the given query."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Reranks search results for improved relevance ordering."""

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[RerankedResult]:
        """Rerank results and return the top_k most relevant."""
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """Generates text using a large language model."""

    async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Generate a single response from the given messages."""
        ...

    async def generate_stream(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream response tokens from the given messages."""
        ...


@runtime_checkable
class Chunker(Protocol):
    """Splits documents into chunks."""

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into a list of chunks."""
        ...


@runtime_checkable
class Parser(Protocol):
    """Parses raw files into Documents."""

    def parse(self, path: str) -> Document:
        """Parse a file and return a Document."""
        ...

    def supports(self, path: str) -> bool:
        """Return True if this parser can handle the given file."""
        ...


@runtime_checkable
class InputGuardrail(Protocol):
    """Validates user queries before processing."""

    async def check(self, query: str, context: QueryContext) -> GuardrailResult:
        """Check a query and return pass/fail with details."""
        ...


@runtime_checkable
class OutputGuardrail(Protocol):
    """Validates generated responses before returning them."""

    async def check(
        self,
        response: str,
        context: QueryContext,
        chunks: list[Chunk],
    ) -> GuardrailResult:
        """Check a generated response and return pass/fail with details."""
        ...
