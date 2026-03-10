"""Core data models used throughout ragmax."""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A raw document before chunking."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str | None = None


class Chunk(BaseModel):
    """A segment of a document, optionally with embeddings."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    document_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    index: int = 0
    embedding: list[float] | None = None
    sparse_embedding: dict[str, float] | None = None

    @property
    def token_estimate(self) -> int:
        """Rough token estimate (1 token ≈ 4 chars)."""
        return len(self.content) // 4


class SearchResult(BaseModel):
    """A chunk returned from a vector search."""

    chunk: Chunk
    score: float
    source: str = "dense"


class RerankedResult(BaseModel):
    """A search result after reranking."""

    chunk: Chunk
    score: float
    original_rank: int
    reranker_score: float


class QueryContext(BaseModel):
    """Context carried through the query pipeline."""

    query: str
    original_query: str | None = None
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    filters: dict[str, Any] | None = None


class GuardrailResult(BaseModel):
    """Result of a guardrail check."""

    passed: bool
    guardrail_name: str
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    modified_text: str | None = None


class Span(BaseModel):
    """A single timed operation within a trace."""

    name: str
    start_time: float = Field(default_factory=time.time)
    end_time: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    children: list[Span] = Field(default_factory=list)

    @property
    def duration_ms(self) -> float | None:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000


class Trace(BaseModel):
    """Full pipeline trace for observability."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    spans: list[Span] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def total_duration_ms(self) -> float | None:
        if not self.spans:
            return None
        start = min(s.start_time for s in self.spans)
        ends = [s.end_time for s in self.spans if s.end_time is not None]
        if not ends:
            return None
        return (max(ends) - start) * 1000


class GenerationResult(BaseModel):
    """The final output of a RAG query."""

    answer: str
    chunks_used: list[Chunk] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    trace: Trace | None = None
