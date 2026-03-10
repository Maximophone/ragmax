"""Relevance guardrail — checks if retrieved chunks are relevant to the query."""

from __future__ import annotations

from ragmax.core.models import Chunk, GuardrailResult, QueryContext


class RelevanceGuardrail:
    """Check if retrieved chunks are sufficiently relevant to the query.

    Uses token overlap as a lightweight heuristic.  For production,
    consider an LLM-based relevance judge.

    Parameters
    ----------
    min_relevance : float
        Minimum relevance score (0-1) for the best chunk.
    """

    def __init__(self, min_relevance: float = 0.1) -> None:
        self.min_relevance = min_relevance

    async def check(
        self,
        response: str,
        context: QueryContext,
        chunks: list[Chunk],
    ) -> GuardrailResult:
        if not chunks:
            return GuardrailResult(
                passed=False,
                guardrail_name="relevance",
                message="No chunks retrieved",
            )

        query_tokens = set(context.query.lower().split())
        if not query_tokens:
            return GuardrailResult(passed=True, guardrail_name="relevance")

        max_relevance = 0.0
        for chunk in chunks:
            chunk_tokens = set(chunk.content.lower().split())
            if chunk_tokens:
                overlap = len(query_tokens & chunk_tokens) / len(query_tokens)
                max_relevance = max(max_relevance, overlap)

        passed = max_relevance >= self.min_relevance
        return GuardrailResult(
            passed=passed,
            guardrail_name="relevance",
            message=f"Max relevance score: {max_relevance:.2f}",
            details={"max_relevance": max_relevance, "num_chunks": len(chunks)},
        )
