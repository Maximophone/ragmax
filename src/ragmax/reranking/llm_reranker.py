"""LLM-based listwise reranker.

Uses a large language model to rerank passages in a single prompt.
Higher quality but slower (4-6 seconds).  Best used as a final
reranking stage on a small candidate set.
"""

from __future__ import annotations

import json
import re

from ragmax.core.exceptions import RerankingError
from ragmax.core.models import RerankedResult, SearchResult
from ragmax.core.protocols import LLMProvider

RERANK_PROMPT = """You are a search relevance expert. Given a query and a list of passages,
rank the passages by relevance to the query. Return a JSON array of passage indices
ordered from most to least relevant. Return ONLY the JSON array.

Query: {query}

Passages:
{passages}

Return the ranking as a JSON array of indices (0-based), e.g. [2, 0, 4, 1, 3]"""


class LLMReranker:
    """Rerank passages using an LLM as a listwise judge.

    This implements the listwise reranking approach from Ch. 6
    of the book.  Most accurate but highest latency.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[RerankedResult]:
        if not results:
            return []
        if len(results) <= 1:
            return [
                RerankedResult(
                    chunk=results[0].chunk,
                    score=results[0].score,
                    original_rank=0,
                    reranker_score=1.0,
                )
            ]

        try:
            passages = "\n".join(
                f"[{i}] {r.chunk.content[:500]}" for i, r in enumerate(results)
            )
            prompt = RERANK_PROMPT.format(query=query, passages=passages)
            response = await self.llm.generate([{"role": "user", "content": prompt}])

            # Parse the JSON array from the response
            match = re.search(r"\[[\d\s,]+\]", response)
            if match:
                ranking = json.loads(match.group())
            else:
                # Fallback to original order
                ranking = list(range(len(results)))

            reranked: list[RerankedResult] = []
            for rank, idx in enumerate(ranking):
                if idx < len(results):
                    score = 1.0 - (rank / len(ranking))
                    reranked.append(
                        RerankedResult(
                            chunk=results[idx].chunk,
                            score=score,
                            original_rank=idx,
                            reranker_score=score,
                        )
                    )

            return reranked[:top_k]
        except Exception as exc:
            raise RerankingError(f"LLM reranking failed: {exc}") from exc
